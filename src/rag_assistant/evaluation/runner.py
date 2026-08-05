"""
Değerlendirme koşucusu.

NE YAPAR:
  Golden set'teki her soruyu sisteme sorar, retrieval ve cevap metriklerini
  hesaplar, sonucu diske yazar.

NEDEN KOŞU BAĞLAMI (config snapshot) KAYDEDİLİR:
  "Recall@5 = 0.82" tek başına hiçbir şey ifade etmez. Hangi embedding
  modeliyle? Hangi chunk boyutuyla? Reranker açık mıydı? Bu bilgiler
  kaydedilmezse iki koşuyu karşılaştırmak imkânsızdır ve "iyileştirdim"
  iddiası kanıtlanamaz.

  Bu, değerlendirmenin en çok atlanan ve en çok zarar veren eksiğidir:
  insanlar ölçer, sonra ayar değiştirir, tekrar ölçer — ama arada BAŞKA
  şeyler de değiştiği için farkı neyin yarattığını bilemez.

RETRIEVAL İKİ KEZ ÇAĞRILMAZ:
  Cevap üretimi zaten retrieval yapıyor. Metrikler için ayrıca retrieve
  etmek hem iki kat yavaş olur hem de (rastgelelik varsa) farklı sonuç
  ölçmene yol açar. Bu yüzden `Answer.used_chunks` üzerinden ölçüyoruz —
  yani modele GERÇEKTEN giden bağlamı.
"""

from __future__ import annotations

import json
import time
from datetime import UTC, datetime
from pathlib import Path

from rag_assistant.composition import RagSystem
from rag_assistant.evaluation.golden_set import GoldenQuestion, GoldenSet
from rag_assistant.evaluation.metrics import (
    EvaluationSummary,
    QuestionResult,
    contains_all,
    contains_any,
    first_relevant_rank,
    hit_at_k,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
)
from rag_assistant.observability import get_logger

logger = get_logger(__name__)


class EvaluationError(RuntimeError):
    """Değerlendirme koşulamadı."""


def _config_snapshot(system: RagSystem) -> dict[str, object]:
    """
    Koşuyu tekrar üretilebilir kılan tüm ayarlar.

    Buradaki her alan, sonucu etkileyebilecek bir karardır. Biri değişirse
    iki koşu karşılaştırılamaz — rapor bunu görünür kılar.
    """
    s = system.settings
    reranker = getattr(system.retriever, "reranker", None)
    reranker_active = bool(reranker) and bool(getattr(reranker, "is_active", True))

    return {
        "embedder": s.embedding.model,
        "embedder_max_tokens": s.embedding.max_tokens,
        "chunk_target_tokens": s.chunking.target_tokens,
        "chunk_overlap_tokens": s.chunking.overlap_tokens,
        "retrieval_strategy": system.retriever.name,
        "fetch_k": s.retrieval.fetch_k,
        "top_k": s.retrieval.top_k,
        "rrf_k": s.retrieval.rrf_k,
        "reranker": (reranker.model_id if reranker_active else None),
        "reranker_configured_but_inactive": bool(reranker) and not reranker_active,
        "llm_provider": s.llm.provider,
        "llm_model": s.llm.model,
        "llm_temperature": s.llm.temperature,
        "prompt_version": s.generation.prompt_version,
        "index_vectors": system.store.count,
    }


def evaluate_question(
    system: RagSystem, question: GoldenQuestion, *, k: int
) -> QuestionResult:
    """Tek soruyu değerlendir."""
    answer = system.answerer.answer(question.question, top_k=k)

    # Modele GERÇEKTEN giden bağlam üzerinden ölçüyoruz (yeniden retrieve yok).
    retrieved = list(answer.used_chunks)

    facts_present, missing = contains_all(answer.text, question.must_contain)
    forbidden_present, forbidden = contains_any(answer.text, question.must_not_contain)

    # ATIFLAR DOĞRU KAYNAĞI MI GÖSTERİYOR?
    # Atıf varlığı yeterli değil: model doğru cevabı verip yanlış sayfaya
    # atıf yapabilir. Bu, doğrulanabilir GÖRÜNEN ama denetlendiğinde
    # tutmayan bir cevap üretir — güveni tamamen yıkan hata biçimi.
    wrong_citations: list[str] = []
    if question.expected_sources:
        for citation in answer.citations:
            src = citation.source
            if not any(
                exp.matches(src.file_name, src.page) for exp in question.expected_sources
            ):
                wrong_citations.append(f"[{citation.marker}] {src.label()}")

    result = QuestionResult(
        question=question,
        answer=answer,
        recall_at_k=recall_at_k(question, retrieved, k),
        hit=hit_at_k(question, retrieved, k),
        reciprocal_rank=reciprocal_rank(question, retrieved),
        precision_at_k=precision_at_k(question, retrieved, k),
        first_relevant_rank=first_relevant_rank(question, retrieved),
        facts_present=facts_present,
        missing_facts=missing,
        forbidden_present=forbidden_present,
        forbidden_found=forbidden,
        citations_correct=not wrong_citations,
        wrong_citations=wrong_citations,
    )

    logger.info(
        "eval.question_done",
        id=question.id,
        kind=str(question.kind),
        passed=result.passed,
        recall=round(result.recall_at_k, 3),
        rank=result.first_relevant_rank,
        abstained=answer.abstained,
        hallucinated=result.hallucinated,
        citations=len(answer.citations),
        wrong_citations=len(wrong_citations),
        ms=answer.latency_ms,
    )
    return result


def run_evaluation(
    system: RagSystem,
    golden_set: GoldenSet,
    *,
    k: int | None = None,
    strict: bool = True,
) -> EvaluationSummary:
    """
    Golden set'in tamamını koş.

    Args:
        k: değerlendirmede kullanılacak top_k. None ise yapılandırmadaki değer.
        strict: golden set'in kendisi tutarsızsa hata ver.
            Ölçü aleti bozuksa ölçüm yapmak, yanlış sonuca güvenmekten
            daha kötüdür — bu yüzden varsayılan True.
    """
    problems = golden_set.validate()
    if problems:
        message = "Golden set tutarsız:\n  - " + "\n  - ".join(problems)
        if strict:
            raise EvaluationError(message)
        logger.warning("eval.golden_set_problems", problems=problems)

    if system.store.count == 0:
        raise EvaluationError("Index boş — önce ingestion çalıştırın.")

    effective_k = k or system.settings.retrieval.top_k
    started = time.perf_counter()

    logger.info(
        "eval.started",
        questions=len(golden_set.questions),
        answerable=len(golden_set.answerable),
        unanswerable=len(golden_set.unanswerable),
        k=effective_k,
    )

    results = [
        evaluate_question(system, q, k=effective_k) for q in golden_set.questions
    ]

    summary = EvaluationSummary(
        results=results,
        k=effective_k,
        config_snapshot=_config_snapshot(system),
        duration_seconds=time.perf_counter() - started,
    )

    logger.info(
        "eval.finished",
        pass_rate=round(summary.pass_rate, 3),
        recall=round(summary.mean_recall, 3),
        mrr=round(summary.mrr, 3),
        hallucination_rate=round(summary.hallucination_rate, 3),
        seconds=round(summary.duration_seconds, 1),
    )
    return summary


def save_summary(summary: EvaluationSummary, directory: Path) -> Path:
    """
    Özeti zaman damgalı bir dosyaya yaz.

    Zaman damgası neden: koşular ÜST ÜSTE YAZILMAZ. Geçmiş koşular
    saklanmazsa "geriledi mi?" sorusu cevaplanamaz. Değerlendirmenin değeri
    tek bir sayıda değil, sayının ZAMAN İÇİNDEKİ hareketindedir.
    """
    directory.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    path = directory / f"eval_{stamp}.json"

    payload = summary.to_dict()
    payload["questions"] = [
        {
            "id": r.question.id,
            "kind": str(r.question.kind),
            "question": r.question.question,
            "passed": r.passed,
            "answer": r.answer.text,
            "abstained": r.answer.abstained,
            "hallucinated": r.hallucinated,
            "citations": [c.source.label() for c in r.answer.citations],
            "retrieved_sources": [
                s.chunk.source.label() for s in r.answer.used_chunks
            ],
            "recall_at_k": round(r.recall_at_k, 4),
            "first_relevant_rank": r.first_relevant_rank,
            "missing_facts": r.missing_facts,
            "wrong_citations": r.wrong_citations,
            "forbidden_found": r.forbidden_found,
            "latency_ms": r.answer.latency_ms,
        }
        for r in summary.results
    ]

    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("eval.summary_saved", path=str(path))
    return path
