"""
Faz 7 testleri: golden set doğrulama ve metrik hesabı.

Metriklerin kendisi test edilmeli — çünkü METRİK BOZUKSA ÖLÇÜM BOZUKTUR
ve bunu fark etmek neredeyse imkânsızdır. Yanlış bir Recall hesabı sistemi
olduğundan iyi/kötü gösterir; sen de yanlış yerde emek harcarsın.

Ölçü aletini kalibre etmeden ölçüm yapılmaz.
"""

from __future__ import annotations

import pytest

from rag_assistant.domain.models import (
    Answer,
    Chunk,
    Citation,
    RetrievalStage,
    ScoredChunk,
    SourceRef,
)
from rag_assistant.evaluation.golden_set import (
    ExpectedSource,
    GoldenQuestion,
    GoldenSet,
    QuestionKind,
)
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


def chunk_on(file: str, page: int) -> ScoredChunk:
    c = Chunk(text=f"{file} sayfa {page} metni", source=SourceRef(file, page), token_count=5)
    return ScoredChunk(c, 0.5, RetrievalStage.FUSED, 1)


def question(
    *,
    kind: QuestionKind = QuestionKind.ANSWERABLE,
    sources: tuple[ExpectedSource, ...] = (ExpectedSource("a.pdf", 1),),
    must_contain: tuple[str, ...] = (),
    must_not_contain: tuple[str, ...] = (),
) -> GoldenQuestion:
    return GoldenQuestion(
        id="q",
        question="soru?",
        kind=kind,
        expected_sources=() if kind is QuestionKind.UNANSWERABLE else sources,
        must_contain=must_contain,
        must_not_contain=must_not_contain,
    )


def answer(*, text: str = "Cevap [1].", abstained: bool = False, citations: int = 1) -> Answer:
    src = SourceRef("a.pdf", 1)
    return Answer(
        question="soru?",
        text=text,
        citations=tuple(Citation(i, src, "cid") for i in range(1, citations + 1)),
        used_chunks=(chunk_on("a.pdf", 1),),
        abstained=abstained,
        model="m",
        prompt_version="v1",
        latency_ms=10,
    )


# ---------------------------------------------------------------------------
# Beklenen kaynak eşleştirme — chunking'den bağımsız çıpa
# ---------------------------------------------------------------------------
class TestExpectedSource:
    def test_dosya_ve_sayfa_eslesir(self) -> None:
        assert ExpectedSource("a.pdf", 3).matches("a.pdf", 3)

    def test_yanlis_sayfa_eslesmez(self) -> None:
        assert not ExpectedSource("a.pdf", 3).matches("a.pdf", 4)

    def test_yanlis_dosya_eslesmez(self) -> None:
        assert not ExpectedSource("a.pdf", 3).matches("b.pdf", 3)

    def test_sayfa_none_ise_yalnizca_dosyaya_bakilir(self) -> None:
        """Bazı sorular için 'şu dokümandan gelmesi yeter' demek yeterlidir."""
        exp = ExpectedSource("a.pdf", None)
        assert exp.matches("a.pdf", 1)
        assert exp.matches("a.pdf", 99)
        assert not exp.matches("b.pdf", 1)


# ---------------------------------------------------------------------------
# Retrieval metrikleri
# ---------------------------------------------------------------------------
class TestRetrievalMetrics:
    def test_ilk_ilgili_sira(self) -> None:
        q = question(sources=(ExpectedSource("a.pdf", 2),))
        retrieved = [chunk_on("a.pdf", 1), chunk_on("a.pdf", 2), chunk_on("a.pdf", 3)]
        assert first_relevant_rank(q, retrieved) == 2

    def test_bulunamazsa_none(self) -> None:
        q = question(sources=(ExpectedSource("a.pdf", 9),))
        assert first_relevant_rank(q, [chunk_on("a.pdf", 1)]) is None

    def test_recall_oran_olarak_hesaplanir(self) -> None:
        """
        'En az biri' değil ORAN. Üç sayfaya dayanan bir soruda yalnızca biri
        getirildiyse cevap eksik kalabilir; bunu 1.0 saymak yanıltıcı olur.
        """
        q = question(
            sources=(
                ExpectedSource("a.pdf", 1),
                ExpectedSource("a.pdf", 2),
                ExpectedSource("a.pdf", 3),
            )
        )
        retrieved = [chunk_on("a.pdf", 1), chunk_on("a.pdf", 2), chunk_on("b.pdf", 9)]
        assert recall_at_k(q, retrieved, 3) == pytest.approx(2 / 3)

    def test_recall_k_ile_sinirlanir(self) -> None:
        q = question(sources=(ExpectedSource("a.pdf", 3),))
        retrieved = [chunk_on("a.pdf", 1), chunk_on("a.pdf", 2), chunk_on("a.pdf", 3)]
        assert recall_at_k(q, retrieved, 2) == 0.0
        assert recall_at_k(q, retrieved, 3) == 1.0

    def test_hit_en_az_biri(self) -> None:
        q = question(sources=(ExpectedSource("a.pdf", 1), ExpectedSource("a.pdf", 5)))
        assert hit_at_k(q, [chunk_on("a.pdf", 1)], 5) is True

    def test_reciprocal_rank(self) -> None:
        q = question(sources=(ExpectedSource("a.pdf", 3),))
        retrieved = [chunk_on("a.pdf", 1), chunk_on("a.pdf", 2), chunk_on("a.pdf", 3)]
        assert reciprocal_rank(q, retrieved) == pytest.approx(1 / 3)

    def test_bulunamazsa_rr_sifir(self) -> None:
        q = question(sources=(ExpectedSource("z.pdf", 1),))
        assert reciprocal_rank(q, [chunk_on("a.pdf", 1)]) == 0.0

    def test_recall_ve_mrr_farkli_seyler_olcer(self) -> None:
        """
        Bu ayrım kritik: Recall 1.0 ama MRR düşükse doğru sonuç listenin
        dibinde demektir — reranker'ın çözeceği problem budur.
        """
        q = question(sources=(ExpectedSource("a.pdf", 5),))
        retrieved = [chunk_on("a.pdf", i) for i in (1, 2, 3, 4, 5)]
        assert recall_at_k(q, retrieved, 5) == 1.0
        assert reciprocal_rank(q, retrieved) == pytest.approx(0.2)

    def test_precision_gurultuyu_olcer(self) -> None:
        q = question(sources=(ExpectedSource("a.pdf", 1),))
        retrieved = [chunk_on("a.pdf", 1), chunk_on("b.pdf", 1), chunk_on("c.pdf", 1)]
        assert precision_at_k(q, retrieved, 3) == pytest.approx(1 / 3)


# ---------------------------------------------------------------------------
# Metin kontrolleri — Türkçe güvenli
# ---------------------------------------------------------------------------
class TestTextChecks:
    def test_tum_olgular_var(self) -> None:
        ok, missing = contains_all("17 PPE sınıfı ve 9 sektör", ("17", "9"))
        assert ok and missing == []

    def test_eksik_olgu_bildirilir(self) -> None:
        ok, missing = contains_all("17 PPE sınıfı", ("17", "65M"))
        assert not ok and missing == ["65M"]

    def test_turkce_buyuk_harf_duyarsiz(self) -> None:
        """
        `str.lower()` "İ" harfini bozar ve eşleşme sessizce başarısız olur.
        Metrik bu yüzden Türkçe-güvenli katlama kullanır.
        """
        ok, _ = contains_all("BİLGİ İSTANBUL", ("bilgi", "istanbul"))
        assert ok

    def test_yasakli_ifade_yakalanir(self) -> None:
        found, items = contains_any("Titanik 1912 yılında battı", ("1912",))
        assert found and items == ["1912"]

    def test_yasakli_ifade_yoksa(self) -> None:
        found, items = contains_any("Bu bilgi dokümanlarda yok", ("1912",))
        assert not found and items == []


# ---------------------------------------------------------------------------
# Karar mantığı: çekimserlik ve uydurma
# ---------------------------------------------------------------------------
def result(
    *, kind: QuestionKind, abstained: bool, forbidden: bool = False, facts: bool = True
) -> QuestionResult:
    q = question(kind=kind)
    return QuestionResult(
        question=q,
        answer=answer(abstained=abstained, citations=0 if abstained else 1),
        recall_at_k=1.0,
        hit=True,
        reciprocal_rank=1.0,
        precision_at_k=1.0,
        first_relevant_rank=1,
        facts_present=facts,
        missing_facts=[] if facts else ["x"],
        forbidden_present=forbidden,
        forbidden_found=["1912"] if forbidden else [],
    )


class TestDecisionLogic:
    def test_cevaplanamazda_cekimserlik_dogru(self) -> None:
        r = result(kind=QuestionKind.UNANSWERABLE, abstained=True)
        assert r.abstention_correct and not r.hallucinated and r.passed

    def test_cevaplanamazda_cevap_vermek_uydurmadir(self) -> None:
        """En kritik tespit: bağlamda yokken cevap üretmek."""
        r = result(kind=QuestionKind.UNANSWERABLE, abstained=False)
        assert not r.abstention_correct
        assert r.hallucinated
        assert not r.passed

    def test_cevaplanabilirde_gereksiz_cekimserlik(self) -> None:
        """Uydurma değil ama ayrı bir hata: bilgi varken 'bilmiyorum' demek."""
        r = result(kind=QuestionKind.ANSWERABLE, abstained=True)
        assert not r.abstention_correct
        assert not r.hallucinated
        assert not r.passed

    def test_yasakli_ifade_uydurma_sayilir(self) -> None:
        """Dünya bilgisi bağlamı ezmiş demektir."""
        r = result(kind=QuestionKind.ANSWERABLE, abstained=False, forbidden=True)
        assert r.hallucinated
        assert not r.passed

    def test_cekimserlikte_atif_beklenmez(self) -> None:
        r = result(kind=QuestionKind.UNANSWERABLE, abstained=True)
        assert r.cited, "çekimser cevapta atıf aranmamalı"

    def test_eksik_olgu_gecersiz_kilar(self) -> None:
        r = result(kind=QuestionKind.ANSWERABLE, abstained=False, facts=False)
        assert not r.passed


# ---------------------------------------------------------------------------
# Özet: cevaplanamaz sorular retrieval metriklerine karışmamalı
# ---------------------------------------------------------------------------
class TestSummary:
    def test_retrieval_metrikleri_yalnizca_cevaplanabilirlerden(self) -> None:
        """
        Cevaplanamaz sorunun beklenen kaynağı yoktur; recall'ı 0'dır.
        Ortalamaya katılırsa retrieval olduğundan kötü görünür ve yanlış
        yerde emek harcarsın.
        """
        s = EvaluationSummary(
            results=[
                result(kind=QuestionKind.ANSWERABLE, abstained=False),
                result(kind=QuestionKind.UNANSWERABLE, abstained=True),
            ]
        )
        assert s.mean_recall == 1.0, "cevaplanamaz soru recall ortalamasını bozmuş"
        assert s.abstention_accuracy == 1.0
        assert s.hallucination_rate == 0.0

    def test_uydurma_orani_tum_sorulardan(self) -> None:
        s = EvaluationSummary(
            results=[
                result(kind=QuestionKind.ANSWERABLE, abstained=False),
                result(kind=QuestionKind.UNANSWERABLE, abstained=False),
            ]
        )
        assert s.hallucination_rate == pytest.approx(0.5)

    def test_ozet_serilestirilebilir(self) -> None:
        """Koşular arası karşılaştırma için diske yazılabilir olmalı."""
        s = EvaluationSummary(
            results=[result(kind=QuestionKind.ANSWERABLE, abstained=False)],
            config_snapshot={"embedder": "test"},
        )
        d = s.to_dict()
        assert d["retrieval"]["recall_at_k"] == 1.0
        assert d["config"]["embedder"] == "test"

    def test_basarisizlar_listelenir(self) -> None:
        s = EvaluationSummary(
            results=[
                result(kind=QuestionKind.ANSWERABLE, abstained=False),
                result(kind=QuestionKind.UNANSWERABLE, abstained=False),
            ]
        )
        assert len(s.failures) == 1


# ---------------------------------------------------------------------------
# Golden set doğrulama — ölçü aletinin kalibrasyonu
# ---------------------------------------------------------------------------
class TestGoldenSetValidation:
    def test_negatif_ornek_yoksa_uyarir(self) -> None:
        """Negatif örnek olmadan uydurma oranı ÖLÇÜLEMEZ."""
        gs = GoldenSet(questions=[question()])
        assert any("negatif" in p for p in gs.validate())

    def test_cevaplanabilirde_kaynak_zorunlu(self) -> None:
        gs = GoldenSet(
            questions=[
                GoldenQuestion("q1", "s?", QuestionKind.ANSWERABLE),
                GoldenQuestion("u1", "s?", QuestionKind.UNANSWERABLE),
            ]
        )
        assert any("beklenen kaynak yok" in p for p in gs.validate())

    def test_cevaplanamazda_kaynak_celiskidir(self) -> None:
        gs = GoldenSet(
            questions=[
                GoldenQuestion(
                    "u1",
                    "s?",
                    QuestionKind.UNANSWERABLE,
                    expected_sources=(ExpectedSource("a.pdf", 1),),
                )
            ]
        )
        assert any("çelişki" in p for p in gs.validate())

    def test_tekrarlanan_id_yakalanir(self) -> None:
        q1 = GoldenQuestion("ayni", "s?", QuestionKind.ANSWERABLE, (ExpectedSource("a.pdf"),))
        gs = GoldenSet(questions=[q1, q1])
        assert any("tekrarlanan id" in p for p in gs.validate())

    def test_diske_yazip_okuma(self, tmp_path) -> None:  # type: ignore[no-untyped-def]
        gs = GoldenSet(
            questions=[
                GoldenQuestion(
                    "q1",
                    "SmartSafe kaç PPE sınıfı?",
                    QuestionKind.ANSWERABLE,
                    (ExpectedSource("a.pdf", 1),),
                    must_contain=("17",),
                ),
                GoldenQuestion(
                    "u1", "Titanik?", QuestionKind.UNANSWERABLE, must_not_contain=("1912",)
                ),
            ],
            description="test",
        )
        gs.save(tmp_path)
        loaded = GoldenSet.load(tmp_path)
        assert len(loaded.questions) == 2
        assert loaded.questions[0].must_contain == ("17",)
        assert loaded.questions[1].kind is QuestionKind.UNANSWERABLE
        assert loaded.validate() == []

    def test_gercek_golden_set_gecerli(self) -> None:
        """Projenin kendi golden set'i tutarlı mı? (regresyon koruması)"""
        from rag_assistant.config import get_settings

        gs = GoldenSet.load(get_settings().paths.eval_dir)
        assert gs.validate() == []
        assert len(gs.unanswerable) >= 3, "yeterli negatif örnek yok"
