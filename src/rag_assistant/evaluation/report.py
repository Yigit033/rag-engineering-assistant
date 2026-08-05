"""
Değerlendirme raporu (terminal).

TASARIM İLKESİ: RAPOR TEK BİR SAYI GÖSTERMEZ.

  "%82 doğruluk" bir rapor değil, bir slogandır. Eyleme dönüştürülebilir
  rapor şu soruyu cevaplar: "Şimdi neyi düzeltmeliyim?"

  Bunun için üç şey gerekir:
    1. Retrieval ve generation AYRI gösterilir → hangi katman bozuk?
    2. BAŞARISIZ sorular tek tek listelenir → hangi vaka bozuk?
    3. Koşu ayarları yazılır → bu sonuç neyle elde edildi?

  Ayrıca teşhis satırı ekliyoruz: metriklerin BİRLİKTE okunuşundan çıkan
  yorum. Örneğin Recall yüksek + MRR düşük = doğru sonuç listenin dibinde
  → reranker'ın çözeceği problem. Bu yorumu raporun kendisi yapmalı;
  kullanıcının metrik ilişkilerini ezberlemesi gerekmemeli.
"""

from __future__ import annotations

from rag_assistant.evaluation.metrics import EvaluationSummary
from rag_assistant.evaluation.golden_set import QuestionKind

_LINE = "─" * 78


def _bar(value: float, width: int = 20) -> str:
    filled = round(max(0.0, min(1.0, value)) * width)
    return "█" * filled + "░" * (width - filled)


def _metric(label: str, value: float, *, invert: bool = False) -> str:
    """
    Metrik satırı.

    `invert=True`: düşük olması İYİ olan metrikler için (uydurma oranı).
    Aynı görsel dilde yüksek çubuğun bazen iyi bazen kötü olması kafa
    karıştırır; bu yüzden yön açıkça işaretleniyor.
    """
    good = (1 - value) if invert else value
    mark = "✓" if good >= 0.8 else "!" if good >= 0.5 else "✗"
    return f"  {mark} {label:<26} {_bar(value)} {value:6.1%}"


def _diagnose(s: EvaluationSummary) -> list[str]:
    """Metriklerin birlikte okunuşundan çıkan teşhisler."""
    notes: list[str] = []

    if s.hallucination_rate > 0:
        notes.append(
            f"UYDURMA VAR ({s.hallucination_rate:.0%}). Diğer her şeyden önce bu "
            "düzeltilir: prompt'ta çekimser kalma izni ve 'sadece bağlamı kullan' "
            "kuralı yeterince net mi?"
        )

    if s.hit_rate < 0.8:
        notes.append(
            f"RETRIEVAL DARBOĞAZ (hit oranı {s.hit_rate:.0%}). Doğru chunk adaylar "
            "arasına girmiyor — prompt iyileştirmek BOŞA emek. Önce fetch_k'yı "
            "artır, chunk boyutunu ve embedding modelini gözden geçir."
        )
    elif s.mrr < 0.5 <= s.hit_rate:
        notes.append(
            f"SIRALAMA PROBLEMİ (hit {s.hit_rate:.0%} ama MRR {s.mrr:.2f}). Doğru "
            "sonuç getiriliyor ama listenin dibinde kalıyor — reranker'ın çözeceği "
            "tam problem budur."
        )

    if s.mean_precision < 0.4 and s.hit_rate >= 0.8:
        notes.append(
            f"BAĞLAM GÜRÜLTÜLÜ (precision {s.mean_precision:.0%}). Doğru chunk var "
            "ama yanında çok alakasız chunk gidiyor; 'lost in the middle' riski. "
            "top_k'yı düşürmeyi veya reranker eşiği koymayı dene."
        )

    if s.citation_accuracy < 0.9:
        notes.append(
            f"ATIF YANLIŞ ({s.citation_accuracy:.0%} doğru). Cevap doğru olsa bile "
            "yanlış sayfa gösteriliyor; denetleyen kişi bilgiyi bulamaz. Bağlamdaki "
            "numaralandırma prompt'ta daha belirgin vurgulanmalı."
        )

    if s.citation_rate < 0.9:
        notes.append(
            f"ATIF EKSİK ({s.citation_rate:.0%}). Atıfsız cevap doğrulanamaz; "
            "prompt'ta atıf zorunluluğu yeterince baskın değil."
        )

    if s.fact_accuracy < 0.8 and s.hit_rate >= 0.8:
        notes.append(
            "BİLGİ KAYBI: doğru bağlam gidiyor ama beklenen olgular cevapta yok. "
            "Bu bir GENERATION problemi — prompt veya model kapasitesi."
        )

    if not notes:
        notes.append("Belirgin bir darboğaz yok. İyileştirme için golden set'i büyüt.")

    return notes


def render_report(summary: EvaluationSummary) -> str:
    """Terminal için okunabilir rapor üret."""
    out: list[str] = []
    a = len([r for r in summary.results if r.question.kind is QuestionKind.ANSWERABLE])
    u = len(summary.results) - a

    out.append(_LINE)
    out.append("  DEĞERLENDİRME RAPORU")
    out.append(_LINE)
    out.append(
        f"  {len(summary.results)} soru ({a} cevaplanabilir, {u} cevaplanamaz) · "
        f"top_k={summary.k} · {summary.duration_seconds:.1f}s"
    )
    out.append("")

    # ---- Koşu ayarları: sonucun neyle elde edildiği
    out.append("  KOŞU AYARLARI (sonuçlar ancak aynı ayarlarla karşılaştırılabilir)")
    cfg = summary.config_snapshot
    for key in (
        "embedder",
        "chunk_target_tokens",
        "retrieval_strategy",
        "fetch_k",
        "top_k",
        "reranker",
        "llm_model",
        "prompt_version",
        "index_vectors",
    ):
        if key in cfg:
            out.append(f"    {key:<24} {cfg[key]}")
    if cfg.get("reranker_configured_but_inactive"):
        out.append("    ⚠ reranker yapılandırılmış ama AKTİF DEĞİL (belleğe sığmadı)")
    out.append("")

    # ---- Retrieval önce: hataların çoğu buradadır
    out.append("  1) RETRIEVAL  (doğru chunk getirildi mi?)")
    out.append(_metric("Hit@k (en az bir kaynak)", summary.hit_rate))
    out.append(_metric("Recall@k (kaynak oranı)", summary.mean_recall))
    out.append(_metric("MRR (sıra kalitesi)", summary.mrr))
    out.append(_metric("Precision@k (gürültüsüzlük)", summary.mean_precision))
    out.append("")

    out.append("  2) GENERATION  (bağlam doğru kullanıldı mı?)")
    out.append(_metric("Çekimserlik doğruluğu", summary.abstention_accuracy))
    out.append(_metric("Uydurma oranı (düşük iyi)", summary.hallucination_rate, invert=True))
    out.append(_metric("Atıf oranı (var mı)", summary.citation_rate))
    out.append(_metric("Atıf doğruluğu (doğru mu)", summary.citation_accuracy))
    out.append(_metric("Olgu doğruluğu", summary.fact_accuracy))
    out.append("")

    out.append("  3) GENEL")
    out.append(_metric("Geçme oranı", summary.pass_rate))
    out.append(f"    ortalama gecikme        {summary.mean_latency_ms:.0f} ms")
    out.append("")

    # ---- Başarısızlıklar tek tek: "neyi düzelteyim?"
    failures = summary.failures
    if failures:
        out.append(f"  BAŞARISIZ SORULAR ({len(failures)})")
        for r in failures:
            out.append(f"    [{r.question.id}] {r.question.question[:60]}")
            reasons: list[str] = []
            if not r.abstention_correct:
                reasons.append(
                    "cevaplanamaz soruya cevap verdi"
                    if r.question.kind is QuestionKind.UNANSWERABLE
                    else "cevaplanabilir soruda gereksiz çekimser kaldı"
                )
            if r.forbidden_found:
                reasons.append(f"yasaklı ifade: {r.forbidden_found}")
            if r.missing_facts:
                reasons.append(f"eksik olgu: {r.missing_facts}")
            if not r.cited:
                reasons.append("atıf yok")
            if r.wrong_citations:
                reasons.append(f"yanlış kaynağa atıf: {r.wrong_citations}")
            if not r.hit and r.question.kind is QuestionKind.ANSWERABLE:
                reasons.append("doğru kaynak hiç getirilmedi")
            for reason in reasons:
                out.append(f"        → {reason}")
        out.append("")

    out.append("  TEŞHİS")
    for note in _diagnose(summary):
        out.append(f"    • {note}")
    out.append(_LINE)

    return "\n".join(out)
