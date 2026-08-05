"""
Değerlendirme metrikleri.

TEMEL İLKE: RETRIEVAL VE GENERATION AYRI ÖLÇÜLÜR.

  Bir RAG sistemi iki farklı sebeple kötü cevap verir:
    A) Doğru chunk hiç getirilmedi  → retrieval hatası
    B) Doğru chunk getirildi ama model kötü kullandı → generation hatası

  Bu ikisi tek bir "doğruluk" sayısında birleştirilirse hangisini
  iyileştireceğini bilemezsin. Daha kötüsü: retrieval bozukken prompt'u
  iyileştirmeye çalışırsın ve hiçbir şey değişmez — çünkü model elinde
  olmayan bilgiyi üretemez.

  Bu yüzden ÖNCE retrieval ölçülür. Recall@k düşükse generation
  metriklerine bakmak anlamsızdır.

METRİKLERİN OKUNUŞU:
  Recall@k → "doğru kaynağı adaylar arasına alabildim mi?" (kaçırma oranı)
  MRR      → "doğru kaynak kaçıncı sıradaydı?" (üstte olmayı ödüllendirir)
  Bu ikisi farklı şeyler ölçer: Recall 1.0 ama MRR 0.2 ise doğru sonuç
  hep listenin dibinde demektir — reranker'ın çözeceği tam problem budur.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from rag_assistant.domain.models import Answer, ScoredChunk
from rag_assistant.evaluation.golden_set import GoldenQuestion, QuestionKind
from rag_assistant.text import tr_lower


# ---------------------------------------------------------------------------
# Retrieval metrikleri
# ---------------------------------------------------------------------------
def first_relevant_rank(
    question: GoldenQuestion, retrieved: list[ScoredChunk]
) -> int | None:
    """
    Beklenen kaynaklardan ilkinin kaçıncı sırada geldiği (1'den başlar).

    Bulunamazsa None. MRR hesabı buna dayanır.
    """
    for rank, scored in enumerate(retrieved, start=1):
        src = scored.chunk.source
        if any(exp.matches(src.file_name, src.page) for exp in question.expected_sources):
            return rank
    return None


def recall_at_k(question: GoldenQuestion, retrieved: list[ScoredChunk], k: int) -> float:
    """
    Beklenen kaynaklardan kaçı ilk k sonuçta var? (0-1)

    Not: "en az biri" değil, ORAN ölçüyoruz. Bir soru üç sayfaya dayanıyorsa
    ve yalnızca biri getirildiyse cevap eksik kalabilir — bunu 1.0 saymak
    yanıltıcı olur.
    """
    if not question.expected_sources:
        return 0.0

    top = retrieved[:k]
    found = sum(
        1
        for exp in question.expected_sources
        if any(exp.matches(s.chunk.source.file_name, s.chunk.source.page) for s in top)
    )
    return found / len(question.expected_sources)


def hit_at_k(question: GoldenQuestion, retrieved: list[ScoredChunk], k: int) -> bool:
    """İlk k sonuçta beklenen kaynaklardan EN AZ BİRİ var mı?"""
    rank = first_relevant_rank(question, retrieved[:k])
    return rank is not None


def reciprocal_rank(question: GoldenQuestion, retrieved: list[ScoredChunk]) -> float:
    """1/sıra. Bulunamazsa 0. Ortalaması MRR'dir."""
    rank = first_relevant_rank(question, retrieved)
    return 1.0 / rank if rank else 0.0


def precision_at_k(question: GoldenQuestion, retrieved: list[ScoredChunk], k: int) -> float:
    """İlk k sonucun kaçı gerçekten ilgili? (gürültü ölçüsü)"""
    top = retrieved[:k]
    if not top:
        return 0.0
    relevant = sum(
        1
        for s in top
        if any(
            exp.matches(s.chunk.source.file_name, s.chunk.source.page)
            for exp in question.expected_sources
        )
    )
    return relevant / len(top)


# ---------------------------------------------------------------------------
# Cevap (generation) metrikleri
# ---------------------------------------------------------------------------
def contains_all(text: str, needles: tuple[str, ...]) -> tuple[bool, list[str]]:
    """
    Beklenen ifadelerin hepsi metinde var mı?

    Karşılaştırma Türkçe-güvenli küçük harfle yapılır (`tr_lower`):
    `str.lower()` "İ" harfini bozar ve eşleşme sessizce başarısız olur.
    """
    haystack = tr_lower(text)
    missing = [n for n in needles if tr_lower(n) not in haystack]
    return not missing, missing


def contains_any(text: str, needles: tuple[str, ...]) -> tuple[bool, list[str]]:
    """Yasaklı ifadelerden herhangi biri metinde var mı?"""
    haystack = tr_lower(text)
    found = [n for n in needles if tr_lower(n) in haystack]
    return bool(found), found


@dataclass(frozen=True, slots=True)
class QuestionResult:
    """Tek bir sorunun değerlendirme sonucu."""

    question: GoldenQuestion
    answer: Answer

    # Retrieval
    recall_at_k: float
    hit: bool
    reciprocal_rank: float
    precision_at_k: float
    first_relevant_rank: int | None

    # Generation
    facts_present: bool
    missing_facts: list[str]
    forbidden_present: bool
    forbidden_found: list[str]

    # Atıfların DOĞRU kaynağı gösterip göstermediği.
    #
    # NEDEN AYRI BİR ÖLÇÜM: atıfın VARLIĞI ile DOĞRULUĞU farklı şeylerdir.
    # Model doğru cevabı verip yanlış sayfayı gösterebilir (ölçüldü: "65M USD"
    # cevabı doğruydu ama s.3 yerine s.2'ye atıf yapıldı).
    # Yanlış kaynağa yapılan atıf, atıfsız cevaptan DAHA KÖTÜDÜR: doğrulanabilir
    # görünür ama denetleyen kişi gösterilen yerde bilgiyi bulamaz ve sisteme
    # olan güven tamamen kaybolur.
    citations_correct: bool = True
    wrong_citations: list[str] = field(default_factory=list)

    @property
    def abstention_correct(self) -> bool:
        """
        Çekimserlik kararı doğru mu?

        Bu, sistemin en önemli davranışsal metriğidir:
          * Cevaplanamaz soruda çekimser kaldıysa → DOĞRU
          * Cevaplanabilir soruda cevap verdiyse  → DOĞRU
        Tersleri iki farklı hata: biri uydurma, diğeri gereksiz suskunluk.
        """
        if self.question.kind is QuestionKind.UNANSWERABLE:
            return self.answer.abstained
        return not self.answer.abstained

    @property
    def hallucinated(self) -> bool:
        """
        Uydurdu mu?

        İki işaretten biri yeterli:
          * Cevaplanamaz soruya cevap verdi (çekimser kalmadı)
          * Yasaklı ifade cevapta geçti (dünya bilgisinden konuştu)
        """
        answered_unanswerable = (
            self.question.kind is QuestionKind.UNANSWERABLE and not self.answer.abstained
        )
        return answered_unanswerable or self.forbidden_present

    @property
    def cited(self) -> bool:
        """Cevaplanabilir soruda atıf verdi mi? (çekimserlikte anlamsız)"""
        if self.question.kind is QuestionKind.UNANSWERABLE:
            return True
        return len(self.answer.citations) > 0

    @property
    def passed(self) -> bool:
        """
        Bu soru genel olarak geçti mi?

        Sıkı tanım: çekimserlik doğru, uydurma yok, gerekli olgular var,
        atıf var. "Kısmen doğru" diye bir şey yok — sistem ya güvenilir ya değil.
        """
        return (
            self.abstention_correct
            and not self.hallucinated
            and self.facts_present
            and self.cited
            and self.citations_correct
        )


@dataclass(slots=True)
class EvaluationSummary:
    """Tüm koşunun özeti."""

    results: list[QuestionResult] = field(default_factory=list)
    k: int = 5

    # Koşu bağlamı — SONUÇLARIN KARŞILAŞTIRILABİLİR OLMASI İÇİN ZORUNLU.
    # Hangi ayarla ölçüldüğü kaydedilmezse iki koşuyu kıyaslamak anlamsızdır.
    config_snapshot: dict[str, object] = field(default_factory=dict)
    duration_seconds: float = 0.0

    # ---- Retrieval ----
    @property
    def mean_recall(self) -> float:
        return self._mean(r.recall_at_k for r in self._answerable)

    @property
    def hit_rate(self) -> float:
        return self._mean(float(r.hit) for r in self._answerable)

    @property
    def mrr(self) -> float:
        return self._mean(r.reciprocal_rank for r in self._answerable)

    @property
    def mean_precision(self) -> float:
        return self._mean(r.precision_at_k for r in self._answerable)

    # ---- Generation ----
    @property
    def abstention_accuracy(self) -> float:
        return self._mean(float(r.abstention_correct) for r in self.results)

    @property
    def hallucination_rate(self) -> float:
        """En kritik metrik: kaç cevapta uydurma tespit edildi?"""
        return self._mean(float(r.hallucinated) for r in self.results)

    @property
    def citation_rate(self) -> float:
        """Atıf VAR mı? (varlık)"""
        return self._mean(float(r.cited) for r in self._answerable)

    @property
    def citation_accuracy(self) -> float:
        """Atıflar DOĞRU kaynağı mı gösteriyor? (doğruluk — varlıktan farklı)"""
        return self._mean(float(r.citations_correct) for r in self._answerable)

    @property
    def fact_accuracy(self) -> float:
        return self._mean(float(r.facts_present) for r in self._answerable)

    @property
    def pass_rate(self) -> float:
        return self._mean(float(r.passed) for r in self.results)

    @property
    def mean_latency_ms(self) -> float:
        return self._mean(float(r.answer.latency_ms) for r in self.results)

    # ---- Yardımcılar ----
    @property
    def _answerable(self) -> list[QuestionResult]:
        return [r for r in self.results if r.question.kind is QuestionKind.ANSWERABLE]

    @property
    def failures(self) -> list[QuestionResult]:
        return [r for r in self.results if not r.passed]

    @staticmethod
    def _mean(values) -> float:  # type: ignore[no-untyped-def] # noqa: ANN001
        items = list(values)
        return sum(items) / len(items) if items else 0.0

    def to_dict(self) -> dict[str, object]:
        """Diske yazılabilir özet — koşular arası karşılaştırma için."""
        return {
            "k": self.k,
            "config": self.config_snapshot,
            "duration_seconds": round(self.duration_seconds, 2),
            "question_count": len(self.results),
            "answerable_count": len(self._answerable),
            "unanswerable_count": len(self.results) - len(self._answerable),
            "retrieval": {
                "recall_at_k": round(self.mean_recall, 4),
                "hit_rate": round(self.hit_rate, 4),
                "mrr": round(self.mrr, 4),
                "precision_at_k": round(self.mean_precision, 4),
            },
            "generation": {
                "abstention_accuracy": round(self.abstention_accuracy, 4),
                "hallucination_rate": round(self.hallucination_rate, 4),
                "citation_rate": round(self.citation_rate, 4),
                "citation_accuracy": round(self.citation_accuracy, 4),
                "fact_accuracy": round(self.fact_accuracy, 4),
            },
            "overall": {
                "pass_rate": round(self.pass_rate, 4),
                "mean_latency_ms": round(self.mean_latency_ms, 1),
            },
        }
