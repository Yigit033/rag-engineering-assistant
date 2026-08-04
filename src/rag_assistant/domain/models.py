"""
Alan (domain) modelleri — sistemin ortak dili.

TASARIM KURALI: Bu dosya HİÇBİR üçüncü parti kütüphaneye bağlı değildir.
Ne FAISS, ne sentence-transformers, ne FastAPI, ne LangChain.

Neden bu kadar katı:
  Domain modelleri sistemin en içteki katmanıdır ve her katman ona bakar.
  Buraya bir kütüphane sızarsa, o kütüphaneyi değiştirmek TÜM projeyi
  değiştirmek anlamına gelir. Örneğin LangChain'in `Document` sınıfını
  domain modeli olarak kullanan bir proje, LangChain'e ömür boyu bağlıdır.

`frozen=True`: modeller değişmez (immutable). Bir chunk üretildikten sonra
kimse içeriğini değiştiremez — pipeline'da sessiz mutasyon hatalarını
tamamen ortadan kaldırır.

`slots=True`: bellek tasarrufu ve yanlış alan adına yazmayı engelleme.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum


class DocumentStatus(StrEnum):
    """
    Bir kaynak dokümanın işlenme sonucu.

    NO_TEXT_LAYER ayrı bir durum olarak var, çünkü "0 chunk üretildi"
    bir BAŞARI değildir. Taranmış PDF'ler buraya düşer ve OCR gerektirir.
    Bu ayrımı yapmayan sistemlerde o dosyalar sessizce kaybolur.
    """

    OK = "ok"
    NO_TEXT_LAYER = "no_text_layer"
    FAILED = "failed"
    SKIPPED = "skipped"  # değişmemiş, yeniden işlenmedi (idempotency)


class RetrievalStage(StrEnum):
    """Bir sonucun hangi aşamadan geldiği — gözlemlenebilirlik için."""

    DENSE = "dense"
    SPARSE = "sparse"
    FUSED = "fused"
    RERANKED = "reranked"


@dataclass(frozen=True, slots=True)
class SourceRef:
    """Bir metin parçasının nereden geldiği. Atıf (citation) bunun üzerine kurulur."""

    file_name: str
    page: int | None = None
    section: str | None = None

    def label(self) -> str:
        """İnsan tarafından okunabilir kaynak etiketi."""
        parts = [self.file_name]
        if self.page is not None:
            parts.append(f"s.{self.page}")
        if self.section:
            parts.append(self.section)
        return " · ".join(parts)


@dataclass(frozen=True, slots=True)
class Chunk:
    """
    Aranabilir en küçük metin birimi.

    `id` İÇERİKTEN türetilir (content-addressed). Bunun üç faydası var:
      1. Aynı içerik iki kez indekslenemez (doğal tekilleştirme).
      2. Farklı süreçlerde/çalıştırmalarda aynı chunk aynı kimliği alır —
         bu yüzden RRF gibi birleştirme algoritmaları güvenle çalışır.
         (Nesne kimliği `id()` kullanmak tam olarak burada çöker.)
      3. Yeniden indeksleme sonrası eski değerlendirme setleri hâlâ geçerlidir.
    """

    text: str
    source: SourceRef
    token_count: int
    id: str = field(default="")

    def __post_init__(self) -> None:
        if not self.id:
            # frozen dataclass'ta alan atamak için object.__setattr__ gerekir.
            object.__setattr__(self, "id", self.compute_id(self.text, self.source))

    @staticmethod
    def compute_id(text: str, source: SourceRef) -> str:
        """
        İçerik + kaynak → kararlı kimlik.

        Kaynağı da karışıma katıyoruz: aynı cümle iki farklı dokümanda
        geçebilir ve bunlar atıf açısından FARKLI parçalardır.
        """
        h = hashlib.sha256()
        h.update(text.encode("utf-8"))
        h.update(b"\x00")
        h.update(source.file_name.encode("utf-8"))
        h.update(b"\x00")
        h.update(str(source.page).encode("utf-8"))
        return h.hexdigest()[:32]


@dataclass(frozen=True, slots=True)
class ScoredChunk:
    """
    Bir arama sonucu: chunk + skor + hangi aşamadan geldiği.

    `retriever_hits`: kaç farklı retriever bu chunk'ı bulmuş.
    Bu alan bir lüks değil — hybrid aramanın GERÇEKTEN çalışıp çalışmadığını
    gösteren tek gözlemlenebilir sinyaldir. Sürekli 1 ise birleştirme bozuktur.
    """

    chunk: Chunk
    score: float
    stage: RetrievalStage
    rank: int = 0
    retriever_hits: int = 1


@dataclass(frozen=True, slots=True)
class Citation:
    """Cevaptaki bir iddianın dayanağı."""

    marker: int  # cevap metnindeki [1], [2] numarası
    source: SourceRef
    chunk_id: str
    quote: str | None = None  # dayanak alıntı (varsa)


@dataclass(frozen=True, slots=True)
class Answer:
    """
    Üretilen cevap ve onu DENETLENEBİLİR kılan her şey.

    `abstained`: model "bu bilgi dokümanlarda yok" dediyse True.
    Bu bir başarısızlık değil, DOĞRU davranıştır ve ayrıca ölçülmelidir.
    Halüsinasyon kontrolünün temel taşı: modelin susma hakkı olmalı.

    `groundedness`: cevabın verilen bağlama dayanma oranı (0-1).
    """

    question: str
    text: str
    citations: tuple[Citation, ...]
    used_chunks: tuple[ScoredChunk, ...]
    abstained: bool
    model: str
    prompt_version: str
    latency_ms: int
    groundedness: float | None = None

    @property
    def is_grounded(self) -> bool:
        """Cevap ya atıf içeriyor ya da bilinçli olarak çekimser kalmış olmalı."""
        return self.abstained or len(self.citations) > 0


@dataclass(frozen=True, slots=True)
class DocumentReport:
    """Tek bir dokümanın işlenme raporu."""

    file_name: str
    status: DocumentStatus
    page_count: int = 0
    chunk_count: int = 0
    error: str | None = None

    @property
    def needs_ocr(self) -> bool:
        return self.status is DocumentStatus.NO_TEXT_LAYER


@dataclass(frozen=True, slots=True)
class IngestReport:
    """Bir ingestion çalıştırmasının tam sonucu."""

    documents: tuple[DocumentReport, ...]
    total_chunks_added: int
    started_at: datetime
    finished_at: datetime

    @property
    def duration_seconds(self) -> float:
        return (self.finished_at - self.started_at).total_seconds()

    @property
    def failed(self) -> tuple[DocumentReport, ...]:
        return tuple(d for d in self.documents if d.status is DocumentStatus.FAILED)

    @property
    def needs_ocr(self) -> tuple[DocumentReport, ...]:
        """Metin katmanı olmayan dosyalar — sessizce kaybolmasınlar."""
        return tuple(d for d in self.documents if d.needs_ocr)


@dataclass(frozen=True, slots=True)
class LoadedPage:
    """Bir dokümandan çıkarılmış tek sayfa."""

    page_number: int
    text: str


@dataclass(frozen=True, slots=True)
class LoadedDocument:
    """
    Parse edilmiş doküman: sayfalar + kaynak bilgisi.

    Chunker'a giden ara temsil. Loader hangi kütüphaneyi kullandığını
    burada saklar — üst katmanlar pypdf'i hiç bilmez.
    """

    file_name: str
    pages: tuple[LoadedPage, ...]
    content_hash: str  # dosya içeriğinin hash'i — idempotency için

    @property
    def has_text_layer(self) -> bool:
        """
        Anlamlı bir metin katmanı var mı?

        Eşik neden var: taranmış PDF'lerden bazen birkaç karakterlik çöp
        (sayfa numarası, filigran artığı) çıkar. Bu, metin katmanı sayılmaz.
        """
        return sum(len(p.text.strip()) for p in self.pages) >= 100

    @property
    def total_chars(self) -> int:
        return sum(len(p.text) for p in self.pages)
