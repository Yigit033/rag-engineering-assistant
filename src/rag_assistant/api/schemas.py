"""
HTTP sözleşmesi (API şemaları).

NEDEN DOMAIN MODELLERİ DOĞRUDAN DÖNDÜRÜLMÜYOR:
  Domain modelleri (`Answer`, `Chunk`, `ScoredChunk`) sistemin İÇ yapısıdır.
  Bunları doğrudan HTTP cevabı yapmak iki yönlü bir bağ kurar:

    * İç bir alan adını değiştirmek → istemcileri KIRAR. Yani artık iç
      yapıyı özgürce yeniden düzenleyemezsin; API uyumluluğu seni kilitler.
    * İç alanlar farkında olmadan dışarı sızar. Örneğin chunk'ın TAM metni,
      dosya sistemi yolları veya skorlama detayları istemciye gitmemeli.

  Ayrı bir şema katmanı bu bağı koparır: iç yapı serbestçe değişir, HTTP
  sözleşmesi kendi hızında ve BİLİNÇLİ olarak sürümlenir.

  Bu, "modüler mimari"nin en somut sınavıdır: iki katman birbirinden
  gerçekten ayrık mı, yoksa sadece farklı dosyalarda mı duruyor?
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from rag_assistant.domain.models import Answer, DocumentReport, IngestReport


class AskRequest(BaseModel):
    """Soru isteği."""

    question: str = Field(
        min_length=3,
        max_length=1000,
        description="Dokümanlara sorulacak soru",
        examples=["SmartSafe AI hangi sektöre yönelik?"],
    )
    # İstemci top_k'yı isteyebilir ama SINIRSIZ değil: üst sınır sunucuda.
    # Aksi halde bir istemci top_k=10000 yollayıp sunucuyu meşgul edebilir.
    top_k: int | None = Field(default=None, ge=1, le=20)


class CitationOut(BaseModel):
    """Cevaptaki bir atıf."""

    marker: int
    source: str = Field(description="İnsan tarafından okunabilir kaynak etiketi")
    chunk_id: str = Field(description="Dayanak chunk'ın kimliği (izlenebilirlik)")


class SourceOut(BaseModel):
    """Cevapta kullanılan bir bağlam parçası."""

    marker: int
    source: str
    chunk_id: str
    score: float
    retriever_hits: int = Field(
        description="Kaç retriever bu parçayı buldu (1'den büyükse uzlaşma var)"
    )
    preview: str = Field(description="Metnin ilk kısmı — tam metin dönmez")


class AskResponse(BaseModel):
    """Soru cevabı ve onu denetlenebilir kılan bilgiler."""

    question: str
    answer: str
    abstained: bool = Field(description="Model bilgi olmadığını mı söyledi?")
    grounded: bool = Field(description="Cevap ya atıflı ya da bilinçli çekimser mi?")
    citations: list[CitationOut]
    sources: list[SourceOut]
    groundedness: float | None = None
    model: str
    prompt_version: str
    latency_ms: int

    @classmethod
    def from_domain(cls, answer: Answer, *, preview_chars: int = 300) -> AskResponse:
        """
        Domain nesnesini HTTP cevabına çevir.

        Dönüşüm BİLİNÇLİ olarak burada yapılıyor: hangi iç alanın dışarı
        çıktığına tek bir yerde karar veriliyor. Chunk'ın tam metni yerine
        yalnızca önizleme gönderiyoruz — cevap boyutunu ve gereksiz veri
        paylaşımını sınırlamak için.
        """
        return cls(
            question=answer.question,
            answer=answer.text,
            abstained=answer.abstained,
            grounded=answer.is_grounded,
            citations=[
                CitationOut(
                    marker=c.marker, source=c.source.label(), chunk_id=c.chunk_id
                )
                for c in answer.citations
            ],
            sources=[
                SourceOut(
                    marker=i,
                    source=sc.chunk.source.label(),
                    chunk_id=sc.chunk.id,
                    score=round(sc.score, 4),
                    retriever_hits=sc.retriever_hits,
                    preview=sc.chunk.text[:preview_chars],
                )
                for i, sc in enumerate(answer.used_chunks, start=1)
            ],
            groundedness=answer.groundedness,
            model=answer.model,
            prompt_version=answer.prompt_version,
            latency_ms=answer.latency_ms,
        )


class IngestRequest(BaseModel):
    """Yeniden indeksleme isteği."""

    force: bool = Field(
        default=False,
        description=(
            "Idempotency kontrolünü atla. Chunk ayarları veya normalizasyon "
            "mantığı değiştiğinde gerekir: bunlar dosya içeriğine yansımadığı "
            "için hash aynı kalır ve dosya normalde atlanır."
        ),
    )


class DocumentReportOut(BaseModel):
    file_name: str
    status: str
    page_count: int
    chunk_count: int
    error: str | None = None
    needs_ocr: bool


class IngestResponse(BaseModel):
    documents: list[DocumentReportOut]
    chunks_added: int
    index_total: int
    duration_seconds: float
    # Bu iki alan bilinçli olarak AYRI raporlanıyor: "başarılı" sayısının
    # içinde kaybolmamaları gerekir.
    failed_count: int
    needs_ocr_count: int

    @classmethod
    def from_domain(cls, report: IngestReport, *, index_total: int) -> IngestResponse:
        def convert(d: DocumentReport) -> DocumentReportOut:
            return DocumentReportOut(
                file_name=d.file_name,
                status=str(d.status),
                page_count=d.page_count,
                chunk_count=d.chunk_count,
                error=d.error,
                needs_ocr=d.needs_ocr,
            )

        return cls(
            documents=[convert(d) for d in report.documents],
            chunks_added=report.total_chunks_added,
            index_total=index_total,
            duration_seconds=round(report.duration_seconds, 2),
            failed_count=len(report.failed),
            needs_ocr_count=len(report.needs_ocr),
        )


class ComponentHealth(BaseModel):
    name: str
    ok: bool
    detail: str


class HealthResponse(BaseModel):
    """
    Hazırlık (readiness) durumu.

    LIVENESS ile READINESS AYRIMI:
      * liveness  → süreç ayakta mı? (yeniden başlatılmalı mı?)
      * readiness → trafik alabilir mi? (modeller yüklü, index dolu mu?)
      İkisini karıştırmak, model yüklenirken gelen isteklerin hata almasına
      veya sağlıklı bir sürecin gereksiz yeniden başlatılmasına yol açar.
    """

    ready: bool
    components: list[ComponentHealth]
    index_vectors: int
    embedder_model: str
    llm_model: str
    retrieval_strategy: str


class ErrorResponse(BaseModel):
    """
    Tek biçimli hata cevabı.

    `detail` istemciye gösterilebilir bir açıklama içerir; yığın izi (stack
    trace) ASLA dışarı verilmez — dosya yolları ve iç yapı sızdırır.
    """

    error: str = Field(description="Makine tarafından okunabilir hata kodu")
    detail: str = Field(description="İnsan tarafından okunabilir açıklama")
    request_id: str | None = None
