"""
Ingestion pipeline: klasör → index.

Bu sınıf ORKESTRASYON yapar, iş yapmaz. Loader, chunker, embedder ve store
kendi işlerini bilir; pipeline yalnızca sırayı, idempotency kararını, hata
yalıtımını ve raporlamayı yönetir.

Neden bu ayrım: her bileşen tek başına test edilebilir kalır ve pipeline
okunduğunda VERİ AKIŞI tek bakışta görülür — hangi adım ne yapıyor, hata
nerede yutuluyor, hangi koşulda atlanıyor.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path

from rag_assistant.domain.models import (
    Chunk,
    DocumentReport,
    DocumentStatus,
    IngestReport,
)
from rag_assistant.domain.protocols import Chunker, Embedder, VectorStore
from rag_assistant.ingestion.loaders import (
    DocumentLoadError,
    LoaderRegistry,
    file_content_hash,
)
from rag_assistant.ingestion.manifest import IngestManifest, ManifestEntry
from rag_assistant.observability import get_logger

logger = get_logger(__name__)


class IngestionPipeline:
    """Klasördeki dokümanları yükler, parçalar, vektörleştirir ve indeksler."""

    def __init__(
        self,
        *,
        embedder: Embedder,
        chunker: Chunker,
        store: VectorStore,
        manifest: IngestManifest,
        loaders: LoaderRegistry | None = None,
    ) -> None:
        self._embedder = embedder
        self._chunker = chunker
        self._store = store
        self._manifest = manifest
        self._loaders = loaders or LoaderRegistry()

    # ------------------------------------------------------------------
    def run(self, source_dir: Path, *, force: bool = False) -> IngestReport:
        """
        Klasördeki tüm desteklenen dosyaları işle.

        `force=True`: manifest kontrolünü atla, her şeyi yeniden işle.
        (Chunker ayarları veya normalizasyon mantığı değiştiğinde gerekir —
        bunlar dosya içeriğine yansımadığı için hash aynı kalır.)
        """
        started = datetime.now(UTC)
        reports: list[DocumentReport] = []
        total_added = 0

        files = sorted(
            p
            for p in source_dir.glob("*")
            if p.is_file() and self._loaders.find(p) is not None
        )
        logger.info("ingest.started", directory=str(source_dir), files=len(files))

        for path in files:
            report, added = self._process_file(path, force=force)
            reports.append(report)
            total_added += added

        self._manifest.save()
        finished = datetime.now(UTC)

        result = IngestReport(
            documents=tuple(reports),
            total_chunks_added=total_added,
            started_at=started,
            finished_at=finished,
        )

        logger.info(
            "ingest.finished",
            files=len(files),
            chunks_added=total_added,
            index_total=self._store.count,
            failed=len(result.failed),
            needs_ocr=len(result.needs_ocr),
            seconds=round(result.duration_seconds, 2),
        )
        return result

    # ------------------------------------------------------------------
    def _process_file(self, path: Path, *, force: bool) -> tuple[DocumentReport, int]:
        """
        Tek dosyayı işle.

        HATA YALITIMI: Tek bir bozuk dosya TÜM ingestion'ı durdurmamalı.
        Hata yakalanır, FAILED olarak raporlanır, diğer dosyalar işlenmeye
        devam eder. Ama hata SESSİZCE yutulmaz — hem loglanır hem rapora girer.
        """
        content_hash = file_content_hash(path)

        if not force and self._manifest.is_current(
            path.name,
            content_hash,
            self._embedder.model_id,
            self._store.contains_all,
        ):
            entry = self._manifest.get(path.name)
            logger.debug("ingest.skipped", file=path.name)
            return (
                DocumentReport(
                    file_name=path.name,
                    status=DocumentStatus.SKIPPED,
                    page_count=entry.page_count if entry else 0,
                    chunk_count=entry.chunk_count if entry else 0,
                ),
                0,
            )

        loader = self._loaders.find(path)
        if loader is None:  # pragma: no cover - run() zaten filtreliyor
            return DocumentReport(path.name, DocumentStatus.FAILED, error="loader yok"), 0

        try:
            document = loader.load(path)
        except DocumentLoadError as exc:
            logger.error("ingest.load_failed", file=path.name, error=str(exc))
            self._manifest.record(
                ManifestEntry(
                    file_name=path.name,
                    content_hash=content_hash,
                    status=DocumentStatus.FAILED,
                    page_count=0,
                    chunk_count=0,
                    embedder_model_id=self._embedder.model_id,
                    error=str(exc),
                )
            )
            return DocumentReport(path.name, DocumentStatus.FAILED, error=str(exc)), 0

        # Metin katmanı yoksa: bu bir BAŞARI DEĞİL, ayrı bir durum.
        # Manifest'e OK yazmıyoruz ki sonraki çalıştırmada tekrar denensin
        # (aradaki zamanda OCR desteği eklenmiş olabilir).
        if not document.has_text_layer:
            logger.warning(
                "ingest.no_text_layer",
                file=path.name,
                pages=len(document.pages),
                hint="taranmış PDF — OCR gerekiyor",
            )
            self._manifest.record(
                ManifestEntry(
                    file_name=path.name,
                    content_hash=content_hash,
                    status=DocumentStatus.NO_TEXT_LAYER,
                    page_count=len(document.pages),
                    chunk_count=0,
                    embedder_model_id=self._embedder.model_id,
                )
            )
            return (
                DocumentReport(
                    file_name=path.name,
                    status=DocumentStatus.NO_TEXT_LAYER,
                    page_count=len(document.pages),
                ),
                0,
            )

        chunks = self._chunker.split(document)
        if not chunks:
            self._manifest.record(
                ManifestEntry(
                    file_name=path.name,
                    content_hash=content_hash,
                    status=DocumentStatus.NO_TEXT_LAYER,
                    page_count=len(document.pages),
                    chunk_count=0,
                    embedder_model_id=self._embedder.model_id,
                )
            )
            return (
                DocumentReport(path.name, DocumentStatus.NO_TEXT_LAYER, len(document.pages)),
                0,
            )

        # Dosya güncellendiyse ESKİ chunk'ları temizle. Yapılmazsa index'te
        # artık hiçbir dosyaya ait olmayan "hayalet" chunk'lar kalır ve
        # aramada eski/yanlış bilgi döner.
        if self._manifest.get(path.name) is not None:
            self._store.remove_by_file(path.name)

        added = self._embed_and_store(chunks)

        self._manifest.record(
            ManifestEntry(
                file_name=path.name,
                content_hash=content_hash,
                status=DocumentStatus.OK,
                page_count=len(document.pages),
                chunk_count=len(chunks),
                chunk_ids=[c.id for c in chunks],
                embedder_model_id=self._embedder.model_id,
            )
        )

        return (
            DocumentReport(
                file_name=path.name,
                status=DocumentStatus.OK,
                page_count=len(document.pages),
                chunk_count=len(chunks),
            ),
            added,
        )

    def _embed_and_store(self, chunks: Sequence[Chunk]) -> int:
        vectors = self._embedder.embed_documents([c.text for c in chunks])
        return self._store.add(chunks, vectors)
