"""
Ingestion manifestosu — hangi dosya, hangi içerikle, ne zaman işlendi?

BU DOSYANIN VAR OLMA SEBEBİ: IDEMPOTENCY.
  Aynı ingestion komutunu iki kez çalıştırmak, bir kez çalıştırmakla aynı
  sonucu vermelidir. Bu olmadan:
    * Her çalıştırmada tüm embedding'ler yeniden hesaplanır (boşa CPU)
    * Ya da daha kötüsü, aynı chunk index'e iki kez girer (bozuk arama)

TASARIM KARARLARI:

  1. Kimlik ölçütü DOSYA İÇERİĞİNİN HASH'İ, değiştirilme zamanı değil.
     mtime; kopyalama, git checkout, yedekten dönme gibi işlemlerde içerik
     hiç değişmeden değişir → gereksiz yeniden işleme. Tersi de mümkündür.

  2. Manifest, index'in kendisiyle DOĞRULANIR (`is_current` içinde
     `index_has_chunks` çağrısı). Manifest "işlendi" derken index boş
     olabilir (silinmiş, taşınmış, bozulmuş). İki kaynağı karşılaştırmadan
     "işlenmiş" demek, sistemi sessizce boş index'le çalıştırmaktır.

  3. `NO_TEXT_LAYER` durumu "işlenmiş" SAYILMAZ. Taranmış bir PDF sonraki
     çalıştırmada tekrar denenir; çünkü aradaki zamanda OCR desteği eklenmiş
     olabilir. "0 chunk üretildi" hiçbir zaman başarı değildir.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from rag_assistant.domain.models import DocumentStatus
from rag_assistant.observability import get_logger

logger = get_logger(__name__)

MANIFEST_VERSION = 1


@dataclass(slots=True)
class ManifestEntry:
    """Tek bir kaynak dosyanın işlenme kaydı."""

    file_name: str
    content_hash: str
    status: DocumentStatus
    page_count: int
    chunk_count: int
    chunk_ids: list[str] = field(default_factory=list)
    embedder_model_id: str = ""
    processed_at: str = ""
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "file_name": self.file_name,
            "content_hash": self.content_hash,
            "status": str(self.status),
            "page_count": self.page_count,
            "chunk_count": self.chunk_count,
            "chunk_ids": self.chunk_ids,
            "embedder_model_id": self.embedder_model_id,
            "processed_at": self.processed_at,
            "error": self.error,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ManifestEntry:
        return cls(
            file_name=data["file_name"],
            content_hash=data["content_hash"],
            status=DocumentStatus(data["status"]),
            page_count=data.get("page_count", 0),
            chunk_count=data.get("chunk_count", 0),
            chunk_ids=list(data.get("chunk_ids", [])),
            embedder_model_id=data.get("embedder_model_id", ""),
            processed_at=data.get("processed_at", ""),
            error=data.get("error"),
        )


class IngestManifest:
    """
    Diskte JSON olarak tutulan ingestion defteri.

    Atomik yazma kullanıyor: geçici dosyaya yaz, sonra yerine taşı.
    Neden — yazma sırasında süreç ölürse (Ctrl+C, güç kesintisi) yarı yazılmış
    bir JSON kalır ve manifest bir daha hiç okunamaz; sistem tüm geçmişini
    kaybeder. `replace` işletim sistemi düzeyinde atomiktir.
    """

    FILE_NAME = "ingest_manifest.json"

    def __init__(self, directory: Path) -> None:
        self._path = directory / self.FILE_NAME
        self._entries: dict[str, ManifestEntry] = {}
        self._load()

    # ------------------------------------------------------------------
    def _load(self) -> None:
        if not self._path.exists():
            return
        try:
            data = json.loads(self._path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            # Bozuk manifest sistemi durdurmamalı: en kötü senaryoda her şey
            # yeniden işlenir (yavaş ama doğru). Sessizce yutmuyoruz, logluyoruz.
            logger.warning("manifest.corrupt", path=str(self._path), error=str(exc))
            return

        if data.get("version") != MANIFEST_VERSION:
            logger.warning(
                "manifest.version_mismatch",
                found=data.get("version"),
                expected=MANIFEST_VERSION,
            )
            return

        self._entries = {
            e["file_name"]: ManifestEntry.from_dict(e) for e in data.get("documents", [])
        }
        logger.debug("manifest.loaded", entries=len(self._entries))

    def save(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "version": MANIFEST_VERSION,
            "updated_at": datetime.now(UTC).isoformat(),
            "documents": [e.to_dict() for e in self._entries.values()],
        }
        tmp = self._path.with_suffix(".json.tmp")
        tmp.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        tmp.replace(self._path)  # atomik

    # ------------------------------------------------------------------
    def get(self, file_name: str) -> ManifestEntry | None:
        return self._entries.get(file_name)

    def record(self, entry: ManifestEntry) -> None:
        entry.processed_at = datetime.now(UTC).isoformat()
        self._entries[entry.file_name] = entry

    def entries(self) -> list[ManifestEntry]:
        return list(self._entries.values())

    def remove(self, file_name: str) -> None:
        self._entries.pop(file_name, None)

    # ------------------------------------------------------------------
    def is_current(
        self,
        file_name: str,
        content_hash: str,
        embedder_model_id: str,
        index_has_chunks: "IndexChunkCheck",
    ) -> bool:
        """
        Bu dosya yeniden işlenmeli mi? (False → işle)

        Beş koşulun HEPSİ sağlanmalı ki atlayalım:
          1. Manifest'te kaydı var
          2. İçerik hash'i aynı            → dosya değişmemiş
          3. Durumu OK                     → NO_TEXT_LAYER/FAILED tekrar denenir
          4. Embedding modeli aynı         → model değiştiyse index geçersiz
          5. Chunk'ları GERÇEKTEN index'te → manifest ile index tutarlı

        5. koşul kritik: manifest'e güvenip index'i doğrulamamak, sistemin
        boş bir index'le "her şey işlendi" demesine yol açar.
        """
        entry = self._entries.get(file_name)
        if entry is None:
            return False
        if entry.content_hash != content_hash:
            logger.debug("manifest.changed", file=file_name)
            return False
        if entry.status is not DocumentStatus.OK:
            logger.debug("manifest.retry", file=file_name, status=str(entry.status))
            return False
        if entry.embedder_model_id != embedder_model_id:
            logger.info(
                "manifest.embedder_changed",
                file=file_name,
                old=entry.embedder_model_id,
                new=embedder_model_id,
            )
            return False
        if not entry.chunk_ids or not index_has_chunks(entry.chunk_ids):
            logger.warning("manifest.index_mismatch", file=file_name)
            return False
        return True


# Manifest'in index'e olan tek bağı: "şu chunk'lar var mı?" sorusu.
# Bir Protocol yerine basit bir çağrılabilir tip kullanıyoruz — en dar arayüz.
from collections.abc import Callable, Sequence  # noqa: E402

IndexChunkCheck = Callable[[Sequence[str]], bool]
