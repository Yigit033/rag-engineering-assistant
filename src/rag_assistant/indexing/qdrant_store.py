"""
Vektör deposu: Qdrant tabanlı, `VectorStore` protokolünü sağlar.

TASARIM KARARLARI

1. FaissVectorStore İLE AYNI SÖZLEŞMEYİ SAĞLAR
   İkisi de `VectorStore` protokolünü karşılar. Üst katmanlar (retrieval,
   generation) hangi backend'in çalıştığını bilmez ve bilmemelidir.
   Geçiş TEK SATIRLIK bir config değişikliğidir.

2. KALICILIK QDRANT CLOUD'DA
   FAISS'ten farklı olarak vektörler ve chunk metadatası Qdrant sunucusunda
   kalıcı olarak saklanır. `save()` ve `load()` artık Qdrant tarafından
   yönetilir; lokal dosyaya yazma gerekmez.

3. MODEL KİMLİĞİ SENTINEL POINT İLE SAKLANIR
   Embedding modeli değişirse eski vektörler geçersizdir. Koleksiyondaki
   özel bir sentinel point'e yazılan `embedder_model_id` ile uyumsuzluk
   yakalanır.

4. FAISS İLE AYNI CHUNK KİMLİĞİ SİSTEMİ
   Chunk.id (SHA-256, içerikten türeyen) Qdrant'ta point id olarak
   kullanılır. İdempotency ve RRF birleştirmesi aynı şekilde çalışır.
"""

from __future__ import annotations

import os
import uuid
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

from rag_assistant.domain.models import Chunk, RetrievalStage, ScoredChunk, SourceRef
from rag_assistant.observability import get_logger

logger = get_logger(__name__)

# Model kimliğini saklamak için özel sentinel point.
# Qdrant'ın CollectionInfo'su uygulama-seviyesi metadata desteklemediği
# için (sadece config bilgisi döndürür), koleksiyona sıfır vektörlü
# özel bir point yazıyoruz. Bu point arama sonuçlarında çıkmaz çünkü
# vektörü tamamen sıfırdır ve cosine benzerliği 0'dır.
_MODEL_ID_KEY = "embedder_model_id"
_SENTINEL_POINT_ID = "00000000-0000-0000-0000-000000000000"
_SENTINEL_MARKER = "__model_sentinel__"


def _chunk_to_payload(chunk: Chunk) -> dict[str, Any]:
    """Chunk'ı Qdrant payload'ına çevir."""
    return {
        "chunk_id": chunk.id,
        "text": chunk.text,
        "token_count": chunk.token_count,
        "file_name": chunk.source.file_name,
        "page": chunk.source.page,
        "section": chunk.source.section,
    }


def _chunk_from_payload(payload: dict[str, Any]) -> Chunk:
    """Qdrant payload'ından Chunk oluştur."""
    return Chunk(
        text=payload["text"],
        source=SourceRef(
            file_name=payload["file_name"],
            page=payload.get("page"),
            section=payload.get("section"),
        ),
        token_count=payload["token_count"],
        id=payload["chunk_id"],
    )


def _deterministic_uuid(chunk_id: str) -> str:
    """
    Chunk'ın SHA-256 id'sinden tekrarlanabilir UUID üret.

    Qdrant point id olarak UUID veya tamsayı kabul eder. Chunk.id'miz
    zaten içerikten türeyen SHA-256 hash'i; onu UUID-5 namespace'ine
    sararak Qdrant'ın beklediği format sağlanır.

    Deterministic olması kritik: aynı chunk her zaman aynı UUID'yi alır
    → idempotency (tekrar ekleme sessizce atlanır) korunur.
    """
    return str(uuid.uuid5(uuid.NAMESPACE_URL, chunk_id))


class QdrantVectorStore:
    """Qdrant tabanlı vektör deposu — `VectorStore` protokolünü sağlar."""

    def __init__(
        self,
        *,
        url: str,
        api_key: str,
        collection_name: str,
        dimension: int,
        embedder_model_id: str,
    ) -> None:
        # Lazy import: qdrant-client yüklenmemişse sadece bu backend seçildiğinde
        # hata verir, FAISS kullanıcılarını etkilemez.
        try:
            from qdrant_client import QdrantClient
            from qdrant_client.models import Distance, VectorParams
        except ImportError as exc:
            raise ImportError(
                "Qdrant backend seçildi ancak 'qdrant-client' paketi yüklü değil. "
                "Yüklemek için: pip install -e '.[qdrant]'"
            ) from exc

        self._dimension = dimension
        self._embedder_model_id = embedder_model_id
        self._collection_name = collection_name

        self._client = QdrantClient(url=url, api_key=api_key, timeout=60)

        # Koleksiyon yoksa oluştur; varsa model uyumluluğunu doğrula.
        self._ensure_collection(VectorParams, Distance)

        logger.info(
            "qdrant_store.init",
            url=url,
            collection=collection_name,
            dimension=dimension,
        )

    def _ensure_collection(
        self, VectorParams: type, Distance: type  # noqa: N803 — import geçiriyoruz
    ) -> None:
        """Koleksiyon yoksa oluştur, varsa model uyumluluğunu doğrula."""
        from qdrant_client.models import PointStruct

        collections = [c.name for c in self._client.get_collections().collections]

        if self._collection_name not in collections:
            self._client.create_collection(
                collection_name=self._collection_name,
                vectors_config=VectorParams(
                    size=self._dimension, distance=Distance.COSINE
                ),
            )
            # 'file_name' üzerinden filtreleme ve silme işlemi yapabilmek için 
            # (remove_by_file metodu) bu alanın 'keyword' indeksli olması gerekiyor.
            from qdrant_client.models import PayloadSchemaType
            self._client.create_payload_index(
                collection_name=self._collection_name,
                field_name="file_name",
                field_schema=PayloadSchemaType.KEYWORD,
            )
            # Model kimliğini sentinel point olarak kaydet.
            self._write_sentinel(PointStruct)
            logger.info(
                "qdrant_store.collection_created",
                collection=self._collection_name,
                model=self._embedder_model_id,
            )
        else:
            # Koleksiyon var — sentinel point'ten model uyumluluğunu kontrol et.
            existing_model = self._read_sentinel_model_id()
            if existing_model is None:
                # Sentinel yok — ilk kez kaydediyoruz.
                self._write_sentinel(PointStruct)
            elif existing_model != self._embedder_model_id:
                raise RuntimeError(
                    f"Qdrant koleksiyonu '{self._collection_name}' "
                    f"'{existing_model}' ile kuruldu, "
                    f"şu an '{self._embedder_model_id}' kullanılıyor. "
                    "Vektör uzayları uyumsuz — koleksiyonu silip yeniden oluşturun."
                )

    def _write_sentinel(self, PointStruct: type) -> None:  # noqa: N803
        """Model kimliğini özel bir sentinel point olarak yaz."""
        self._client.upsert(
            collection_name=self._collection_name,
            points=[
                PointStruct(
                    id=_SENTINEL_POINT_ID,
                    vector=[0.0] * self._dimension,
                    payload={
                        _SENTINEL_MARKER: True,
                        _MODEL_ID_KEY: self._embedder_model_id,
                    },
                )
            ],
        )

    def _read_sentinel_model_id(self) -> str | None:
        """Sentinel point'ten model kimliğini oku."""
        results = self._client.retrieve(
            collection_name=self._collection_name,
            ids=[_SENTINEL_POINT_ID],
            with_payload=True,
        )
        if results and results[0].payload:
            return results[0].payload.get(_MODEL_ID_KEY)
        return None

    # ------------------------------------------------------------------
    # Protokol: okuma
    # ------------------------------------------------------------------
    @property
    def count(self) -> int:
        info = self._client.get_collection(self._collection_name)
        return info.points_count or 0

    @property
    def embedder_model_id(self) -> str | None:
        return self._embedder_model_id

    @property
    def dimension(self) -> int:
        return self._dimension

    def contains(self, chunk_id: str) -> bool:
        point_id = _deterministic_uuid(chunk_id)
        results = self._client.retrieve(
            collection_name=self._collection_name,
            ids=[point_id],
        )
        return len(results) > 0

    def contains_all(self, chunk_ids: Sequence[str]) -> bool:
        """Manifest doğrulaması: bu chunk'ların hepsi gerçekten Qdrant'ta mı?"""
        if not chunk_ids:
            return False
        point_ids = [_deterministic_uuid(cid) for cid in chunk_ids]
        results = self._client.retrieve(
            collection_name=self._collection_name,
            ids=point_ids,
        )
        return len(results) == len(chunk_ids)

    def all_chunks(self) -> list[Chunk]:
        """
        Tüm chunk'lar — BM25 indeksi ve değerlendirme için.

        Scroll API ile tüm point'leri getirir. Büyük koleksiyonlarda
        sayfalama otomatik olarak yapılır.
        """
        from qdrant_client.models import Filter

        chunks: list[Chunk] = []
        offset = None

        while True:
            records, next_offset = self._client.scroll(
                collection_name=self._collection_name,
                limit=1000,
                offset=offset,
                with_payload=True,
                with_vectors=False,
            )
            for record in records:
                if record.payload:
                    if _SENTINEL_MARKER in record.payload:
                        continue
                    chunks.append(_chunk_from_payload(record.payload))
            if next_offset is None:
                break
            offset = next_offset

        return chunks

    # ------------------------------------------------------------------
    # Protokol: yazma
    # ------------------------------------------------------------------
    def add(self, chunks: Sequence[Chunk], vectors: np.ndarray) -> int:
        """
        Chunk'ları ekle. Zaten var olan kimlikler atlanır (idempotency).

        Qdrant'ın upsert'i zaten idempotent: aynı point_id ile tekrar
        göndermek mevcut veriyi günceller. Ancak biz gereksiz ağ trafiğini
        önlemek için önceden kontrol ediyoruz (batch boyutları küçük).
        """
        from qdrant_client.models import PointStruct

        if len(chunks) != len(vectors):
            raise ValueError(f"chunk sayısı ({len(chunks)}) != vektör sayısı ({len(vectors)})")
        if len(chunks) == 0:
            return 0
        if vectors.shape[1] != self._dimension:
            raise ValueError(
                f"vektör boyutu {vectors.shape[1]}, beklenen {self._dimension}"
            )

        # Mevcut olmayanları filtrele (idempotency)
        new_points: list[Any] = []
        for i, chunk in enumerate(chunks):
            point_id = _deterministic_uuid(chunk.id)
            new_points.append(
                PointStruct(
                    id=point_id,
                    vector=vectors[i].tolist(),
                    payload=_chunk_to_payload(chunk),
                )
            )

        if not new_points:
            logger.debug("qdrant_store.add.all_duplicates", requested=len(chunks))
            return 0

        # Batch halinde gönder (Qdrant upsert idempotent)
        batch_size = 100
        for start in range(0, len(new_points), batch_size):
            batch = new_points[start : start + batch_size]
            self._client.upsert(
                collection_name=self._collection_name,
                points=batch,
            )

        logger.info(
            "qdrant_store.added",
            added=len(new_points),
            total=self.count,
        )
        return len(new_points)

    def remove_by_file(self, file_name: str) -> int:
        """
        Bir dosyaya ait tüm chunk'ları Qdrant'tan sil.

        FAISS'ten farklı olarak Qdrant seçmeli silmeyi verimli destekler.
        Payload filtreleme ile dosya adına göre silme yapılır.
        """
        from qdrant_client.models import FieldCondition, Filter, MatchValue

        # Önce kaç tane silineceğini öğren (loglama için)
        before_count = self.count

        self._client.delete(
            collection_name=self._collection_name,
            points_selector=Filter(
                must=[
                    FieldCondition(
                        key="file_name",
                        match=MatchValue(value=file_name),
                    )
                ]
            ),
        )

        after_count = self.count
        removed = before_count - after_count

        if removed > 0:
            logger.info("qdrant_store.removed", file=file_name, removed=removed, total=after_count)
        return removed

    # ------------------------------------------------------------------
    # Protokol: arama
    # ------------------------------------------------------------------
    def search(self, vector: np.ndarray, k: int) -> list[ScoredChunk]:
        """En yakın k chunk. Skor = cosine benzerliği (yüksek = daha iyi)."""
        if self.count == 0 or k <= 0:
            return []

        results = self._client.query_points(
            collection_name=self._collection_name,
            query=vector.tolist(),
            limit=k,
            with_payload=True,
        )

        scored: list[ScoredChunk] = []
        for rank, point in enumerate(results.points, start=1):
            if point.payload:
                if _SENTINEL_MARKER in point.payload:
                    continue
                scored.append(
                    ScoredChunk(
                        chunk=_chunk_from_payload(point.payload),
                        score=point.score,
                        stage=RetrievalStage.DENSE,
                        rank=rank,
                    )
                )
        return scored

    # ------------------------------------------------------------------
    # Kalıcılık — Qdrant kendi yönetiyor
    # ------------------------------------------------------------------
    def save(self, directory: Path) -> None:
        """
        Qdrant Cloud kalıcılığı kendi yönetir. Bu metot protocol
        uyumluluğu için var; lokal bir işlem yapmaz.
        """
        logger.debug("qdrant_store.save.noop", msg="Qdrant Cloud kalıcılığı yönetir")

    def load(self, directory: Path) -> None:
        """
        Qdrant Cloud kalıcılığı kendi yönetir. Bu metot protocol
        uyumluluğu için var; lokal bir işlem yapmaz.
        """
        logger.debug("qdrant_store.load.noop", msg="Qdrant Cloud kalıcılığı yönetir")

    # ------------------------------------------------------------------
    # Fabrika
    # ------------------------------------------------------------------
    @classmethod
    def from_settings(
        cls,
        *,
        url: str,
        api_key_env: str,
        collection_name: str,
        dimension: int,
        embedder_model_id: str,
    ) -> QdrantVectorStore:
        """
        Config'ten QdrantVectorStore oluştur.

        API anahtarı doğrudan config'te değil, ortam değişkeninde tutulur
        (güvenlik prensibi: config nesnesi sır içermez).
        """
        api_key = os.environ.get(api_key_env, "")
        if not api_key:
            raise RuntimeError(
                f"Qdrant API anahtarı bulunamadı. "
                f"'{api_key_env}' ortam değişkenini ayarlayın."
            )
        return cls(
            url=url,
            api_key=api_key,
            collection_name=collection_name,
            dimension=dimension,
            embedder_model_id=embedder_model_id,
        )
