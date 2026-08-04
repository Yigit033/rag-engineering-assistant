"""
Vektör deposu: FAISS tabanlı, `VectorStore` protokolünü sağlar.

TASARIM KARARLARI

1. `IndexFlatIP` — inner product, `IndexFlatL2` DEĞİL
   Vektörler L2-normalize geldiği için inner product DOĞRUDAN cosine
   benzerliğine eşittir (‖a‖=‖b‖=1 ⇒ a·b = cos). Skor okunabilir hale gelir:
   1.0 = birebir aynı, 0 = ilgisiz. L2 mesafesiyle çalışıp sonra `1/(1+d)`
   gibi dönüşümler uydurmaya gerek kalmaz.
   `Flat` = kaba kuvvet, %100 kesin. ~100K vektöre kadar doğru tercih;
   bu ölçekte ANN (IVF/HNSW) kullanmak gereksiz karmaşıklıktır.

2. KALICILIK JSON İLE, PICKLE İLE DEĞİL
   Pickle açmak KOD ÇALIŞTIRMAKTIR. Bir index dosyası başka bir makineden,
   bir yedekten veya bir iş arkadaşından geliyorsa pickle onu güvenlik açığına
   çevirir (LangChain'in FAISS sarmalayıcısı bu yüzden
   `allow_dangerous_deserialization=True` istemek zorundadır).
   JSON: güvenli, insan tarafından okunabilir, sürüm kontrolünde diff'lenebilir,
   dile bağımsız.

3. MODEL KİMLİĞİ INDEX İLE BİRLİKTE SAKLANIR
   Embedding modeli değişirse eski vektörler geçersizdir — vektör uzayları
   uyumsuzdur. Bu kontrol olmadan sistem hata vermeden ÇÖP sonuç döndürür.
   `load()` uyumsuzluğu yakalar ve açıkça hata verir.

4. CHUNK'LARIN TEK DOĞRULUK KAYNAĞI BURASI
   BM25 indeksi de aynı chunk kümesine ihtiyaç duyar. İki ayrı yerden iki
   ayrı kopya yüklenirse kimliğe dayalı birleştirme (RRF) sessizce bozulur.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import faiss
import numpy as np

from rag_assistant.domain.models import Chunk, RetrievalStage, ScoredChunk, SourceRef
from rag_assistant.observability import get_logger

logger = get_logger(__name__)

STORE_VERSION = 1
INDEX_FILE = "index.faiss"
CHUNKS_FILE = "chunks.json"
META_FILE = "meta.json"


class IndexCompatibilityError(RuntimeError):
    """Diskteki index, mevcut yapılandırmayla uyumsuz."""


def _chunk_to_dict(chunk: Chunk) -> dict[str, Any]:
    return {
        "id": chunk.id,
        "text": chunk.text,
        "token_count": chunk.token_count,
        "source": {
            "file_name": chunk.source.file_name,
            "page": chunk.source.page,
            "section": chunk.source.section,
        },
    }


def _chunk_from_dict(data: dict[str, Any]) -> Chunk:
    src = data["source"]
    return Chunk(
        text=data["text"],
        source=SourceRef(
            file_name=src["file_name"], page=src.get("page"), section=src.get("section")
        ),
        token_count=data["token_count"],
        id=data["id"],
    )


class FaissVectorStore:
    """Bellek içi FAISS index + diskte JSON chunk deposu."""

    def __init__(self, dimension: int, embedder_model_id: str) -> None:
        self._dimension = dimension
        self._embedder_model_id = embedder_model_id
        self._index = faiss.IndexFlatIP(dimension)
        # Satır sırası FAISS'teki sıra ile birebir aynı olmak ZORUNDA.
        self._chunks: list[Chunk] = []
        self._id_to_row: dict[str, int] = {}

    # ------------------------------------------------------------------
    # Protokol: okuma
    # ------------------------------------------------------------------
    @property
    def count(self) -> int:
        return int(self._index.ntotal)

    @property
    def embedder_model_id(self) -> str | None:
        return self._embedder_model_id

    @property
    def dimension(self) -> int:
        return self._dimension

    def contains(self, chunk_id: str) -> bool:
        return chunk_id in self._id_to_row

    def contains_all(self, chunk_ids: Sequence[str]) -> bool:
        """Manifest doğrulaması: bu chunk'ların hepsi gerçekten index'te mi?"""
        return bool(chunk_ids) and all(cid in self._id_to_row for cid in chunk_ids)

    def all_chunks(self) -> list[Chunk]:
        return list(self._chunks)

    # ------------------------------------------------------------------
    # Protokol: yazma
    # ------------------------------------------------------------------
    def add(self, chunks: Sequence[Chunk], vectors: np.ndarray) -> int:
        """
        Chunk'ları ekle. Zaten var olan kimlikler ATLANIR (idempotency).

        Aynı ingestion'ı iki kez çalıştırmak index'i bozmamalıdır. Kimlik
        içerikten türediği için tekilleştirme doğal olarak çalışır.
        """
        if len(chunks) != len(vectors):
            raise ValueError(f"chunk sayısı ({len(chunks)}) != vektör sayısı ({len(vectors)})")
        if len(chunks) == 0:
            return 0
        if vectors.shape[1] != self._dimension:
            raise ValueError(
                f"vektör boyutu {vectors.shape[1]}, beklenen {self._dimension}"
            )

        fresh_rows = [i for i, c in enumerate(chunks) if c.id not in self._id_to_row]
        if not fresh_rows:
            logger.debug("store.add.all_duplicates", requested=len(chunks))
            return 0

        new_vectors = np.ascontiguousarray(vectors[fresh_rows], dtype=np.float32)
        self._index.add(new_vectors)

        for i in fresh_rows:
            self._id_to_row[chunks[i].id] = len(self._chunks)
            self._chunks.append(chunks[i])

        logger.info(
            "store.added",
            added=len(fresh_rows),
            skipped=len(chunks) - len(fresh_rows),
            total=self.count,
        )
        return len(fresh_rows)

    def remove_by_file(self, file_name: str) -> int:
        """
        Bir dosyaya ait tüm chunk'ları sil.

        FAISS `IndexFlat` seçmeli silmeyi verimli desteklemediği için index'i
        kalan vektörlerle yeniden kuruyoruz. Bu ölçekte (<100K) maliyeti
        önemsiz; karşılığında "güncellenen dosyanın eski chunk'ları index'te
        hayalet olarak kalır" hatasını tamamen ortadan kaldırıyor.
        """
        keep = [i for i, c in enumerate(self._chunks) if c.source.file_name != file_name]
        removed = len(self._chunks) - len(keep)
        if removed == 0:
            return 0

        vectors = self._index.reconstruct_n(0, self._index.ntotal)
        kept_chunks = [self._chunks[i] for i in keep]

        self._index = faiss.IndexFlatIP(self._dimension)
        if keep:
            self._index.add(np.ascontiguousarray(vectors[keep], dtype=np.float32))

        self._chunks = kept_chunks
        self._id_to_row = {c.id: i for i, c in enumerate(self._chunks)}

        logger.info("store.removed", file=file_name, removed=removed, total=self.count)
        return removed

    # ------------------------------------------------------------------
    # Protokol: arama
    # ------------------------------------------------------------------
    def search(self, vector: np.ndarray, k: int) -> list[ScoredChunk]:
        """
        En yakın k chunk. Skor = cosine benzerliği (yüksek = daha iyi).
        """
        if self.count == 0 or k <= 0:
            return []

        query = np.ascontiguousarray(vector.reshape(1, -1), dtype=np.float32)
        scores, indices = self._index.search(query, min(k, self.count))

        results: list[ScoredChunk] = []
        for rank, (row, score) in enumerate(zip(indices[0], scores[0], strict=True), start=1):
            if row < 0:  # FAISS yetersiz sonuçta -1 döndürür
                continue
            results.append(
                ScoredChunk(
                    chunk=self._chunks[row],
                    score=float(score),
                    stage=RetrievalStage.DENSE,
                    rank=rank,
                )
            )
        return results

    # ------------------------------------------------------------------
    # Kalıcılık
    # ------------------------------------------------------------------
    def save(self, directory: Path) -> None:
        """
        Diske yaz. Önce geçici dosyalara, sonra yerine taşı (atomik).

        Neden atomik: yazma sırasında süreç ölürse yarım index + tam chunk
        listesi kalır ve ikisi tutarsız olur. Tutarsız index, hatalı sonuçtan
        daha kötüdür çünkü fark edilmez.
        """
        directory.mkdir(parents=True, exist_ok=True)

        tmp_index = directory / f"{INDEX_FILE}.tmp"
        faiss.write_index(self._index, str(tmp_index))

        tmp_chunks = directory / f"{CHUNKS_FILE}.tmp"
        tmp_chunks.write_text(
            json.dumps([_chunk_to_dict(c) for c in self._chunks], ensure_ascii=False),
            encoding="utf-8",
        )

        tmp_meta = directory / f"{META_FILE}.tmp"
        tmp_meta.write_text(
            json.dumps(
                {
                    "version": STORE_VERSION,
                    "dimension": self._dimension,
                    "embedder_model_id": self._embedder_model_id,
                    "count": self.count,
                    "index_type": type(self._index).__name__,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

        tmp_index.replace(directory / INDEX_FILE)
        tmp_chunks.replace(directory / CHUNKS_FILE)
        tmp_meta.replace(directory / META_FILE)

        logger.info("store.saved", directory=str(directory), vectors=self.count)

    def load(self, directory: Path) -> None:
        """
        Diskten oku ve UYUMLULUĞU DOĞRULA.

        Üç kontrol: sürüm, boyut, model kimliği. Üçü de sessiz bozulma
        kaynağıdır — hiçbiri istisna atmadan yanlış sonuç üretir.
        """
        meta_path = directory / META_FILE
        if not meta_path.exists():
            raise FileNotFoundError(f"Index bulunamadı: {directory}")

        meta = json.loads(meta_path.read_text(encoding="utf-8"))

        if meta.get("version") != STORE_VERSION:
            raise IndexCompatibilityError(
                f"Index sürümü {meta.get('version')}, beklenen {STORE_VERSION}. "
                "Index'i yeniden oluşturun."
            )
        if meta["dimension"] != self._dimension:
            raise IndexCompatibilityError(
                f"Index boyutu {meta['dimension']}, model boyutu {self._dimension}."
            )
        if meta["embedder_model_id"] != self._embedder_model_id:
            raise IndexCompatibilityError(
                f"Index '{meta['embedder_model_id']}' ile kuruldu, "
                f"şu an '{self._embedder_model_id}' kullanılıyor. "
                "Vektör uzayları uyumsuz — index'i yeniden oluşturun."
            )

        self._index = faiss.read_index(str(directory / INDEX_FILE))
        raw = json.loads((directory / CHUNKS_FILE).read_text(encoding="utf-8"))
        self._chunks = [_chunk_from_dict(d) for d in raw]
        self._id_to_row = {c.id: i for i, c in enumerate(self._chunks)}

        if self._index.ntotal != len(self._chunks):
            raise IndexCompatibilityError(
                f"Tutarsız index: {self._index.ntotal} vektör, {len(self._chunks)} chunk."
            )

        logger.info("store.loaded", directory=str(directory), vectors=self.count)

    @classmethod
    def open_or_create(
        cls, directory: Path, *, dimension: int, embedder_model_id: str
    ) -> FaissVectorStore:
        """Varsa yükle, yoksa boş oluştur. Uyumsuzsa açıkça hata ver."""
        store = cls(dimension=dimension, embedder_model_id=embedder_model_id)
        if (directory / META_FILE).exists():
            store.load(directory)
        return store
