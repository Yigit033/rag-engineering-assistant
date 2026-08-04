"""
Reciprocal Rank Fusion (RRF) — farklı retriever sonuçlarını birleştirme.

PROBLEM:
  BM25 skorları sınırsız pozitif sayılardır (0 … ∞), cosine benzerliği ise
  [-1, 1] aralığındadır. Bu iki skoru doğrudan toplamak anlamsızdır; önce
  normalize etmek gerekir, normalizasyon ise veri dağılımına duyarlıdır ve
  her veri setinde yeniden ayar ister.

RRF ÇÖZÜMÜ:
  Skorları tamamen AT, yalnızca SIRAYI kullan:

      RRF(d) = Σ_retriever  1 / (k + rank(d))

  Ölçek problemi kökten ortadan kalkar. Ayar (tuning) gerektirmez.

`k = 60` NE İŞE YARAR:
  k=0 olsaydı 1. sıra 1.0, 2. sıra 0.5 puan alırdı — tek retriever'ın birinci
  sırası her şeyi ezerdi. k=60 ile 1. sıra 0.01639, 2. sıra 0.01613 alır;
  aralar daralır ve İKİ LİSTEDE BİRDEN geçen bir doküman, tek listede birinci
  olanı GEÇER. RRF'in bütün değeri bu "uzlaşma ödülü"ndedir.

  Örnek (k=60):
      A: dense #1, sparse yok  → 1/61            = 0.01639
      B: dense #3, sparse #2   → 1/63 + 1/62     = 0.03200  ← kazanır
  A hiçbir listede geçilmemiş olmasına rağmen B öne çıkar, çünkü iki farklı
  yöntem birbirinden bağımsız olarak B'yi işaret etmiştir.

BİRLEŞTİRME ANAHTARI:
  `chunk.id` — yani İÇERİKTEN türeyen kararlı hash.
  Python nesne kimliği (`id()`) kullanmak bu algoritmayı sessizce çalışmaz
  hale getirir: aynı içerik farklı nesnelerde tutulduğunda hiçbir eşleşme
  bulunmaz, skorlar toplanmaz ve fusion fiilen "iki listeyi arka arkaya
  ekleme"ye dönüşür. Hata mesajı da alınmaz.
"""

from __future__ import annotations

from collections.abc import Sequence

from rag_assistant.domain.models import RetrievalStage, ScoredChunk
from rag_assistant.observability import get_logger

logger = get_logger(__name__)

DEFAULT_RRF_K = 60


def reciprocal_rank_fusion(
    ranked_lists: Sequence[Sequence[ScoredChunk]],
    *,
    k: int = DEFAULT_RRF_K,
    top_k: int | None = None,
) -> list[ScoredChunk]:
    """
    N adet sıralı listeyi tek sıralamada birleştir.

    Args:
        ranked_lists: her retriever'ın sıralı sonuçları
        k: RRF yumuşatma sabiti
        top_k: döndürülecek sonuç sayısı (None = hepsi)

    Returns:
        RRF skoruna göre azalan sıralı sonuçlar. `retriever_hits` alanı,
        her chunk'ın kaç farklı listede göründüğünü taşır — hybrid aramanın
        gerçekten çalışıp çalışmadığını gösteren tek gözlemlenebilir sinyal.
    """
    accumulated: dict[str, dict[str, object]] = {}

    for ranked in ranked_lists:
        for rank, scored in enumerate(ranked, start=1):
            key = scored.chunk.id  # içerik tabanlı — nesne kimliği DEĞİL
            slot = accumulated.setdefault(
                key, {"chunk": scored.chunk, "score": 0.0, "hits": 0}
            )
            slot["score"] = float(slot["score"]) + 1.0 / (k + rank)  # type: ignore[arg-type]
            slot["hits"] = int(slot["hits"]) + 1  # type: ignore[arg-type]

    ordered = sorted(accumulated.values(), key=lambda s: float(s["score"]), reverse=True)  # type: ignore[arg-type]
    if top_k is not None:
        ordered = ordered[:top_k]

    results = [
        ScoredChunk(
            chunk=slot["chunk"],  # type: ignore[arg-type]
            score=float(slot["score"]),  # type: ignore[arg-type]
            stage=RetrievalStage.FUSED,
            rank=rank,
            retriever_hits=int(slot["hits"]),  # type: ignore[arg-type]
        )
        for rank, slot in enumerate(ordered, start=1)
    ]

    logger.debug(
        "fusion.rrf",
        lists=len(ranked_lists),
        unique=len(accumulated),
        returned=len(results),
        consensus=sum(1 for r in results if r.retriever_hits > 1),
    )
    return results
