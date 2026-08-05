"""
Retrieval pipeline: birden çok retriever + RRF + isteğe bağlı reranking.

Bu sınıf da ORKESTRASYON yapar, arama yapmaz. Kendisi de `Retriever`
protokolünü sağlar — yani bir HybridRetriever, sıradan bir retriever gibi
kullanılabilir ve gerekirse başka bir pipeline'ın içine konabilir
(kompozisyon).

AKIŞ:
    sorgu
      ├─→ dense  (fetch_k aday)  ─┐
      └─→ sparse (fetch_k aday)  ─┴─→ RRF ─→ [reranker] ─→ top_k
"""

from __future__ import annotations

import time
from collections.abc import Sequence

from rag_assistant.domain.models import ScoredChunk
from rag_assistant.domain.protocols import Reranker, Retriever
from rag_assistant.observability import get_logger
from rag_assistant.retrieval.fusion import reciprocal_rank_fusion

logger = get_logger(__name__)


class HybridRetriever:
    """Çok retriever'lı, RRF ile birleştiren, isteğe bağlı rerank eden pipeline."""

    def __init__(
        self,
        retrievers: Sequence[Retriever],
        *,
        fetch_k: int,
        rrf_k: int = 60,
        reranker: Reranker | None = None,
        rerank_input_size: int | None = None,
    ) -> None:
        if not retrievers:
            raise ValueError("en az bir retriever gerekli")
        self._retrievers = list(retrievers)
        self._fetch_k = fetch_k
        self._rrf_k = rrf_k
        self._reranker = reranker
        # Reranker'a kaç aday verilecek. Fusion'dan çıkanların hepsini
        # vermek pahalı olabilir; None ise fetch_k kadarı verilir.
        self._rerank_input_size = rerank_input_size or fetch_k

    @property
    def name(self) -> str:
        parts = "+".join(r.name for r in self._retrievers)
        return f"hybrid({parts}){'+rerank' if self._reranker else ''}"

    @property
    def reranker(self) -> Reranker | None:
        """
        Yapılandırılmış reranker (varsa).

        `/ready` bunu okuyup gerçekten aktif olup olmadığını raporlar:
        yapılandırmada açık olması, belleğe sığdığı anlamına gelmez.
        """
        return self._reranker

    def retrieve(self, query: str, k: int) -> list[ScoredChunk]:
        started = time.perf_counter()

        # 1) Her retriever bağımsız olarak geniş bir aday kümesi getirir.
        #    fetch_k > k olması TASARIM GEREĞİ: recall'ı burada kazanırsın,
        #    kaybedersen bir daha telafi edemezsin — hiç aday olmayan bir
        #    chunk'ı reranker kurtaramaz.
        ranked_lists = [r.retrieve(query, self._fetch_k) for r in self._retrievers]
        per_retriever = {
            r.name: len(lst) for r, lst in zip(self._retrievers, ranked_lists, strict=True)
        }

        # 2) Sıra tabanlı birleştirme (skor ölçeklerinden bağımsız).
        fused = reciprocal_rank_fusion(ranked_lists, k=self._rrf_k)
        consensus = sum(1 for c in fused if c.retriever_hits > 1)

        # 3) İsteğe bağlı yeniden sıralama (precision aşaması).
        if self._reranker is not None and fused:
            results = self._reranker.rerank(query, fused[: self._rerank_input_size], k)
        else:
            results = fused[:k]

        elapsed_ms = int((time.perf_counter() - started) * 1000)

        # Bu log satırı sistemin sağlık göstergesidir:
        #   consensus sürekli 0 ise -> birleştirme çalışmıyor demektir
        #   sparse sürekli 0 ise     -> tokenizasyon veya indeks bozuk
        logger.info(
            "retrieval.completed",
            query=query[:80],
            strategy=self.name,
            per_retriever=per_retriever,
            fused=len(fused),
            consensus=consensus,
            returned=len(results),
            top_score=round(results[0].score, 4) if results else None,
            ms=elapsed_ms,
        )
        return results
