"""
Reranking — cross-encoder ile yeniden sıralama.

RETRIEVER İLE FARKI (bunu iyi anlamak gerekir):

  Bi-encoder (retriever):
    Sorgu ve doküman AYRI AYRI vektörlenir, sonra karşılaştırılır.
    Doküman vektörleri önceden hesaplandığı için arama milisaniyeler sürer.
    Ama model, sorgu ile dokümanı hiçbir zaman BİRLİKTE görmez —
    aralarındaki ince ilişkiyi kaçırır.

  Cross-encoder (reranker):
    Sorgu ve doküman TEK bir girdi olarak modele verilir; model ikisi
    arasındaki etkileşimi doğrudan modelleyip tek bir alaka skoru üretir.
    Çok daha isabetli, ama her çift için model çalıştığından çok yavaş —
    milyonlarca doküman üzerinde kullanılamaz.

  Bu yüzden mimari iki aşamalıdır:
    geniş tara (bi-encoder, ucuz) → daralt (cross-encoder, pahalı ama isabetli)

  Pratikte kalite/emek oranı en yüksek iyileştirme genellikle budur.

TEMBEL YÜKLEME: model ~2.3 GB. Reranker kapalıysa veya hiç sorgu gelmediyse
belleğe hiç alınmaz. 13.7 GB RAM'de embedder + reranker + LLM'in bir arada
yaşaması buna bağlı.
"""

from __future__ import annotations

from collections.abc import Sequence
from functools import cached_property

from rag_assistant.domain.models import RetrievalStage, ScoredChunk
from rag_assistant.observability import get_logger
from rag_assistant.resources import has_room_for

logger = get_logger(__name__)


class CrossEncoderReranker:
    """sentence-transformers CrossEncoder tabanlı. `Reranker` protokolünü sağlar."""

    def __init__(
        self,
        model_name: str,
        *,
        device: str = "cpu",
        batch_size: int = 8,
        min_score: float | None = None,
        required_ram_gb: float = 2.3,
    ) -> None:
        self._model_name = model_name
        self._device = device
        self._batch_size = batch_size
        self._min_score = min_score
        self._required_ram_gb = required_ram_gb
        # Yükleme bir kez denenir; başarısız olursa devre dışı kalır ve
        # her sorguda tekrar denenmez.
        self._disabled = False

    @property
    def is_active(self) -> bool:
        """Reranker gerçekten kullanılabilir durumda mı? (/ready raporlar)"""
        return not self._disabled

    @cached_property
    def _model(self):  # type: ignore[no-untyped-def] # noqa: ANN202
        """
        Modeli tembel yükle — ÖNCE bellek kontrolü ile.

        Kontrol neden `try/except`'ten önce geliyor: bellek yetmediğinde
        işletim sistemi süreci Python istisnası fırlatmadan öldürebilir
        (ölçüldü). Bu durumda `except` bloğuna hiç ulaşılmaz. Bu yüzden
        DENEMEDEN ÖNCE yer olup olmadığına bakıyoruz.
        """
        if self._device == "cpu" and not has_room_for(
            self._required_ram_gb, label=f"reranker:{self._model_name}"
        ):
            logger.warning(
                "reranker.disabled",
                reason="yetersiz bellek",
                model=self._model_name,
                impact="sistem çalışmaya devam eder; yalnızca sıralama hassasiyeti düşer",
                hint="daha küçük bir reranker (RAG_RETRIEVAL__RERANKER_MODEL) veya "
                "RAG_RETRIEVAL__USE_RERANKER=false",
            )
            self._disabled = True
            return None

        from sentence_transformers import CrossEncoder

        logger.info("reranker.loading", model=self._model_name, device=self._device)
        try:
            model = CrossEncoder(self._model_name, device=self._device)
        except Exception as exc:  # noqa: BLE001 - hangi hata olursa olsun servis düşmemeli
            logger.warning(
                "reranker.disabled", reason=str(exc)[:200], model=self._model_name
            )
            self._disabled = True
            return None

        logger.info("reranker.loaded", model=self._model_name)
        return model

    @property
    def model_id(self) -> str:
        return self._model_name

    def rerank(
        self, query: str, candidates: Sequence[ScoredChunk], top_k: int
    ) -> list[ScoredChunk]:
        """
        Adayları alaka skoruna göre yeniden sırala.

        `min_score` verilmişse, eşiğin altındaki adaylar ATILIR. Bu, "hiç
        alakalı sonuç yok" durumunu LLM'e boş bağlam olarak iletmeyi mümkün
        kılar — ve boş bağlamda modelin çekimser kalması, uydurmasından
        çok daha iyidir. Eşik ampiriktir; golden set üzerinde belirlenir.
        """
        if not candidates:
            return []

        # ZARİF DÜŞÜŞ: reranker yüklenemediyse adayları OLDUĞU GİBİ geçir.
        # Kalite bir miktar düşer (RRF sıralaması kalır) ama servis ayakta
        # kalır. İsteğe bağlı bir iyileştirmenin zorunlu işlevi düşürmesi
        # kabul edilemez.
        model = self._model
        if model is None:
            return list(candidates[:top_k])

        pairs = [(query, c.chunk.text) for c in candidates]
        scores = model.predict(
            pairs, batch_size=self._batch_size, show_progress_bar=False
        )

        scored = sorted(
            zip(candidates, (float(s) for s in scores), strict=True),
            key=lambda pair: pair[1],
            reverse=True,
        )

        if self._min_score is not None:
            kept = [(c, s) for c, s in scored if s >= self._min_score]
            if len(kept) < len(scored):
                logger.debug(
                    "reranker.filtered",
                    dropped=len(scored) - len(kept),
                    threshold=self._min_score,
                )
            scored = kept

        results = [
            ScoredChunk(
                chunk=candidate.chunk,
                score=score,
                stage=RetrievalStage.RERANKED,
                rank=rank,
                # Fusion aşamasındaki uzlaşma bilgisini KORUYORUZ.
                # Kaybedilirse "hybrid gerçekten çalışıyor mu?" sorusunu
                # cevaplayacak sinyal yok olur.
                retriever_hits=candidate.retriever_hits,
            )
            for rank, (candidate, score) in enumerate(scored[:top_k], start=1)
        ]

        logger.debug(
            "reranker.done",
            candidates=len(candidates),
            returned=len(results),
            top_score=results[0].score if results else None,
        )
        return results
