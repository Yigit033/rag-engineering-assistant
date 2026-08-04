"""
Retriever'lar: dense (anlam) ve sparse (kelime).

İkisi de aynı `Retriever` protokolünü sağlar — yani birbirinin yerine
geçebilir ve üst katman hangisinin kullanıldığını bilmez.

NEDEN İKİSİ BİRDEN (hybrid):
  Hata modları FARKLIDIR ve birbirini tamamlar.
    * Dense: eşanlam ve parafrazı yakalar ("araç" ↔ "otomobil"),
             ama nadir token'ları (seri no, kısaltma, kod) vektörde "eritir".
    * Sparse: tam kelime örtüşmesinde kusursuz ("VF7OBBHYHJE513871"),
             ama eşanlamı hiç göremez.
  Biri tamamen bozulsa bile diğeri sistemi ayakta tutar — bu, ölçülmüş bir
  gerçektir, teorik bir iddia değil.
"""

from __future__ import annotations

import numpy as np
from rank_bm25 import BM25Okapi

from rag_assistant.domain.models import Chunk, RetrievalStage, ScoredChunk
from rag_assistant.domain.protocols import Embedder, VectorStore
from rag_assistant.observability import get_logger
from rag_assistant.retrieval.tokenize import tokenize

logger = get_logger(__name__)


class DenseRetriever:
    """Embedding tabanlı anlamsal arama. `Retriever` protokolünü sağlar."""

    def __init__(self, embedder: Embedder, store: VectorStore) -> None:
        self._embedder = embedder
        self._store = store

    @property
    def name(self) -> str:
        return "dense"

    def retrieve(self, query: str, k: int) -> list[ScoredChunk]:
        if self._store.count == 0:
            return []
        vector = self._embedder.embed_query(query)
        results = self._store.search(vector, k)
        logger.debug("retrieve.dense", query=query[:60], results=len(results))
        return results


class BM25Retriever:
    """
    BM25 tabanlı kelime araması. `Retriever` protokolünü sağlar.

    KRİTİK TASARIM NOTU — chunk'lar STORE'DAN alınır:
      BM25 korpusu ile dense aramanın döndürdüğü chunk'lar AYNI nesnelerden
      gelmeli. Ayrı bir kaynaktan ikinci bir kopya yüklenirse, birleştirme
      aşaması iki kümeyi eşleştiremez.
      Bu projede kimlik içerikten türediği için (Chunk.id = sha256) böyle bir
      hata olsa bile birleştirme çalışır — yani iki ayrı güvenlik katmanımız
      var. Yine de tek kaynak kullanmak hem daha hızlı hem daha doğru.
    """

    def __init__(self, store: VectorStore, *, stem: bool = True) -> None:
        self._store = store
        self._stem = stem
        self._chunks: list[Chunk] = []
        self._bm25: BM25Okapi | None = None
        # Her dokümanın token kümesi. "Gerçekten eşleşti mi?" sorusunu
        # skordan bağımsız cevaplamak için tutuluyor (bkz. retrieve()).
        self._token_sets: list[set[str]] = []
        self._built_for_count = -1

    @property
    def name(self) -> str:
        return "sparse"

    def _ensure_index(self) -> None:
        """
        BM25 indeksini gerektiğinde (yeniden) kur.

        Store'daki chunk sayısı değiştiyse indeks bayattır. Bu kontrol
        olmadan, ingestion sonrası yeni eklenen chunk'lar sparse aramada
        HİÇ görünmez — ve bu sessiz bir kayıptır.
        """
        current = self._store.count
        if self._bm25 is not None and self._built_for_count == current:
            return

        self._chunks = self._store.all_chunks()
        if not self._chunks:
            self._bm25 = None
            self._token_sets = []
            self._built_for_count = current
            return

        corpus = [tokenize(c.text, stem=self._stem) for c in self._chunks]
        self._bm25 = BM25Okapi(corpus)
        self._token_sets = [set(tokens) for tokens in corpus]
        self._built_for_count = current
        logger.info("bm25.built", documents=len(corpus))

    def retrieve(self, query: str, k: int) -> list[ScoredChunk]:
        self._ensure_index()
        if self._bm25 is None:
            return []

        tokens = tokenize(query, stem=self._stem)
        if not tokens:
            return []

        scores = self._bm25.get_scores(tokens)
        query_tokens = set(tokens)

        # ELEME ÖLÇÜTÜ SKOR DEĞİL, GERÇEK TOKEN ÖRTÜŞMESİDİR.
        #
        # Neden: BM25'in IDF terimi log((N-n+0.5)/(n+0.5)) biçimindedir ve
        # n/N ≈ 0.5 olduğunda SIFIR olur. Küçük korpuslarda bu sürekli olur:
        # N=2 doküman, terim 1'inde geçiyorsa idf = log(1) = 0 → skor 0.
        # Yani GERÇEK bir eşleşme 0 puan alır. `score > 0` ile elemek bu
        # eşleşmeleri sessizce atar ve sparse retriever küçük korpuslarda
        # fiilen çalışmaz hale gelir. (Ölçüldü: 6 chunk'lık index'te sparse
        # sonuçların bir kısmı bu yüzden kayboluyordu.)
        #
        # Doğru soru "skoru pozitif mi?" değil, "sorgu kelimesi bu chunk'ta
        # geçiyor mu?" sorusudur. Sıralama yine BM25 skoruna göre yapılır.
        candidates = [
            i for i in range(len(self._chunks)) if query_tokens & self._token_sets[i]
        ]
        if not candidates:
            return []

        # argsort O(n log n); bu ölçekte yeterli. Milyonlarca chunk'ta
        # np.argpartition ile O(n) yapılır.
        order = sorted(candidates, key=lambda i: float(scores[i]), reverse=True)[:k]

        results = [
            ScoredChunk(
                chunk=self._chunks[i],
                score=float(scores[i]),
                stage=RetrievalStage.SPARSE,
                rank=rank,
            )
            for rank, i in enumerate(order, start=1)
        ]
        logger.debug("retrieve.sparse", query=query[:60], results=len(results))
        return results
