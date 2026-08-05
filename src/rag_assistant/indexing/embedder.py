"""
Embedding: metin → vektör.

Bu dosya, sentence-transformers'ın projede bilindiği TEK yerdir.
Üst katmanlar yalnızca `Embedder` protokolünü görür; hangi kütüphanenin
kullanıldığını, modelin nereden yüklendiğini, önek gerekip gerekmediğini
bilmezler. Modeli değiştirmek = burayı değiştirmek.
"""

from __future__ import annotations

from collections.abc import Sequence
from functools import cached_property

import numpy as np

from rag_assistant.observability import get_logger

logger = get_logger(__name__)


class SentenceTransformerEmbedder:
    """
    sentence-transformers tabanlı embedder. `Embedder` protokolünü sağlar.

    TASARIM KARARLARI

    1. TEMBEL YÜKLEME (lazy loading)
       Model 2.3 GB. `__init__` içinde yüklersek bu sınıfı içeren her modül
       import edildiğinde bellek doluyor — testler yavaşlıyor, CLI açılışı
       saniyeler sürüyor. `cached_property` ile ilk gerçek kullanımda yükleniyor.

    2. `embed_documents` / `embed_query` AYRI
       Retrieval asimetriktir. Bazı modeller sorgu ve pasajı ayırt etmek için
       önek ister (E5 ailesinde zorunlu). Öneki burada, tek yerde uyguluyoruz;
       çağıran taraf hiç bilmiyor. Bu, "öneki unutmak" hatasını mimari olarak
       imkânsız kılıyor — ve o hata sessizdir, sadece kaliteyi düşürür.

    3. NORMALİZE EDİLMİŞ ÇIKTI SÖZLEŞMEsi
       Protokol, çıktının L2-normalize olmasını şart koşar. Böylece store
       tarafında inner product doğrudan cosine benzerliğine eşit olur ve
       "normalize etmeyi unutma" hatası ortadan kalkar.
    """

    def __init__(
        self,
        model_name: str,
        *,
        device: str = "cpu",
        batch_size: int = 8,
        max_tokens: int = 1024,
        query_prefix: str = "",
        document_prefix: str = "",
        normalize: bool = True,
    ) -> None:
        self._model_name = model_name
        self._device = device
        self._batch_size = batch_size
        self._max_tokens = max_tokens
        self._query_prefix = query_prefix
        self._document_prefix = document_prefix
        self._normalize = normalize

    # ------------------------------------------------------------------
    # Model erişimi (tembel)
    # ------------------------------------------------------------------
    @cached_property
    def _model(self):  # type: ignore[no-untyped-def] # noqa: ANN202
        from sentence_transformers import SentenceTransformer

        logger.info("embedder.loading", model=self._model_name, device=self._device)

        # ÖNCE YEREL ÖNBELLEKTEN YÜKLEMEYİ DENE.
        #
        # Neden: model önbellekte olmasına rağmen sentence-transformers her
        # açılışta HuggingFace Hub'a sorar. Hub yavaşsa veya hız sınırına
        # takılırsa yükleme BAŞARISIZ olur — yani tamamen yerel çalışan bir
        # servis, dış bir servisin erişilebilirliğine bağımlı hale gelir.
        # Bu fiilen yaşandı: model dakikalar önce yüklenmişken sonraki
        # denemede "Unrecognized processing class" hatası alındı; sebep
        # eksik model dosyası değil, Hub'a ulaşılamamasıydı.
        #
        # İlk indirmeden sonra ağ bağımlılığı tamamen ortadan kalkar.
        try:
            model = SentenceTransformer(
                self._model_name, device=self._device, local_files_only=True
            )
            logger.debug("embedder.loaded_from_cache", model=self._model_name)
        except Exception as exc:  # noqa: BLE001 - önbellekte yoksa indirmeye düş
            logger.info(
                "embedder.cache_miss",
                model=self._model_name,
                reason=str(exc)[:120],
                action="Hub'dan indiriliyor (ilk çalıştırma)",
            )
            model = SentenceTransformer(self._model_name, device=self._device)

        # Modelin kendi limiti (bge-m3'te 8192) genelde bizim istediğimizden
        # büyüktür. Bilinçli olarak kısıtlıyoruz: çok uzun chunk arama
        # hassasiyetini düşürür. Modelin limitini AŞMAK ise sessiz kesme
        # demektir; bu yüzden ikisinin küçüğünü alıyoruz.
        model.max_seq_length = min(self._max_tokens, model.max_seq_length)

        logger.info(
            "embedder.loaded",
            model=self._model_name,
            dimension=self._dimension_of(model),
            max_seq_length=model.max_seq_length,
        )
        return model

    @staticmethod
    def _dimension_of(model: object) -> int:
        """
        Boyut okuma — sentence-transformers 5.x'te metot adı değişti.
        Her iki adı da destekleyerek sürüm oynamalarına dayanıklı kalıyoruz.
        """
        for attr in ("get_embedding_dimension", "get_sentence_embedding_dimension"):
            fn = getattr(model, attr, None)
            if callable(fn):
                return int(fn())
        raise RuntimeError("Embedding boyutu okunamadı")

    # ------------------------------------------------------------------
    # Protokol
    # ------------------------------------------------------------------
    @property
    def model_id(self) -> str:
        return self._model_name

    @cached_property
    def dimension(self) -> int:
        return self._dimension_of(self._model)

    @property
    def max_tokens(self) -> int:
        return self._max_tokens

    def count_tokens(self, text: str) -> int:
        """
        Metnin token sayısı — chunker'ın bütçe hesabı buna dayanır.

        `add_special_tokens=False`: [CLS]/[SEP] gibi özel token'ları saymıyoruz
        çünkü chunker metnin kendi uzunluğuyla ilgileniyor. Sert limitte
        bunların payını bırakmak için `max_tokens` zaten modelin gerçek
        limitinin altında tutuluyor.
        """
        return len(self._model.tokenizer.encode(text, add_special_tokens=False))

    def embed_documents(self, texts: Sequence[str]) -> np.ndarray:
        """Pasajları vektörleştir (n, dim), L2-normalize."""
        if not texts:
            return np.zeros((0, self.dimension), dtype=np.float32)

        prepared = [f"{self._document_prefix}{t}" for t in texts]
        vectors = self._model.encode(
            prepared,
            batch_size=self._batch_size,
            normalize_embeddings=self._normalize,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        return np.asarray(vectors, dtype=np.float32)

    def embed_query(self, text: str) -> np.ndarray:
        """Tek sorguyu vektörleştir (dim,), L2-normalize."""
        vector = self._model.encode(
            f"{self._query_prefix}{text}",
            normalize_embeddings=self._normalize,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        return np.asarray(vector, dtype=np.float32)
