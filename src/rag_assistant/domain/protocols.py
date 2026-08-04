"""
Protokoller — katmanlar arası sözleşmeler.

Bu dosya mimarinin belkemiğidir. Buradaki her `Protocol`, "bir şeyin ne
YAPTIĞINI" tanımlar, "nasıl yaptığını" tanımlamaz.

NEDEN PROTOCOL, NEDEN ABC DEĞİL:
  `Protocol` yapısal tipleme (structural typing) sağlar: bir sınıfın bu
  protokolü sağlaması için ondan MİRAS ALMASI gerekmez, sadece aynı imzaya
  sahip metotları olması yeterlidir. Sonuç: üçüncü parti bir sınıfı bile
  hiç dokunmadan protokolümüze uyduramazsak ince bir sarmalayıcı yazarız,
  ama kütüphane bizim sınıf hiyerarşimize girmek zorunda kalmaz.
  Bağımlılık yönü tersine döner: kütüphane bize uyar, biz kütüphaneye değil.

BUNUN PRATİK KARŞILIĞI:
  * Embedding modelini değiştirmek → tek bir sınıfı değiştirmek.
  * Testte gerçek modeli yüklemek yerine 5 satırlık bir fake yazmak.
  * FAISS'ten Qdrant'a geçmek → `VectorStore` protokolünün yeni bir
    implementasyonu; retrieval katmanı tek satır değişmez.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Protocol, runtime_checkable

import numpy as np

from rag_assistant.domain.models import (
    Chunk,
    LoadedDocument,
    ScoredChunk,
)

# Embedding matrisi için tip takma adı: (n_metin, boyut), float32
EmbeddingMatrix = np.ndarray


@runtime_checkable
class Embedder(Protocol):
    """
    Metni vektöre çeviren bileşen.

    DİKKAT — `embed_documents` ve `embed_query` AYRI metotlar:
      Retrieval asimetriktir. Sorgu kısa ve soru biçimindedir; pasaj uzun ve
      cevap biçimindedir. Bazı modeller (E5 ailesi) bu ikisini ayırt etmek
      için metnin başına farklı önekler ("query:" / "passage:") koymanı
      ZORUNLU kılar; koymazsan kalite düşer ve HİÇBİR HATA ALMAZSIN.
      Bu iki metodu ayırmak, o hatayı mimari olarak imkânsız kılar —
      önek mantığı implementasyonun içinde kalır, çağıran tarafın bilmesi
      gerekmez.
    """

    @property
    def model_id(self) -> str:
        """Model kimliği. Index metadata'sına yazılır: index hangi modelle kuruldu?"""
        ...

    @property
    def dimension(self) -> int:
        """Vektör boyutu. Store bunu doğrulamak için kullanır."""
        ...

    @property
    def max_tokens(self) -> int:
        """Modelin kabul ettiği azami token. Chunker bunu AŞMAMAK zorundadır."""
        ...

    def count_tokens(self, text: str) -> int:
        """
        Metnin token sayısı.

        Chunker'ın karakter değil TOKEN saymasını mümkün kılan metot.
        Bu olmadan "chunk_size=1000 karakter" gibi bir ayar yaparsın ve
        modelin 512 token limiti sessizce metnin sonunu keser.
        """
        ...

    def embed_documents(self, texts: Sequence[str]) -> EmbeddingMatrix:
        """Pasajları vektörleştir. Çıktı L2-normalize edilmiş olmalıdır."""
        ...

    def embed_query(self, text: str) -> np.ndarray:
        """Tek bir sorguyu vektörleştir. Çıktı L2-normalize edilmiş olmalıdır."""
        ...


@runtime_checkable
class VectorStore(Protocol):
    """
    Vektörleri saklayan ve en yakın komşuyu bulan bileşen.

    Store aynı zamanda chunk'ların TEK doğruluk kaynağıdır (`all_chunks`).
    Neden önemli: BM25 indeksi de aynı chunk kümesine ihtiyaç duyar. İki
    ayrı yerden iki ayrı kopya yüklenirse, kimliğe dayalı her birleştirme
    sessizce bozulur. Tek kaynak → bu hata sınıfı imkânsız hale gelir.
    """

    @property
    def count(self) -> int:
        """Depodaki vektör sayısı."""
        ...

    @property
    def embedder_model_id(self) -> str | None:
        """
        Bu index hangi embedding modeliyle kuruldu?

        Model değişirse index geçersizdir (vektör uzayları uyumsuz).
        Bu alan olmadan uyumsuzluk sessizce çöp sonuç üretir.
        """
        ...

    def add(self, chunks: Sequence[Chunk], vectors: EmbeddingMatrix) -> int:
        """Chunk'ları ekle. Zaten var olan id'ler atlanır. Eklenen sayıyı döndürür."""
        ...

    def search(self, vector: np.ndarray, k: int) -> list[ScoredChunk]:
        """En yakın k chunk'ı skorla birlikte döndür (skor: yüksek = daha iyi)."""
        ...

    def all_chunks(self) -> list[Chunk]:
        """Tüm chunk'lar — BM25 indeksi ve değerlendirme için."""
        ...

    def contains(self, chunk_id: str) -> bool:
        """Idempotency kontrolü: bu chunk zaten indekste mi?"""
        ...

    def save(self, directory: Path) -> None: ...

    def load(self, directory: Path) -> None: ...


@runtime_checkable
class Retriever(Protocol):
    """
    Sorguya en ilgili chunk'ları getiren bileşen.

    Tek metot, tek sorumluluk. Dense, sparse, hybrid ve reranked retriever
    HEPSİ bu protokolü sağlar — bu yüzden birbirinin yerine geçebilir ve
    kompozisyonla birleştirilebilir. Üst katman (generation) hangi
    stratejinin kullanıldığını hiç bilmez.
    """

    @property
    def name(self) -> str:
        """Loglama ve değerlendirmede stratejiyi ayırt etmek için."""
        ...

    def retrieve(self, query: str, k: int) -> list[ScoredChunk]: ...


@runtime_checkable
class Reranker(Protocol):
    """
    Adayları yeniden sıralayan cross-encoder.

    Retriever'dan ayrı bir protokol, çünkü farklı bir işi var: retriever
    ARAR (milyonlarca aday arasından), reranker SIRALAR (onlarca aday
    arasında). İkisini ayırmak, reranker'ı isteğe bağlı kılmayı ve
    açık/kapalı ölçüm yapmayı mümkün kılar.
    """

    @property
    def model_id(self) -> str: ...

    def rerank(
        self, query: str, candidates: Sequence[ScoredChunk], top_k: int
    ) -> list[ScoredChunk]: ...


@runtime_checkable
class LLM(Protocol):
    """
    Metin üreten model.

    `generate` ve `stream` ayrı: API katmanı token token akıtmak isterken
    değerlendirme katmanı tam metni bir kerede ister. İkisini tek metotta
    birleştirmek her iki tarafı da rahatsız eder.
    """

    @property
    def model_id(self) -> str: ...

    def generate(self, prompt: str, *, temperature: float | None = None) -> str: ...

    def stream(self, prompt: str, *, temperature: float | None = None) -> Iterator[str]: ...


@runtime_checkable
class DocumentLoader(Protocol):
    """
    Dosyayı okunabilir sayfalara çeviren bileşen.

    Hangi formatları desteklediğini `supports` ile bildirir — yeni bir format
    eklemek yeni bir loader yazmak demektir, mevcut kodu değiştirmek değil
    (açık/kapalı ilkesi).
    """

    def supports(self, path: Path) -> bool: ...

    def load(self, path: Path) -> LoadedDocument: ...


@runtime_checkable
class Chunker(Protocol):
    """Dokümanı chunk'lara bölen bileşen."""

    def split(self, document: LoadedDocument) -> list[Chunk]: ...
