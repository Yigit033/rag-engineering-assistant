"""
Faz 3 testleri: Türkçe tokenizasyon, RRF birleştirme, retriever davranışı.

Bu testler de model yüklemiyor. Retriever'lar `Embedder` ve `VectorStore`
protokollerine bağlı olduğu için testte 20 satırlık sahte implementasyonlar
geçiyoruz. Protokol tabanlı mimarinin en somut getirisi budur.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pytest

from rag_assistant.domain.models import Chunk, RetrievalStage, ScoredChunk, SourceRef
from rag_assistant.retrieval.fusion import reciprocal_rank_fusion
from rag_assistant.retrieval.pipeline import HybridRetriever
from rag_assistant.retrieval.retrievers import BM25Retriever, DenseRetriever
from rag_assistant.retrieval.tokenize import light_stem, normalize, tokenize


def chunk(text: str, *, file: str = "a.pdf", page: int = 1) -> Chunk:
    return Chunk(text=text, source=SourceRef(file, page), token_count=len(text.split()))


# ---------------------------------------------------------------------------
# Türkçe tokenizasyon
# ---------------------------------------------------------------------------
class TestTurkishNormalize:
    def test_buyuk_I_dogru_kucultulur(self) -> None:
        """
        Python'da "İ".lower() -> 'i̇' (i + birleşen nokta) = İKİ karakter.
        Naif .lower() kullanan her Türkçe kod burada sessizce bozulur.
        """
        assert len("İ".lower()) == 2, "Python davranışı değişmiş — testi gözden geçir"
        assert normalize("İSTANBUL") == "istanbul"
        assert len(normalize("İ")) == 1

    def test_noktasiz_I_dogru_kucultulur(self) -> None:
        assert normalize("IŞIK") == "ışık"

    def test_tum_turkce_harfler(self) -> None:
        assert normalize("ÇĞİÖŞÜ") == "çğiöşü"

    def test_noktalama_atilir(self) -> None:
        assert tokenize("poliçe, tarih.") == tokenize("poliçe tarih")


class TestTurkishStemmer:
    @pytest.mark.parametrize(
        ("base", "inflected"),
        [
            ("poliçe", "poliçenin"),
            ("poliçe", "poliçeler"),
            ("gemi", "gemiler"),
            ("teminat", "teminatı"),
            ("gövde", "gövdesi"),
            ("tarih", "tarihinde"),
            ("rapor", "raporların"),
            ("sistem", "sistemleri"),
            ("güvenlik", "güvenliğin"),  # ünsüz yumuşaması k→ğ
            ("kitap", "kitabın"),  # ünsüz yumuşaması p→b
        ],
    )
    def test_ekli_ve_eksiz_ayni_govdeye_iner(self, base: str, inflected: str) -> None:
        """
        Stemmer'ın TEK amacı bu. Tek geçişli kırpma bunu SAĞLAMIYORDU:
        "poliçenin"→"poliçe" ama "poliçe"→"poliç" — asimetrik gövde.
        Yinelemeli kırpma + ünsüz sertleştirme ile çözüldü.
        """
        assert tokenize(base) == tokenize(inflected)

    @pytest.mark.parametrize(
        ("a", "b"),
        [("kalem", "kale"), ("gemi", "gemlik"), ("rapor", "rapel"), ("bina", "binlik")],
    )
    def test_farkli_kelimeler_birlesmez(self, a: str, b: str) -> None:
        """Aşırı kırpma alakasız sonucu ÜST sıraya taşır — kaçırmaktan kötüdür."""
        assert tokenize(a) != tokenize(b)

    def test_kisa_kelime_kirpilmaz(self) -> None:
        """4 harf altı gövdelerde kırpma yapılmaz (bilinçli sınır)."""
        assert light_stem("kod") == "kod"

    def test_sonek_listesi_uzundan_kisaya_denenir(self) -> None:
        """
        "gövdesi" için önce "si" denenmeli, "i" değil. Liste elle sıralanırsa
        biri araya kısa ek ekleyince bu kural sessizce bozulur; sıralama
        koda bırakıldı.
        """
        assert light_stem("gövdesi") == light_stem("gövde")


# ---------------------------------------------------------------------------
# RRF — birleştirmenin kalbi
# ---------------------------------------------------------------------------
class TestRRF:
    def test_uzlasma_tek_listedeki_birinciyi_geçer(self) -> None:
        """
        RRF'in var olma sebebi. A dense'te #1 ama tek listede;
        B ikisinde de var → B kazanmalı.
        """
        a, b, c = chunk("alfa"), chunk("beta"), chunk("gama")
        dense = [
            ScoredChunk(a, 0.9, RetrievalStage.DENSE, 1),
            ScoredChunk(c, 0.8, RetrievalStage.DENSE, 2),
            ScoredChunk(b, 0.7, RetrievalStage.DENSE, 3),
        ]
        sparse = [
            ScoredChunk(c, 5.0, RetrievalStage.SPARSE, 1),
            ScoredChunk(b, 3.0, RetrievalStage.SPARSE, 2),
        ]
        fused = reciprocal_rank_fusion([dense, sparse])

        assert fused[0].chunk.id in {b.id, c.id}
        assert fused[-1].chunk.id == a.id, "tek listedeki #1 en sonda olmalı"

    def test_ayri_nesneler_ayni_icerik_birlesir(self) -> None:
        """
        Eski sistemdeki kritik hata: anahtar olarak Python nesne kimliği
        kullanmak. Aynı içerik iki farklı nesnede olduğunda hiç eşleşmez
        ve fusion sessizce çalışmaz hale gelir.
        """
        first = chunk("ayni metin")
        second = chunk("ayni metin")  # AYRI nesne
        assert id(first) != id(second)

        fused = reciprocal_rank_fusion(
            [
                [ScoredChunk(first, 1.0, RetrievalStage.DENSE, 1)],
                [ScoredChunk(second, 1.0, RetrievalStage.SPARSE, 1)],
            ]
        )
        assert len(fused) == 1, "aynı içerik birleşmedi — fusion bozuk"
        assert fused[0].retriever_hits == 2

    def test_hits_sayaci_dogru(self) -> None:
        a, b = chunk("alfa"), chunk("beta")
        fused = reciprocal_rank_fusion(
            [
                [ScoredChunk(a, 1.0, RetrievalStage.DENSE, 1), ScoredChunk(b, 0.5, RetrievalStage.DENSE, 2)],
                [ScoredChunk(a, 2.0, RetrievalStage.SPARSE, 1)],
            ]
        )
        hits = {c.chunk.id: c.retriever_hits for c in fused}
        assert hits[a.id] == 2
        assert hits[b.id] == 1

    def test_k_sabiti_ilk_sira_ustunlugunu_kirar(self) -> None:
        """k büyüdükçe sıralar arası fark daralır → uzlaşma daha çok kazanır."""
        a, b = chunk("alfa"), chunk("beta")
        lists = [
            [ScoredChunk(a, 1.0, RetrievalStage.DENSE, 1), ScoredChunk(b, 0.5, RetrievalStage.DENSE, 2)],
            [ScoredChunk(b, 1.0, RetrievalStage.SPARSE, 1)],
        ]
        with_small_k = reciprocal_rank_fusion(lists, k=1)
        with_large_k = reciprocal_rank_fusion(lists, k=60)
        # k=1: a -> 1/2=0.5 ; b -> 1/3 + 1/2 = 0.833  -> b kazanır
        # k=60 ile de b kazanır ama farklar daralır
        assert with_small_k[0].chunk.id == b.id
        assert with_large_k[0].chunk.id == b.id

    def test_bos_listeler(self) -> None:
        assert reciprocal_rank_fusion([[], []]) == []

    def test_top_k_uygulanir(self) -> None:
        chunks = [chunk(f"metin{i}") for i in range(10)]
        ranked = [ScoredChunk(c, 1.0, RetrievalStage.DENSE, i) for i, c in enumerate(chunks, 1)]
        assert len(reciprocal_rank_fusion([ranked], top_k=3)) == 3

    def test_sonuclar_fused_asamasi_ile_isaretlenir(self) -> None:
        a = chunk("alfa")
        fused = reciprocal_rank_fusion([[ScoredChunk(a, 1.0, RetrievalStage.DENSE, 1)]])
        assert fused[0].stage is RetrievalStage.FUSED


# ---------------------------------------------------------------------------
# Sahte bileşenler — protokol tabanlı mimarinin getirisi
# ---------------------------------------------------------------------------
class FakeEmbedder:
    """20 satırlık sahte embedder. 2.3 GB model yüklemeye gerek yok."""

    model_id = "fake"
    dimension = 3
    max_tokens = 100

    def count_tokens(self, text: str) -> int:
        return len(text.split())

    def embed_documents(self, texts: Sequence[str]) -> np.ndarray:
        return np.array([self._vec(t) for t in texts], dtype=np.float32)

    def embed_query(self, text: str) -> np.ndarray:
        return self._vec(text)

    @staticmethod
    def _vec(text: str) -> np.ndarray:
        v = np.zeros(3, dtype=np.float32)
        for i, word in enumerate(text.lower().split()[:3]):
            v[i] = len(word)
        norm = np.linalg.norm(v)
        return v / norm if norm else v


class FakeStore:
    """Bellek içi sahte store."""

    def __init__(self, chunks: list[Chunk]) -> None:
        self._chunks = chunks

    @property
    def count(self) -> int:
        return len(self._chunks)

    @property
    def embedder_model_id(self) -> str | None:
        return "fake"

    def all_chunks(self) -> list[Chunk]:
        return list(self._chunks)

    def contains(self, chunk_id: str) -> bool:
        return any(c.id == chunk_id for c in self._chunks)

    def search(self, vector: np.ndarray, k: int) -> list[ScoredChunk]:
        return [
            ScoredChunk(c, 1.0 - i * 0.1, RetrievalStage.DENSE, i + 1)
            for i, c in enumerate(self._chunks[:k])
        ]

    def add(self, chunks, vectors) -> int:  # type: ignore[no-untyped-def] # noqa: ANN001
        raise NotImplementedError

    def save(self, directory) -> None:  # type: ignore[no-untyped-def] # noqa: ANN001
        raise NotImplementedError

    def load(self, directory) -> None:  # type: ignore[no-untyped-def] # noqa: ANN001
        raise NotImplementedError


class TestRetrievers:
    def test_bos_store_bos_sonuc(self) -> None:
        dense = DenseRetriever(FakeEmbedder(), FakeStore([]))  # type: ignore[arg-type]
        assert dense.retrieve("soru", 5) == []

    def test_bm25_ekli_kelimeyi_bulur(self) -> None:
        """Stemmer'ın retrieval'a gerçek katkısı."""
        store = FakeStore(
            [
                chunk("Poliçenin başlangıç tarihi 2025 yılıdır"),
                chunk("Gemi gövdesi dayanım hesapları"),
            ]
        )
        results = BM25Retriever(store).retrieve("poliçe", 5)  # type: ignore[arg-type]
        assert results, "ekli biçim bulunamadı — stemmer çalışmıyor"
        assert "Poliçenin" in results[0].chunk.text

    def test_bm25_eslesme_yoksa_bos_doner(self) -> None:
        """Skoru 0 olan sonuç 'eşleşme yok' demektir, döndürülmemeli."""
        store = FakeStore([chunk("tamamen alakasiz icerik")])
        assert BM25Retriever(store).retrieve("kuantum kromodinamik", 5) == []  # type: ignore[arg-type]

    def test_bm25_index_store_degisince_yenilenir(self) -> None:
        """
        Ingestion sonrası yeni chunk'lar sparse aramada görünmeli.
        Görünmezse bu SESSİZ bir kayıptır.
        """
        chunks = [chunk("birinci metin")]
        store = FakeStore(chunks)
        retriever = BM25Retriever(store)  # type: ignore[arg-type]
        retriever.retrieve("birinci", 5)

        chunks.append(chunk("ikinci metin farkli"))
        results = retriever.retrieve("ikinci", 5)
        assert results, "yeni chunk sparse aramada görünmüyor"


class TestHybridPipeline:
    def _store(self) -> FakeStore:
        return FakeStore(
            [
                chunk("Poliçenin başlangıç tarihi bilgisi"),
                chunk("Gemi gövdesi dayanım analizi"),
                chunk("Sistem güvenliği ve raporlama"),
            ]
        )

    def test_hybrid_iki_retrieveri_birlestirir(self) -> None:
        store = self._store()
        hybrid = HybridRetriever(
            [DenseRetriever(FakeEmbedder(), store), BM25Retriever(store)],  # type: ignore[arg-type]
            fetch_k=3,
        )
        results = hybrid.retrieve("poliçe tarih", 3)
        assert results
        assert any(r.retriever_hits > 1 for r in results), "hiç uzlaşma yok — fusion bozuk"

    def test_retriever_olmadan_kurulamaz(self) -> None:
        with pytest.raises(ValueError, match="en az bir retriever"):
            HybridRetriever([], fetch_k=5)

    def test_isim_stratejiyi_gosterir(self) -> None:
        store = self._store()
        hybrid = HybridRetriever(
            [DenseRetriever(FakeEmbedder(), store), BM25Retriever(store)],  # type: ignore[arg-type]
            fetch_k=3,
        )
        assert hybrid.name == "hybrid(dense+sparse)"

    def test_hybrid_kendisi_de_retrieverdir(self) -> None:
        """Kompozisyon: bir pipeline başka bir pipeline'ın içine konabilir."""
        store = self._store()
        inner = HybridRetriever([DenseRetriever(FakeEmbedder(), store)], fetch_k=3)  # type: ignore[arg-type]
        outer = HybridRetriever([inner, BM25Retriever(store)], fetch_k=3)  # type: ignore[arg-type]
        assert outer.retrieve("poliçe", 2)
