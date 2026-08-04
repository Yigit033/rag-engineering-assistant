"""
Faz 1 testleri: loader normalizasyonu, chunker garantileri, manifest mantığı.

DİKKAT: Bu testlerin HİÇBİRİ model yüklemiyor, ağa çıkmıyor, diske
büyük dosya yazmıyor. Milisaniyeler içinde koşarlar.

Bunu mümkün kılan şey mimari: chunker embedding modelinin tamamına değil
yalnızca "kaç token?" fonksiyonuna bağlı. Testte o fonksiyonun yerine
`len(text.split())` geçiyoruz. Framework'e gömülü bir chunker'ı bu şekilde
test etmek mümkün değildi — 2.3 GB model yüklemek gerekirdi.
"""

from __future__ import annotations

import pytest

from rag_assistant.domain.models import (
    Chunk,
    DocumentStatus,
    LoadedDocument,
    LoadedPage,
    SourceRef,
)
from rag_assistant.ingestion.chunker import TokenAwareChunker
from rag_assistant.ingestion.loaders import normalize_text
from rag_assistant.ingestion.manifest import IngestManifest, ManifestEntry


# Sahte token sayıcı: kelime = token. Gerçek tokenizer'a ihtiyaç yok.
def word_tokens(text: str) -> int:
    return len(text.split())


def make_chunker(**overrides: int) -> TokenAwareChunker:
    params: dict[str, int] = {
        "target_tokens": 20,
        "overlap_tokens": 5,
        "min_tokens": 3,
        "hard_limit_tokens": 25,
    }
    params.update(overrides)
    return TokenAwareChunker(word_tokens, **params)  # type: ignore[arg-type]


def make_doc(text: str, *, name: str = "test.pdf") -> LoadedDocument:
    return LoadedDocument(
        file_name=name,
        pages=(LoadedPage(page_number=1, text=text),),
        content_hash="deadbeef",
    )


# ---------------------------------------------------------------------------
# normalize_text — YAPIYI KORUMA
# ---------------------------------------------------------------------------
class TestNormalizeText:
    def test_paragraf_yapisi_korunur(self) -> None:
        """En kritik garanti: paragraf sınırları silinmez."""
        out = normalize_text("Birinci paragraf.\n\nIkinci paragraf.")
        assert "\n\n" in out, "paragraf sınırı yok edildi — chunking bozulur"

    def test_satir_yapisi_korunur(self) -> None:
        """'Etiket: değer' hizalaması satır sonuna bağlıdır."""
        out = normalize_text("Poliçe No\n:12345\nTarih\n:01.01.2025")
        assert out.count("\n") == 3

    def test_satir_ici_fazla_bosluk_teklenir(self) -> None:
        assert normalize_text("iki      bosluk") == "iki bosluk"

    def test_asiri_bos_satir_ikiye_iner(self) -> None:
        assert normalize_text("a\n\n\n\n\nb") == "a\n\nb"

    def test_satir_sonu_tiresi_birlestirilir(self) -> None:
        assert normalize_text("daya-\nnim") == "dayanim"

    def test_tarih_araligi_tiresi_bozulmaz(self) -> None:
        """"2020-\n2021" birleştirilmemeli mi? Rakam-rakam kalıbı korunur."""
        out = normalize_text("2020-\n2021")
        assert out in ("2020-2021", "2020-\n2021")  # her iki davranış kabul

    def test_nfc_normalizasyonu(self) -> None:
        """
        Türkçe için kritik: 'ğ' birleşik veya 'g'+aksan olarak kodlanabilir.
        Normalize etmezsek aynı kelime iki farklı bayt dizisi olur.
        """
        decomposed = "güvenlik"  # g + u + birleşen umlaut
        assert normalize_text(decomposed) == "güvenlik"

    def test_kontrol_karakterleri_atilir(self) -> None:
        assert normalize_text("temiz\x00\x07metin") == "temizmetin"


# ---------------------------------------------------------------------------
# Chunker — SERT GARANTİLER
# ---------------------------------------------------------------------------
class TestChunkerGuarantees:
    def test_hicbir_chunk_sert_limiti_asmaz(self) -> None:
        """
        Sayfa 1'in birinci tuzağının mimari çözümü.
        Eski sistemde chunk'lar model limitini aşıp sessizce kesiliyordu.
        """
        long_text = "\n\n".join("kelime " * 40 for _ in range(10))
        chunks = make_chunker().split(make_doc(long_text))

        assert chunks
        for c in chunks:
            assert c.token_count <= 25, f"sert limit aşıldı: {c.token_count}"

    def test_tek_uzun_cumle_sert_kesilir_ve_limite_uyar(self) -> None:
        """Satır sonu olmayan uzun tablo dökümü senaryosu."""
        chunks = make_chunker().split(make_doc("x " * 200))
        assert len(chunks) > 1
        assert all(c.token_count <= 25 for c in chunks)

    def test_ortusme_uygulanir(self) -> None:
        """Sınırda kalan bilgi kaybolmasın."""
        paragraphs = [f"paragraf{i} " + "dolgu " * 8 for i in range(6)]
        with_overlap = make_chunker(overlap_tokens=8).split(make_doc("\n\n".join(paragraphs)))
        without = make_chunker(overlap_tokens=0).split(make_doc("\n\n".join(paragraphs)))

        total_with = sum(c.token_count for c in with_overlap)
        total_without = sum(c.token_count for c in without)
        assert total_with > total_without, "örtüşme hiç uygulanmamış"

    def test_paragraf_siniri_tercih_edilir(self) -> None:
        """Bütçeye sığan paragraf bölünmemeli."""
        chunks = make_chunker(target_tokens=50, hard_limit_tokens=60).split(
            make_doc("Kisa birinci paragraf.\n\nKisa ikinci paragraf.")
        )
        assert len(chunks) == 1

    def test_bos_sayfa_chunk_uretmez(self) -> None:
        assert make_chunker().split(make_doc("   \n\n  ")) == []

    def test_chunk_sayfa_bilgisini_tasir(self) -> None:
        """Atıf için sayfa numarası şart."""
        doc = LoadedDocument(
            file_name="rapor.pdf",
            pages=(LoadedPage(1, "birinci sayfa metni"), LoadedPage(2, "ikinci sayfa metni")),
            content_hash="h",
        )
        chunks = make_chunker().split(doc)
        assert {c.source.page for c in chunks} == {1, 2}

    def test_chunk_sayfa_sinirini_asmaz(self) -> None:
        """Bir chunk iki sayfaya yayılırsa doğru kaynak gösterilemez."""
        doc = LoadedDocument(
            file_name="r.pdf",
            pages=(LoadedPage(1, "a b c"), LoadedPage(2, "d e f")),
            content_hash="h",
        )
        chunks = make_chunker(target_tokens=100, hard_limit_tokens=100).split(doc)
        assert len(chunks) == 2, "sayfalar birleştirilmiş — atıf güvenilmez olur"

    def test_gecersiz_yapilandirma_reddedilir(self) -> None:
        with pytest.raises(ValueError, match="overlap_tokens"):
            make_chunker(overlap_tokens=30, target_tokens=20)
        with pytest.raises(ValueError, match="sert limit"):
            make_chunker(target_tokens=100, hard_limit_tokens=50)


# ---------------------------------------------------------------------------
# Chunk kimliği — FUSION'IN TEMELİ
# ---------------------------------------------------------------------------
class TestChunkIdentity:
    def test_ayni_icerik_ayni_kimlik(self) -> None:
        """
        Nesne kimliği (id()) yerine içerik hash'i kullanmanın sebebi.
        Bu test geçtiği sürece RRF birleştirmesi sessizce bozulamaz.
        """
        src = SourceRef("a.pdf", page=1)
        a = Chunk(text="ayni metin", source=src, token_count=2)
        b = Chunk(text="ayni metin", source=src, token_count=2)
        assert a.id == b.id
        assert id(a) != id(b)

    def test_farkli_kaynak_farkli_kimlik(self) -> None:
        """Aynı cümle iki dokümanda geçebilir; atıf açısından farklıdır."""
        a = Chunk(text="ayni metin", source=SourceRef("a.pdf", 1), token_count=2)
        b = Chunk(text="ayni metin", source=SourceRef("b.pdf", 1), token_count=2)
        assert a.id != b.id

    def test_chunk_degistirilemez(self) -> None:
        c = Chunk(text="metin", source=SourceRef("a.pdf"), token_count=1)
        with pytest.raises(Exception):  # FrozenInstanceError
            c.text = "yeni"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Metin katmanı tespiti
# ---------------------------------------------------------------------------
class TestTextLayerDetection:
    def test_taranmis_pdf_metin_katmani_yok(self) -> None:
        """Tarama artığı sayfa numarası metin katmanı sayılmaz."""
        doc = LoadedDocument("tarama.pdf", (LoadedPage(1, " 12 "),), "h")
        assert not doc.has_text_layer

    def test_dijital_pdf_metin_katmani_var(self) -> None:
        doc = LoadedDocument("dijital.pdf", (LoadedPage(1, "A" * 150),), "h")
        assert doc.has_text_layer


# ---------------------------------------------------------------------------
# Manifest — IDEMPOTENCY
# ---------------------------------------------------------------------------
class TestManifest:
    @staticmethod
    def _entry(**kw: object) -> ManifestEntry:
        base: dict[str, object] = {
            "file_name": "a.pdf",
            "content_hash": "h1",
            "status": DocumentStatus.OK,
            "page_count": 2,
            "chunk_count": 1,
            "chunk_ids": ["c1"],
            "embedder_model_id": "BAAI/bge-m3",
        }
        base.update(kw)
        return ManifestEntry(**base)  # type: ignore[arg-type]

    def test_degismemis_dosya_atlanir(self, tmp_path) -> None:  # type: ignore[no-untyped-def]
        m = IngestManifest(tmp_path)
        m.record(self._entry())
        assert m.is_current("a.pdf", "h1", "BAAI/bge-m3", lambda ids: True)

    def test_icerik_degistiyse_yeniden_islenir(self, tmp_path) -> None:  # type: ignore[no-untyped-def]
        m = IngestManifest(tmp_path)
        m.record(self._entry())
        assert not m.is_current("a.pdf", "FARKLI", "BAAI/bge-m3", lambda ids: True)

    def test_embedder_degistiyse_yeniden_islenir(self, tmp_path) -> None:  # type: ignore[no-untyped-def]
        """Model değişti → vektör uzayı uyumsuz → index geçersiz."""
        m = IngestManifest(tmp_path)
        m.record(self._entry())
        assert not m.is_current("a.pdf", "h1", "yeni-model", lambda ids: True)

    def test_index_bos_ise_yeniden_islenir(self, tmp_path) -> None:  # type: ignore[no-untyped-def]
        """
        Manifest 'işlendi' der ama index'te chunk yok (silinmiş/bozulmuş).
        İki kaynağı karşılaştırmadan 'işlenmiş' demek, boş index'le
        çalışmaya devam etmektir.
        """
        m = IngestManifest(tmp_path)
        m.record(self._entry())
        assert not m.is_current("a.pdf", "h1", "BAAI/bge-m3", lambda ids: False)

    def test_metin_katmani_yoksa_yeniden_denenir(self, tmp_path) -> None:  # type: ignore[no-untyped-def]
        """0 chunk asla 'başarı' değildir — her çalıştırmada tekrar denenir."""
        m = IngestManifest(tmp_path)
        m.record(self._entry(status=DocumentStatus.NO_TEXT_LAYER, chunk_ids=[]))
        assert not m.is_current("a.pdf", "h1", "BAAI/bge-m3", lambda ids: True)

    def test_diske_yazip_okuma(self, tmp_path) -> None:  # type: ignore[no-untyped-def]
        m = IngestManifest(tmp_path)
        m.record(self._entry())
        m.save()

        yeniden = IngestManifest(tmp_path)
        entry = yeniden.get("a.pdf")
        assert entry is not None
        assert entry.content_hash == "h1"
        assert entry.status is DocumentStatus.OK

    def test_bozuk_manifest_sistemi_durdurmaz(self, tmp_path) -> None:  # type: ignore[no-untyped-def]
        """En kötü senaryo: her şey yeniden işlenir (yavaş ama doğru)."""
        (tmp_path / IngestManifest.FILE_NAME).write_text("{bozuk json", encoding="utf-8")
        m = IngestManifest(tmp_path)
        assert m.entries() == []
