"""
Doküman kütüphanesi testleri.

Buradaki testlerin çoğu GÜVENLİK testidir. Dosya yükleme, bir web
uygulamasının en çok istismar edilen uç noktasıdır; her doğrulamanın
gerçekten çalıştığı kanıtlanmalı — "yazdım, herhalde çalışıyor" yeterli
değil.
"""

from __future__ import annotations

import pytest

from rag_assistant.domain.models import Chunk, DocumentStatus, SourceRef
from rag_assistant.ingestion.manifest import IngestManifest, ManifestEntry
from rag_assistant.library import (
    DocumentExistsError,
    DocumentLibrary,
    DocumentNotFoundError,
    FileTooLargeError,
    InvalidFileNameError,
    UnsupportedFileTypeError,
    sanitize_file_name,
)


# ---------------------------------------------------------------------------
# Dosya adı güvenliği — EN KRİTİK BÖLÜM
# ---------------------------------------------------------------------------
class TestPathTraversal:
    """
    Dizin aşımı: istemci `../../../etc/passwd` gönderirse hedef klasörün
    DIŞINA yazılır. Sistem dosyalarının üzerine yazmak buradan başlar.
    """

    @pytest.mark.parametrize(
        ("attack", "expected"),
        [
            ("../../../etc/passwd", "passwd"),
            ("../../secret.pdf", "secret.pdf"),
            ("/etc/shadow", "shadow"),
            ("..\\..\\Windows\\System32\\hosts", "hosts"),
            ("C:\\Windows\\win.ini", "win.ini"),
            ("dir/alt/dosya.pdf", "dosya.pdf"),
        ],
    )
    def test_dizin_bilesenleri_atilir(self, attack: str, expected: str) -> None:
        assert sanitize_file_name(attack) == expected

    def test_sonucta_hic_ayirici_kalmaz(self) -> None:
        """Çift kontrol: bu hatanın bedeli yüksek olduğu için iki kez bakılır."""
        for attack in ["../../a/b/c.pdf", "..\\..\\a\\b.pdf", "//etc//x.pdf"]:
            result = sanitize_file_name(attack)
            assert "/" not in result
            assert "\\" not in result
            assert ".." not in result

    def test_sadece_nokta_reddedilir(self) -> None:
        for bad in ["..", ".", "../"]:
            with pytest.raises(InvalidFileNameError):
                sanitize_file_name(bad)


class TestFileNameValidation:
    def test_null_bayt_reddedilir(self) -> None:
        """Null bayt, C tabanlı dosya sistemi çağrılarında adı kesebilir."""
        with pytest.raises(InvalidFileNameError, match="kontrol karakteri"):
            sanitize_file_name("rapor\x00.pdf")

    def test_gorunmez_karakter_reddedilir(self) -> None:
        """Görünüşte aynı iki farklı dosya adı (homograph) engellenir."""
        with pytest.raises(InvalidFileNameError, match="kontrol karakteri"):
            sanitize_file_name("rapor\u200b.pdf")  # zero-width space

    def test_gizli_dosya_reddedilir(self) -> None:
        with pytest.raises(InvalidFileNameError, match="nokta ile başlayamaz"):
            sanitize_file_name(".gizli.pdf")

    @pytest.mark.parametrize("reserved", ["CON.pdf", "nul.pdf", "COM1.pdf", "LPT9.pdf"])
    def test_windows_ayrilmis_adlari(self, reserved: str) -> None:
        with pytest.raises(InvalidFileNameError, match="ayrılmış"):
            sanitize_file_name(reserved)

    def test_uzunluk_siniri(self) -> None:
        with pytest.raises(InvalidFileNameError, match="çok uzun"):
            sanitize_file_name("a" * 300 + ".pdf")

    def test_bos_ad(self) -> None:
        for bad in ["", "   ", "\t"]:
            with pytest.raises(InvalidFileNameError):
                sanitize_file_name(bad)

    @pytest.mark.parametrize("bad", ["rapor<>.pdf", "a|b.pdf", 'x".pdf', "a*b.pdf"])
    def test_tehlikeli_karakterler(self, bad: str) -> None:
        with pytest.raises(InvalidFileNameError, match="izin verilmeyen"):
            sanitize_file_name(bad)

    @pytest.mark.parametrize(
        "ok",
        [
            "rapor.pdf",
            "SmartSafe AI - Kopya.pdf",
            "Poliçe (2024) [v2].pdf",
            "ürün&fiyat.pdf",
            "İŞ_planı.pdf",
        ],
    )
    def test_gecerli_adlar_kabul_edilir(self, ok: str) -> None:
        """Türkçe karakterler ve yaygın noktalama ÇALIŞMALI — aşırı katı olamayız."""
        assert sanitize_file_name(ok) == ok

    def test_nfc_normalize_edilir(self) -> None:
        """Aynı görünen iki ad aynı bayt dizisine inmeli."""
        assert sanitize_file_name("güvenlik.pdf") == "güvenlik.pdf"


# ---------------------------------------------------------------------------
# Sahte store
# ---------------------------------------------------------------------------
class FakeStore:
    def __init__(self) -> None:
        self.count = 0
        self.removed: list[str] = []
        self.saved = False

    def remove_by_file(self, file_name: str) -> int:
        self.removed.append(file_name)
        return 3

    def save(self, directory) -> None:  # type: ignore[no-untyped-def] # noqa: ANN001
        self.saved = True


@pytest.fixture
def library(tmp_path):  # type: ignore[no-untyped-def]
    from rag_assistant.config import Settings

    settings = Settings(paths={"data_dir": tmp_path})
    settings.paths.ensure()
    store = FakeStore()
    manifest = IngestManifest(settings.paths.index_dir)
    return DocumentLibrary(settings, store=store, manifest=manifest), settings, store, manifest


def blocks(data: bytes, size: int = 8192):  # type: ignore[no-untyped-def]
    for i in range(0, len(data), size):
        yield data[i : i + size]


# ---------------------------------------------------------------------------
# Yükleme
# ---------------------------------------------------------------------------
class TestUpload:
    def test_basarili_yukleme(self, library) -> None:  # type: ignore[no-untyped-def]
        lib, settings, _, _ = library
        result = lib.save_upload("rapor.pdf", blocks(b"%PDF-1.4 icerik"))

        assert result.file_name == "rapor.pdf"
        assert result.size_bytes == 15
        assert (settings.paths.raw_dir / "rapor.pdf").exists()

    def test_boyut_siniri_akista_uygulanir(self, library) -> None:  # type: ignore[no-untyped-def]
        """
        Content-Length'e GÜVENİLMEZ. Sınır, yazarken sayılan bayta göre
        uygulanır — aksi halde 50 MB sınırı olan uca 10 GB gönderilebilirdi.
        """
        lib, settings, _, _ = library
        settings.upload.max_size_bytes = 100

        with pytest.raises(FileTooLargeError):
            lib.save_upload("buyuk.pdf", blocks(b"x" * 500, size=50))

        # Yarım dosya BIRAKILMAMALI
        assert list(settings.paths.raw_dir.glob("*")) == []

    def test_desteklenmeyen_uzanti(self, library) -> None:  # type: ignore[no-untyped-def]
        lib, _, _, _ = library
        with pytest.raises(UnsupportedFileTypeError):
            lib.save_upload("zararli.exe", blocks(b"MZ"))

    def test_uzantisiz_dosya(self, library) -> None:  # type: ignore[no-untyped-def]
        lib, _, _, _ = library
        with pytest.raises(UnsupportedFileTypeError):
            lib.save_upload("dosya", blocks(b"veri"))

    def test_bos_dosya_reddedilir(self, library) -> None:  # type: ignore[no-untyped-def]
        lib, settings, _, _ = library
        with pytest.raises(InvalidFileNameError, match="boş"):
            lib.save_upload("bos.pdf", blocks(b""))
        assert list(settings.paths.raw_dir.glob("*")) == []

    def test_ayni_ad_varsayilan_olarak_reddedilir(self, library) -> None:  # type: ignore[no-untyped-def]
        lib, _, _, _ = library
        lib.save_upload("rapor.pdf", blocks(b"birinci"))
        with pytest.raises(DocumentExistsError):
            lib.save_upload("rapor.pdf", blocks(b"ikinci"))

    def test_overwrite_ile_uzerine_yazilir(self, library) -> None:  # type: ignore[no-untyped-def]
        lib, settings, _, _ = library
        lib.save_upload("rapor.pdf", blocks(b"birinci"))
        lib.save_upload("rapor.pdf", blocks(b"ikinci icerik"), overwrite=True)
        assert (settings.paths.raw_dir / "rapor.pdf").read_bytes() == b"ikinci icerik"

    def test_dizin_asimi_hedef_klasorde_kalir(self, library) -> None:  # type: ignore[no-untyped-def]
        """Saldırı denemesi hedef klasörün DIŞINA yazamaz."""
        lib, settings, _, _ = library
        result = lib.save_upload("../../../kacti.pdf", blocks(b"veri"))

        assert result.file_name == "kacti.pdf"
        written = (settings.paths.raw_dir / "kacti.pdf").resolve()
        assert written.parent == settings.paths.raw_dir.resolve()

    def test_ayni_icerik_farkli_ad_bildirilir(self, library) -> None:  # type: ignore[no-untyped-def]
        """
        Aynı doküman iki adla yüklenirse iki kez indekslenir ve aynı bilgi
        sonuçlarda iki kez çıkar. Engellemiyoruz ama BİLDİRİYORUZ.
        """
        lib, _, _, manifest = library
        first = lib.save_upload("birinci.pdf", blocks(b"ayni icerik"))
        manifest.record(
            ManifestEntry(
                file_name="birinci.pdf",
                content_hash=first.content_hash,
                status=DocumentStatus.OK,
                page_count=1,
                chunk_count=2,
            )
        )

        second = lib.save_upload("ikinci.pdf", blocks(b"ayni icerik"))
        assert second.duplicate_of == "birinci.pdf"

    def test_yarim_yukleme_temizlenir(self, library) -> None:  # type: ignore[no-untyped-def]
        """Hata durumunda geçici dosya kalmamalı — ingestion onu işlemeye kalkar."""
        lib, settings, _, _ = library

        def patlayan():  # type: ignore[no-untyped-def]
            yield b"basladi"
            raise OSError("disk hatası")

        with pytest.raises(OSError):
            lib.save_upload("yarim.pdf", patlayan())

        assert list(settings.paths.raw_dir.glob("*")) == []


# ---------------------------------------------------------------------------
# Listeleme
# ---------------------------------------------------------------------------
class TestList:
    def test_bos_kutuphane(self, library) -> None:  # type: ignore[no-untyped-def]
        lib, _, _, _ = library
        assert lib.list_documents() == []

    def test_islenmemis_dosya_da_listelenir(self, library) -> None:  # type: ignore[no-untyped-def]
        """Kullanıcı yüklediği dosyayı görmeli, henüz indekslenmemiş olsa bile."""
        lib, _, _, _ = library
        lib.save_upload("yeni.pdf", blocks(b"veri"))

        docs = lib.list_documents()
        assert len(docs) == 1
        assert docs[0].file_name == "yeni.pdf"
        assert not docs[0].is_searchable

    def test_manifest_durumu_yansitilir(self, library) -> None:  # type: ignore[no-untyped-def]
        lib, _, _, manifest = library
        lib.save_upload("islenmis.pdf", blocks(b"veri"))
        manifest.record(
            ManifestEntry(
                file_name="islenmis.pdf",
                content_hash="h",
                status=DocumentStatus.OK,
                page_count=5,
                chunk_count=6,
            )
        )

        doc = lib.list_documents()[0]
        assert doc.status is DocumentStatus.OK
        assert doc.chunk_count == 6
        assert doc.is_searchable

    def test_taranmis_pdf_aranabilir_degil(self, library) -> None:  # type: ignore[no-untyped-def]
        """0 chunk üreten doküman aramaya DAHİL DEĞİL — sessizce 'ok' görünmemeli."""
        lib, _, _, manifest = library
        lib.save_upload("tarama.pdf", blocks(b"veri"))
        manifest.record(
            ManifestEntry(
                file_name="tarama.pdf",
                content_hash="h",
                status=DocumentStatus.NO_TEXT_LAYER,
                page_count=7,
                chunk_count=0,
            )
        )

        doc = lib.list_documents()[0]
        assert doc.needs_ocr
        assert not doc.is_searchable

    def test_gizli_dosyalar_listelenmez(self, library) -> None:  # type: ignore[no-untyped-def]
        lib, settings, _, _ = library
        (settings.paths.raw_dir / ".gecici.upload").write_bytes(b"x")
        assert lib.list_documents() == []


# ---------------------------------------------------------------------------
# Silme — TUTARLILIK
# ---------------------------------------------------------------------------
class TestDelete:
    def test_uc_yerden_birden_silinir(self, library) -> None:  # type: ignore[no-untyped-def]
        """
        Index + manifest + disk. Biri atlanırsa sistem tutarsız kalır:
        silinmiş dokümandan alıntı yapan cevaplar üretilir.
        """
        lib, settings, store, manifest = library
        lib.save_upload("silinecek.pdf", blocks(b"veri"))
        manifest.record(
            ManifestEntry(
                file_name="silinecek.pdf",
                content_hash="h",
                status=DocumentStatus.OK,
                page_count=1,
                chunk_count=3,
            )
        )

        lib.delete("silinecek.pdf")

        assert store.removed == ["silinecek.pdf"]           # index
        assert store.saved                                   # index diske yazıldı
        assert manifest.get("silinecek.pdf") is None          # manifest
        assert not (settings.paths.raw_dir / "silinecek.pdf").exists()  # disk

    def test_olmayan_dokuman(self, library) -> None:  # type: ignore[no-untyped-def]
        lib, _, _, _ = library
        with pytest.raises(DocumentNotFoundError):
            lib.delete("yok.pdf")

    def test_silmede_de_dizin_asimi_engellenir(self, library) -> None:  # type: ignore[no-untyped-def]
        """
        Silme yolu da temizlenmeli. Aksi halde `DELETE /documents/../../x`
        ile hedef klasör dışındaki dosyalar silinebilirdi.
        """
        lib, _, _, _ = library
        with pytest.raises(DocumentNotFoundError):
            lib.delete("../../../etc/passwd")

    def test_hepsini_sil(self, library) -> None:  # type: ignore[no-untyped-def]
        lib, _, _, _ = library
        lib.save_upload("a.pdf", blocks(b"a"))
        lib.save_upload("b.pdf", blocks(b"b"))
        assert lib.delete_all() == 2
        assert lib.list_documents() == []
