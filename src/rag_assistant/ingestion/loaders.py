"""
Doküman yükleyiciler: dosya → yapısı korunmuş metin.

TEMEL İLKE: Metin temizlerken YAPIYI KORU.
  Belgelerde anlamın büyük kısmı düzenden gelir: paragraf sınırları, başlık
  satırları, "Etiket: değer" hizalamaları. Bütün boşlukları tek boşluğa
  indiren bir "temizleme" adımı bu bilginin tamamını yok eder ve hem
  chunking'i hem de sonraki her aşamayı bozar.

  Bu yüzden burada yaptığımız normalizasyon CERRAHİ:
    * Satır sonu tirelemesini birleştir  ("daya-\nnım" → "dayanım")
    * Satır içi fazla boşluğu tekle       (ama satır sonlarını KORU)
    * 3+ boş satırı 2'ye indir            (paragraf sınırı olarak 2 yeter)
    * Görünmez/kontrol karakterlerini at
  Paragraf ve satır yapısı olduğu gibi kalır.
"""

from __future__ import annotations

import hashlib
import logging
import re
import unicodedata
from pathlib import Path

from pypdf import PdfReader
from pypdf.errors import PdfReadError

from rag_assistant.domain.models import LoadedDocument, LoadedPage
from rag_assistant.observability import get_logger

logger = get_logger(__name__)

# pypdf bozuk PDF'lerde onlarca satır iç uyarı basar
# ("incorrect startxref pointer", "Error -3 while decompressing data" ...).
# Bunları BASTIRIYORUZ çünkü:
#   * Aynı bilgiyi kendimiz zaten yapılandırılmış biçimde raporluyoruz
#     (page.extract_failed / document.loaded / no_text_layer).
#   * Kütüphanenin ham gürültüsü, bizim anlamlı çıktımızı boğuyor.
# ERROR seviyesi açık bırakılır: gerçek bir kütüphane hatası hâlâ görünür.
logging.getLogger("pypdf").setLevel(logging.ERROR)


class DocumentLoadError(RuntimeError):
    """Dosya okunamadı. Çağıran taraf bunu FAILED durumuna çevirir."""


# Satır sonunda tirelenmiş kelime: "daya-\nnım" → "dayanım"
#
# YALNIZCA HARF-tire-satırsonu-HARF kalıbı birleştirilir. `\w` kullanmak
# rakamları da yakalar ve "2020-\n2021" gibi aralıkları "20202021" yapar —
# yani tarih/sayı aralıklarını sessizce bozar. Bu yüzden `[^\W\d_]`
# (harf, ama rakam ve alt çizgi değil) kullanıyoruz.
# Lookbehind/lookahead ile sadece "-\n" siliniyor, harfler yerinde kalıyor.
_HYPHEN_BREAK = re.compile(r"(?<=[^\W\d_])-\n(?=[^\W\d_])", flags=re.UNICODE)

# Satır İÇİNDEKİ fazla boşluk (\n hariç! bu yüzden [^\S\n] kullanıyoruz).
_INLINE_SPACES = re.compile(r"[^\S\n]{2,}")

# 3 veya daha fazla satır sonu → 2 (tek paragraf sınırı yeter)
_EXCESS_NEWLINES = re.compile(r"\n{3,}")

# Yazdırılamayan kontrol karakterleri (sekme ve satır sonu hariç)
_CONTROL_CHARS = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")


def normalize_text(raw: str) -> str:
    """
    PDF'ten çıkan ham metni, YAPISINI BOZMADAN temizle.

    Unicode NFC normalizasyonu Türkçe için özellikle önemli: aynı karakter
    (ör. "ğ") birleşik tek kod noktası veya "g + birleşen aksan" olarak
    kodlanmış olabilir. NFC'ye çevirmezsek aynı kelime iki farklı bayt
    dizisi olur ve hem arama hem tekilleştirme sessizce başarısız olur.
    """
    text = unicodedata.normalize("NFC", raw)
    text = _CONTROL_CHARS.sub("", text)
    text = _HYPHEN_BREAK.sub("", text)
    text = _INLINE_SPACES.sub(" ", text)
    text = _EXCESS_NEWLINES.sub("\n\n", text)
    # Her satırın sağındaki boşluğu at, satır yapısını koru
    text = "\n".join(line.rstrip() for line in text.split("\n"))
    return text.strip()


def file_content_hash(path: Path) -> str:
    """
    Dosya İÇERİĞİNİN hash'i — idempotency'nin temeli.

    Neden değiştirilme zamanı (mtime) değil:
      mtime kopyalama, git checkout, yedekten geri yükleme, dosya
      senkronizasyonu gibi işlemlerde içerik hiç değişmeden değişir. O zaman
      sistem dosyayı "değişmiş" sayıp baştan işler — boşa iş. Tersi de olur:
      farklı içerik aynı mtime'la gelirse hiç işlenmez.
      İçerik hash'i tek doğru ölçüttür.
    """
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()[:32]


class PdfLoader:
    """
    pypdf tabanlı PDF yükleyici. `DocumentLoader` protokolünü sağlar.

    Bu sınıf, pypdf'in projede bilindiği TEK yerdir. Yarın PyMuPDF veya
    docling'e geçilecekse yalnızca burası değişir; hiçbir üst katman
    hangi kütüphanenin kullanıldığını bilmez.
    """

    extensions = frozenset({".pdf"})

    def supports(self, path: Path) -> bool:
        return path.suffix.lower() in self.extensions

    def load(self, path: Path) -> LoadedDocument:
        content_hash = file_content_hash(path)

        try:
            reader = PdfReader(str(path), strict=False)
            raw_pages = list(reader.pages)
        except (PdfReadError, OSError, ValueError) as exc:
            raise DocumentLoadError(f"{path.name}: PDF açılamadı ({exc})") from exc

        pages: list[LoadedPage] = []
        failed_pages = 0

        for index, page in enumerate(raw_pages, start=1):
            # Tek bir bozuk sayfa TÜM dokümanı düşürmemeli. Bozuk PDF'ler
            # yaygındır; kısmi başarı, tam başarısızlıktan iyidir.
            try:
                text = normalize_text(page.extract_text() or "")
            except Exception as exc:  # noqa: BLE001 - pypdf çeşitli hata tipleri atar
                failed_pages += 1
                logger.warning(
                    "page.extract_failed", file=path.name, page=index, error=str(exc)
                )
                text = ""
            pages.append(LoadedPage(page_number=index, text=text))

        document = LoadedDocument(
            file_name=path.name,
            pages=tuple(pages),
            content_hash=content_hash,
        )

        logger.info(
            "document.loaded",
            file=path.name,
            pages=len(pages),
            failed_pages=failed_pages,
            chars=document.total_chars,
            has_text_layer=document.has_text_layer,
        )
        return document


class LoaderRegistry:
    """
    Uzantıya göre doğru loader'ı bulan kayıt defteri.

    Yeni format desteği eklemek = yeni bir loader yazıp buraya kaydetmek.
    Mevcut kodun hiçbir satırı değişmez (açık/kapalı ilkesi). Örneğin
    taranmış PDF'ler için bir OCR loader'ı ileride buraya eklenebilir.
    """

    def __init__(self, loaders: list[PdfLoader] | None = None) -> None:
        self._loaders: list[PdfLoader] = loaders if loaders is not None else [PdfLoader()]

    def find(self, path: Path) -> PdfLoader | None:
        return next((ld for ld in self._loaders if ld.supports(path)), None)

    def supported_extensions(self) -> set[str]:
        return {ext for ld in self._loaders for ext in ld.extensions}
