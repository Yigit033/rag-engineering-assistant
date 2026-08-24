"""
Doküman kütüphanesi — kaynak dosyaların yaşam döngüsü.

NEDEN AYRI BİR SERVİS KATMANI (uç noktanın içine yazmak yerine):
  Dosya yükleme, listeleme ve silme İŞ MANTIĞIDIR, HTTP ayrıntısı değil.
  HTTP uç noktasının içine yazılırsa:
    * CLI aynı işi yapmak için kodu KOPYALAMAK zorunda kalır
    * Test etmek için HTTP sunucusu ayağa kaldırmak gerekir
    * Güvenlik kontrolleri (dosya adı temizleme, boyut sınırı) her
      çağıran yerde TEKRAR yazılır — ve biri unutulur

  Servis katmanı sayesinde API ve CLI aynı kodu çağırır, güvenlik
  kontrolleri TEK yerde durur ve testler HTTP olmadan koşar.

GÜVENLİK — DOSYA YÜKLEME EN ÇOK İSTİSMAR EDİLEN UÇ NOKTADIR:
  Bu dosyadaki her doğrulama bir saldırı yüzeyini kapatır. Hiçbiri
  "kullanıcı dostu hata mesajı" için değil.
"""

from __future__ import annotations

import re
import shutil
import unicodedata
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

from rag_assistant.config import Settings
from rag_assistant.domain.models import DocumentStatus, StoredDocument
from rag_assistant.domain.protocols import VectorStore
from rag_assistant.ingestion.loaders import file_content_hash
from rag_assistant.ingestion.manifest import IngestManifest
from rag_assistant.observability import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Hatalar — çağıran katman bunları kendi diline çevirir (HTTP kodu vb.)
# ---------------------------------------------------------------------------
class LibraryError(RuntimeError):
    """Kütüphane işlemi başarısız."""


class InvalidFileNameError(LibraryError):
    """Dosya adı güvenli değil veya kabul edilemez."""


class UnsupportedFileTypeError(LibraryError):
    """Bu uzantı desteklenmiyor."""


class FileTooLargeError(LibraryError):
    """Dosya boyut sınırını aştı."""


class DocumentNotFoundError(LibraryError):
    """Böyle bir doküman yok."""


class DocumentExistsError(LibraryError):
    """Aynı isimde doküman zaten var."""


# ---------------------------------------------------------------------------
# Dosya adı güvenliği
# ---------------------------------------------------------------------------
# Windows'ta ayrılmış cihaz adları. "CON.pdf" adlı dosya oluşturulamaz ve
# bazı işlemler beklenmedik davranır. Platformdan bağımsız reddediyoruz.
_RESERVED_NAMES = frozenset(
    {"con", "prn", "aux", "nul"}
    | {f"com{i}" for i in range(1, 10)}
    | {f"lpt{i}" for i in range(1, 10)}
)

# Dosya adında kabul edilen karakterler. Beyaz liste — bilmediğimiz her
# karakteri reddediyoruz. Türkçe harfler bilinçli olarak dahil.
_SAFE_NAME_RE = re.compile(r"^[\w\s.,()\[\]+&#@!'çğıöşüÇĞİÖŞÜ-]+$", re.UNICODE)

# Görünmez / kontrol karakterleri — dosya adında gizli karakter,
# görünüşte aynı iki farklı dosya üretebilir (homograph saldırısı).
_CONTROL_CHARS = re.compile(r"[\x00-\x1f\x7f-\x9f​-‏ -‮]")


def sanitize_file_name(raw_name: str, *, max_length: int = 200) -> str:
    """
    Yüklenen dosya adını güvenli hale getir veya REDDET.

    EN KRİTİK KONTROL — DİZİN AŞIMI (path traversal):
      İstemci dosya adı olarak `../../../etc/passwd` veya
      `..\\..\\Windows\\System32\\drivers\\etc\\hosts` gönderebilir.
      Bu ad doğrudan bir yola eklenirse, hedef klasörün DIŞINA yazarsın.
      Sistem dosyalarının üzerine yazmak buradan başlar.

      Savunma: `Path(name).name` ile YALNIZCA taban adı al. Bu, tüm dizin
      bileşenlerini (`../`, `/`, `\\`) atar. Ardından sonucun hâlâ ayırıcı
      içermediğini DOĞRULA — çift kontrol, çünkü bu hatanın bedeli yüksek.

    Diğer savunmalar:
      * Unicode NFC normalizasyonu → görünüşte aynı, baytça farklı adlar
      * Kontrol/görünmez karakter reddi → homograph ve gizleme saldırıları
      * Nokta ile başlayan ad reddi → gizli dosya oluşturma
      * Windows ayrılmış adları reddi
      * Uzunluk sınırı → dosya sistemi hataları
    """
    if not raw_name or not raw_name.strip():
        raise InvalidFileNameError("Dosya adı boş olamaz.")

    name = unicodedata.normalize("NFC", raw_name).strip()

    if _CONTROL_CHARS.search(name):
        raise InvalidFileNameError("Dosya adı görünmez veya kontrol karakteri içeriyor.")

    # DİZİN AŞIMI SAVUNMASI — yalnızca taban adı al.
    # Hem POSIX hem Windows ayırıcılarını elle temizliyoruz: Linux'ta
    # çalışan bir süreçte `Path("..\\..\\x").name` ters eğik çizgiyi
    # ayırıcı SAYMAZ ve tüm dize tek bir ad olarak geçer.
    had_separator = "/" in name or "\\" in name or ".." in name
    name = name.replace("\\", "/").split("/")[-1]
    name = Path(name).name

    if had_separator:
        # Temizlik yeterli, ama SESSİZ KALMIYORUZ.
        # Normal bir tarayıcı yüklemesinde dosya adında ayırıcı OLMAZ.
        # Varsa bu ya bozuk bir istemci ya da dizin aşımı denemesidir;
        # ikisini de görmek isteriz. Güvenlik olayları görünür olmalı.
        logger.warning(
            "library.suspicious_filename",
            received=raw_name[:120],
            sanitized=name,
            reason="dosya adında dizin ayırıcısı vardı",
        )

    if not name or name in {".", ".."}:
        raise InvalidFileNameError("Geçersiz dosya adı.")

    # Çift kontrol: buradan sonra hiçbir ayırıcı kalmamalı.
    if "/" in name or "\\" in name or "\x00" in name:
        raise InvalidFileNameError("Dosya adı dizin ayırıcısı içeremez.")

    if name.startswith("."):
        raise InvalidFileNameError("Dosya adı nokta ile başlayamaz.")

    if len(name) > max_length:
        raise InvalidFileNameError(f"Dosya adı çok uzun (en fazla {max_length} karakter).")

    if Path(name).stem.lower() in _RESERVED_NAMES:
        raise InvalidFileNameError(f"'{Path(name).stem}' ayrılmış bir addır.")

    if not _SAFE_NAME_RE.match(name):
        raise InvalidFileNameError(
            "Dosya adı izin verilmeyen karakter içeriyor. "
            "Harf, rakam, boşluk ve . , ( ) [ ] - + & # @ ! ' kullanılabilir."
        )

    return name


def validate_upload_name(
    raw_name: str,
    *,
    allowed_extensions: tuple[str, ...],
    max_length: int = 200,
) -> str:
    """
    Yükleme için tam doğrulama: güvenli ad + izinli uzantı.

    NEDEN AYRI BİR FONKSİYON (save_upload içine gömmek yerine):
      Doğrulama, yazma işleminden BAĞIMSIZ olarak çağrılabilmeli —
      örneğin bir istemci "bu adı kabul eder misin?" diye önden sorabilir,
      ya da bir test sahtesi (fake) gerçekle AYNI kuralı uygulayabilir.

      Sahteler doğrulamayı kendi yeniden yazarsa gerçekten sapar ve
      testler yanlış güven verir. Bu fiilen yaşandı: sahte kütüphane
      uzantı kontrolünü atlıyordu ve API testi ".exe kabul edildi"
      diyerek hatayı yakaladı.
    """
    name = sanitize_file_name(raw_name, max_length=max_length)
    suffix = Path(name).suffix.lower()
    if suffix not in allowed_extensions:
        raise UnsupportedFileTypeError(
            f"'{suffix or 'uzantısız'}' desteklenmiyor. "
            f"Desteklenen: {', '.join(allowed_extensions)}"
        )
    return name


# ---------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class UploadResult:
    """Yükleme sonucu."""

    file_name: str
    size_bytes: int
    content_hash: str
    # Aynı İÇERİK başka bir adla zaten varsa burada bildirilir.
    # Kullanıcı bunu bilmeli: aynı doküman iki kez indekslenmiş olur.
    duplicate_of: str | None = None


class DocumentLibrary:
    """
    Kaynak dokümanların yaşam döngüsünü yönetir.

    Sorumluluk: dosya sistemi + manifest + index arasındaki TUTARLILIK.
    Bir doküman silindiğinde üçünden de silinmeli; biri atlanırsa sistem
    tutarsız hale gelir (hayalet chunk veya yeniden indekslenen dosya).
    """

    def __init__(
        self,
        settings: Settings,
        *,
        store: VectorStore,
        manifest: IngestManifest,
    ) -> None:
        self._settings = settings
        self._store = store
        self._manifest = manifest
        self._raw_dir = settings.paths.raw_dir

    # ------------------------------------------------------------------
    # Listeleme
    # ------------------------------------------------------------------
    def list_documents(self) -> list[StoredDocument]:
        """
        Kütüphanedeki tüm dokümanlar.

        Kaynak: DİSK (gerçek dosyalar) + manifest (işlenme durumu).
        Diski esas alıyoruz: manifest'te olup diskte olmayan bir kayıt
        artık bir doküman değildir. Tersi de mümkün — henüz işlenmemiş
        dosya diskte vardır ama manifest'te yoktur; o da listelenmeli
        (kullanıcı yüklediği dosyayı görmeli).
        """
        documents: list[StoredDocument] = []

        for path in sorted(self._raw_dir.glob("*")):
            if not path.is_file() or path.name.startswith("."):
                continue

            entry = self._manifest.get(path.name)
            size = path.stat().st_size

            if entry is None:
                # Diskte var, henüz işlenmemiş.
                documents.append(
                    StoredDocument(
                        file_name=path.name,
                        status=DocumentStatus.SKIPPED,
                        size_bytes=size,
                        page_count=0,
                        chunk_count=0,
                        content_hash="",
                        processed_at=None,
                        error="henüz indekslenmedi",
                    )
                )
                continue

            documents.append(
                StoredDocument(
                    file_name=path.name,
                    status=entry.status,
                    size_bytes=size,
                    page_count=entry.page_count,
                    chunk_count=entry.chunk_count,
                    content_hash=entry.content_hash,
                    embedder_model_id=entry.embedder_model_id,
                    processed_at=entry.processed_at or None,
                    error=entry.error,
                )
            )

        return documents

    def get(self, file_name: str) -> StoredDocument:
        """Tek doküman. Yoksa `DocumentNotFoundError`."""
        safe = sanitize_file_name(file_name, max_length=self._settings.upload.max_filename_length)
        for doc in self.list_documents():
            if doc.file_name == safe:
                return doc
        raise DocumentNotFoundError(f"Doküman bulunamadı: {safe}")

    # ------------------------------------------------------------------
    # Yükleme
    # ------------------------------------------------------------------
    def save_upload(
        self,
        file_name: str,
        chunks: Iterator[bytes],
        *,
        overwrite: bool = False,
    ) -> UploadResult:
        """
        Yüklenen dosyayı diske yaz.

        Args:
            file_name: istemcinin bildirdiği ad (GÜVENİLMEZ, temizlenir)
            chunks: dosya içeriğini parça parça veren akış
            overwrite: aynı isimde dosya varsa üzerine yaz

        AKIŞ HALİNDE YAZMA + BOYUT SAYIMI:
          Tüm dosyayı belleğe almıyoruz. Ayrıca `Content-Length` başlığına
          GÜVENMİYORUZ — istemci yalan söyleyebilir. Bayt sayısı yazarken
          sayılıyor, sınır aşılınca yazma durduruluyor ve geçici dosya
          siliniyor. Aksi halde 50 MB sınırı olan bir uç noktaya 10 GB
          gönderilebilirdi.

        ATOMİK YERLEŞTİRME:
          Önce geçici dosyaya yazılır, tamamlanınca hedefe taşınır.
          Yarım kalan bir yükleme, `data/raw/` içinde geçerli görünen
          ama bozuk bir PDF bırakmaz — yoksa ingestion onu işlemeye
          çalışır ve tuhaf hatalar üretir.
        """
        upload = self._settings.upload
        safe_name = validate_upload_name(
            file_name,
            allowed_extensions=upload.allowed_extensions,
            max_length=upload.max_filename_length,
        )

        target = self._raw_dir / safe_name
        if target.exists() and not overwrite:
            raise DocumentExistsError(
                f"'{safe_name}' zaten var. Üzerine yazmak için overwrite=true gönderin."
            )

        self._raw_dir.mkdir(parents=True, exist_ok=True)
        temp = self._raw_dir / f".{safe_name}.upload"

        written = 0
        try:
            with temp.open("wb") as out:
                for block in chunks:
                    written += len(block)
                    if written > upload.max_size_bytes:
                        raise FileTooLargeError(
                            f"Dosya {upload.max_size_bytes // (1024 * 1024)} MB sınırını aştı."
                        )
                    out.write(block)

            if written == 0:
                raise InvalidFileNameError("Dosya boş.")

            content_hash = file_content_hash(temp)
            duplicate = self._find_duplicate(content_hash, exclude=safe_name)

            temp.replace(target)  # atomik
        except Exception:
            temp.unlink(missing_ok=True)  # yarım dosya bırakma
            raise

        logger.info(
            "library.uploaded",
            file=safe_name,
            bytes=written,
            duplicate_of=duplicate,
            overwritten=overwrite,
        )
        return UploadResult(
            file_name=safe_name,
            size_bytes=written,
            content_hash=content_hash,
            duplicate_of=duplicate,
        )

    def _find_duplicate(self, content_hash: str, *, exclude: str) -> str | None:
        """
        Aynı İÇERİK başka bir adla var mı?

        Dosya adı farklı ama içerik aynıysa doküman iki kez indekslenir;
        aynı bilgi arama sonuçlarında iki kez çıkar. Engellemiyoruz
        (kullanıcının bilinçli tercihi olabilir) ama BİLDİRİYORUZ.
        """
        for entry in self._manifest.entries():
            if entry.file_name != exclude and entry.content_hash == content_hash:
                return entry.file_name
        return None

    # ------------------------------------------------------------------
    # Silme
    # ------------------------------------------------------------------
    def delete(self, file_name: str) -> StoredDocument:
        """
        Dokümanı ÜÇ yerden birden sil: index, manifest, disk.

        SIRA ÖNEMLİ ve bilinçli:
          1. Index'ten chunk'lar    → arama sonuçlarında anında kaybolur
          2. Manifest kaydı         → yeniden işlenme geçmişi temizlenir
          3. Diskten dosya          → en son

        Neden bu sıra: herhangi bir adımda hata olursa kalan durum
        "fazladan dosya var" olur — bu ZARARSIZ ve düzeltilebilir
        (yeniden ingestion). Ters sırada olsaydı "dosya yok ama index'te
        chunk'ları var" durumu kalırdı: kullanıcı silinmiş bir dokümandan
        alıntı yapan cevaplar alır ve kaynağı bulamaz.
        """
        doc = self.get(file_name)
        safe_name = doc.file_name

        removed_chunks = self._store.remove_by_file(safe_name)
        if removed_chunks:
            self._store.save(self._settings.paths.index_dir)

        self._manifest.remove(safe_name)
        self._manifest.save()

        (self._raw_dir / safe_name).unlink(missing_ok=True)

        logger.info(
            "library.deleted",
            file=safe_name,
            removed_chunks=removed_chunks,
            index_total=self._store.count,
        )
        return doc

    def delete_all(self) -> int:
        """Tüm kütüphaneyi temizle. Silinen doküman sayısını döndürür."""
        documents = self.list_documents()
        for doc in documents:
            self.delete(doc.file_name)
        return len(documents)

    # ------------------------------------------------------------------
    @property
    def raw_dir(self) -> Path:
        return self._raw_dir

    def disk_usage_bytes(self) -> int:
        return sum(p.stat().st_size for p in self._raw_dir.glob("*") if p.is_file())

    def free_space_bytes(self) -> int:
        """
        Diskte kalan yer.

        Yükleme öncesi kontrol için: dolu diske yazmaya çalışmak
        yarım dosya ve anlaşılmaz hatalar üretir.
        """
        return shutil.disk_usage(self._raw_dir).free
