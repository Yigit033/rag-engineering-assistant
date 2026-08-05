"""
Türkçe-güvenli metin işlemleri — kesişen (cross-cutting) yardımcılar.

NEDEN AYRI BİR MODÜL:
  Türkçe büyük/küçük harf dönüşümü retrieval'a özgü bir konu değil.
  Metin karşılaştıran HER katman aynı tuzağa düşebilir — ve düştü:
  `answerer.py` içinde çekimserlik tespiti `str.casefold()` kullanıyordu ve
  "BU BİLGİ ... YOK" ile "Bu bilgi ... yok" eşleşmiyordu. Aynı hatanın iki
  farklı katmanda çıkması, bunun ortak bir yere ait olduğunun kanıtıdır.

TUZAK:
  Python'un `lower()` / `casefold()` metotları Unicode'un genel kurallarını
  uygular ve Türkçe'nin noktalı/noktasız I ayrımını BİLMEZ:

      "İ".lower()     -> 'i̇'   (i + U+0307 BİRLEŞEN NOKTA)  = 2 karakter
      "I".lower()     -> 'i'    (olması gereken: 'ı')

  Sonuç: karşılaştırma sessizce başarısız olur, istisna atılmaz.
"""

from __future__ import annotations

import unicodedata

# Türkçe'ye özgü büyük harfler → doğru küçük karşılıkları.
# Bunları ÖNCE elle eşliyoruz, ardından standart lower() güvenle çalışabilir.
_TR_UPPER_TO_LOWER = str.maketrans(
    {
        "İ": "i",
        "I": "ı",
        "Ş": "ş",
        "Ğ": "ğ",
        "Ü": "ü",
        "Ö": "ö",
        "Ç": "ç",
    }
)


def tr_lower(text: str) -> str:
    """
    Türkçe-güvenli küçük harfe çevirme + Unicode NFC normalizasyonu.

    NFC neden gerekli: aynı karakter ("ğ") tek birleşik kod noktası olarak da,
    "g + birleşen aksan" olarak da kodlanabilir. Normalize edilmezse aynı
    kelime iki farklı bayt dizisi olur ve hiçbir karşılaştırmada eşleşmez.
    """
    return unicodedata.normalize("NFC", text).translate(_TR_UPPER_TO_LOWER).lower()


def tr_equal_fold(left: str, right: str) -> bool:
    """İki metni Türkçe-güvenli biçimde büyük/küçük harf duyarsız karşılaştır."""
    return tr_lower(left) == tr_lower(right)


def tr_contains(haystack: str, needle: str) -> bool:
    """`needle`, `haystack` içinde geçiyor mu? (Türkçe-güvenli, harf duyarsız)"""
    return tr_lower(needle) in tr_lower(haystack)
