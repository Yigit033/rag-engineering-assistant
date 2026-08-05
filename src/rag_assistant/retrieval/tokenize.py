"""
Türkçe-güvenli tokenizasyon — BM25 (sparse) arama için.

NEDEN AYRI BİR MODÜL:
  Dense arama modelin kendi tokenizer'ını kullanır. Sparse arama (BM25) ise
  kelime örtüşmesine bakar ve tokenizasyonu BİZ yaparız. Bu adım yanlış
  yapılırsa BM25 sessizce zayıflar — hata vermez, sadece daha az eşleşme
  bulur ve hybrid aramanın yarısı boşa çalışır.

TÜRKÇE'DE ÜÇ TUZAK:

  1. `"İ".lower()` Python'da 'i̇' üretir — yani i + BİRLEŞEN NOKTA,
     TEK karakter değil İKİ karakter. Naif `.lower()` kullanan her Türkçe
     metin işleme kodu burada sessizce bozulur.
     Çözüm: Türkçe'ye özgü büyük harfleri önce elle eşle, sonra lower().

  2. Noktalama: "poliçe," ile "poliçe" BM25 için farklı kelimedir.
     Çözüm: yalnızca alfanümerik dizileri token say.

  3. Türkçe EKLEMELİ bir dildir: "poliçe / poliçenin / poliçeye / poliçeler"
     BM25 için dört ayrı kelimedir. Bu, recall'ı ciddi düşürür.
     Çözüm: hafif bir sonek kırpma (light stemming). Tam bir morfolojik
     çözümleyici (zeyrek gibi) daha doğru sonuç verir ama ağır bir bağımlılık
     getirir; bu ölçekte sonek kırpma maliyetsiz ve belirgin fayda sağlar.
"""

from __future__ import annotations

import re

from rag_assistant.text import tr_lower

# Alfanümerik diziler. Türkçe harfler açıkça listelenmeli — `\w` yeterli
# görünse de sınıf davranışı yerel ayara/derlemeye göre değişebilir.
_TOKEN_RE = re.compile(r"[0-9a-zçğıöşü]+")

# Sık kullanılan çekim ekleri. Amaç dilbilimsel doğruluk değil,
# "poliçenin" ile "poliçe"yi aynı gövdede buluşturmak.
#
# SIRALAMA KODA BIRAKILIR (aşağıda uzunluğa göre sıralanıyor).
# Elle sıralamaya güvenmek kırılgandır: liste büyüdükçe biri araya kısa bir
# ek ekler ve "en uzun eşleşme kazanır" kuralı sessizce bozulur. Ölçtüm:
# "i" eki "si"den önce denendiğinde "gövdesi" → "gövdes" kalıyor ve
# "gövde" ile hiç eşleşmiyor.
_SUFFIX_LITERALS = (
    "lerinden", "larından", "lerine", "larına", "lerini", "larını",
    "lerin", "ların", "leri", "ları", "ler", "lar",
    "inden", "ından", "unden", "ündan",
    "ince", "ınca", "unca", "ünce",
    "nin", "nın", "nun", "nün",
    "den", "dan", "ten", "tan",
    "in", "ın", "un", "ün",
    "de", "da", "te", "ta",
    "ye", "ya", "e", "a",
    "yi", "yı", "yu", "yü", "i", "ı", "u", "ü",
    "si", "sı", "su", "sü",
    "le", "la",
)

# EN UZUN EŞLEŞME KAZANIR: sıralamayı elle değil kodla garanti ediyoruz.
_SUFFIXES: tuple[str, ...] = tuple(
    sorted(set(_SUFFIX_LITERALS), key=len, reverse=True)
)

# Bu uzunluğun altına inen gövde anlamını kaybeder; kırpma yapılmaz.
_MIN_STEM_LENGTH = 4


# Türkçe harf katlama tek bir yerde tanımlıdır (rag_assistant.text).
# Burada yeniden yazmıyoruz: aynı mantığın iki kopyası, birini düzeltip
# diğerini unutmanın garantisidir. (Bu hata bu projede fiilen yaşandı:
# tokenize.py doğru yapıyordu, answerer.py casefold() kullanıp bozuluyordu.)
normalize = tr_lower


# Kırpma en fazla bu kadar tur döner. Uzunluk her turda azaldığı için
# döngü zaten sonlanır; sınır öngörülebilirlik ve maliyet içindir.
_MAX_STEM_PASSES = 3


def light_stem(token: str) -> str:
    """
    Hafif sonek kırpma — SABİT NOKTAYA KADAR YİNELEMELİ.

    NEDEN TEK GEÇİŞ YETMİYOR (ölçülmüş bir hata):
      Türkçe'de sesli harfle biten gövdeye ek gelirken araya kaynaştırma
      harfi girer: "poliçe" + "in" → "poliçe-N-in".
      Tek geçişle kırpınca:
          "poliçenin" → "nin" atılır → "poliçe"
          "poliçe"    → "e"   atılır → "poliç"
      Sonuç: aynı kelimenin iki biçimi FARKLI gövdelere iniyor ve BM25'te
      hiç eşleşmiyor — yani stemmer amacının tam tersini yapıyor.

      Yinelemeli kırpmada ikisi de aynı sabit noktaya iner:
          "poliçenin" → "poliçe" → "poliç"
          "poliçe"    → "poliç"            ✓ eşleşir

    KABUL EDİLEN ÖDÜNLEŞİM:
      Yinelemeli kırpma dilbilimsel olarak daha kabadır (gövdeler gerçek
      kök olmayabilir). Ama BM25 için önemli olan gövdenin DOĞRU olması
      değil, sorgu ile dokümanda AYNI olmasıdır. Tutarlılık > doğruluk.
      `_MIN_STEM_LENGTH` aşırı kırpmanın zararını sınırlar.
    """
    stem = token
    for _ in range(_MAX_STEM_PASSES):
        if len(stem) <= _MIN_STEM_LENGTH:
            break
        for suffix in _SUFFIXES:
            if stem.endswith(suffix) and len(stem) - len(suffix) >= _MIN_STEM_LENGTH:
                stem = stem[: -len(suffix)]
                break
        else:
            break  # bu turda hiçbir ek eşleşmedi → sabit noktadayız
    return _harden_final(stem)


# Ünsüz yumuşaması: Türkçe'de sesliyle başlayan ek gelince gövde sonundaki
# sert ünsüz yumuşar (k→ğ, p→b, ç→c, t→d):
#     "güvenlik" + "in"  → "güvenliğin"
#     "kitap"    + "ın"  → "kitabın"
# Ek atıldıktan sonra geriye YUMUŞAK biçim kalır ve eksiz halle eşleşmez.
# Sert biçime geri çevirerek ikisini buluşturuyoruz.
_SOFTENED_TO_HARD = {"ğ": "k", "b": "p", "c": "ç", "d": "t"}


def _harden_final(stem: str) -> str:
    """
    Gövde sonundaki yumuşak ünsüzü sertleştir.

    KOŞULSUZ uygulanır (ek atılmış olsun olmasın). Nedeni önemli:
    yalnızca ek atıldığında uygulasaydık, gerçekten "d" ile biten bir gövde
    ("kod") ekli halinde sertleşip ("kodun" → "kod" → "kot") eksiz halinden
    ("kod") ayrılırdı — yani yeni bir asimetri yaratırdık.
    Koşulsuz uygulamak "kod" ve "kot"u birleştirir; bu bir kayıptır ama
    tutarlılık, BM25 için doğruluktan daha değerlidir.
    """
    if len(stem) <= _MIN_STEM_LENGTH:
        return stem
    last = stem[-1]
    return stem[:-1] + _SOFTENED_TO_HARD[last] if last in _SOFTENED_TO_HARD else stem


def tokenize(text: str, *, stem: bool = True) -> list[str]:
    """Metni BM25 için token listesine çevir."""
    tokens = _TOKEN_RE.findall(normalize(text))
    if not stem:
        return tokens
    return [light_stem(t) for t in tokens]
