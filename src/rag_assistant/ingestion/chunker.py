"""
Chunking: dokümanı aranabilir parçalara bölme.

İKİ TEMEL GARANTİ:
  1. Hiçbir chunk embedding modelinin token limitini AŞMAZ.
     Bu bir "hedef" değil, sert bir garantidir. Aşan chunk sessizce kesilir
     ve o metin bir daha asla bulunamaz — hata mesajı da almazsın.
  2. Bölme noktaları YAPIYI takip eder: önce paragraf, sonra satır, sonra
     cümle, en son (mecbur kalırsa) kelime. Cümleyi ortadan bölmek anlamı
     ikiye ayırır ve iki chunk'ın ikisini de faydasız hale getirebilir.

TASARIM NOTU — neden `count_tokens` bir fonksiyon olarak enjekte ediliyor:
  Chunker'ın embedding modelinin tamamına ihtiyacı yok; sadece "bu metin kaç
  token?" sorusunu sorabilmesi gerekiyor. En dar arayüzü talep etmek
  (interface segregation) chunker'ı model yüklemeden test edilebilir kılar:
  testte `lambda t: len(t.split())` geçmek yeterli.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Sequence

from rag_assistant.domain.models import Chunk, LoadedDocument, LoadedPage, SourceRef
from rag_assistant.observability import get_logger

logger = get_logger(__name__)

TokenCounter = Callable[[str], int]

# Paragraf sınırı: bir veya daha fazla boş satır.
_PARAGRAPH_SPLIT = re.compile(r"\n\s*\n")

# Cümle sonu: . ! ? veya : ardından boşluk. Türkçe kısaltmalar yanlış
# bölmeye yol açtığı için aşağıdaki listeyle koruyoruz.
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?:])\s+")

# Bunlarla BİTEN bir parça cümle sonu sayılmaz (yanlış bölmeyi engeller).
_ABBREVIATIONS = (
    "vb.", "vs.", "bkz.", "örn.", "yy.", "no.", "sy.", "md.", "bkz",
    "dr.", "prof.", "doç.", "av.", "sn.", "mm.", "cm.", "km.", "kg.",
    "a.ş.", "ltd.", "şti.", "tl.", "usd.",
)


def _split_paragraphs(text: str) -> list[str]:
    return [p.strip() for p in _PARAGRAPH_SPLIT.split(text) if p.strip()]


def _split_sentences(text: str) -> list[str]:
    """
    Metni cümlelere ayır, Türkçe kısaltmaları koruyarak.

    Yaklaşım: kaba bir bölme yap, sonra kısaltmayla biten parçaları bir
    sonrakine geri yapıştır. Mükemmel bir cümle ayrıştırıcı değil ama
    bağımlılık eklemeden yanlış bölmelerin büyük kısmını önler.
    """
    rough = [s for s in _SENTENCE_SPLIT.split(text) if s.strip()]
    if not rough:
        return []

    merged: list[str] = []
    for part in rough:
        lowered = part.strip().lower()
        if merged and lowered.startswith(tuple(a for a in _ABBREVIATIONS)):
            merged[-1] = f"{merged[-1]} {part}"
        elif merged and merged[-1].strip().lower().endswith(_ABBREVIATIONS):
            merged[-1] = f"{merged[-1]} {part}"
        else:
            merged.append(part)
    return merged


class TokenAwareChunker:
    """
    Token bütçesine uyan, yapı-farkında chunker. `Chunker` protokolünü sağlar.

    Algoritma:
      1. Sayfayı paragraflara ayır.
      2. Paragrafları token bütçesi dolana kadar biriktir.
      3. Bütçeye sığmayan tek bir paragraf varsa onu cümlelere ayır.
      4. Bütçeye sığmayan tek bir cümle varsa (nadir: tablo dökümü, uzun
         liste) kelime bazında sert kes ve bunu UYARI olarak logla —
         sessizce yapılan hiçbir kesme kabul edilebilir değildir.
      5. Ardışık chunk'lar arasında `overlap_tokens` kadar örtüşme bırak.
    """

    def __init__(
        self,
        count_tokens: TokenCounter,
        *,
        target_tokens: int,
        overlap_tokens: int,
        min_tokens: int,
        hard_limit_tokens: int,
    ) -> None:
        if overlap_tokens >= target_tokens:
            raise ValueError("overlap_tokens < target_tokens olmalı")
        if target_tokens > hard_limit_tokens:
            raise ValueError("target_tokens, modelin sert limitini aşamaz")

        self._count = count_tokens
        self._target = target_tokens
        self._overlap = overlap_tokens
        self._min = min_tokens
        self._hard_limit = hard_limit_tokens

    # ------------------------------------------------------------------
    # Genel API
    # ------------------------------------------------------------------
    def split(self, document: LoadedDocument) -> list[Chunk]:
        """
        Dokümanı chunk'lara böl.

        Sayfa sınırlarını aşmıyoruz: bir chunk tek bir sayfaya aittir.
        Neden — atıf. "rapor.pdf s.7" diyebilmek için chunk'ın hangi sayfadan
        geldiği kesin olmalı. İki sayfayı birleştiren bir chunk'a doğru
        sayfa numarası veremezsin ve kaynak gösterimi güvenilmez olur.
        """
        chunks: list[Chunk] = []
        for page in document.pages:
            chunks.extend(self._split_page(document.file_name, page))

        logger.info(
            "document.chunked",
            file=document.file_name,
            chunks=len(chunks),
            tokens_total=sum(c.token_count for c in chunks),
            tokens_max=max((c.token_count for c in chunks), default=0),
        )
        return chunks

    # ------------------------------------------------------------------
    # İç işleyiş
    # ------------------------------------------------------------------
    def _split_page(self, file_name: str, page: LoadedPage) -> list[Chunk]:
        text = page.text.strip()
        if not text:
            return []

        units = self._to_units(text)
        if not units:
            return []

        source = SourceRef(file_name=file_name, page=page.page_number)
        return [
            self._make_chunk(group, source)
            for group in self._group_units(units)
        ]

    def _to_units(self, text: str) -> list[str]:
        """
        Metni, hiçbiri bütçeyi aşmayan en büyük anlamlı birimlere indir.

        Paragraf sığıyorsa paragraf kalır (en iyi durum: anlam bütünlüğü tam).
        Sığmıyorsa cümlelere, gerekirse kelimelere iner.
        """
        units: list[str] = []
        for paragraph in _split_paragraphs(text):
            if self._count(paragraph) <= self._target:
                units.append(paragraph)
                continue

            for sentence in _split_sentences(paragraph):
                if self._count(sentence) <= self._target:
                    units.append(sentence)
                else:
                    units.extend(self._hard_split(sentence))
        return units

    def _hard_split(self, text: str) -> list[str]:
        """
        Son çare: kelime bazında sert kesme.

        Buraya düşmek bir uyarı işaretidir (genellikle tablo dökümü veya
        satır sonu olmayan uzun liste). Bu yüzden loglanıyor — sessiz
        kesme yapmıyoruz.
        """
        words = text.split()
        pieces: list[str] = []
        current: list[str] = []

        for word in words:
            candidate = [*current, word]
            if self._count(" ".join(candidate)) > self._target and current:
                pieces.append(" ".join(current))
                current = [word]
            else:
                current = candidate
        if current:
            pieces.append(" ".join(current))

        logger.warning(
            "chunker.hard_split",
            reason="tek cümle token bütçesini aştı",
            tokens=self._count(text),
            budget=self._target,
            pieces=len(pieces),
            preview=text[:80],
        )
        return pieces

    def _group_units(self, units: Sequence[str]) -> list[list[str]]:
        """
        Birimleri bütçeye göre grupla, gruplar arasında örtüşme bırak.

        ÖNEMLİ — token sayımı BİRLEŞTİRİLMİŞ metin üzerinden yapılır.
        Birimlerin token sayılarını tek tek toplamak, birleştirme ayırıcısının
        ("\\n\\n") token maliyetini saymaz ve toplamı OLDUĞUNDAN AZ gösterir.
        Sert limit koruması bu yüzden yanıltıcı hale gelir. Tek tek toplamak
        hızlıdır ama yanlıştır; chunk başına birim sayısı az olduğu için
        kesin sayım pratikte bedavaya gelir.
        """
        groups: list[list[str]] = []
        current: list[str] = []

        for unit in units:
            if current and self._count(self._join([*current, unit])) > self._target:
                groups.append(current)
                current = self._overlap_tail(current)

            # SERT LİMİT KORUMASI: örtüşme taşındıktan sonra bile birim
            # sığmıyorsa örtüşmeyi feda et. Sınırda küçük bir bağlam kaybı,
            # limiti aşıp metnin sessizce kesilmesinden her zaman iyidir.
            if current and self._count(self._join([*current, unit])) > self._hard_limit:
                current = []

            current.append(unit)

        if current:
            # Son grup çok küçükse öncekiyle birleştir — tek başına anlamsız
            # bir kırıntı chunk üretmek arama gürültüsüdür.
            if groups and self._count(self._join(current)) < self._min:
                merged = [*groups[-1], *current]
                if self._count(self._join(merged)) <= self._hard_limit:
                    groups[-1] = merged
                else:
                    groups.append(current)
            else:
                groups.append(current)

        return groups

    def _overlap_tail(self, group: list[str]) -> list[str]:
        """
        Bir sonraki chunk'a taşınacak örtüşme metnini üret.

        Örtüşme neden var: bir bilgi tam olarak chunk sınırına düşerse
        (cümlenin öznesi bir chunk'ta, yüklemi diğerinde) her iki chunk da
        o bilgiyi tek başına cevaplayamaz. Örtüşme bu sınır kaybını kapatır.

        ÖRTÜŞME BİRİM DÜZEYİNDE DEĞİL, METİN SONEKİ OLARAK ALINIR.
        Nedeni önemli: örtüşmeyi "tam paragraflar" olarak taşımaya çalışırsan,
        paragraf örtüşme bütçesinden büyük olduğu her durumda örtüşme HİÇ
        uygulanmaz. Gerçek ayarlarda (hedef 400, örtüşme 80 token) paragrafların
        çoğu bütçeden büyüktür — yani örtüşme sessizce devre dışı kalır.

        Bu yüzden kademeli iniyoruz:
          1. Cümle sınırından sonek al (anlam bütünlüğü korunur)   ← tercih
          2. Tek cümle bile bütçeden büyükse kelime bazında sonek al
        Örtüşme metni bağlam dolgusudur; yarım cümle olması kabul edilebilir.
        Bütçe her iki durumda da KESİNDİR — asla aşılmaz.
        """
        if self._overlap <= 0:
            return []

        text = self._join(group)

        # 1) Cümle sınırından
        tail: list[str] = []
        for sentence in reversed(_split_sentences(text)):
            candidate = [sentence, *tail]
            if self._count(" ".join(candidate)) > self._overlap:
                break
            tail = candidate
        if tail:
            return [" ".join(tail)]

        # 2) Kelime bazında sonek
        words: list[str] = []
        for word in reversed(text.split()):
            candidate = [word, *words]
            if self._count(" ".join(candidate)) > self._overlap:
                break
            words = candidate
        return [" ".join(words)] if words else []

    @staticmethod
    def _join(units: Sequence[str]) -> str:
        return "\n\n".join(units)

    def _make_chunk(self, units: Sequence[str], source: SourceRef) -> Chunk:
        text = self._join(units)
        tokens = self._count(text)

        # SERT GARANTİ: buradan limiti aşan bir chunk çıkmaz.
        # Çıkarsa bu bir programlama hatasıdır ve gürültüsüzce geçmemeli.
        if tokens > self._hard_limit:
            raise AssertionError(
                f"chunk sert limiti aştı ({tokens} > {self._hard_limit}) — "
                "chunker mantığında hata var"
            )

        return Chunk(text=text, source=source, token_count=tokens)
