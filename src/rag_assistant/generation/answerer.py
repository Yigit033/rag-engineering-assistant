"""
Kaynak gösteren cevap üretimi ve halüsinasyon kontrolü.

Bu dosya, sistemin "uydurmama" sözünü tuttuğu yerdir. Beş savunma katmanı var
ve her biri farklı bir hata biçimini hedefliyor:

  1. BOŞ BAĞLAMDA MODELİ HİÇ ÇAĞIRMA
     Hiç aday bulunamadıysa LLM'e boş bağlam göndermek uydurma için
     davetiyedir. Model hiç çağrılmaz, doğrudan çekimser cevap dönülür.

  2. PROMPT'TA AÇIK ÇEKİMSER KALMA İZNİ
     Modele "bilmiyorsan şunu yaz" demezsen, model her koşulda bir cevap
     üretmek zorunda hisseder ve boşluğu uydurmayla doldurur.

  3. NUMARALI BAĞLAM + ZORUNLU ATIF
     Her chunk [1], [2] diye numaralanır ve modelden iddialarını bu
     numaralara bağlaması istenir. Atıfsız cevap doğrulanamaz.

  4. ATIF DOĞRULAMA (uydurulmuş atıfı yakalar)
     Model var olmayan bir numaraya atıf yapabilir — kaynak göstermiş gibi
     görünen ama dayanağı olmayan bir cevap. Geçersiz numaralar ayıklanır
     ve loglanır. Bu, en sinsi halüsinasyon biçimlerinden biridir.

  5. GROUNDEDNESS DENETİMİ (isteğe bağlı, ikinci LLM çağrısı)
     Cevaptaki iddiaların kaçının bağlamda gerçekten desteklendiğini ölçer.

TASARIM İLKESİ — MODELİN ÇIKTISI SESSİZCE DEĞİŞTİRİLMEZ:
  Atıfsız bir cevap geldiğinde onu silip yerine kendi metnimizi koymuyoruz.
  Bunun yerine sonucu İŞARETLİYORUZ (`Answer.is_grounded`) ve logluyoruz.
  Sebep: sessiz düzeltme, sorunu gizler ve ölçümü bozar. Görünür bir
  başarısızlık, gizli bir "düzeltme"den her zaman iyidir.
"""

from __future__ import annotations

import re
import time
from collections.abc import Iterator, Sequence

from rag_assistant.domain.models import Answer, Citation, ScoredChunk
from rag_assistant.domain.protocols import LLM, Retriever
from rag_assistant.generation.prompt import PromptLibrary
from rag_assistant.observability import get_logger
from rag_assistant.text import tr_contains

logger = get_logger(__name__)

# Cevap metnindeki [1], [12] gibi atıf işaretleri
_CITATION_RE = re.compile(r"\[(\d{1,2})\]")


class GroundedAnswerer:
    """Retrieval + prompt + LLM'i birleştirip DENETLENEBİLİR cevap üretir."""

    def __init__(
        self,
        *,
        retriever: Retriever,
        llm: LLM,
        prompts: PromptLibrary,
        top_k: int = 5,
        abstain_phrase: str = "Bu bilgi verilen dokümanlarda yok",
        abstain_when_no_context: bool = True,
        require_citations: bool = True,
        check_groundedness: bool = False,
        max_context_tokens: int = 4000,
    ) -> None:
        self._retriever = retriever
        self._llm = llm
        self._prompts = prompts
        self._top_k = top_k
        self._abstain_phrase = abstain_phrase
        self._abstain_when_no_context = abstain_when_no_context
        self._require_citations = require_citations
        self._check_groundedness = check_groundedness
        self._max_context_tokens = max_context_tokens

    # ------------------------------------------------------------------
    # Bağlam kurulumu
    # ------------------------------------------------------------------
    def _build_context(self, chunks: Sequence[ScoredChunk]) -> tuple[str, list[ScoredChunk]]:
        """
        Numaralandırılmış bağlam metni üret ve token bütçesine sığdır.

        Bütçe neden var: bağlam modelin penceresini aşarsa model metnin
        başını veya sonunu görmez — ve bunu sana söylemez. Ayrıca "lost in
        the middle" etkisiyle çok uzun bağlamda ortadaki bilgi kaybolur.
        Az ve isabetli bağlam, çok ve gürültülü bağlamdan iyidir.

        Not: chunk.token_count embedding modelinin tokenizer'ıyla hesaplandı;
        LLM'in tokenizer'ı farklıdır. Bu bir YAKLAŞIKTIR, bu yüzden bütçeyi
        pencerenin epey altında tutuyoruz.
        """
        parts: list[str] = []
        used: list[ScoredChunk] = []
        budget = self._max_context_tokens

        for scored in chunks:
            if scored.chunk.token_count > budget and used:
                logger.debug(
                    "context.budget_reached", used=len(used), skipped=len(chunks) - len(used)
                )
                break
            marker = len(used) + 1
            parts.append(
                f"[{marker}] (kaynak: {scored.chunk.source.label()})\n{scored.chunk.text}"
            )
            used.append(scored)
            budget -= scored.chunk.token_count

        return "\n\n".join(parts), used

    # ------------------------------------------------------------------
    # Atıf ayrıştırma ve doğrulama
    # ------------------------------------------------------------------
    def _extract_citations(
        self, text: str, used: Sequence[ScoredChunk]
    ) -> tuple[tuple[Citation, ...], list[int]]:
        """
        Cevaptaki atıf numaralarını çöz.

        Returns:
            (geçerli atıflar, uydurulmuş numaralar)

        Uydurulmuş numara = bağlamda olmayan bir kaynağa yapılan atıf.
        Bu, "kaynak göstermiş gibi görünen" bir halüsinasyondur ve normal
        gözle bakışta fark edilmez — bu yüzden ayrıca raporlanır.
        """
        seen: dict[int, Citation] = {}
        hallucinated: list[int] = []

        for match in _CITATION_RE.finditer(text):
            marker = int(match.group(1))
            if marker in seen:
                continue
            if 1 <= marker <= len(used):
                scored = used[marker - 1]
                seen[marker] = Citation(
                    marker=marker,
                    source=scored.chunk.source,
                    chunk_id=scored.chunk.id,
                )
            elif marker not in hallucinated:
                hallucinated.append(marker)

        return tuple(seen[m] for m in sorted(seen)), hallucinated

    def _is_abstention(self, text: str) -> bool:
        """
        Model çekimser mi kaldı?

        Karşılaştırma `tr_contains` ile yapılır, `str.casefold()` ile DEĞİL.
        Sebebi ölçüldü: `"BİLGİ".casefold()` → `"bi̇lgi̇"` (i + birleşen nokta)
        üretir ve `"bilgi"` ile eşleşmez. Yani büyük harfle yazılmış bir
        çekimser cevap "cevap verilmiş" sanılır — uydurma olarak raporlanır.

        Uzunluk koşulu: model hem cevap verip hem "şu kısım yok" diyorsa
        cevap VERMİŞTİR. Bunu çekimserlik saymak, uydurmayı "bilmiyorum"
        diye raporlamak olur.
        """
        return (
            tr_contains(text, self._abstain_phrase)
            and len(text) < len(self._abstain_phrase) * 3
        )

    # ------------------------------------------------------------------
    # Ana akış
    # ------------------------------------------------------------------
    def answer(self, question: str) -> Answer:
        started = time.perf_counter()

        chunks = self._retriever.retrieve(question, self._top_k)

        # SAVUNMA 1: bağlam yoksa modeli hiç çağırma.
        if not chunks and self._abstain_when_no_context:
            logger.info("answer.no_context", question=question[:80])
            return Answer(
                question=question,
                text=self._abstain_phrase + ".",
                citations=(),
                used_chunks=(),
                abstained=True,
                model=self._llm.model_id,
                prompt_version=self._prompts.version,
                latency_ms=int((time.perf_counter() - started) * 1000),
                groundedness=None,
            )

        context, used = self._build_context(chunks)
        prompt = self._prompts.render(
            "answer",
            context=context,
            question=question,
            abstain_phrase=self._abstain_phrase,
        )

        text = self._llm.generate(prompt)
        abstained = self._is_abstention(text)
        citations, hallucinated = self._extract_citations(text, used)

        # ÇEKİMSER CEVAPTA ATIF ANLAMSIZDIR.
        # Gözlem: bazı modeller "bu bilgi dokümanlarda yok [1], [2], [3]"
        # biçiminde cevap veriyor — yani "bilgi yok" derken kaynak gösteriyor.
        # Bu atıflar hiçbir iddiayı desteklemiyor; bırakılırsa arayüzde
        # yanıltıcı kaynak listesi çıkar ve değerlendirmede atıf sayısını
        # şişirir. Çekimserlikte atıf listesini boşaltıyoruz.
        if abstained and citations:
            logger.debug("answer.abstention_citations_dropped", dropped=len(citations))
            citations = ()

        if hallucinated:
            # SAVUNMA 4: uydurulmuş atıf. Cevabı değiştirmiyoruz ama
            # görünür kılıyoruz — sessiz düzeltme ölçümü bozar.
            logger.warning(
                "answer.hallucinated_citation",
                question=question[:80],
                invalid_markers=hallucinated,
                available=len(used),
            )

        if self._require_citations and not citations and not abstained:
            logger.warning(
                "answer.missing_citations",
                question=question[:80],
                context_chunks=len(used),
                hint="model bağlam verilmesine rağmen kaynak göstermedi",
            )

        groundedness = (
            self._measure_groundedness(context, text)
            if self._check_groundedness and not abstained
            else None
        )

        answer = Answer(
            question=question,
            text=text,
            citations=citations,
            used_chunks=tuple(used),
            abstained=abstained,
            model=self._llm.model_id,
            prompt_version=self._prompts.version,
            latency_ms=int((time.perf_counter() - started) * 1000),
            groundedness=groundedness,
        )

        logger.info(
            "answer.completed",
            question=question[:80],
            chunks=len(used),
            citations=len(citations),
            hallucinated_citations=len(hallucinated),
            abstained=abstained,
            grounded=answer.is_grounded,
            groundedness=groundedness,
            ms=answer.latency_ms,
        )
        return answer

    def stream(self, question: str) -> Iterator[str]:
        """
        Cevabı parça parça üret (arayüz için).

        Atıf doğrulama akış bittikten sonra yapılabilir; akış sırasında
        yapılamaz çünkü metnin tamamı henüz yoktur.
        """
        chunks = self._retriever.retrieve(question, self._top_k)

        if not chunks and self._abstain_when_no_context:
            yield self._abstain_phrase + "."
            return

        context, _ = self._build_context(chunks)
        prompt = self._prompts.render(
            "answer",
            context=context,
            question=question,
            abstain_phrase=self._abstain_phrase,
        )
        yield from self._llm.stream(prompt)

    # ------------------------------------------------------------------
    # SAVUNMA 5: groundedness
    # ------------------------------------------------------------------
    def _measure_groundedness(self, context: str, answer_text: str) -> float | None:
        """
        Cevaptaki iddiaların kaç tanesi bağlamda destekleniyor? (0-1)

        İkinci bir LLM çağrısı gerektirir — bu yüzden varsayılan olarak
        kapalıdır. Değerlendirme koşularında ve şüpheli cevaplarda açılır.

        UYARI: Bu bir LLM-as-judge ölçümüdür; yön verici, mutlak değil.
        Hakem model de yanılabilir. Trend takibi için değerli, tek karar
        mercii olarak değil.
        """
        prompt = self._prompts.render("groundedness", context=context, answer=answer_text)
        try:
            verdict = self._llm.generate(prompt, temperature=0.0)
        except Exception as exc:  # noqa: BLE001 - denetim hatası ana akışı düşürmemeli
            logger.warning("groundedness.failed", error=str(exc))
            return None

        supported = re.search(r"DESTEKLENEN:\s*(\d+)", verdict, re.IGNORECASE)
        total = re.search(r"TOPLAM:\s*(\d+)", verdict, re.IGNORECASE)
        if not supported or not total:
            logger.warning("groundedness.unparsable", preview=verdict[:120])
            return None

        total_value = int(total.group(1))
        if total_value == 0:
            return None
        return min(1.0, int(supported.group(1)) / total_value)
