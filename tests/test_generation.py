"""
Faz 4 testleri: prompt yönetimi ve halüsinasyon kontrolleri.

Sahte bir LLM kullanıyoruz. Bu sayede "model uydurulmuş atıf yaptığında ne
oluyor?" gibi soruları DETERMİNİSTİK olarak test edebiliyoruz — gerçek bir
modelle bunu tetiklemek şansa kalırdı.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence

import pytest

from rag_assistant.domain.models import Chunk, RetrievalStage, ScoredChunk, SourceRef
from rag_assistant.generation.answerer import GroundedAnswerer
from rag_assistant.generation.prompt import PromptLibrary, PromptNotFoundError


def scored(text: str, *, page: int = 1, tokens: int = 10) -> ScoredChunk:
    return ScoredChunk(
        chunk=Chunk(text=text, source=SourceRef("rapor.pdf", page), token_count=tokens),
        score=0.9,
        stage=RetrievalStage.RERANKED,
        rank=1,
    )


class FakeLLM:
    """Ne söyleyeceğini biz belirliyoruz. Çağrı sayısını da sayıyor."""

    model_id = "fake-llm"

    def __init__(self, response: str = "Cevap [1].") -> None:
        self.response = response
        self.calls: list[str] = []

    def generate(self, prompt: str, *, temperature: float | None = None) -> str:
        self.calls.append(prompt)
        return self.response

    def stream(self, prompt: str, *, temperature: float | None = None) -> Iterator[str]:
        self.calls.append(prompt)
        yield self.response


class FakeRetriever:
    name = "fake"

    def __init__(self, results: Sequence[ScoredChunk]) -> None:
        self._results = list(results)

    def retrieve(self, query: str, k: int) -> list[ScoredChunk]:
        return self._results[:k]


def make_answerer(
    llm: FakeLLM, chunks: Sequence[ScoredChunk], **kwargs: object
) -> GroundedAnswerer:
    params: dict[str, object] = {
        "retriever": FakeRetriever(chunks),
        "llm": llm,
        "prompts": PromptLibrary("v1"),
        "top_k": 5,
    }
    params.update(kwargs)
    return GroundedAnswerer(**params)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Prompt yönetimi
# ---------------------------------------------------------------------------
class TestPromptLibrary:
    def test_surum_yuklenir(self) -> None:
        assert PromptLibrary("v1").version == "v1"

    def test_olmayan_surum_hata_verir(self) -> None:
        with pytest.raises(PromptNotFoundError, match="v99"):
            PromptLibrary("v99")

    def test_olmayan_prompt_adi_hata_verir(self) -> None:
        with pytest.raises(PromptNotFoundError, match="yok_boyle"):
            PromptLibrary("v1").render("yok_boyle")

    def test_eksik_degisken_sessizce_gecmez(self) -> None:
        """
        Yarısı boş bir prompt'la modeli çağırmak = sessizce bozuk sistem.
        Açılışta patlamak daha iyidir.
        """
        with pytest.raises(KeyError, match="eksik değişken"):
            PromptLibrary("v1").render("answer", context="x")  # question ve abstain eksik

    def test_degiskenler_yerlesir(self) -> None:
        out = PromptLibrary("v1").render(
            "answer", context="BAGLAM_METNI", question="SORU_METNI", abstain_phrase="YOK"
        )
        assert "BAGLAM_METNI" in out
        assert "SORU_METNI" in out
        assert "YOK" in out


# ---------------------------------------------------------------------------
# SAVUNMA 1: boş bağlamda modeli hiç çağırma
# ---------------------------------------------------------------------------
class TestNoContextAbstention:
    def test_bos_baglamda_llm_cagrilmaz(self) -> None:
        """Boş bağlamla model çağırmak uydurma davetiyesidir."""
        llm = FakeLLM("Kesinlikle şöyle olmuştur...")
        answer = make_answerer(llm, [], abstain_when_no_context=True).answer("soru?")

        assert llm.calls == [], "bağlam yokken LLM çağrıldı"
        assert answer.abstained
        assert answer.citations == ()
        assert answer.is_grounded, "çekimserlik de geçerli bir grounded durumdur"


# ---------------------------------------------------------------------------
# SAVUNMA 3-4: atıf ayrıştırma ve doğrulama
# ---------------------------------------------------------------------------
class TestCitations:
    def test_gecerli_atiflar_cozulur(self) -> None:
        llm = FakeLLM("Sistem İSG odaklıdır [1]. Pazar 2.5 milyar USD [2].")
        answer = make_answerer(llm, [scored("a", page=1), scored("b", page=2)]).answer("s?")

        assert [c.marker for c in answer.citations] == [1, 2]
        assert answer.citations[0].source.page == 1
        assert answer.citations[1].source.page == 2
        assert answer.is_grounded

    def test_uydurulmus_atif_ayiklanir(self) -> None:
        """
        En sinsi halüsinasyon: kaynak göstermiş GİBİ görünen cevap.
        Bağlamda 2 chunk var ama model [7]'ye atıf yapıyor.
        """
        llm = FakeLLM("Şu doğrudur [1] ve şu da [7].")
        answer = make_answerer(llm, [scored("a"), scored("b")]).answer("s?")

        markers = [c.marker for c in answer.citations]
        assert markers == [1], "geçersiz atıf ayıklanmadı"
        assert all(c.marker <= 2 for c in answer.citations)

    def test_tekrarli_atif_bir_kez_sayilir(self) -> None:
        llm = FakeLLM("Şu [1] ve yine şu [1] ve tekrar [1].")
        answer = make_answerer(llm, [scored("a")]).answer("s?")
        assert len(answer.citations) == 1

    def test_atifsiz_cevap_isaretlenir_ama_degistirilmez(self) -> None:
        """
        TASARIM İLKESİ: modelin çıktısını sessizce değiştirmiyoruz.
        Sorunu görünür kılıyoruz; gizli 'düzeltme' ölçümü bozar.
        """
        original = "Kaynak göstermeden iddia ediyorum."
        llm = FakeLLM(original)
        answer = make_answerer(llm, [scored("a")], require_citations=True).answer("s?")

        assert answer.text == original, "cevap sessizce değiştirilmiş"
        assert answer.citations == ()
        assert not answer.is_grounded, "atıfsız cevap grounded sayılmamalı"

    def test_atif_chunk_kimligini_tasir(self) -> None:
        """Atıftan chunk'a geri izlenebilirlik — denetim için şart."""
        chunk_a = scored("içerik a")
        llm = FakeLLM("Cevap [1].")
        answer = make_answerer(llm, [chunk_a]).answer("s?")
        assert answer.citations[0].chunk_id == chunk_a.chunk.id


# ---------------------------------------------------------------------------
# SAVUNMA 2: çekimser kalma
# ---------------------------------------------------------------------------
class TestAbstention:
    def test_cekimser_cevap_tespit_edilir(self) -> None:
        llm = FakeLLM("Bu bilgi verilen dokümanlarda yok.")
        answer = make_answerer(llm, [scored("alakasiz")]).answer("s?")
        assert answer.abstained
        assert answer.is_grounded

    def test_buyuk_kucuk_harf_duyarsiz(self) -> None:
        llm = FakeLLM("BU BİLGİ VERİLEN DOKÜMANLARDA YOK.")
        assert make_answerer(llm, [scored("x")]).answer("s?").abstained

    def test_uzun_cevap_icindeki_ifade_cekimserlik_sayilmaz(self) -> None:
        """
        Model hem cevap verip hem 'şu kısım yok' diyorsa, cevap VERMİŞTİR.
        Bunu çekimserlik saymak, uydurmayı 'bilmiyorum' diye raporlamaktır.
        """
        llm = FakeLLM(
            "Sistem İSG odaklıdır [1] ve pazar büyüktür [1]. "
            "Ancak maliyet detayı için Bu bilgi verilen dokümanlarda yok "
            "diyebilirim, yine de tahminimce yüksektir."
        )
        answer = make_answerer(llm, [scored("a")]).answer("s?")
        assert not answer.abstained


# ---------------------------------------------------------------------------
# Bağlam bütçesi
# ---------------------------------------------------------------------------
class TestContextBudget:
    def test_butce_asilinca_kesilir(self) -> None:
        chunks = [scored(f"metin {i}", page=i, tokens=100) for i in range(1, 11)]
        llm = FakeLLM("Cevap [1].")
        answer = make_answerer(llm, chunks, top_k=10, max_context_tokens=250).answer("s?")
        assert len(answer.used_chunks) < 10
        assert sum(c.chunk.token_count for c in answer.used_chunks) <= 250

    def test_baglam_numaralandirilir(self) -> None:
        llm = FakeLLM("Cevap [1].")
        make_answerer(llm, [scored("birinci"), scored("ikinci", page=2)]).answer("s?")
        prompt = llm.calls[0]
        assert "[1]" in prompt and "[2]" in prompt
        assert "rapor.pdf · s.2" in prompt, "kaynak etiketi bağlamda yok"

    def test_prompt_cekimser_ifadeyi_icerir(self) -> None:
        """Modele 'bilmiyorsan şunu yaz' demezsen boşluğu uydurmayla doldurur."""
        llm = FakeLLM("x")
        make_answerer(llm, [scored("a")], abstain_phrase="BILMIYORUM").answer("s?")
        assert "BILMIYORUM" in llm.calls[0]


# ---------------------------------------------------------------------------
# SAVUNMA 5: groundedness
# ---------------------------------------------------------------------------
class TestGroundedness:
    def test_oran_hesaplanir(self) -> None:
        class JudgeLLM(FakeLLM):
            def generate(self, prompt: str, *, temperature: float | None = None) -> str:
                self.calls.append(prompt)
                if "denetlemek" in prompt:  # groundedness prompt'u
                    return "DESTEKLENEN: 3\nTOPLAM: 4"
                return "Cevap [1]."

        llm = JudgeLLM()
        answer = make_answerer(llm, [scored("a")], check_groundedness=True).answer("s?")
        assert answer.groundedness == pytest.approx(0.75)

    def test_okunamayan_yanit_none_doner(self) -> None:
        """Denetim başarısız olursa ana akış düşmemeli."""

        class BadJudge(FakeLLM):
            def generate(self, prompt: str, *, temperature: float | None = None) -> str:
                self.calls.append(prompt)
                return "Cevap [1]." if "denetlemek" not in prompt else "anlamsiz cikti"

        answer = make_answerer(BadJudge(), [scored("a")], check_groundedness=True).answer("s?")
        assert answer.groundedness is None
        assert answer.text == "Cevap [1]."

    def test_cekimser_cevapta_denetim_yapilmaz(self) -> None:
        """Çekimser cevabı denetlemek anlamsız — gereksiz LLM çağrısı."""
        llm = FakeLLM("Bu bilgi verilen dokümanlarda yok.")
        answer = make_answerer(llm, [scored("a")], check_groundedness=True).answer("s?")
        assert answer.groundedness is None
        assert len(llm.calls) == 1, "gereksiz ikinci çağrı yapıldı"


# ---------------------------------------------------------------------------
# İzlenebilirlik
# ---------------------------------------------------------------------------
class TestTraceability:
    def test_cevap_model_ve_prompt_surumunu_tasir(self) -> None:
        """
        Hangi cevap hangi model + hangi prompt sürümüyle üretildi?
        Bu olmadan değerlendirme sonuçları karşılaştırılamaz.
        """
        answer = make_answerer(FakeLLM("Cevap [1]."), [scored("a")]).answer("s?")
        assert answer.model == "fake-llm"
        assert answer.prompt_version == "v1"
        assert answer.latency_ms >= 0

    def test_kullanilan_chunklar_kaydedilir(self) -> None:
        chunks = [scored("a"), scored("b", page=2)]
        answer = make_answerer(FakeLLM("Cevap [1][2]."), chunks).answer("s?")
        assert len(answer.used_chunks) == 2
