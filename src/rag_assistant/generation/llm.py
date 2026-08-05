"""
LLM istemcisi — Ollama HTTP API'si üzerinden.

NEDEN SDK DEĞİL, DOĞRUDAN HTTP:
  Ollama'nın Python SDK'sı ince bir HTTP sarmalayıcıdan ibaret. Doğrudan
  `httpx` kullanmak bize şunları veriyor:
    * Bir bağımlılık daha az (SDK sürüm uyumsuzluğu riski yok)
    * Zaman aşımı, yeniden deneme ve akış (streaming) üzerinde tam kontrol
    * Ne gönderdiğimizin ve ne aldığımızın tam olarak görünür olması
  Bu, `LLM` protokolünün arkasında durduğu için üst katmanlar etkilenmez;
  yarın OpenAI/Anthropic uyumlu bir uca geçilecekse yalnızca bu dosya değişir.

DETERMİNİZM:
  `temperature=0` varsayılan. Bilgi çıkarma işlerinde zorunludur: aynı soru
  aynı cevabı vermezse ne hata ayıklayabilirsin ne de değerlendirebilirsin.
  `seed` de sabitleniyor — Ollama'da bu, tekrarlanabilirliği belirgin artırır.
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterator

import httpx

from rag_assistant.observability import get_logger

logger = get_logger(__name__)

DEFAULT_SEED = 42

# ---------------------------------------------------------------------------
# Düşünen (reasoning) model çıktısını temizleme
# ---------------------------------------------------------------------------
# 2026'da yaygın modellerin çoğu cevaptan önce bir "düşünme" izi üretir:
#     <think>Kullanıcı X soruyor. Bağlamda [1]'de şu var...</think>
#     Asıl cevap [1].
#
# BU İZ AYIKLANMAK ZORUNDA. İki sebeple:
#   1. ATIF GÜVENLİĞİ: düşünme bloğunun içinde de "[1]" geçer. Ayıklanmazsa
#      atıf çözücü onu gerçek bir atıf sayar → dayanağı olmayan bir cevap
#      "kaynak gösterilmiş" gibi görünür. En sinsi hata biçimi.
#   2. ÇEKİMSERLİK TESPİTİ: model düşünürken "bu bilgi dokümanlarda yok"
#      diye akıl yürütüp sonra yine de cevap uydurabilir. İz ayıklanmazsa
#      sistem bunu "çekimser kaldı" diye raporlar.
_THINK_CLOSED = re.compile(r"<think\b[^>]*>.*?</think\s*>", re.DOTALL | re.IGNORECASE)
_THINK_UNCLOSED = re.compile(r"<think\b[^>]*>.*\Z", re.DOTALL | re.IGNORECASE)
_THINK_OPEN_TAG = re.compile(r"<think\b[^>]*>", re.IGNORECASE)
_THINK_CLOSE_TAG = re.compile(r"</think\s*>", re.IGNORECASE)


def strip_reasoning(text: str) -> str:
    """
    Düşünme izini çıkar, yalnızca görünür cevabı döndür.

    Kapanmamış blok da temizlenir: token bütçesi düşünme sırasında biterse
    çıktı `<think>` ile başlar ve hiç kapanmaz. O durumda geriye boş metin
    kalır — bu bir hata değil, bütçe yetersizliğinin işaretidir ve çağıran
    katman bunu görüp uyarır.
    """
    cleaned = _THINK_CLOSED.sub("", text)
    cleaned = _THINK_UNCLOSED.sub("", cleaned)
    # Artık kalan tekil etiketler (bozuk çıktı) da gitsin.
    cleaned = _THINK_CLOSE_TAG.sub("", _THINK_OPEN_TAG.sub("", cleaned))
    return cleaned.strip()


class _ReasoningStreamFilter:
    """
    Akış sırasında düşünme izini bastırır.

    Akışta metin parça parça gelir, dolayısıyla `<think>` etiketi iki parçaya
    bölünebilir. Bu yüzden durum tutuyoruz: etiket görene kadar bekleyen küçük
    bir tampon ve "şu an düşünme içindeyiz" bayrağı.
    """

    def __init__(self) -> None:
        self._buffer = ""
        self._inside = False

    def feed(self, piece: str) -> str:
        """Yeni parçayı ver, dışarıya yazılacak (görünür) kısmı al."""
        self._buffer += piece
        visible: list[str] = []

        while self._buffer:
            if self._inside:
                match = _THINK_CLOSE_TAG.search(self._buffer)
                if not match:
                    # Kapanış henüz gelmedi. Etiketin bölünmüş olabileceği
                    # kadarını sakla, gerisini at.
                    self._buffer = self._buffer[-16:]
                    break
                self._buffer = self._buffer[match.end() :]
                self._inside = False
                continue

            match = _THINK_OPEN_TAG.search(self._buffer)
            if not match:
                # Açılış etiketi yok. "<" ile başlayan olası bölünmüş etiketi
                # tamponda bırak, öncesini yaz.
                cut = self._buffer.rfind("<")
                if cut == -1:
                    visible.append(self._buffer)
                    self._buffer = ""
                else:
                    visible.append(self._buffer[:cut])
                    self._buffer = self._buffer[cut:]
                    if len(self._buffer) > 16:  # "<think ...>" den uzun → etiket değil
                        visible.append(self._buffer)
                        self._buffer = ""
                break

            visible.append(self._buffer[: match.start()])
            self._buffer = self._buffer[match.end() :]
            self._inside = True

        return "".join(visible)


class LLMError(RuntimeError):
    """LLM çağrısı başarısız."""


class LLMUnavailableError(LLMError):
    """Servise hiç ulaşılamadı — model yüklü değil veya Ollama kapalı."""


class OllamaLLM:
    """Ollama üzerinden metin üretimi. `LLM` protokolünü sağlar."""

    def __init__(
        self,
        model: str,
        *,
        base_url: str = "http://localhost:11434",
        temperature: float = 0.0,
        max_tokens: int = 1024,
        context_window: int = 8192,
        timeout_seconds: float = 120.0,
        seed: int = DEFAULT_SEED,
        keep_alive: str = "30m",
    ) -> None:
        self._model = model
        self._base_url = base_url.rstrip("/")
        self._temperature = temperature
        self._max_tokens = max_tokens
        self._context_window = context_window
        self._timeout = timeout_seconds
        self._seed = seed
        self._keep_alive = keep_alive

    @property
    def model_id(self) -> str:
        return self._model

    # ------------------------------------------------------------------
    def _options(self, temperature: float | None) -> dict[str, object]:
        return {
            "temperature": self._temperature if temperature is None else temperature,
            "num_predict": self._max_tokens,
            "num_ctx": self._context_window,
            "seed": self._seed,
        }

    def warmup(self) -> bool:
        """
        Modeli belleğe yükle ve orada tut.

        NEDEN BU BİR AÇILIŞ ADIMI (ölçülmüş bir zorunluluk):
          Bellek kısıtlı bir makinede embedding modeli (2.3 GB) zaten
          yüklüyken LLM yüklenmeye çalışırsa Ollama şunu döndürür:
              "model requires more system memory (3.8 GiB) than is
               available (1.1 GiB)"
          Sıra tersine çevrilince — önce LLM ısıtılır, sonra embedder
          yüklenir — ikisi bir arada çalışır. Ölçüldü.

        `keep_alive` modeli bellekte tutar; aksi halde Ollama birkaç dakika
        sonra boşaltır ve aynı çakışma bir sonraki soruda geri gelir.

        Bu yüzden uygulama açılışında (FastAPI `lifespan`, CLI başlangıcı)
        embedding modelinden ÖNCE çağrılır.
        """
        payload = {
            "model": self._model,
            "prompt": "hazır",
            "stream": False,
            "options": {"num_predict": 1, "num_ctx": self._context_window},
            "keep_alive": self._keep_alive,
        }
        try:
            with httpx.Client(timeout=self._timeout) as client:
                response = client.post(f"{self._base_url}/api/generate", json=payload)
                response.raise_for_status()
        except httpx.HTTPError as exc:
            logger.warning("llm.warmup_failed", model=self._model, error=str(exc))
            return False
        logger.info("llm.warmed_up", model=self._model, keep_alive=self._keep_alive)
        return True

    def health(self) -> bool:
        """Servis ayakta ve model yüklü mü?"""
        try:
            with httpx.Client(timeout=5.0) as client:
                response = client.get(f"{self._base_url}/api/tags")
                response.raise_for_status()
                names = {m["name"] for m in response.json().get("models", [])}
        except (httpx.HTTPError, ValueError, KeyError) as exc:
            logger.warning("llm.health_failed", error=str(exc))
            return False

        # "qwen3:8b" ile "qwen3:8b-instruct-q4" gibi etiket varyantlarını da kabul et
        available = self._model in names or any(n.startswith(f"{self._model}:") for n in names)
        if not available:
            logger.warning("llm.model_missing", model=self._model, available=sorted(names))
        return available

    # ------------------------------------------------------------------
    def generate(self, prompt: str, *, temperature: float | None = None) -> str:
        """Tam cevabı tek seferde üret."""
        payload = {
            "model": self._model,
            "prompt": prompt,
            "stream": False,
            "options": self._options(temperature),
            "keep_alive": self._keep_alive,
        }
        try:
            with httpx.Client(timeout=self._timeout) as client:
                response = client.post(f"{self._base_url}/api/generate", json=payload)
                response.raise_for_status()
                data = response.json()
        except httpx.ConnectError as exc:
            raise LLMUnavailableError(
                f"Ollama'ya bağlanılamadı ({self._base_url}). Servis çalışıyor mu?"
            ) from exc
        except httpx.HTTPStatusError as exc:
            raise LLMError(f"Ollama hatası {exc.response.status_code}: {exc.response.text[:200]}") from exc
        except httpx.TimeoutException as exc:
            raise LLMError(f"Ollama {self._timeout} saniyede yanıt vermedi") from exc

        raw = str(data.get("response", ""))
        text = strip_reasoning(raw)

        # Boş görünür cevap = token bütçesinin tamamı düşünmeye gitti.
        # Sessizce boş cevap döndürmek yerine ne yapılacağını söylüyoruz.
        if raw.strip() and not text:
            logger.warning(
                "llm.reasoning_consumed_budget",
                model=self._model,
                num_predict=self._max_tokens,
                hint="düşünen model; RAG_LLM__MAX_TOKENS değerini artır",
            )

        logger.info(
            "llm.generated",
            model=self._model,
            prompt_tokens=data.get("prompt_eval_count"),
            output_tokens=data.get("eval_count"),
            reasoning_stripped=len(raw) - len(text),
            ms=int(data.get("total_duration", 0) / 1e6),
        )
        return text

    def stream(self, prompt: str, *, temperature: float | None = None) -> Iterator[str]:
        """
        Cevabı parça parça üret.

        Toplam süre aynı olsa bile ilk token'ın erken gelmesi algılanan hızı
        belirgin değiştirir; kullanıcı bekleyen her arayüzde tercih edilir.
        """
        payload = {
            "model": self._model,
            "prompt": prompt,
            "stream": True,
            "options": self._options(temperature),
            "keep_alive": self._keep_alive,
        }
        # Düşünme izi akışta da bastırılır; kullanıcı modelin iç monologunu
        # görmemeli. Etiket iki parçaya bölünebildiği için durum tutulur.
        reasoning_filter = _ReasoningStreamFilter()
        try:
            with httpx.Client(timeout=self._timeout) as client, client.stream(
                "POST", f"{self._base_url}/api/generate", json=payload
            ) as response:
                response.raise_for_status()
                for line in response.iter_lines():
                    if not line:
                        continue
                    try:
                        chunk = json.loads(line)
                    except json.JSONDecodeError:
                        # Bozuk tek satır tüm akışı düşürmemeli.
                        logger.warning("llm.stream_bad_line", preview=line[:80])
                        continue
                    piece = chunk.get("response", "")
                    if piece:
                        visible = reasoning_filter.feed(piece)
                        if visible:
                            yield visible
                    if chunk.get("done"):
                        break
        except httpx.ConnectError as exc:
            raise LLMUnavailableError(
                f"Ollama'ya bağlanılamadı ({self._base_url})."
            ) from exc
        except httpx.HTTPError as exc:
            raise LLMError(f"Ollama akış hatası: {exc}") from exc
