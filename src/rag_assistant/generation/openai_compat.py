"""
OpenAI-uyumlu LLM adaptörü — tek sağlayıcıya kilitlenmemek için.

NEDEN BU ADAPTÖR TEK BAŞINA BÜYÜK BİR MİMARİ KAZANÇ:
  `/v1/chat/completions` fiilen sektör standardı oldu. Bu tek adaptör ile
  aşağıdakilerin HEPSİ kod değişmeden kullanılabilir:

      OpenAI · Groq · Together · OpenRouter · Fireworks · DeepInfra
      vLLM (kendi sunucun) · LM Studio · llama.cpp server
      Ollama'nın kendi /v1 ucu

  Yani "yerelde çalışsın, sunucuda büyük model kullansın" kararı bir
  YAPILANDIRMA kararına iner. Bu, donanım kısıtının mimariyi belirlemesini
  engeller: dizüstünde 4B model, sunucuda 70B model, aynı kod.

SIR YÖNETİMİ:
  API anahtarı yapılandırma nesnesinde TUTULMAZ. Yalnızca anahtarı taşıyan
  ortam değişkeninin ADI tutulur; değer çağrı anında okunur. Böylece
  yapılandırma loglansa, hata izine girse veya diske yazılsa bile sır sızmaz.
"""

from __future__ import annotations

import json
import os
from collections.abc import Iterator

import httpx

from rag_assistant.generation.llm import (
    LLMError,
    LLMUnavailableError,
    _ReasoningStreamFilter,
    strip_reasoning,
)
from rag_assistant.observability import get_logger

logger = get_logger(__name__)


class OpenAICompatLLM:
    """OpenAI-uyumlu bir uca konuşur. `LLM` protokolünü sağlar."""

    def __init__(
        self,
        model: str,
        *,
        base_url: str,
        api_key_env: str = "RAG_LLM_API_KEY",
        temperature: float = 0.0,
        max_tokens: int = 1024,
        timeout_seconds: float = 120.0,
        seed: int | None = 42,
    ) -> None:
        self._model = model
        # Kullanıcı hem ".../v1" hem de kök adres verebilir; ikisini de kabul et.
        self._base_url = base_url.rstrip("/")
        if not self._base_url.endswith("/v1"):
            self._base_url = f"{self._base_url}/v1"
        self._api_key_env = api_key_env
        self._temperature = temperature
        self._max_tokens = max_tokens
        self._timeout = timeout_seconds
        self._seed = seed

    @property
    def model_id(self) -> str:
        return self._model

    def _headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        # Anahtar çağrı anında okunur, nesnede saklanmaz.
        api_key = os.environ.get(self._api_key_env)
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        return headers

    def _payload(self, prompt: str, temperature: float | None, *, stream: bool) -> dict[str, object]:
        payload: dict[str, object] = {
            "model": self._model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": self._temperature if temperature is None else temperature,
            "max_tokens": self._max_tokens,
            "stream": stream,
        }
        if self._seed is not None:
            payload["seed"] = self._seed
        return payload

    # ------------------------------------------------------------------
    def health(self) -> bool:
        try:
            with httpx.Client(timeout=10.0) as client:
                response = client.get(f"{self._base_url}/models", headers=self._headers())
                response.raise_for_status()
        except httpx.HTTPError as exc:
            logger.warning("llm.health_failed", provider="openai_compat", error=str(exc))
            return False
        return True

    def generate(self, prompt: str, *, temperature: float | None = None) -> str:
        try:
            with httpx.Client(timeout=self._timeout) as client:
                response = client.post(
                    f"{self._base_url}/chat/completions",
                    json=self._payload(prompt, temperature, stream=False),
                    headers=self._headers(),
                )
                response.raise_for_status()
                data = response.json()
        except httpx.ConnectError as exc:
            raise LLMUnavailableError(f"Uca bağlanılamadı: {self._base_url}") from exc
        except httpx.HTTPStatusError as exc:
            raise LLMError(
                f"LLM hatası {exc.response.status_code}: {exc.response.text[:200]}"
            ) from exc
        except httpx.TimeoutException as exc:
            raise LLMError(f"LLM {self._timeout} saniyede yanıt vermedi") from exc

        usage = data.get("usage", {})
        logger.info(
            "llm.generated",
            provider="openai_compat",
            model=self._model,
            prompt_tokens=usage.get("prompt_tokens"),
            output_tokens=usage.get("completion_tokens"),
        )
        # Düşünen modeller burada da <think> bloğu üretir (bkz. llm.py).
        # Ayıklanmazsa iz içindeki "[1]" gerçek atıf sanılır.
        return strip_reasoning(str(data["choices"][0]["message"]["content"]))

    def stream(self, prompt: str, *, temperature: float | None = None) -> Iterator[str]:
        """Server-Sent Events akışını çöz."""
        reasoning_filter = _ReasoningStreamFilter()
        try:
            with httpx.Client(timeout=self._timeout) as client, client.stream(
                "POST",
                f"{self._base_url}/chat/completions",
                json=self._payload(prompt, temperature, stream=True),
                headers=self._headers(),
            ) as response:
                response.raise_for_status()
                for line in response.iter_lines():
                    if not line.startswith("data: "):
                        continue
                    body = line[6:].strip()
                    if body == "[DONE]":
                        break
                    try:
                        chunk = json.loads(body)
                        delta = chunk["choices"][0].get("delta", {}).get("content")
                    except (json.JSONDecodeError, KeyError, IndexError):
                        # Tek bozuk olay tüm akışı düşürmemeli.
                        logger.warning("llm.stream_bad_event", preview=body[:80])
                        continue
                    if delta:
                        visible = reasoning_filter.feed(delta)
                        if visible:
                            yield visible
        except httpx.ConnectError as exc:
            raise LLMUnavailableError(f"Uca bağlanılamadı: {self._base_url}") from exc
        except httpx.HTTPError as exc:
            raise LLMError(f"LLM akış hatası: {exc}") from exc
