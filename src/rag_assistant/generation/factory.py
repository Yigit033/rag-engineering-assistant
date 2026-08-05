"""
LLM fabrikası ve açılış öncesi kaynak denetimi (preflight).

İKİ SORUMLULUK:

  1. YAPILANDIRMADAN DOĞRU İMPLEMENTASYONU ÜRET
     Üst katmanlar `LLM` protokolünü görür; hangi sağlayıcının seçildiğini
     yalnızca burası bilir. Sağlayıcı eklemek = buraya bir dal eklemek.

  2. SORUNU ÇALIŞMA ANINDA DEĞİL, AÇILIŞTA VE ANLAŞILIR BİÇİMDE BİLDİR
     Ham hâlinde sistem şunu veriyordu:

         Ollama hatası 500: model requires more system memory
         (6.1 GiB) than is available (3.2 GiB)

     Bu mesaj DOĞRU ama kullanıcı için eyleme dönüştürülebilir değil ve
     ancak ilk soru sorulduğunda ortaya çıkıyor. Preflight bunu açılışta,
     ne yapılacağını söyleyerek verir.
"""

from __future__ import annotations

from dataclasses import dataclass

from rag_assistant.config import LLMSettings
from rag_assistant.domain.protocols import LLM
from rag_assistant.generation.llm import OllamaLLM
from rag_assistant.generation.openai_compat import OpenAICompatLLM
from rag_assistant.observability import get_logger

logger = get_logger(__name__)


class LLMConfigurationError(RuntimeError):
    """LLM yapılandırması bu ortamda çalışamaz."""


def build_llm(settings: LLMSettings) -> LLM:
    """Yapılandırmaya göre `LLM` protokolünü sağlayan bir nesne üret."""
    if settings.provider == "ollama":
        return OllamaLLM(
            settings.model,
            base_url=settings.base_url,
            temperature=settings.temperature,
            max_tokens=settings.max_tokens,
            context_window=settings.context_window,
            timeout_seconds=settings.timeout_seconds,
            seed=settings.seed,
            keep_alive=settings.keep_alive,
        )
    if settings.provider == "openai_compat":
        return OpenAICompatLLM(
            settings.model,
            base_url=settings.base_url,
            api_key_env=settings.api_key_env,
            temperature=settings.temperature,
            max_tokens=settings.max_tokens,
            timeout_seconds=settings.timeout_seconds,
            seed=settings.seed,
        )
    raise LLMConfigurationError(f"Bilinmeyen sağlayıcı: {settings.provider}")


@dataclass(frozen=True, slots=True)
class PreflightResult:
    """Açılış denetiminin sonucu."""

    ok: bool
    provider: str
    model: str
    detail: str

    def raise_if_failed(self) -> None:
        if not self.ok:
            raise LLMConfigurationError(self.detail)


def preflight(settings: LLMSettings) -> PreflightResult:
    """
    LLM gerçekten kullanılabilir mi?

    Yalnızca "servis ayakta mı"ya bakmıyor; model gerçekten yüklü mü, ona da
    bakıyor. Servis ayakta ama model yok durumu, ilk soruda 404 ile patlar —
    bunu açılışta yakalamak çok daha ucuzdur.
    """
    llm = build_llm(settings)
    healthy = bool(getattr(llm, "health", lambda: True)())

    if healthy:
        # ISITMA SIRASI KRİTİK: modeli ŞİMDİ belleğe al, embedding modeli
        # yüklenmeden önce. Ters sırada bellek kısıtlı makinede LLM'e yer
        # kalmıyor (ölçüldü). Bu yüzden preflight sadece kontrol etmiyor,
        # aynı zamanda kaynağı REZERVE ediyor.
        warmup = getattr(llm, "warmup", None)
        if callable(warmup) and not warmup():
            detail = (
                f"'{settings.model}' belleğe yüklenemedi. Muhtemel sebep: yetersiz "
                f"boş RAM/VRAM.\n"
                f"  • Daha küçük model:   RAG_LLM__MODEL=qwen3:4b\n"
                f"  • Uzak/bulut uç:      RAG_LLM__PROVIDER=openai_compat"
            )
            logger.error("llm.preflight_warmup_failed", model=settings.model)
            return PreflightResult(False, settings.provider, settings.model, detail)

        logger.info("llm.preflight_ok", provider=settings.provider, model=settings.model)
        return PreflightResult(True, settings.provider, settings.model, "hazır")

    if settings.provider == "ollama":
        detail = (
            f"Ollama'da '{settings.model}' kullanılamıyor ({settings.base_url}).\n"
            f"  • Servis çalışıyor mu?           ollama serve\n"
            f"  • Model yüklü mü?                ollama pull {settings.model}\n"
            f"  • Bellek yetiyor mu?             8B model ~6 GB boş RAM/VRAM ister.\n"
            f"    Yetmiyorsa daha küçük bir modele geç:\n"
            f"        RAG_LLM__MODEL=qwen3:4b\n"
            f"  • Ya da uzak/bulut bir uca geç (kod değişmez):\n"
            f"        RAG_LLM__PROVIDER=openai_compat\n"
            f"        RAG_LLM__BASE_URL=https://<saglayici>/v1\n"
            f"        RAG_LLM__MODEL=<model-adi>\n"
            f"        {settings.api_key_env}=<anahtar>"
        )
    else:
        detail = (
            f"OpenAI-uyumlu uca ulaşılamadı: {settings.base_url}\n"
            f"  • Adres doğru mu? (sonu /v1 olmalı veya otomatik eklenir)\n"
            f"  • Anahtar tanımlı mı? Ortam değişkeni: {settings.api_key_env}\n"
            f"  • Model adı sağlayıcıda mevcut mu? ({settings.model})"
        )

    logger.error("llm.preflight_failed", provider=settings.provider, model=settings.model)
    return PreflightResult(False, settings.provider, settings.model, detail)
