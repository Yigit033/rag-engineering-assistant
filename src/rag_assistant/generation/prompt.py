"""
Prompt yönetimi — versiyonlu, dosya tabanlı.

NEDEN PROMPT KODA GÖMÜLMEZ:
  Prompt bir MODEL PARAMETRESİDİR. Onu değiştirmek sistemin davranışını
  değiştirir — tıpkı chunk boyutunu veya top_k'yı değiştirmek gibi.
  Koda gömülü bir prompt şu üç şeyi imkânsız kılar:

    1. SÜRÜMLEME    → hangi prompt hangi skoru üretti, bilemezsin.
    2. KARŞILAŞTIRMA → v1 ile v2'yi aynı golden set üzerinde koşamazsın.
    3. DENETLENEBİLİRLİK → üretimde hangi talimatın verildiğini kanıtlayamazsın.

  Bu yüzden prompt'lar `prompts/<sürüm>/<ad>.txt` altında durur ve üretilen
  her cevap hangi sürümle üretildiğini `Answer.prompt_version` alanında taşır.

EKSİK DEĞİŞKEN = HATA:
  `str.format` eksik bir alanı sessizce boş bırakmaz, KeyError atar. Bunu
  bilerek yakalamıyoruz: yarısı boş bir prompt'la modeli çağırmak, sessizce
  bozuk bir sistem demektir. Açılışta patlamak daha iyidir.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from rag_assistant.observability import get_logger

logger = get_logger(__name__)

PROMPTS_ROOT = Path(__file__).parent / "prompts"


class PromptNotFoundError(FileNotFoundError):
    """İstenen prompt sürümü veya adı bulunamadı."""


class PromptLibrary:
    """Versiyonlu prompt şablonlarını yükler ve doldurur."""

    def __init__(self, version: str, *, root: Path | None = None) -> None:
        self._version = version
        self._root = (root or PROMPTS_ROOT) / version
        if not self._root.is_dir():
            available = (
                sorted(p.name for p in (root or PROMPTS_ROOT).iterdir() if p.is_dir())
                if (root or PROMPTS_ROOT).is_dir()
                else []
            )
            raise PromptNotFoundError(
                f"Prompt sürümü '{version}' bulunamadı. Mevcut sürümler: {available}"
            )

    @property
    def version(self) -> str:
        return self._version

    @lru_cache(maxsize=16)  # noqa: B019 - örnek başına şablon sayısı sabit ve küçük
    def _template(self, name: str) -> str:
        path = self._root / f"{name}.txt"
        if not path.is_file():
            available = sorted(p.stem for p in self._root.glob("*.txt"))
            raise PromptNotFoundError(
                f"'{name}' prompt'u '{self._version}' sürümünde yok. Mevcut: {available}"
            )
        return path.read_text(encoding="utf-8")

    def render(self, name: str, /, **variables: str) -> str:
        """
        Şablonu doldur.

        Eksik değişken KeyError ile patlar — sessizce yarım prompt üretmez.
        """
        template = self._template(name)
        try:
            return template.format(**variables)
        except KeyError as exc:
            raise KeyError(
                f"'{name}' prompt'u için eksik değişken: {exc}. "
                f"Verilenler: {sorted(variables)}"
            ) from exc
