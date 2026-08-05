"""
Kaynak (bellek) farkındalığı.

NEDEN GEREKLİ — ÖLÇÜLMÜŞ BİR ÇÖKME:
  Süreç, embedding modeli (2.3 GB) yüklüyken reranker'ı (2.2 GB) yüklemeye
  çalıştı ve işletim sistemi tarafından ÖLDÜRÜLDÜ. Dikkat: Python istisnası
  fırlatılmadı — yani `try/except` ile yakalanamaz. Süreç sessizce gitti.

  Bu, "önce dene, hata olursa yakala" yaklaşımının işlemediği bir durumdur.
  Tek doğru yöntem DENEMEDEN ÖNCE kontrol etmektir.

TASARIM İLKESİ:
  İsteğe bağlı ve kaliteyi artıran bir bileşen (reranker), zorunlu olan
  çekirdek işlevi (soru cevaplama) ASLA düşürmemeli. Yer yoksa o bileşen
  devre dışı kalır, sistem çalışmaya devam eder ve bu durum açıkça
  raporlanır — sessizce değil.
"""

from __future__ import annotations

import psutil

from rag_assistant.observability import get_logger

logger = get_logger(__name__)

# Model dosya boyutu ile bellek ihtiyacı aynı değildir: ağırlıklar float32'ye
# açılır, ayrıca ara tensörler ve tokenizer için ek yer gerekir. Ölçüme dayalı
# güvenlik payı.
MEMORY_OVERHEAD_FACTOR = 1.4


def available_ram_gb() -> float:
    """Şu anda kullanılabilir fiziksel bellek (GB)."""
    return psutil.virtual_memory().available / 1e9


def total_ram_gb() -> float:
    return psutil.virtual_memory().total / 1e9


def has_room_for(required_gb: float, *, label: str = "model") -> bool:
    """
    `required_gb` boyutunda bir model için yeterli boş bellek var mı?

    Güvenlik payı uygulanır: dosya boyutu kadar bellek asla yetmez.
    """
    needed = required_gb * MEMORY_OVERHEAD_FACTOR
    available = available_ram_gb()
    ok = available >= needed

    if not ok:
        logger.warning(
            "resources.insufficient_ram",
            component=label,
            needed_gb=round(needed, 2),
            available_gb=round(available, 2),
            total_gb=round(total_ram_gb(), 2),
        )
    return ok
