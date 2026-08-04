"""
Yapılandırılmış loglama.

NEDEN `print` DEĞİL, NEDEN DÜZ `logging` DEĞİL:
  Bir RAG sisteminde "neden bu cevabı verdi?" sorusunu cevaplayabilmen gerekir.
  Bunun için gereken şey serbest metin log değil, SORGULANABİLİR olaydır:
      retrieval.completed  query=... dense=20 sparse=14 fused=27 both=6 ms=142
  Bu satırı `grep` ile değil, alan adıyla filtreleyebilirsin. Prod'da JSON'a
  çevirip herhangi bir log toplayıcıya (Loki, CloudWatch, Datadog) verirsin.

  `print` ise: seviyesi yok, zaman damgası yok, bağlam yok, kapatılamaz,
  kütüphane kodunda stdout'u kirletir. Kütüphane kodu ASLA print etmez.
"""

from __future__ import annotations

import logging
import sys
from typing import Any

import structlog

_configured = False


def configure_logging(*, level: str = "INFO", json_format: bool = False) -> None:
    """
    Loglamayı yapılandır. Uygulama giriş noktasında BİR KEZ çağrılır
    (CLI, API veya test), kütüphane modüllerinde asla.
    """
    global _configured
    if _configured:
        return

    # Windows konsolu cp1254 kullanır; UTF-8'e sabitlemezsek Türkçe karakter
    # veya emoji içeren bir log satırı UnicodeEncodeError ile programı çökertir.
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[union-attr]

    shared: list[Any] = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_log_level,
        structlog.processors.TimeStamper(fmt="iso", utc=False),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
    ]

    renderer: Any = (
        structlog.processors.JSONRenderer()
        if json_format
        else structlog.dev.ConsoleRenderer(colors=not json_format)
    )

    structlog.configure(
        processors=[*shared, renderer],
        wrapper_class=structlog.make_filtering_bound_logger(
            logging.getLevelNamesMapping()[level.upper()]
        ),
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )
    _configured = True


def get_logger(name: str) -> structlog.stdlib.BoundLogger:
    """
    Modül düzeyinde logger.

    Kullanım:  logger = get_logger(__name__)
    """
    return structlog.get_logger(name)  # type: ignore[no-any-return]
