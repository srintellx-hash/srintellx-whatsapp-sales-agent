"""Logging setup, phone normalisation, and small helpers."""
from __future__ import annotations

import logging
import re
import sys
from datetime import datetime
from zoneinfo import ZoneInfo

from app.config import settings


def configure_logging() -> None:
    level = getattr(logging, settings.log_level.upper(), logging.INFO)
    root = logging.getLogger()
    root.setLevel(level)
    # Avoid duplicate handlers on reload.
    for h in list(root.handlers):
        root.removeHandler(h)
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(
        logging.Formatter(
            fmt="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
            datefmt="%Y-%m-%dT%H:%M:%S%z",
        )
    )
    root.addHandler(handler)
    # Quieten noisy libraries.
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("hpack").setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)


def now_tz() -> datetime:
    return datetime.now(ZoneInfo(settings.timezone))


def tz() -> ZoneInfo:
    return ZoneInfo(settings.timezone)


_PHONE_RE = re.compile(r"\D+")


def normalise_phone(raw: str) -> str:
    """Return digits only with a leading country code.

    WhatsApp sends MSISDN without '+' (e.g. '919876543210'). We keep that
    canonical digits-only form as the contact key.
    """
    digits = _PHONE_RE.sub("", raw or "")
    return digits


def truncate(text: str, limit: int = 4000) -> str:
    """WhatsApp text bodies are capped at 4096 chars; stay safely under."""
    if text is None:
        return ""
    text = text.strip()
    if len(text) <= limit:
        return text
    return text[: limit - 1].rstrip() + "…"
