"""Project-wide logging: console + rotating file in logs/emotionlens.log."""
from __future__ import annotations

import logging
import sys
from logging.handlers import RotatingFileHandler

from config import LOG_DIR

_ROOT_NAME = "emotionlens"
_configured = False


def _configure(level: int) -> logging.Logger:
    global _configured
    root = logging.getLogger(_ROOT_NAME)
    if _configured:
        return root

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(fmt)

    file_handler = RotatingFileHandler(
        LOG_DIR / "emotionlens.log",
        maxBytes=2_000_000,
        backupCount=3,
        encoding="utf-8",
    )
    file_handler.setFormatter(fmt)

    root.addHandler(console)
    root.addHandler(file_handler)
    root.setLevel(level)
    # Streamlit installs its own root handlers; do not duplicate our lines there.
    root.propagate = False
    _configured = True
    return root


def get_logger(name: str = _ROOT_NAME, level: int = logging.INFO) -> logging.Logger:
    """Return a child logger, e.g. get_logger("detector") -> 'emotionlens.detector'."""
    root = _configure(level)
    if name in ("", _ROOT_NAME):
        return root
    return root.getChild(name.removeprefix(f"{_ROOT_NAME}."))
