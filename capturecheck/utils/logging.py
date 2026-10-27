"""Consistent logging for the CLI, scripts and the library."""

from __future__ import annotations

import logging
import os

_CONFIGURED = False
DEFAULT_FORMAT = "%(asctime)s %(levelname)-7s %(name)s | %(message)s"


def setup_logging(level: int | str | None = None) -> None:
    """Configure the root logger once. ``CAPTURECHECK_LOG_LEVEL`` overrides the level."""
    global _CONFIGURED
    if level is None:
        level = os.environ.get("CAPTURECHECK_LOG_LEVEL", "INFO")
    if isinstance(level, str):
        level = getattr(logging, level.upper(), logging.INFO)

    if not _CONFIGURED:
        logging.basicConfig(level=level, format=DEFAULT_FORMAT, datefmt="%H:%M:%S")
        _CONFIGURED = True
    else:
        logging.getLogger().setLevel(level)


def get_logger(name: str) -> logging.Logger:
    """Module logger; sets up logging on first use."""
    setup_logging()
    return logging.getLogger(name)
