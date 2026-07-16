"""
ai/logging - Structured logging helpers for the AI engine

VERSION: 1.0.0
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Optional

from ai.config.settings import AIConfig


def get_ai_logger(name: str = "ai") -> logging.Logger:
    """Return a module logger under the ai namespace."""
    if not name.startswith("ai"):
        name = f"ai.{name}"
    return logging.getLogger(name)


def setup_ai_logging(
    config: Optional[AIConfig] = None,
    level: int = logging.INFO,
    log_file: Optional[Path] = None,
) -> logging.Logger:
    """
    Configure AI logging once. Safe to call multiple times.
    """
    logger = logging.getLogger("ai")
    if logger.handlers:
        return logger

    logger.setLevel(level)
    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    stream = logging.StreamHandler(sys.stdout)
    stream.setFormatter(formatter)
    stream.setLevel(level)
    logger.addHandler(stream)

    if config is not None:
        config.ensure_directories()
        target = Path(config.storage.root_dir) / config.storage.logs_dir / "ai_engine.log"
    elif log_file is not None:
        target = Path(log_file)
    else:
        target = None

    if target is not None:
        target.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(target, encoding="utf-8")
        file_handler.setFormatter(formatter)
        file_handler.setLevel(level)
        logger.addHandler(file_handler)

    logger.propagate = False
    return logger


__all__ = ["get_ai_logger", "setup_ai_logging"]
