"""
ai/__init__.py - eTrade Production AI Trading Engine

RESPONSIBILITY:
Public entry point for the complete AI trading intelligence layer.

VERSION: 1.0.0
"""

from typing import Any

from ai.config.settings import AIConfig, create_ai_config

try:
    from ai.services.pipeline import AIPipeline, create_ai_pipeline
except ModuleNotFoundError as exc:
    if exc.name not in {"ai.services", "ai.services.pipeline"}:
        raise
    AIPipeline = Any
    create_ai_pipeline = None

__version__ = "1.0.0"

__all__ = [
    "AIConfig",
    "create_ai_config",
    "AIPipeline",
    "create_ai_pipeline",
    "__version__",
]
