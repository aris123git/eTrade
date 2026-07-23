"""
=========================================================
MarketAI Configuration
=========================================================
Author : Aristide & ChatGPT
Version: 2.0.0 — Phase 2 infrastructure stabilization
=========================================================

Provides both module-level constants (legacy) and a typed
`Config` class expected throughout the codebase.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

# ---------------------------------------------------------------------------
# Optional MetaTrader5 — required on Windows trading hosts, optional elsewhere
# ---------------------------------------------------------------------------
try:
    import MetaTrader5 as mt5  # type: ignore
except ImportError:  # pragma: no cover - Linux / CI environments
    mt5 = None


# =========================================================
# PROJECT
# =========================================================

PROJECT_NAME = "MarketAI"
VERSION = "1.0.0"
# Project root is the parent of core/
ROOT = Path(__file__).resolve().parent.parent

# =========================================================
# DATABASE
# =========================================================

DATABASE_FOLDER = ROOT / "data" / "database"
DATABASE_FOLDER.mkdir(parents=True, exist_ok=True)

DATABASE_NAME = "market_ai.db"
DATABASE_PATH = DATABASE_FOLDER / DATABASE_NAME

# =========================================================
# DATA / LOGS
# =========================================================

DATA_FOLDER = ROOT / "data"
DATA_FOLDER.mkdir(parents=True, exist_ok=True)

LOG_FOLDER = ROOT / "logs"
LOG_FOLDER.mkdir(parents=True, exist_ok=True)

# =========================================================
# MT5 TIMEFRAMES
# =========================================================

if mt5 is not None:
    TIMEFRAMES: Dict[str, int] = {
        "M1": mt5.TIMEFRAME_M1,
        "M5": mt5.TIMEFRAME_M5,
        "M15": mt5.TIMEFRAME_M15,
        "M30": mt5.TIMEFRAME_M30,
        "H1": mt5.TIMEFRAME_H1,
        "H4": mt5.TIMEFRAME_H4,
        "D1": mt5.TIMEFRAME_D1,
        "W1": mt5.TIMEFRAME_W1,
        "MN1": mt5.TIMEFRAME_MN1,
    }
else:
    # Standard MT5 integer constants (documented by MetaQuotes)
    TIMEFRAMES = {
        "M1": 1,
        "M5": 5,
        "M15": 15,
        "M30": 30,
        "H1": 16385,
        "H4": 16388,
        "D1": 16408,
        "W1": 32769,
        "MN1": 49153,
    }

TIMEFRAME_SECONDS: Dict[str, int] = {
    "M1": 60,
    "M5": 300,
    "M15": 900,
    "M30": 1800,
    "H1": 3600,
    "H4": 14400,
    "D1": 86400,
    "W1": 604800,
    "MN1": 2592000,
}

DEFAULT_TIMEFRAMES = list(TIMEFRAMES.keys())

# =========================================================
# DOWNLOAD
# =========================================================

DOWNLOAD_SLEEP = 0.10
RETRY = 3
CHUNK_SIZE = 100000

# =========================================================
# VALIDATION
# =========================================================

REMOVE_DUPLICATES = False
MARK_INVALID = True

# =========================================================
# AI FLAGS
# =========================================================

SAVE_RAW_DATA = True
SAVE_FEATURES = False
SAVE_PATTERNS = True

# =========================================================
# TYPED CONFIG CLASS
# =========================================================


@dataclass
class Config:
    """
    Typed configuration object used across mt5/, discovery/,
    preprocessing/, knowledge/, validation/, and collector/.

    Defaults mirror the module-level constants so existing call
    sites that do `Config()` keep working.
    """

    project_name: str = PROJECT_NAME
    version: str = VERSION
    root: Path = field(default_factory=lambda: ROOT)
    database_path: Path = field(default_factory=lambda: DATABASE_PATH)
    data_folder: Path = field(default_factory=lambda: DATA_FOLDER)
    log_folder: Path = field(default_factory=lambda: LOG_FOLDER)
    timeframes: Dict[str, int] = field(default_factory=lambda: dict(TIMEFRAMES))
    default_timeframes: list = field(default_factory=lambda: list(DEFAULT_TIMEFRAMES))
    download_sleep: float = DOWNLOAD_SLEEP
    retry: int = RETRY
    chunk_size: int = CHUNK_SIZE
    remove_duplicates: bool = REMOVE_DUPLICATES
    mark_invalid: bool = MARK_INVALID
    save_raw_data: bool = SAVE_RAW_DATA
    save_features: bool = SAVE_FEATURES
    save_patterns: bool = SAVE_PATTERNS
    mt5_available: bool = field(default_factory=lambda: mt5 is not None)
    extra: Dict[str, Any] = field(default_factory=dict)

    def ensure_directories(self) -> None:
        """Create data and log directories if missing."""
        Path(self.database_path).parent.mkdir(parents=True, exist_ok=True)
        Path(self.data_folder).mkdir(parents=True, exist_ok=True)
        Path(self.log_folder).mkdir(parents=True, exist_ok=True)

    def get(self, key: str, default: Any = None) -> Any:
        """Dict-like access for legacy callers."""
        if hasattr(self, key):
            return getattr(self, key)
        return self.extra.get(key, default)

    def __getitem__(self, key: str) -> Any:
        value = self.get(key)
        if value is None and key not in self.extra and not hasattr(self, key):
            raise KeyError(key)
        return value


# Singleton-style default used by some modules: `from core.config import config`
config = Config()


def get_config(**overrides: Any) -> Config:
    """Factory returning a Config with optional field overrides."""
    base = Config()
    for key, value in overrides.items():
        if hasattr(base, key):
            setattr(base, key, value)
        else:
            base.extra[key] = value
    return base
