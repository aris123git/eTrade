"""
=========================================================
MarketAI Configuration
=========================================================
Author : Aristide & ChatGPT
Version: 1.0
=========================================================
"""

from pathlib import Path
import MetaTrader5 as mt5

# =========================================================
# PROJECT
# =========================================================

PROJECT_NAME = "MarketAI"

VERSION = "1.0.0"

ROOT = Path(__file__).parent

# =========================================================
# DATABASE
# =========================================================

DATABASE_FOLDER = ROOT / "database"

DATABASE_FOLDER.mkdir(exist_ok=True)

DATABASE_NAME = "market_ai.db"

DATABASE_PATH = DATABASE_FOLDER / DATABASE_NAME

# =========================================================
# DATA
# =========================================================

DATA_FOLDER = ROOT / "data"

DATA_FOLDER.mkdir(exist_ok=True)

LOG_FOLDER = ROOT / "logs"

LOG_FOLDER.mkdir(exist_ok=True)

# =========================================================
# MT5
# =========================================================

TIMEFRAMES = {

    "M1": mt5.TIMEFRAME_M1,
    "M5": mt5.TIMEFRAME_M5,
    "M15": mt5.TIMEFRAME_M15,
    "M30": mt5.TIMEFRAME_M30,
    "H1": mt5.TIMEFRAME_H1,
    "H4": mt5.TIMEFRAME_H4,
    "D1": mt5.TIMEFRAME_D1,
    "W1": mt5.TIMEFRAME_W1,
    "MN1": mt5.TIMEFRAME_MN1

}

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
# AI
# =========================================================

SAVE_RAW_DATA = True

SAVE_FEATURES = False

SAVE_PATTERNS = True