"""
core/constants.py - Global Constants for MarketAI

RESPONSIBILITY:
Define all global constants used across MarketAI components.

ARCHITECTURAL DECISIONS:
1. Single source of truth - All constants in one place
2. Type-hinted constants - For IDE support
3. Grouped by category - Easy to find and maintain
4. No hardcoded values in business logic - Use constants instead
5. Immutable - Constants should not be modified at runtime

VERSION: 1.0.1
"""

from typing import Dict, Set, List, Tuple, Final
from enum import Enum


# ==============================================================================
# TIME CONSTANTS
# ==============================================================================

# Timeframes in seconds
TIMEFRAMES: Final[Dict[str, int]] = {
    "M1": 60,
    "M2": 120,
    "M3": 180,
    "M4": 240,
    "M5": 300,
    "M6": 360,
    "M10": 600,
    "M12": 720,
    "M15": 900,
    "M20": 1200,
    "M30": 1800,
    "H1": 3600,
    "H2": 7200,
    "H3": 10800,
    "H4": 14400,
    "H6": 21600,
    "H8": 28800,
    "H12": 43200,
    "D1": 86400,
    "W1": 604800,
    "MN1": 2592000,
}

# Valid timeframe names
VALID_TIMEFRAMES: Final[Set[str]] = set(TIMEFRAMES.keys())

# Timeframe display names
TIMEFRAME_DISPLAY: Final[Dict[str, str]] = {
    "M1": "1 Minute",
    "M5": "5 Minutes",
    "M15": "15 Minutes",
    "M30": "30 Minutes",
    "H1": "1 Hour",
    "H4": "4 Hours",
    "D1": "1 Day",
    "W1": "1 Week",
    "MN1": "1 Month",
}

# Default timeframes for data collection
DEFAULT_TIMEFRAMES: Final[List[str]] = ["M5", "M15", "H1", "H4", "D1"]

# Time constants (in seconds)
SECONDS_PER_MINUTE: Final[int] = 60
SECONDS_PER_HOUR: Final[int] = 3600
SECONDS_PER_DAY: Final[int] = 86400
SECONDS_PER_WEEK: Final[int] = 604800
SECONDS_PER_MONTH: Final[int] = 2592000  # 30 days

# Milliseconds conversions
MS_PER_SECOND: Final[int] = 1000
MS_PER_MINUTE: Final[int] = 60000
MS_PER_HOUR: Final[int] = 3600000


# ==============================================================================
# PRICE CONSTANTS
# ==============================================================================

# Pip sizes for different asset classes
PIP_SIZES: Final[Dict[str, float]] = {
    "forex": 0.0001,        # 4-digit forex
    "forex_5digit": 0.00001, # 5-digit forex
    "index": 0.01,          # Stock indices
    "crypto": 0.01,         # Cryptocurrencies (BTCUSD)
    "crypto_alt": 0.0001,   # Alternative cryptocurrencies (ETHUSD)
    "metal": 0.01,          # Metals (XAUUSD)
    "commodity": 0.001,     # Commodities (WTI)
}

# Price rounding digits by asset class
PRICE_DIGITS: Final[Dict[str, int]] = {
    "forex": 5,
    "index": 2,
    "crypto": 2,
    "metal": 2,
    "commodity": 3,
}


# ==============================================================================
# MAXIMUM CONSTANTS
# ==============================================================================

# Data limits
MAX_CANDLES_PER_REQUEST: Final[int] = 100000
MAX_CANDLES_PER_BATCH: Final[int] = 1000
MAX_HISTORICAL_DAYS: Final[int] = 3650  # 10 years
MAX_SYMBOLS: Final[int] = 10000

# Queue limits
MAX_QUEUE_SIZE: Final[int] = 10000
MAX_RETRY_ATTEMPTS: Final[int] = 3
MAX_CONCURRENT_DOWNLOADS: Final[int] = 5

# Database limits
MAX_SQL_PARAMETERS: Final[int] = 999  # SQLite limit
MAX_BATCH_INSERT: Final[int] = 500

# File limits
MAX_LOG_FILE_SIZE_MB: Final[int] = 100
MAX_BACKUP_FILES: Final[int] = 5


# ==============================================================================
# DEFAULT VALUES
# ==============================================================================

# Default configuration values
DEFAULT_LOG_LEVEL: Final[str] = "INFO"
DEFAULT_LOG_DIR: Final[str] = "logs"
DEFAULT_DB_PATH: Final[str] = "market_ai.db"
DEFAULT_DATA_DIR: Final[str] = "data"

# Default retry settings
DEFAULT_RETRY_ATTEMPTS: Final[int] = 3
DEFAULT_RETRY_DELAY: Final[float] = 1.0
DEFAULT_RETRY_BACKOFF: Final[float] = 2.0

# Default timeout settings (seconds)
DEFAULT_MT5_TIMEOUT: Final[int] = 30
DEFAULT_DB_TIMEOUT: Final[int] = 30
DEFAULT_HTTP_TIMEOUT: Final[int] = 10

# Default scheduler settings
DEFAULT_SCHEDULER_INTERVAL: Final[int] = 10  # seconds
DEFAULT_SCHEDULER_MAX_CONCURRENT: Final[int] = 3

# Default priority values
DEFAULT_PRIORITY: Final[int] = 50
PRIORITY_CRITICAL: Final[int] = 100
PRIORITY_HIGH: Final[int] = 80
PRIORITY_NORMAL: Final[int] = 50
PRIORITY_LOW: Final[int] = 20
PRIORITY_BACKGROUND: Final[int] = 10


# ==============================================================================
# ERROR CODES
# ==============================================================================

# Error code prefixes by component
ERROR_PREFIX: Final[Dict[str, str]] = {
    "config": "CFG",
    "database": "DB",
    "mt5": "MT5",
    "collector": "COL",
    "scheduler": "SCH",
    "discovery": "DSC",
    "validation": "VAL",
    "system": "SYS",
}

# Error codes
ERROR_CODES: Final[Dict[str, str]] = {
    # Configuration errors
    "CONFIG_NOT_FOUND": "CFG-001",
    "CONFIG_INVALID": "CFG-002",
    "CONFIG_VALIDATION_ERROR": "CFG-003",
    
    # Database errors
    "DB_CONNECTION_ERROR": "DB-001",
    "DB_QUERY_ERROR": "DB-002",
    "DB_SCHEMA_ERROR": "DB-003",
    "DB_INTEGRITY_ERROR": "DB-004",
    
    # MT5 errors
    "MT5_CONNECTION_ERROR": "MT5-001",
    "MT5_DISCONNECTED": "MT5-002",
    "MT5_DOWNLOAD_ERROR": "MT5-003",
    "MT5_TIMEOUT_ERROR": "MT5-004",
    "MT5_SYMBOL_NOT_FOUND": "MT5-005",
    
    # Collector errors
    "COLLECTOR_ERROR": "COL-001",
    "QUEUE_FULL": "COL-002",
    "QUEUE_EMPTY": "COL-003",
    "DUPLICATE_TASK": "COL-004",
    
    # Scheduler errors
    "SCHEDULER_ERROR": "SCH-001",
    "SCHEDULER_START_ERROR": "SCH-002",
    "SCHEDULER_STOP_ERROR": "SCH-003",
    
    # Discovery errors
    "DISCOVERY_ERROR": "DSC-001",
    "PATTERN_DISCOVERY_ERROR": "DSC-002",
    "CORRELATION_DISCOVERY_ERROR": "DSC-003",
    
    # Validation errors
    "VALIDATION_ERROR": "VAL-001",
    "LIVE_VALIDATION_ERROR": "VAL-002",
    
    # System errors
    "SYSTEM_ERROR": "SYS-001",
    "INITIALIZATION_ERROR": "SYS-002",
    "SHUTDOWN_ERROR": "SYS-003",
    "RETRY_EXHAUSTED": "SYS-004",
}


# ==============================================================================
# MARKET CATEGORIES
# ==============================================================================

# Market type categories
MARKET_TYPES: Final[Dict[str, str]] = {
    "forex": "Forex",
    "index": "Index",
    "commodity": "Commodity",
    "crypto": "Cryptocurrency",
    "stock": "Stock",
    "etf": "ETF",
    "bond": "Bond",
    "futures": "Futures",
    "option": "Option",
    "cfd": "CFD",
}

# Market type priorities for detection
MARKET_TYPE_PRIORITY: Final[Dict[str, int]] = {
    "forex": 100,
    "crypto": 90,
    "metal": 85,
    "index": 80,
    "commodity": 75,
    "stock": 70,
    "etf": 60,
    "bond": 50,
    "futures": 40,
    "option": 30,
    "cfd": 20,
}

# Fiat currencies
FIAT_CURRENCIES: Final[Set[str]] = {
    "USD", "EUR", "GBP", "JPY", "CHF", "AUD", "CAD", "NZD",
    "MXN", "ZAR", "TRY", "HKD", "SGD", "SEK", "NOK", "DKK",
    "PLN", "CZK", "HUF", "ILS", "KRW", "TWD", "THB", "MYR",
    "IDR", "PHP", "CNY", "RUB", "BRL", "ARS", "CLP", "COP",
    "PEN", "SAR", "AED", "QAR", "KWD", "BHD", "OMR", "JOD",
}

# Cryptocurrencies - Fixed: Removed duplicate "EOS"
CRYPTO_CURRENCIES: Final[Set[str]] = {
    "BTC", "ETH", "SOL", "XRP", "ADA", "DOGE", "BNB", "DOT",
    "LTC", "LINK", "AVAX", "MATIC", "UNI", "ATOM", "XLM",
    "ETC", "VET", "ICP", "FIL", "THETA", "ALGO", "AXS",
    "XMR", "NEO", "IOTA", "DAI", "MKR", "COMP",
    "AAVE", "WBTC", "LEO", "BCH", "LUNC", "FTT", "XDC",
    "STX", "HBAR", "HNT", "XEC", "BSV", "VTHO", "WAVES",
    "CEL", "EGLD", "XTZ", "EOS", "NEXO", "KCS", "XEM",
    "ZEC", "DASH", "NANO", "BAT", "REP", "LRC", "ZIL",
}

# Commodities
COMMODITIES: Final[Set[str]] = {
    "XAU", "XAG", "XPT", "XPD",  # Metals
    "WTI", "BRENT", "OIL", "NG", "GAS", "NATGAS",  # Energy
    "WHEAT", "CORN", "SOY", "COFFEE", "SUGAR", "COTTON", "COCOA",  # Agriculture
    "LUMBER", "LEAN", "CATTLE", "HOG",  # Livestock
}

# Indices patterns
INDICES: Final[Dict[str, str]] = {
    "US30": "Dow Jones Industrial Average",
    "US500": "S&P 500",
    "USTEC": "NASDAQ 100",
    "GER40": "DAX 40",
    "UK100": "FTSE 100",
    "FRA40": "CAC 40",
    "JPN225": "Nikkei 225",
    "HK50": "Hang Seng",
    "CHINA50": "China A50",
    "ITA40": "FTSE MIB",
    "ESP35": "IBEX 35",
    "NETH25": "AEX 25",
    "SWI20": "SMI 20",
    "AUS200": "ASX 200",
    "NZ50": "NZX 50",
    "SGP30": "STI 30",
    "BRA50": "Bovespa",
    "IND50": "Nifty 50",
    "KOR200": "KOSPI 200",
    "RUS50": "RTS 50",
    "TUR100": "BIST 100",
}

# Indices prefix mapping
INDEX_PREFIXES: Final[Set[str]] = {
    "US", "GER", "UK", "FRA", "JPN", "HK", "CHINA", "ITA", "ESP",
    "NETH", "SWI", "AUS", "NZ", "SGP", "BRA", "IND", "KOR", "RUS",
    "TUR", "CAN", "MEX", "SA", "UAE",
}


# ==============================================================================
# BROKER CONSTANTS
# ==============================================================================

# Broker suffix patterns for symbol normalization
BROKER_SUFFIX_PATTERNS: Final[Set[str]] = {
    ".cash", ".pro", ".mini", ".ecn", ".raw",
    ".a", ".b", ".c", ".d", ".e", ".f", ".g", ".h", ".i", ".j",
    ".k", ".l", ".m", ".n", ".o", ".p", ".q", ".r", ".s", ".t",
    ".u", ".v", ".w", ".x", ".y", ".z",
    "_i", "_m", "_a", "_b", "_c", "_d", "_e", "_f",
    "_1", "_2", "_3", "_4", "_5", "_6", "_7", "_8", "_9",
    ".demo", ".live", ".test", ".practice", ".real", ".sim",
    ".simulation", ".swapfree", ".islamic", ".cfd", ".diff", ".swap",
    ".zero", ".standard", ".commission", ".free", ".micro", ".nano",
}


# ==============================================================================
# PATH CONSTANTS
# ==============================================================================

# Directory names
DIR_DATA: Final[str] = "data"
DIR_LOGS: Final[str] = "logs"
DIR_MODELS: Final[str] = "models"
DIR_TEMPLATES: Final[str] = "templates"
DIR_CONFIG: Final[str] = "config"
DIR_DATABASE: Final[str] = "database"

# File names
FILE_DATABASE: Final[str] = "market_ai.db"
FILE_CONFIG: Final[str] = "config.yaml"
FILE_LOG: Final[str] = "market_ai.log"
FILE_README: Final[str] = "README.md"
FILE_REQUIREMENTS: Final[str] = "requirements.txt"


# ==============================================================================
# METRIC CONSTANTS
# ==============================================================================

# Metric names for monitoring
METRIC_NAMES: Final[Dict[str, str]] = {
    "candles_downloaded": "Total candles downloaded",
    "candles_skipped": "Total candles skipped",
    "patterns_discovered": "Total patterns discovered",
    "correlations_found": "Total correlations found",
    "validation_success": "Validations passed",
    "validation_failure": "Validations failed",
    "queue_size": "Current queue size",
    "processing_time": "Average processing time (ms)",
    "memory_usage": "Memory usage (MB)",
    "cpu_usage": "CPU usage (%)",
}


# ==============================================================================
# CANDLE CONSTANTS
# ==============================================================================

# Candle field names
CANDLE_FIELDS: Final[Set[str]] = {
    "open", "high", "low", "close", "volume",
    "tick_volume", "time", "spread", "real_volume",
}

# Required candle fields
CANDLE_REQUIRED_FIELDS: Final[Set[str]] = {"open", "high", "low", "close"}

# Candle field types - Fixed: "spread" changed from int to float
CANDLE_FIELD_TYPES: Final[Dict[str, type]] = {
    "time": int,
    "open": float,
    "high": float,
    "low": float,
    "close": float,
    "volume": int,
    "tick_volume": int,
    "spread": float,      # Some brokers report fractional spreads
    "real_volume": int,
}


# ==============================================================================
# REGEX PATTERNS
# ==============================================================================

# Compiled regex patterns are in utils.py to avoid import issues
# These are the pattern strings

SYMBOL_PATTERN_STR: Final[str] = r'^[A-Z0-9]{2,12}([._][A-Z0-9]{2,6})?$'
INDEX_PATTERN_STR: Final[str] = r'^(US|GER|UK|FRA|JPN|AUS|CHI|HK|ITA|ESP|NETH|SWI|NZD|CAD|AUD|SGP|BRA|IND|KOR|RUS|TUR)([0-9]{1,4}|[A-Z]{1,3})$'
BROKER_SUFFIX_STR: Final[str] = r'(\.(cash|pro|mini|ecn|raw|demo|live|test|practice|real|sim|simulation|swapfree|islamic|cfd|diff|swap|zero|standard|commission|free|micro|nano)|_(swapfree|islamic|demo|live|test|practice|real|sim|a|b|c|d|e|f|g|h|i|j|k|l|m|n|o|p|q|r|s|t|u|v|w|x|y|z))$'


# ==============================================================================
# LOGGING CONSTANTS
# ==============================================================================

# Log format strings
LOG_FORMAT: Final[str] = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
LOG_DATE_FORMAT: Final[str] = "%Y-%m-%d %H:%M:%S"

# Log levels
LOG_LEVEL_NAMES: Final[Dict[int, str]] = {
    10: "DEBUG",
    20: "INFO",
    30: "WARNING",
    40: "ERROR",
    50: "CRITICAL",
}


# ==============================================================================
# EXPORTS
# ==============================================================================

__all__ = [
    # Time constants
    'TIMEFRAMES',
    'VALID_TIMEFRAMES',
    'TIMEFRAME_DISPLAY',
    'DEFAULT_TIMEFRAMES',
    'SECONDS_PER_MINUTE',
    'SECONDS_PER_HOUR',
    'SECONDS_PER_DAY',
    'SECONDS_PER_WEEK',
    'SECONDS_PER_MONTH',
    'MS_PER_SECOND',
    'MS_PER_MINUTE',
    'MS_PER_HOUR',
    
    # Price constants
    'PIP_SIZES',
    'PRICE_DIGITS',
    
    # Maximum constants
    'MAX_CANDLES_PER_REQUEST',
    'MAX_CANDLES_PER_BATCH',
    'MAX_HISTORICAL_DAYS',
    'MAX_SYMBOLS',
    'MAX_QUEUE_SIZE',
    'MAX_RETRY_ATTEMPTS',
    'MAX_CONCURRENT_DOWNLOADS',
    'MAX_SQL_PARAMETERS',
    'MAX_BATCH_INSERT',
    'MAX_LOG_FILE_SIZE_MB',
    'MAX_BACKUP_FILES',
    
    # Default values
    'DEFAULT_LOG_LEVEL',
    'DEFAULT_LOG_DIR',
    'DEFAULT_DB_PATH',
    'DEFAULT_DATA_DIR',
    'DEFAULT_RETRY_ATTEMPTS',
    'DEFAULT_RETRY_DELAY',
    'DEFAULT_RETRY_BACKOFF',
    'DEFAULT_MT5_TIMEOUT',
    'DEFAULT_DB_TIMEOUT',
    'DEFAULT_HTTP_TIMEOUT',
    'DEFAULT_SCHEDULER_INTERVAL',
    'DEFAULT_SCHEDULER_MAX_CONCURRENT',
    'DEFAULT_PRIORITY',
    'PRIORITY_CRITICAL',
    'PRIORITY_HIGH',
    'PRIORITY_NORMAL',
    'PRIORITY_LOW',
    'PRIORITY_BACKGROUND',
    
    # Error codes
    'ERROR_PREFIX',
    'ERROR_CODES',
    
    # Market categories
    'MARKET_TYPES',
    'MARKET_TYPE_PRIORITY',
    'FIAT_CURRENCIES',
    'CRYPTO_CURRENCIES',
    'COMMODITIES',
    'INDICES',
    'INDEX_PREFIXES',
    
    # Broker constants
    'BROKER_SUFFIX_PATTERNS',
    
    # Path constants
    'DIR_DATA',
    'DIR_LOGS',
    'DIR_MODELS',
    'DIR_TEMPLATES',
    'DIR_CONFIG',
    'DIR_DATABASE',
    'FILE_DATABASE',
    'FILE_CONFIG',
    'FILE_LOG',
    'FILE_README',
    'FILE_REQUIREMENTS',
    
    # Metric constants
    'METRIC_NAMES',
    
    # Candle constants
    'CANDLE_FIELDS',
    'CANDLE_REQUIRED_FIELDS',
    'CANDLE_FIELD_TYPES',
    
    # Regex patterns
    'SYMBOL_PATTERN_STR',
    'INDEX_PATTERN_STR',
    'BROKER_SUFFIX_STR',
    
    # Logging constants
    'LOG_FORMAT',
    'LOG_DATE_FORMAT',
    'LOG_LEVEL_NAMES',
]