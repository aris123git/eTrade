"""
seed_database_improved.py

Production-ready database seeder for eTrade Discovery Engine.

Key Improvements:
- Single transaction for all operations (BIG performance gain)
- JSON metadata storage (json.dumps, not str)
- Full broker-market relationship
- Proper pip calculation using MT5's point field
- Extensible detector system (detector pattern)
- Currency extraction via length + validation
- Comprehensive logging with tracebacks
- Clean 3-stage architecture: Discovery → Normalizer → Repository
- Configuration-driven design

Version: 3.0.0
"""

import json
import logging
import sqlite3
import threading
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any, Set, Type
from uuid import uuid4

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ==============================================================================
# CONFIGURATION
# ==============================================================================

@dataclass
class SeederConfig:
    """Configuration for database seeder."""
    
    # Database
    db_path: str = "market_ai.db"
    
    # MT5
    mt5_retry_attempts: int = 3
    mt5_retry_delay: float = 1.0
    
    # Batch
    batch_size: int = 1000
    
    # Timeframes
    default_timeframes: List[str] = field(default_factory=lambda: [
        "M1", "M5", "M15", "M30", "H1", "H4", "D1", "W1", "MN1"
    ])
    
    # Market types
    default_market_types: Dict[str, str] = field(default_factory=lambda: {
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
    })
    
    # Fiat currencies (for validation)
    fiat_currencies: Set[str] = field(default_factory=lambda: {
        "USD", "EUR", "GBP", "JPY", "CHF", "AUD", "CAD", "NZD",
        "MXN", "ZAR", "TRY", "HKD", "SGD", "SEK", "NOK", "DKK",
        "PLN", "CZK", "HUF", "ILS", "KRW", "TWD", "THB", "MYR",
        "IDR", "PHP", "CNY", "RUB", "BRL", "ARS", "CLP", "COP",
        "PEN", "SAR", "AED", "QAR", "KWD", "BHD", "OMR"
    })
    
    # Crypto currencies
    crypto_currencies: Set[str] = field(default_factory=lambda: {
        "BTC", "ETH", "SOL", "XRP", "ADA", "DOGE", "BNB", "DOT",
        "LTC", "LINK", "AVAX", "MATIC", "UNI", "ATOM", "XLM",
        "ETC", "VET", "ICP", "FIL", "THETA", "ALGO", "AXS",
        "XMR", "EOS", "NEO", "IOTA", "DAI", "MKR", "COMP"
    })


config = SeederConfig()


# ==============================================================================
# DATABASE CONTEXT MANAGER
# ==============================================================================

class DatabaseConnection:
    """
    Database connection context manager with single transaction support.
    
    All operations are performed within a single transaction for performance.
    Uses BEGIN IMMEDIATE for atomicity.
    """
    
    def __init__(self, db_path: str = config.db_path):
        self.db_path = db_path
        self.conn: Optional[sqlite3.Connection] = None
        self.cursor: Optional[sqlite3.Cursor] = None
    
    def __enter__(self):
        self.conn = sqlite3.connect(self.db_path)
        self.conn.row_factory = sqlite3.Row
        self.cursor = self.conn.cursor()
        # Start immediate transaction (prevents deadlocks)
        self.cursor.execute("BEGIN IMMEDIATE")
        logger.debug("🔒 Transaction started")
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.conn:
            if exc_type is None:
                self.conn.commit()
                logger.debug("✅ Transaction committed")
            else:
                self.conn.rollback()
                logger.warning(f"⚠️ Transaction rolled back: {exc_type.__name__}")
            self.conn.close()
    
    def execute(self, sql: str, params: Tuple = ()):
        """Execute SQL with parameters."""
        return self.cursor.execute(sql, params)
    
    def executemany(self, sql: str, params: List[Tuple]):
        """Execute many SQL statements."""
        return self.cursor.executemany(sql, params)
    
    def fetchone(self):
        return self.cursor.fetchone()
    
    def fetchall(self):
        return self.cursor.fetchall()
    
    def lastrowid(self):
        return self.cursor.lastrowid


# ==============================================================================
# MT5 MANAGER (Thread-safe)
# ==============================================================================

class MT5Manager:
    """
    Thread-safe MT5 connection manager with retry logic.
    
    Singleton pattern ensures only one connection exists.
    Supports multiple brokers via login/password/server.
    """
    
    _instance: Optional['MT5Manager'] = None
    _lock = threading.Lock()
    
    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._initialized = False
            return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
        
        self.mt5 = None
        self.connected = False
        self._mt5_lock = threading.Lock()
        self._initialized = True
        self._connection_params = {}
    
    def initialize(
        self,
        path: Optional[str] = None,
        login: Optional[int] = None,
        password: Optional[str] = None,
        server: Optional[str] = None,
    ) -> bool:
        """
        Initialize MT5 connection with retry logic.
        
        Supports multiple brokers by accepting connection parameters.
        """
        if self.connected and not self._connection_params:
            return True
        
        # Store parameters for reconnection
        self._connection_params = {
            'path': path,
            'login': login,
            'password': password,
            'server': server,
        }
        
        with self._mt5_lock:
            try:
                import MetaTrader5 as mt5
                self.mt5 = mt5
                
                for attempt in range(config.mt5_retry_attempts):
                    # Build connection parameters
                    kwargs = {}
                    if path:
                        kwargs['path'] = path
                    if login:
                        kwargs['login'] = login
                    if password:
                        kwargs['password'] = password
                    if server:
                        kwargs['server'] = server
                    
                    if mt5.initialize(**kwargs):
                        self.connected = True
                        logger.info(f"✅ MT5 connected (attempt {attempt + 1})")
                        if login:
                            logger.info(f"   Login: {login} on {server or 'default'}")
                        return True
                    
                    logger.warning(f"⚠️ MT5 connection attempt {attempt + 1} failed, retrying...")
                    time.sleep(config.mt5_retry_delay)
                
                logger.error("❌ MT5 connection failed after all retries")
                return False
                
            except ImportError:
                logger.error("❌ MetaTrader5 module not installed")
                return False
            except Exception as e:
                logger.exception(f"❌ MT5 initialization error: {e}")
                return False
    
    def get_symbols(self) -> List[Any]:
        """Get all symbols with retry logic."""
        if not self.connected:
            if not self.initialize():
                return []
        
        with self._mt5_lock:
            for attempt in range(config.mt5_retry_attempts):
                try:
                    symbols = self.mt5.symbols_get()
                    if symbols is not None:
                        return symbols
                except Exception as e:
                    logger.warning(f"⚠️ symbols_get attempt {attempt + 1} failed: {e}")
                
                time.sleep(config.mt5_retry_delay)
            
            logger.error("❌ Failed to get symbols after all retries")
            return []
    
    def get_account_info(self) -> Optional[Dict]:
        """Get account information."""
        if not self.connected:
            if not self.initialize():
                return None
        
        try:
            account = self.mt5.account_info()
            if account:
                return {
                    'login': account.login,
                    'server': account.server,
                    'name': account.name,
                    'company': account.company,
                    'currency': account.currency,
                    'balance': account.balance,
                    'equity': account.equity,
                    'margin': account.margin,
                    'free_margin': account.margin_free,
                    'profit': account.profit,
                }
        except Exception as e:
            logger.exception(f"❌ Error getting account info: {e}")
        
        return None
    
    def get_terminal_info(self) -> Optional[Dict]:
        """Get terminal information."""
        if not self.connected:
            return None
        
        try:
            terminal = self.mt5.terminal_info()
            if terminal:
                return {
                    'name': terminal.name,
                    'company': terminal.company,
                    'path': terminal.path,
                    'build': terminal.build,
                }
        except Exception as e:
            logger.exception(f"❌ Error getting terminal info: {e}")
        
        return None
    
    def shutdown(self):
        """Shutdown MT5 connection."""
        if self.connected and self.mt5:
            try:
                self.mt5.shutdown()
            except Exception as e:
                logger.warning(f"⚠️ Error during MT5 shutdown: {e}")
            finally:
                self.connected = False


# ==============================================================================
# DETECTOR PATTERN (Extensible)
# ==============================================================================

class BaseDetector(ABC):
    """Abstract base class for market type detectors."""
    
    @abstractmethod
    def matches(self, symbol_obj: Any, normalized_name: str) -> bool:
        """Check if this detector matches the symbol."""
        pass
    
    @abstractmethod
    def get_market_type(self) -> str:
        """Return the market type name."""
        pass
    
    @abstractmethod
    def get_priority(self) -> int:
        """Return priority (higher = checked first)."""
        return 50


class ForexDetector(BaseDetector):
    """Detect Forex pairs."""
    
    def matches(self, symbol_obj: Any, normalized_name: str) -> bool:
        # Check MT5 path first
        path = getattr(symbol_obj, 'path', '')
        if path and 'forex' in path.lower():
            return True
        
        # Check if it looks like a currency pair (6 characters)
        if len(normalized_name) == 6:
            base = normalized_name[:3]
            quote = normalized_name[3:6]
            if base in config.fiat_currencies and quote in config.fiat_currencies:
                return True
        
        return False
    
    def get_market_type(self) -> str:
        return "forex"
    
    def get_priority(self) -> int:
        return 100


class CryptoDetector(BaseDetector):
    """Detect Cryptocurrencies."""
    
    def matches(self, symbol_obj: Any, normalized_name: str) -> bool:
        # Check MT5 path
        path = getattr(symbol_obj, 'path', '')
        if path and 'crypto' in path.lower():
            return True
        
        # Check if it starts with crypto currency
        for crypto in config.crypto_currencies:
            if normalized_name.startswith(crypto):
                quote = normalized_name[len(crypto):]
                if quote in config.fiat_currencies or quote in config.crypto_currencies:
                    return True
        
        return False
    
    def get_market_type(self) -> str:
        return "crypto"
    
    def get_priority(self) -> int:
        return 90


class IndexDetector(BaseDetector):
    """Detect Stock Indices."""
    
    INDEX_PATTERN = re.compile(
        r'^(US|GER|UK|FRA|JPN|AUS|CHI|HK|ITA|ESP|NETH|SWI|NZD|CAD|AUD|SGP|BRA|IND|KOR|RUS|TUR)'
        r'([0-9]{1,4}|[A-Z]{1,3}|[A-Z]{1,3}[0-9]{1,2}|[A-Z]{1,3}[0-9]{1,2}[A-Z]?)$'
    )
    
    def matches(self, symbol_obj: Any, normalized_name: str) -> bool:
        # Check MT5 path
        path = getattr(symbol_obj, 'path', '')
        if path and ('index' in path.lower() or 'indices' in path.lower()):
            return True
        
        # Check regex pattern
        return bool(self.INDEX_PATTERN.match(normalized_name))
    
    def get_market_type(self) -> str:
        return "index"
    
    def get_priority(self) -> int:
        return 80


class MetalDetector(BaseDetector):
    """Detect Precious Metals."""
    
    METALS = {'XAU', 'XAG', 'XPT', 'XPD', 'GOLD', 'SILVER'}
    
    def matches(self, symbol_obj: Any, normalized_name: str) -> bool:
        # Check MT5 path
        path = getattr(symbol_obj, 'path', '')
        if path and ('metal' in path.lower() or 'commodity' in path.lower()):
            # Check if it's a metal
            for metal in self.METALS:
                if metal in normalized_name:
                    return True
        
        # Check if it starts with metal symbol
        for metal in self.METALS:
            if normalized_name.startswith(metal):
                return True
        
        return False
    
    def get_market_type(self) -> str:
        return "commodity"
    
    def get_priority(self) -> int:
        return 85


class CommodityDetector(BaseDetector):
    """Detect Commodities (Oil, Gas, Agriculture)."""
    
    COMMODITIES = {
        'WTI', 'BRENT', 'OIL', 'NG', 'GAS', 'NATGAS',
        'WHEAT', 'CORN', 'SOY', 'COFFEE', 'SUGAR', 'COTTON', 'COCOA',
        'LUMBER', 'LEAN', 'CATTLE', 'HOG'
    }
    
    def matches(self, symbol_obj: Any, normalized_name: str) -> bool:
        # Check MT5 path
        path = getattr(symbol_obj, 'path', '')
        if path and 'commodity' in path.lower():
            return True
        
        # Check if it contains commodity name
        for comm in self.COMMODITIES:
            if comm in normalized_name:
                return True
        
        return False
    
    def get_market_type(self) -> str:
        return "commodity"
    
    def get_priority(self) -> int:
        return 75


class StockDetector(BaseDetector):
    """Detect Stocks."""
    
    def matches(self, symbol_obj: Any, normalized_name: str) -> bool:
        # Check MT5 path
        path = getattr(symbol_obj, 'path', '')
        if path and 'stock' in path.lower():
            return True
        
        # Check if it's 4+ uppercase letters
        if re.match(r'^[A-Z]{4,}$', normalized_name):
            # Exclude common forex pairs
            if len(normalized_name) == 6:
                base = normalized_name[:3]
                quote = normalized_name[3:6]
                if base in config.fiat_currencies and quote in config.fiat_currencies:
                    return False
            return True
        
        return False
    
    def get_market_type(self) -> str:
        return "stock"
    
    def get_priority(self) -> int:
        return 70


class DetectorManager:
    """
    Manages all detectors and orchestrates market type detection.
    """
    
    def __init__(self):
        self.detectors: List[BaseDetector] = []
        self._register_detectors()
    
    def _register_detectors(self):
        """Register all detectors in priority order."""
        detectors = [
            ForexDetector(),
            CryptoDetector(),
            MetalDetector(),
            IndexDetector(),
            CommodityDetector(),
            StockDetector(),
        ]
        # Sort by priority (highest first)
        self.detectors = sorted(detectors, key=lambda d: d.get_priority(), reverse=True)
    
    def detect(self, symbol_obj: Any, normalized_name: str) -> Optional[str]:
        """
        Detect market type for a symbol.
        
        Returns None if no detector matches.
        """
        for detector in self.detectors:
            if detector.matches(symbol_obj, normalized_name):
                return detector.get_market_type()
        
        return None


# ==============================================================================
# CURRENCY EXTRACTOR
# ==============================================================================

class CurrencyExtractor:
    """
    Extract base and quote currencies from MT5 symbols.
    
    Uses MT5's native fields first, falls back to name parsing.
    """
    
    @staticmethod
    def extract(symbol_obj: Any) -> Tuple[Optional[str], Optional[str]]:
        """
        Extract base and quote currencies.
        
        Returns (base, quote) or (None, None).
        """
        # Try MT5's native fields first
        base = getattr(symbol_obj, 'currency_base', None)
        quote = getattr(symbol_obj, 'currency_profit', None)
        
        if base and quote:
            return base, quote
        
        # Fallback: parse from symbol name
        normalized = SymbolNormalizer.normalize(symbol_obj.name)
        
        # Check if it's a crypto pair
        for crypto in config.crypto_currencies:
            if normalized.startswith(crypto):
                quote_part = normalized[len(crypto):]
                if quote_part in config.fiat_currencies or quote_part in config.crypto_currencies:
                    return crypto, quote_part
        
        # Check if it's a forex pair (6 characters)
        if len(normalized) == 6:
            base = normalized[:3]
            quote = normalized[3:6]
            if base in config.fiat_currencies and quote in config.fiat_currencies:
                return base, quote
        
        # Check 7+ character pairs (e.g., EURUSDm, GBPUSDm)
        if len(normalized) >= 7 and normalized[:6] in config.fiat_currencies:
            base = normalized[:3]
            quote = normalized[3:6]
            if base in config.fiat_currencies and quote in config.fiat_currencies:
                return base, quote
        
        return None, None


# ==============================================================================
# SYMBOL NORMALIZER
# ==============================================================================

class SymbolNormalizer:
    """
    Normalize MT5 symbol names.
    
    Removes broker-specific suffixes for canonical representation.
    """
    
    # Comprehensive suffix pattern
    SUFFIX_PATTERN = re.compile(
        r'(\.(cash|pro|mini|ecn|raw|a|b|c|d|e|f|g|h|i|j|k|l|m|n|o|p|q|r|s|t|u|v|w|x|y|z|'
        r'demo|live|test|practice|real|sim|simulation|swapfree|islamic|'
        r'cfd|diff|swap|zero|standard|commission|free|micro|nano)|'
        r'_(swapfree|islamic|demo|live|test|practice|real|sim|'
        r'a|b|c|d|e|f|g|h|i|j|k|l|m|n|o|p|q|r|s|t|u|v|w|x|y|z|'
        r'1|2|3|4|5|6|7|8|9|0))$',
        re.IGNORECASE
    )
    
    @classmethod
    def normalize(cls, symbol: str) -> str:
        """Normalize symbol name."""
        if not symbol:
            return symbol
        
        # Remove broker suffixes
        normalized = cls.SUFFIX_PATTERN.sub('', symbol)
        
        return normalized


# ==============================================================================
# PIP CALCULATOR
# ==============================================================================

class PipCalculator:
    """
    Calculate pip size correctly for any instrument.
    
    Uses MT5's point field and instrument type.
    """
    
    @staticmethod
    def calculate(symbol_obj: Any, market_type: Optional[str] = None) -> float:
        """
        Calculate pip size.
        
        Prefers MT5's point field, with market-specific adjustments.
        """
        # Try MT5's point first
        point = getattr(symbol_obj, 'point', None)
        if point is not None and point > 0:
            # For forex with 5-digit quotes, pip = point * 10
            digits = getattr(symbol_obj, 'digits', 5)
            
            if market_type == 'forex':
                if digits == 5 or digits == 3:
                    return point * 10
                return point
            
            # For crypto, indices, commodities, use point directly
            return point
        
        # Try trade_tick_size
        tick_size = getattr(symbol_obj, 'trade_tick_size', None)
        if tick_size is not None and tick_size > 0:
            return tick_size
        
        # Fallback: calculate from digits
        digits = getattr(symbol_obj, 'digits', 5)
        return 10 ** (-digits)


# ==============================================================================
# NORMALIZER LAYER
# ==============================================================================

class Normalizer:
    """
    Normalizes MT5 symbols into database-ready objects.
    
    This is the second stage of the 3-stage architecture.
    """
    
    def __init__(self):
        self.detector_manager = DetectorManager()
        self.currency_extractor = CurrencyExtractor()
        self.pip_calculator = PipCalculator()
    
    def normalize_symbol(self, symbol_obj: Any) -> Dict:
        """
        Normalize a single MT5 symbol into a dictionary.
        
        Returns a dictionary with all normalized fields.
        """
        name = symbol_obj.name
        normalized = SymbolNormalizer.normalize(name)
        
        # Detect market type
        market_type = self.detector_manager.detect(symbol_obj, normalized)
        
        # Extract currencies
        base, quote = self.currency_extractor.extract(symbol_obj)
        
        # Calculate pip size
        pip_size = self.pip_calculator.calculate(symbol_obj, market_type)
        
        # Extract other MT5 fields
        return {
            'symbol': name,
            'normalized_symbol': normalized,
            'base_currency': base,
            'quote_currency': quote,
            'market_type': market_type,
            'pip_size': pip_size,
            'point': getattr(symbol_obj, 'point', None),
            'digits': getattr(symbol_obj, 'digits', 5),
            'contract_size': getattr(symbol_obj, 'trade_contract_size', 1.0),
            'tick_size': getattr(symbol_obj, 'trade_tick_size', None),
            'tick_value': getattr(symbol_obj, 'trade_tick_value', 0.0),
            'path': getattr(symbol_obj, 'path', None),
            'trade_mode': getattr(symbol_obj, 'trade_mode', None),
            'trade_calc_mode': getattr(symbol_obj, 'trade_calc_mode', None),
            'swap_mode': getattr(symbol_obj, 'swap_mode', None),
            'is_synthetic': False,
            'metadata': symbol_obj._asdict() if hasattr(symbol_obj, '_asdict') else {},
            'description': getattr(symbol_obj, 'description', None),
        }


# ==============================================================================
# REPOSITORY LAYER
# ==============================================================================

class CurrencyRegistry:
    """
    Currency registry with caching.
    
    Ensures each currency code exists only once in the database.
    """
    
    def __init__(self, db: DatabaseConnection):
        self.db = db
        self.cache: Dict[str, str] = {}  # code -> id
        self.stats: Dict[str, int] = {
            'total_currencies': 0,
            'cached_lookups': 0,
            'db_lookups': 0,
            'created': 0,
        }
        self._load_currencies()
    
    def _load_currencies(self):
        """Load existing currencies into cache."""
        try:
            self.db.execute("SELECT id, code FROM currencies")
            for row in self.db.fetchall():
                self.cache[row['code']] = row['id']
            self.stats['total_currencies'] = len(self.cache)
            logger.debug(f"✅ Loaded {len(self.cache)} currencies into cache")
        except sqlite3.OperationalError:
            # Table might not exist yet
            pass
    
    def ensure_currency(self, code: str) -> Tuple[Optional[str], bool]:
        """
        Ensure a currency exists in the database.
        
        Returns:
            (currency_id, created) where created is True if new
        """
        if not code:
            return None, False
        
        # Check cache
        if code in self.cache:
            self.stats['cached_lookups'] += 1
            return self.cache[code], False
        
        self.stats['db_lookups'] += 1
        
        # Check database
        self.db.execute(
            "SELECT id FROM currencies WHERE code = ?",
            (code,)
        )
        row = self.db.fetchone()
        
        if row:
            self.cache[code] = row['id']
            return row['id'], False
        
        # Create new currency
        currency_id = str(uuid4())
        self.db.execute(
            "INSERT INTO currencies (id, code, name, created_at) VALUES (?, ?, ?, ?)",
            (currency_id, code, code, datetime.now().isoformat())
        )
        
        self.cache[code] = currency_id
        self.stats['total_currencies'] += 1
        self.stats['created'] += 1
        
        logger.debug(f"✅ Created new currency: {code} (ID: {currency_id})")
        return currency_id, True
    
    def get_stats(self) -> Dict:
        return {
            'total_currencies': self.stats['total_currencies'],
            'cache_hits': self.stats['cached_lookups'],
            'cache_misses': self.stats['db_lookups'],
            'created': self.stats['created'],
        }


class Repository:
    """
    Repository layer for database operations.
    
    This is the third stage of the 3-stage architecture.
    """
    
    def __init__(self, db: DatabaseConnection):
        self.db = db
        self.currencies = CurrencyRegistry(db)
        self._cache = {
            'timeframes': {},
            'market_types': {},
            'brokers': {},
        }
    
    def ensure_timeframe(self, name: str) -> Optional[str]:
        """Ensure timeframe exists, return ID."""
        if name in self._cache['timeframes']:
            return self._cache['timeframes'][name]
        
        tf_id = str(uuid4())
        self.db.execute(
            "INSERT OR IGNORE INTO timeframes (id, name) VALUES (?, ?)",
            (tf_id, name)
        )
        
        result = self.db.execute(
            "SELECT id FROM timeframes WHERE name = ?",
            (name,)
        ).fetchone()
        
        if result:
            self._cache['timeframes'][name] = result['id']
            return result['id']
        return None
    
    def ensure_market_type(self, name: str) -> Optional[str]:
        """Ensure market type exists, return ID."""
        if name in self._cache['market_types']:
            return self._cache['market_types'][name]
        
        mt_id = str(uuid4())
        self.db.execute(
            "INSERT OR IGNORE INTO market_types (id, name) VALUES (?, ?)",
            (mt_id, name)
        )
        
        result = self.db.execute(
            "SELECT id FROM market_types WHERE name = ?",
            (name,)
        ).fetchone()
        
        if result:
            self._cache['market_types'][name] = result['id']
            return result['id']
        return None
    
    def ensure_broker(self, name: str, server: Optional[str] = None, metadata: Optional[Dict] = None) -> Optional[str]:
        """Ensure broker exists, return ID."""
        key = f"{name}_{server or 'unknown'}"
        
        if key in self._cache['brokers']:
            return self._cache['brokers'][key]
        
        # Check existing
        if server:
            self.db.execute(
                "SELECT id FROM brokers WHERE name = ? AND server = ?",
                (name, server)
            )
        else:
            self.db.execute(
                "SELECT id FROM brokers WHERE name = ?",
                (name,)
            )
        
        row = self.db.fetchone()
        if row:
            self._cache['brokers'][key] = row['id']
            return row['id']
        
        # Create new
        broker_id = str(uuid4())
        self.db.execute(
            """INSERT INTO brokers (id, name, server, metadata, created_at)
               VALUES (?, ?, ?, ?, ?)""",
            (broker_id, name, server, json.dumps(metadata) if metadata else None, datetime.now().isoformat())
        )
        
        self._cache['brokers'][key] = broker_id
        return broker_id
    
    def save_market(self, normalized_data: Dict, broker_id: Optional[str]) -> Tuple[Optional[str], bool]:
        """
        Save a normalized market.
        
        Returns:
            (market_id, created) where created is True if new
        """
        # Ensure currencies
        base_id, _ = self.currencies.ensure_currency(normalized_data.get('base_currency'))
        quote_id, _ = self.currencies.ensure_currency(normalized_data.get('quote_currency'))
        
        # Ensure market type
        market_type = normalized_data.get('market_type')
        market_type_id = None
        if market_type:
            type_name = config.default_market_types.get(market_type, market_type)
            market_type_id = self.ensure_market_type(type_name)
        
        symbol = normalized_data['symbol']
        
        # Check if market exists
        if broker_id:
            self.db.execute(
                "SELECT id FROM markets WHERE symbol = ? AND broker_id = ?",
                (symbol, broker_id)
            )
            existing = self.db.fetchone()
            
            if existing:
                # Update existing
                self.db.execute("""
                    UPDATE markets SET
                        normalized_symbol = ?,
                        base_currency_id = ?,
                        quote_currency_id = ?,
                        market_type_id = ?,
                        pip_size = ?,
                        point = ?,
                        digits = ?,
                        contract_size = ?,
                        tick_size = ?,
                        tick_value = ?,
                        path = ?,
                        trade_mode = ?,
                        trade_calc_mode = ?,
                        swap_mode = ?,
                        metadata = ?,
                        description = ?,
                        updated_at = ?
                    WHERE id = ?
                """, (
                    normalized_data['normalized_symbol'],
                    base_id, quote_id,
                    market_type_id,
                    normalized_data['pip_size'],
                    normalized_data['point'],
                    normalized_data['digits'],
                    normalized_data['contract_size'],
                    normalized_data['tick_size'],
                    normalized_data['tick_value'],
                    normalized_data['path'],
                    normalized_data['trade_mode'],
                    normalized_data['trade_calc_mode'],
                    normalized_data['swap_mode'],
                    json.dumps(normalized_data['metadata'], default=str),
                    normalized_data.get('description'),
                    datetime.now().isoformat(),
                    existing['id']
                ))
                return existing['id'], False
        
        # Insert new
        market_id = str(uuid4())
        self.db.execute("""
            INSERT INTO markets (
                id, symbol, normalized_symbol,
                base_currency_id, quote_currency_id,
                broker_id, market_type_id,
                pip_size, point, digits,
                contract_size, tick_size, tick_value,
                path, is_synthetic,
                trade_mode, trade_calc_mode, swap_mode,
                metadata, description, created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            market_id,
            symbol,
            normalized_data['normalized_symbol'],
            base_id, quote_id,
            broker_id, market_type_id,
            normalized_data['pip_size'],
            normalized_data['point'],
            normalized_data['digits'],
            normalized_data['contract_size'],
            normalized_data['tick_size'],
            normalized_data['tick_value'],
            normalized_data['path'],
            normalized_data['is_synthetic'],
            normalized_data['trade_mode'],
            normalized_data['trade_calc_mode'],
            normalized_data['swap_mode'],
            json.dumps(normalized_data['metadata'], default=str),
            normalized_data.get('description'),
            datetime.now().isoformat(),
            datetime.now().isoformat()
        ))
        
        return market_id, True


# ==============================================================================
# MAIN SEEDER
# ==============================================================================

class DatabaseSeeder:
    """
    Main database seeder - Production Ready.
    
    3-Stage Architecture:
    1. Discovery: Collect raw MT5 symbols
    2. Normalizer: Convert to database-ready objects
    3. Repository: Save to database
    
    Single transaction for all operations.
    """
    
    def __init__(self, db_path: str = config.db_path):
        self.db_path = db_path
        self.mt5 = MT5Manager()
        self.normalizer = Normalizer()
        
        self.stats = {
            'timeframes': 0,
            'market_types': 0,
            'brokers': 0,
            'markets_inserted': 0,
            'markets_updated': 0,
            'errors': 0,
            'warnings': 0,
            'processed': 0,
        }
    
    def seed(self) -> bool:
        """Main seeding entry point - single transaction."""
        logger.info("=" * 70)
        logger.info("🚀 Starting Database Seeder v3.0 (Production Ready)")
        logger.info("=" * 70)
        
        # Connect MT5
        if not self.mt5.initialize():
            logger.error("❌ MT5 connection failed - cannot seed")
            return False
        
        # Get account info
        account = self.mt5.get_account_info()
        if account:
            logger.info(f"📊 Account: {account.get('login')} on {account.get('server')}")
        
        try:
            # SINGLE TRANSACTION for all operations
            with DatabaseConnection(self.db_path) as db:
                repo = Repository(db)
                
                # 1. Seed static data
                self._seed_timeframes(db, repo)
                self._seed_market_types(db, repo)
                
                # 2. Seed broker
                broker_id = self._seed_broker(db, repo, account)
                
                # 3. Discover and normalize markets
                symbols = self.mt5.get_symbols()
                if symbols:
                    self._process_markets(db, repo, symbols, broker_id)
                
                # 4. Create indexes
                self._create_indexes(db)
                
                # Transaction commits automatically on exit
                
        except Exception as e:
            logger.exception(f"❌ Seeding failed: {e}")
            return False
        
        finally:
            self.mt5.shutdown()
        
        # Print statistics
        self._print_stats()
        
        return True
    
    def _seed_timeframes(self, db: DatabaseConnection, repo: Repository):
        """Seed default timeframes."""
        logger.info("⏱️ Seeding timeframes...")
        for name in config.default_timeframes:
            if repo.ensure_timeframe(name):
                self.stats['timeframes'] += 1
        logger.info(f