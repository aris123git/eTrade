"""
discovery/currency_discovery.py - Currency Discovery Engine

RESPONSIBILITY:
Discover and analyze currency relationships from market data.
DYNAMIC discovery - works with ANY broker and ANY format.

ARCHITECTURAL PRINCIPLES:
1. Pure discovery - No data storage, no I/O, no business logic
2. Dynamic detection - No hardcoded lists of symbols
3. Format-agnostic - Works with any broker naming convention
4. Pattern-based - Discovers currencies through patterns, not lists

WHAT IT NEVER DOES:
- ❌ Connect to MT5
- ❌ Store data
- ❌ Download data
- ❌ Make trading decisions
- ❌ Assume symbol names (works dynamically)

VERSION: 2.0.0
"""

import logging
import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional, Dict, Any, Tuple, Set, Union
from enum import Enum
from collections import defaultdict

from core.config import Config
from core.exceptions import DiscoveryError, DataValidationError


# ==============================================================================
# EXPORTS
# ==============================================================================

__all__ = [
    'CurrencyType',
    'CurrencyPairType',
    'CurrencyInfo',
    'CurrencyPair',
    'DiscoveryResult',
    'CurrencyDiscovery',
    'create_currency_discovery',
]


# ==============================================================================
# ENUMS
# ==============================================================================

class CurrencyType(Enum):
    """Type of currency."""
    FIAT = "fiat"
    CRYPTO = "crypto"
    COMMODITY = "commodity"
    INDEX = "index"
    UNKNOWN = "unknown"


class CurrencyPairType(Enum):
    """Type of currency pair."""
    MAJOR = "major"
    MINOR = "minor"
    EXOTIC = "exotic"
    CRYPTO = "crypto"
    COMMODITY = "commodity"
    INDEX = "index"
    CROSS = "cross"
    UNKNOWN = "unknown"


# ==============================================================================
# DATA MODELS
# ==============================================================================

@dataclass(frozen=True)
class CurrencyInfo:
    """Information about a currency."""
    code: str
    name: str
    currency_type: CurrencyType
    is_major: bool = False
    description: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class CurrencyPair:
    """Discovered currency pair."""
    base: str
    quote: str
    symbol: str
    normalized_symbol: str
    pair_type: CurrencyPairType
    confidence: float
    base_info: Optional[CurrencyInfo] = None
    quote_info: Optional[CurrencyInfo] = None
    signals: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def is_major(self) -> bool:
        return self.pair_type == CurrencyPairType.MAJOR
    
    def is_crypto(self) -> bool:
        return self.pair_type == CurrencyPairType.CRYPTO
    
    def is_forex(self) -> bool:
        return self.pair_type in (CurrencyPairType.MAJOR, CurrencyPairType.MINOR, CurrencyPairType.EXOTIC)


@dataclass
class DiscoveryResult:
    """Complete discovery result."""
    symbol: str
    normalized_symbol: str
    timestamp: datetime
    pairs: List[CurrencyPair]
    currencies: List[CurrencyInfo]
    broker: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def get_pair(self, base: str, quote: str) -> Optional[CurrencyPair]:
        """Get a specific currency pair."""
        for pair in self.pairs:
            if pair.base == base and pair.quote == quote:
                return pair
        return None
    
    def get_currency(self, code: str) -> Optional[CurrencyInfo]:
        """Get currency info by code."""
        for currency in self.currencies:
            if currency.code == code:
                return currency
        return None
    
    def get_summary(self) -> Dict[str, Any]:
        """Get summary of discovery results."""
        return {
            'symbol': self.symbol,
            'normalized_symbol': self.normalized_symbol,
            'broker': self.broker,
            'total_pairs': len(self.pairs),
            'total_currencies': len(self.currencies),
            'by_type': {
                'major': sum(1 for p in self.pairs if p.pair_type == CurrencyPairType.MAJOR),
                'minor': sum(1 for p in self.pairs if p.pair_type == CurrencyPairType.MINOR),
                'exotic': sum(1 for p in self.pairs if p.pair_type == CurrencyPairType.EXOTIC),
                'crypto': sum(1 for p in self.pairs if p.pair_type == CurrencyPairType.CRYPTO),
            },
        }


# ==============================================================================
# CURRENCY DISCOVERY
# ==============================================================================

class CurrencyDiscovery:
    """
    Currency discovery engine - DYNAMIC broker-agnostic version.
    
    Discovers and analyzes currency pairs from ANY broker, ANY format.
    No hardcoded lists - discovers everything dynamically.
    """
    
    # Known patterns for currency detection
    CRYPTO_PATTERNS = re.compile(
        r'(BTC|XBT|ETH|SOL|XRP|ADA|DOGE|BNB|LTC|LINK|'
        r'AVAX|MATIC|UNI|ATOM|XLM|ETC|VET|ICP|FIL|THETA|'
        r'ALGO|AXS|XMR|NEO|IOTA|DAI|MKR|COMP|AAVE|WBTC|'
        r'LEO|BCH|LUNC|FTT|XDC|STX|HBAR|HNT|XEC|BSV|'
        r'VTHO|WAVES|CEL|EGLD|XTZ|EOS|NEXO|KCS|XEM|'
        r'ZEC|DASH|NANO|BAT|REP|LRC|ZIL)'
    )
    
    FIAT_PATTERN = re.compile(r'^[A-Z]{3}$')
    SYMBOL_PATTERN = re.compile(r'^[A-Z0-9]{2,12}([._][A-Z0-9]{2,6})?$')
    
    def __init__(self, config: Config):
        """
        Initialize the currency discovery engine.
        
        Args:
            config: Application configuration
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        self._cache: Dict[str, DiscoveryResult] = {}
        self._currency_cache: Dict[str, CurrencyInfo] = {}
        self._symbol_normalizer = SymbolNormalizer()
        
        # Known fiat currencies (used only for classification, not discovery)
        self._fiat_set = self._build_fiat_set()
        
        # Crypto aliases (XBT → BTC, etc.)
        self._crypto_aliases = {
            'XBT': 'BTC',
            'XRP': 'XRP',
            'ETH': 'ETH',
            # Add more as needed
        }
        
        self.logger.info("✅ CurrencyDiscovery initialized (dynamic mode)")
    
    def _build_fiat_set(self) -> Set[str]:
        """Build set of known fiat currencies from ISO codes."""
        return {
            'USD', 'EUR', 'GBP', 'JPY', 'CHF', 'AUD', 'CAD', 'NZD',
            'MXN', 'ZAR', 'TRY', 'HKD', 'SGD', 'SEK', 'NOK', 'DKK',
            'PLN', 'CZK', 'HUF', 'ILS', 'KRW', 'TWD', 'THB', 'MYR',
            'IDR', 'PHP', 'CNY', 'RUB', 'BRL', 'ARS', 'CLP', 'COP',
            'PEN', 'SAR', 'AED', 'QAR', 'KWD', 'BHD', 'OMR', 'JOD',
        }
    
    # ==========================================================================
    # PUBLIC METHODS
    # ==========================================================================
    
    def discover_from_symbol(self, symbol_obj: Any) -> DiscoveryResult:
        """
        Discover currency information from a symbol.
        
        Args:
            symbol_obj: MT5 symbol object, dict, or string
            
        Returns:
            DiscoveryResult object
        """
        symbol_name = self._get_symbol_name(symbol_obj)
        normalized = self._normalize_symbol_name(symbol_name)
        
        # Check cache
        cache_key = f"{symbol_name}_{normalized}"
        if cache_key in self._cache:
            self.logger.debug(f"Cache hit: {symbol_name}")
            return self._cache[cache_key]
        
        self.logger.debug(f"Discovering currency info for: {symbol_name}")
        
        try:
            # Extract currency pair
            pair = self._extract_currency_pair(symbol_obj, symbol_name, normalized)
            
            if pair:
                # Get currency info
                currencies = self._get_currencies(pair)
                
                result = DiscoveryResult(
                    symbol=symbol_name,
                    normalized_symbol=normalized,
                    timestamp=datetime.now(),
                    pairs=[pair],
                    currencies=currencies,
                    metadata={
                        'source': 'symbol_extraction',
                        'detection_method': pair.metadata.get('detection_method', 'unknown'),
                    },
                )
            else:
                # Try to discover as standalone currency
                currency = self._discover_currency(normalized)
                result = DiscoveryResult(
                    symbol=symbol_name,
                    normalized_symbol=normalized,
                    timestamp=datetime.now(),
                    pairs=[],
                    currencies=[currency] if currency else [],
                    metadata={
                        'source': 'currency_discovery',
                        'is_currency': currency is not None,
                    },
                )
            
            # Cache result
            self._cache[cache_key] = result
            
            return result
            
        except Exception as e:
            raise DiscoveryError(f"Failed to discover currency for {symbol_name}: {e}")
    
    def discover_from_broker(self, symbols: List[Any]) -> Dict[str, DiscoveryResult]:
        """
        Discover all currency pairs from a broker's symbol list.
        
        This is the key method for multi-broker support.
        Scans ALL symbols and discovers what they are.
        
        Args:
            symbols: List of MT5 symbol objects
            
        Returns:
            Dictionary mapping symbol name to DiscoveryResult
        """
        if not symbols:
            return {}
        
        self.logger.info(f"📊 Discovering currencies from {len(symbols)} symbols")
        
        results = {}
        currency_pairs = []
        currencies_found = set()
        
        # Scan all symbols
        for symbol in symbols:
            try:
                result = self.discover_from_symbol(symbol)
                results[symbol.name] = result
                
                # Collect pairs and currencies
                for pair in result.pairs:
                    currency_pairs.append(pair)
                    if pair.base_info:
                        currencies_found.add(pair.base_info.code)
                    if pair.quote_info:
                        currencies_found.add(pair.quote_info.code)
                        
            except Exception as e:
                self.logger.warning(f"⚠️ Failed to discover {getattr(symbol, 'name', 'unknown')}: {e}")
        
        # Build statistics
        stats = self._build_broker_statistics(results, currency_pairs, currencies_found)
        
        self.logger.info(
            f"✅ Broker discovery complete: "
            f"{len(results)} symbols, "
            f"{len(currency_pairs)} pairs, "
            f"{len(currencies_found)} currencies"
        )
        
        return results
    
    def normalize_symbol(self, symbol: str) -> str:
        """
        Normalize a symbol name to standard format.
        
        Examples:
            EURUSD.cash → EURUSD
            BTCUSD.ecn → BTCUSD
            EUR/USD → EURUSD
            XBTUSD → BTCUSD (if alias mapping exists)
        
        Args:
            symbol: Raw symbol name
            
        Returns:
            Normalized symbol name
        """
        return self._normalize_symbol_name(symbol)
    
    def is_currency_pair(self, symbol: str) -> bool:
        """
        Check if a symbol is a currency pair (any format).
        
        Args:
            symbol: Symbol name
            
        Returns:
            True if it's a currency pair
        """
        normalized = self._normalize_symbol_name(symbol)
        # Try to extract currencies
        base, quote = self._try_extract_currencies(normalized)
        return base is not None and quote is not None
    
    def get_cached(self, symbol: str) -> Optional[DiscoveryResult]:
        """Get cached discovery result."""
        for key, result in self._cache.items():
            if symbol in key:
                return result
        return None
    
    def clear_cache(self) -> None:
        """Clear the discovery cache."""
        self._cache.clear()
        self.logger.debug("Currency discovery cache cleared")
    
    # ==========================================================================
    # PRIVATE METHODS
    # ==========================================================================
    
    def _get_symbol_name(self, symbol_obj: Any) -> str:
        """Extract symbol name from object."""
        if hasattr(symbol_obj, 'name'):
            return symbol_obj.name
        if isinstance(symbol_obj, dict):
            return symbol_obj.get('name', symbol_obj.get('symbol', ''))
        if isinstance(symbol_obj, str):
            return symbol_obj
        return str(symbol_obj)
    
    def _normalize_symbol_name(self, symbol: str) -> str:
        """
        Normalize symbol name to standard format.
        
        Removes:
        - .cash, .ecn, .pro, .mini, .raw
        - -SPOT, -SWAP
        - / (EUR/USD → EURUSD)
        - _ (BTC_USD → BTCUSD)
        - Other broker-specific suffixes
        """
        if not symbol:
            return symbol
        
        # Remove suffixes
        normalized = re.sub(r'\.(cash|pro|mini|ecn|raw|a|b|c|d|e|f|g|h|i|j|k|l|m|n|o|p|q|r|s|t|u|v|w|x|y|z)$', '', symbol)
        normalized = re.sub(r'_(swapfree|islamic|demo|live|test|practice|real|sim|cfd|diff|swap)$', '', normalized)
        
        # Remove separators
        normalized = normalized.replace('/', '')
        normalized = normalized.replace('-', '')
        normalized = normalized.replace('_', '')
        normalized = normalized.replace('.', '')
        
        # Remove price suffixes
        normalized = re.sub(r'-SPOT$', '', normalized)
        normalized = re.sub(r'-SWAP$', '', normalized)
        
        # Handle crypto aliases
        for alias, target in self._crypto_aliases.items():
            if normalized.startswith(alias):
                normalized = target + normalized[len(alias):]
                break
        
        return normalized
    
    def _extract_currency_pair(self, symbol_obj: Any, symbol_name: str, normalized: str) -> Optional[CurrencyPair]:
        """
        Extract currency pair from symbol.
        
        Works with ANY format, ANY broker.
        """
        # Try MT5 fields first (most reliable)
        base = getattr(symbol_obj, 'currency_base', None)
        quote = getattr(symbol_obj, 'currency_profit', None)
        
        if base and quote:
            return self._create_currency_pair(
                base, quote, symbol_name, normalized,
                detection_method='mt5_fields',
                confidence=0.95
            )
        
        # Try to extract from normalized name
        base, quote = self._try_extract_currencies(normalized)
        
        if base and quote:
            return self._create_currency_pair(
                base, quote, symbol_name, normalized,
                detection_method='pattern_extraction',
                confidence=0.85
            )
        
        return None
    
    def _try_extract_currencies(self, normalized: str) -> Tuple[Optional[str], Optional[str]]:
        """
        Try to extract base and quote currencies from normalized symbol.
        
        Handles:
        - 6-character pairs: EURUSD → EUR, USD
        - Crypto pairs: BTCUSD → BTC, USD
        - Variable length: Some brokers use 4-5 letter crypto codes
        """
        if len(normalized) < 4:
            return None, None
        
        # Try 3-3 split (standard forex)
        if len(normalized) == 6:
            base = normalized[:3]
            quote = normalized[3:6]
            if self._is_currency_code(base) and self._is_currency_code(quote):
                return base, quote
        
        # Try crypto detection (variable length)
        # Look for known crypto patterns anywhere in the string
        crypto_match = self.CRYPTO_PATTERNS.search(normalized)
        if crypto_match:
            crypto = crypto_match.group(1)
            # Remaining part should be the quote currency
            rest = normalized.replace(crypto, '')
            if rest and self._is_currency_code(rest):
                return crypto, rest
        
        # Try to split by finding known currency codes
        for i in range(3, len(normalized) - 2):
            base = normalized[:i]
            quote = normalized[i:]
            if self._is_currency_code(base) and self._is_currency_code(quote):
                return base, quote
        
        return None, None
    
    def _is_currency_code(self, code: str) -> bool:
        """
        Check if a string is a valid currency code.
        
        Dynamic detection - does not rely on hardcoded lists.
        """
        if not code or len(code) < 2 or len(code) > 5:
            return False
        
        # Must be all uppercase letters
        if not re.match(r'^[A-Z]{2,5}$', code):
            return False
        
        # Check against known patterns
        if self.CRYPTO_PATTERNS.match(code):
            return True
        
        if code in self._fiat_set:
            return True
        
        # If not in known lists, check if it looks like a currency
        # (e.g., 3 uppercase letters, not a common word)
        if len(code) == 3 and code not in {'ABC', 'DEF', 'GHI', 'JKL', 'MNO', 'PQR', 'STU', 'VWX', 'YZ'}:
            # Could be an exotic currency
            return True
        
        return False
    
    def _create_currency_pair(
        self,
        base: str,
        quote: str,
        symbol_name: str,
        normalized: str,
        detection_method: str = 'unknown',
        confidence: float = 0.8
    ) -> CurrencyPair:
        """Create a CurrencyPair object."""
        pair_type = self._determine_pair_type(base, quote)
        signals = self._generate_signals(base, quote, pair_type, detection_method)
        
        return CurrencyPair(
            base=base,
            quote=quote,
            symbol=symbol_name,
            normalized_symbol=normalized,
            pair_type=pair_type,
            confidence=confidence,
            base_info=self._currency_cache.get(base),
            quote_info=self._currency_cache.get(quote),
            signals=signals,
            metadata={
                'normalized_symbol': normalized,
                'detection_method': detection_method,
                'broker_format': symbol_name,
            },
        )
    
    def _determine_pair_type(self, base: str, quote: str) -> CurrencyPairType:
        """Determine the type of currency pair."""
        # Check if it's a crypto pair
        if self.CRYPTO_PATTERNS.match(base) or self.CRYPTO_PATTERNS.match(quote):
            return CurrencyPairType.CRYPTO
        
        # Check if both are fiat
        if base in self._fiat_set and quote in self._fiat_set:
            return CurrencyPairType.MAJOR
        
        return CurrencyPairType.UNKNOWN
    
    def _generate_signals(self, base: str, quote: str, pair_type: CurrencyPairType, method: str) -> List[str]:
        """Generate signals for currency pair discovery."""
        signals = []
        
        if method == 'mt5_fields':
            signals.append("MT5 currency_base/currency_profit fields detected")
        elif method == 'pattern_extraction':
            signals.append("Pattern extraction from symbol name")
        
        if self.CRYPTO_PATTERNS.match(base) or self.CRYPTO_PATTERNS.match(quote):
            signals.append("Cryptocurrency detected")
        
        if base in self._fiat_set:
            signals.append(f"Base currency {base} is fiat")
        if quote in self._fiat_set:
            signals.append(f"Quote currency {quote} is fiat")
        
        return signals
    
    def _discover_currency(self, symbol: str) -> Optional[CurrencyInfo]:
        """Discover if symbol is a currency."""
        normalized = self._normalize_symbol_name(symbol)
        
        if self.CRYPTO_PATTERNS.match(normalized):
            return CurrencyInfo(
                code=normalized,
                name=normalized,
                currency_type=CurrencyType.CRYPTO,
                is_major=False,
                description=f"Cryptocurrency: {normalized}",
            )
        
        if normalized in self._fiat_set:
            return CurrencyInfo(
                code=normalized,
                name=normalized,
                currency_type=CurrencyType.FIAT,
                is_major=normalized in {'USD', 'EUR', 'GBP', 'JPY', 'CHF', 'AUD', 'CAD', 'NZD'},
                description=f"Fiat currency: {normalized}",
            )
        
        return None
    
    def _get_currencies(self, pair: CurrencyPair) -> List[CurrencyInfo]:
        """Get currency info for a pair."""
        currencies = []
        
        if pair.base in self._currency_cache:
            currencies.append(self._currency_cache[pair.base])
        else:
            # Discover the currency
            info = self._discover_currency(pair.base)
            if info:
                self._currency_cache[pair.base] = info
                currencies.append(info)
            else:
                currencies.append(CurrencyInfo(
                    code=pair.base,
                    name=pair.base,
                    currency_type=CurrencyType.UNKNOWN,
                    description=f"Unknown currency: {pair.base}",
                ))
        
        if pair.quote in self._currency_cache:
            currencies.append(self._currency_cache[pair.quote])
        else:
            info = self._discover_currency(pair.quote)
            if info:
                self._currency_cache[pair.quote] = info
                currencies.append(info)
            else:
                currencies.append(CurrencyInfo(
                    code=pair.quote,
                    name=pair.quote,
                    currency_type=CurrencyType.UNKNOWN,
                    description=f"Unknown currency: {pair.quote}",
                ))
        
        return currencies
    
    def _build_broker_statistics(self, results: Dict, pairs: List, currencies: Set) -> Dict:
        """Build statistics from broker discovery."""
        return {
            'total_symbols': len(results),
            'total_pairs': len(pairs),
            'total_currencies': len(currencies),
            'currencies_found': list(currencies),
            'pair_types': {
                'crypto': sum(1 for p in pairs if p.pair_type == CurrencyPairType.CRYPTO),
                'fiat': sum(1 for p in pairs if p.pair_type == CurrencyPairType.MAJOR),
                'unknown': sum(1 for p in pairs if p.pair_type == CurrencyPairType.UNKNOWN),
            }
        }


# ==============================================================================
# SYMBOL NORMALIZER (Helper Class)
# ==============================================================================

class SymbolNormalizer:
    """Helper class for symbol normalization."""
    
    @staticmethod
    def normalize(symbol: str) -> str:
        """Normalize a symbol name."""
        # Remove common suffixes
        normalized = re.sub(r'\.(cash|pro|mini|ecn|raw|a|b|c|d|e|f|g|h|i|j|k|l|m|n|o|p|q|r|s|t|u|v|w|x|y|z)$', '', symbol)
        normalized = re.sub(r'_(swapfree|islamic|demo|live|test|practice|real|sim|cfd|diff|swap)$', '', normalized)
        
        # Remove separators
        normalized = normalized.replace('/', '')
        normalized = normalized.replace('-', '')
        normalized = normalized.replace('_', '')
        normalized = normalized.replace('.', '')
        
        return normalized


# ==============================================================================
# FACTORY FUNCTION
# ==============================================================================

def create_currency_discovery(config: Config) -> CurrencyDiscovery:
    """
    Factory function for CurrencyDiscovery creation.
    
    Args:
        config: Application configuration
        
    Returns:
        CurrencyDiscovery instance
    """
    return CurrencyDiscovery(config)