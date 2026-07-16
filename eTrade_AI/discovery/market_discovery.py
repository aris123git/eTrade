"""
discovery/market_discovery.py - Market Discovery Engine

RESPONSIBILITY:
Discover and classify markets from MT5 data.

ARCHITECTURAL PRINCIPLES:
1. Pure discovery - No data storage, no I/O, no business logic
2. Classification based on multiple signals (path, currency, patterns)
3. Extensible detector pattern
4. Type-safe results
5. Cache for performance

WHAT IT NEVER DOES:
- ❌ Connect to MT5
- ❌ Store data
- ❌ Download data
- ❌ Make trading decisions

VERSION: 1.0.1
"""

import logging
import re
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, Set, Tuple
from enum import Enum

from core.config import Config
from core.exceptions import DiscoveryError
from core.constants import (
    FIAT_CURRENCIES,
    CRYPTO_CURRENCIES,
    COMMODITIES,
    INDICES,
    INDEX_PREFIXES,
)


# ==============================================================================
# ENUMS
# ==============================================================================

class MarketType(Enum):
    """Market type classification."""
    FOREX = "forex"
    CRYPTO = "crypto"
    INDEX = "index"
    COMMODITY = "commodity"
    STOCK = "stock"
    ETF = "etf"
    BOND = "bond"
    FUTURES = "futures"
    OPTION = "option"
    CFD = "cfd"
    UNKNOWN = "unknown"


class DiscoveryConfidence(Enum):
    """Confidence level for market discovery."""
    HIGH = 1.0
    MEDIUM = 0.7
    LOW = 0.4
    NONE = 0.0


# ==============================================================================
# DATA MODELS
# ==============================================================================

@dataclass(frozen=True)
class DiscoveredMarket:
    """
    Result of market discovery.
    
    Attributes:
        symbol: Original symbol name
        normalized_symbol: Normalized symbol name
        market_type: Detected market type
        confidence: Confidence level (0.0 - 1.0)
        base_currency: Detected base currency (if applicable)
        quote_currency: Detected quote currency (if applicable)
        description: Market description
        signals: List of signals that led to this classification
        metadata: Additional metadata
    """
    symbol: str
    normalized_symbol: str
    market_type: MarketType
    confidence: float
    base_currency: Optional[str] = None
    quote_currency: Optional[str] = None
    description: Optional[str] = None
    signals: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def is_forex(self) -> bool:
        return self.market_type == MarketType.FOREX

    def is_crypto(self) -> bool:
        return self.market_type == MarketType.CRYPTO

    def is_index(self) -> bool:
        return self.market_type == MarketType.INDEX

    def is_commodity(self) -> bool:
        return self.market_type == MarketType.COMMODITY

    def is_futures(self) -> bool:
        return self.market_type == MarketType.FUTURES

    def is_tradable(self) -> bool:
        """Check if market is tradable (has sufficient confidence)."""
        return self.confidence >= DiscoveryConfidence.LOW.value


# ==============================================================================
# DETECTOR BASE CLASS
# ==============================================================================

class BaseDetector:
    """Base class for all market detectors."""
    
    def __init__(self):
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
    
    def detect(self, symbol_obj: Any, normalized_name: str) -> Tuple[Optional[MarketType], float, List[str]]:
        """
        Detect market type from symbol.
        
        Args:
            symbol_obj: MT5 symbol object
            normalized_name: Normalized symbol name
            
        Returns:
            Tuple of (MarketType, confidence, list_of_signals)
        """
        raise NotImplementedError("Subclasses must implement detect()")
    
    def priority(self) -> int:
        """Priority of this detector (higher = checked first)."""
        return 50


# ==============================================================================
# CONCRETE DETECTORS
# ==============================================================================

class ForexDetector(BaseDetector):
    """Detects Forex pairs."""
    
    def priority(self) -> int:
        return 100
    
    def detect(self, symbol_obj: Any, normalized_name: str) -> Tuple[Optional[MarketType], float, List[str]]:
        signals = []
        confidence = 0.0
        
        # Check MT5 path
        path = getattr(symbol_obj, 'path', '')
        if path and 'forex' in path.lower():
            signals.append("MT5 path contains 'forex'")
            confidence += 0.4
        
        # Check if it's a currency pair (6 characters)
        if len(normalized_name) == 6:
            base = normalized_name[:3]
            quote = normalized_name[3:6]
            if base in FIAT_CURRENCIES and quote in FIAT_CURRENCIES:
                signals.append(f"Currency pair: {base}/{quote}")
                confidence += 0.6
        
        # Check MT5 currency fields
        base = getattr(symbol_obj, 'currency_base', '')
        quote = getattr(symbol_obj, 'currency_profit', '')
        if base and quote:
            if base in FIAT_CURRENCIES and quote in FIAT_CURRENCIES:
                signals.append(f"MT5 currency_base/profit: {base}/{quote}")
                confidence += 0.3
        
        if confidence >= 0.5:
            return MarketType.FOREX, min(confidence, 1.0), signals
        
        return None, 0.0, []


class CryptoDetector(BaseDetector):
    """Detects Cryptocurrencies."""
    
    def priority(self) -> int:
        return 90
    
    def detect(self, symbol_obj: Any, normalized_name: str) -> Tuple[Optional[MarketType], float, List[str]]:
        signals = []
        confidence = 0.0
        
        # Check MT5 path
        path = getattr(symbol_obj, 'path', '')
        if path and 'crypto' in path.lower():
            signals.append("MT5 path contains 'crypto'")
            confidence += 0.5
        
        # Check if it starts with crypto currency
        for crypto in CRYPTO_CURRENCIES:
            if normalized_name.startswith(crypto):
                quote = normalized_name[len(crypto):]
                if quote in FIAT_CURRENCIES or quote in CRYPTO_CURRENCIES:
                    signals.append(f"Crypto pair: {crypto}/{quote}")
                    confidence += 0.5
                    break
        
        if confidence >= 0.5:
            return MarketType.CRYPTO, min(confidence, 1.0), signals
        
        return None, 0.0, []


class IndexDetector(BaseDetector):
    """Detects Stock Indices."""
    
    INDEX_PATTERN = re.compile(
        r'^(US|GER|UK|FRA|JPN|AUS|CHI|HK|ITA|ESP|NETH|SWI|NZD|CAD|AUD|SGP|BRA|IND|KOR|RUS|TUR)'
        r'([0-9]{1,4}|[A-Z]{1,3}|[A-Z]{1,3}[0-9]{1,2}|[A-Z]{1,3}[0-9]{1,2}[A-Z]?)$'
    )
    
    def priority(self) -> int:
        return 80
    
    def detect(self, symbol_obj: Any, normalized_name: str) -> Tuple[Optional[MarketType], float, List[str]]:
        signals = []
        confidence = 0.0
        
        # Check MT5 path
        path = getattr(symbol_obj, 'path', '')
        if path and ('index' in path.lower() or 'indices' in path.lower()):
            signals.append("MT5 path contains 'index'")
            confidence += 0.5
        
        # Check regex pattern
        if self.INDEX_PATTERN.match(normalized_name):
            signals.append(f"Matches index pattern: {normalized_name}")
            confidence += 0.5
        
        # Check known indices
        if normalized_name in INDICES:
            signals.append(f"Known index: {INDICES[normalized_name]}")
            confidence += 0.3
        
        if confidence >= 0.5:
            return MarketType.INDEX, min(confidence, 1.0), signals
        
        return None, 0.0, []


class CommodityDetector(BaseDetector):
    """Detects Commodities."""
    
    def priority(self) -> int:
        return 75
    
    def detect(self, symbol_obj: Any, normalized_name: str) -> Tuple[Optional[MarketType], float, List[str]]:
        signals = []
        confidence = 0.0
        
        # Check MT5 path
        path = getattr(symbol_obj, 'path', '')
        if path and 'commodity' in path.lower():
            signals.append("MT5 path contains 'commodity'")
            confidence += 0.5
        
        # Check if it contains commodity name
        for comm in COMMODITIES:
            if comm in normalized_name or normalized_name.startswith(comm):
                signals.append(f"Commodity: {comm}")
                confidence += 0.5
                break
        
        if confidence >= 0.5:
            return MarketType.COMMODITY, min(confidence, 1.0), signals
        
        return None, 0.0, []


class StockDetector(BaseDetector):
    """Detects Stocks."""
    
    def priority(self) -> int:
        return 70
    
    def detect(self, symbol_obj: Any, normalized_name: str) -> Tuple[Optional[MarketType], float, List[str]]:
        signals = []
        confidence = 0.0
        
        # Check MT5 path
        path = getattr(symbol_obj, 'path', '')
        if path and 'stock' in path.lower():
            signals.append("MT5 path contains 'stock'")
            confidence += 0.5
        
        # Check if it's 4+ uppercase letters (stock symbol)
        if re.match(r'^[A-Z]{4,}$', normalized_name):
            # Exclude forex pairs
            if len(normalized_name) != 6 or not (
                normalized_name[:3] in FIAT_CURRENCIES and 
                normalized_name[3:6] in FIAT_CURRENCIES
            ):
                signals.append(f"Looks like stock symbol: {normalized_name}")
                confidence += 0.4
        
        if confidence >= 0.5:
            return MarketType.STOCK, min(confidence, 1.0), signals
        
        return None, 0.0, []


class FuturesDetector(BaseDetector):
    """Detects Futures contracts."""
    
    def priority(self) -> int:
        return 85
    
    def detect(self, symbol_obj: Any, normalized_name: str) -> Tuple[Optional[MarketType], float, List[str]]:
        signals = []
        confidence = 0.0
        
        # Futures often start with /
        if normalized_name.startswith('/'):
            signals.append("Futures prefix '/' detected")
            confidence += 0.6
        
        # Check MT5 path
        path = getattr(symbol_obj, 'path', '')
        if path and 'futures' in path.lower():
            signals.append("MT5 path contains 'futures'")
            confidence += 0.4
        
        # Check for futures-like patterns (e.g., ES, NQ, YM with month/year)
        futures_pattern = re.compile(r'^[A-Z]{1,2}[0-9]{2,4}$')
        if futures_pattern.match(normalized_name):
            signals.append(f"Matches futures pattern: {normalized_name}")
            confidence += 0.3
        
        if confidence >= 0.5:
            return MarketType.FUTURES, min(confidence, 1.0), signals
        
        return None, 0.0, []


# ==============================================================================
# DISCOVERY ENGINE
# ==============================================================================

class MarketDiscovery:
    """
    Market discovery engine.
    
    Discovers and classifies markets from MT5 data using multiple detectors.
    Features caching, filtering, and configurable thresholds.
    """
    
    def __init__(self, config: Config):
        """
        Initialize the discovery engine.
        
        Args:
            config: Application configuration
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        self._detectors: List[BaseDetector] = []
        self._cache: Dict[str, DiscoveredMarket] = {}
        
        # Configuration thresholds from config
        self.min_confidence = getattr(config, 'DISCOVERY_MIN_CONFIDENCE', 0.3)
        self.high_threshold = getattr(config, 'DISCOVERY_HIGH_THRESHOLD', 0.8)
        self.medium_threshold = getattr(config, 'DISCOVERY_MEDIUM_THRESHOLD', 0.5)
        
        self._registered = False
        
        # Register default detectors
        self._register_detectors()
    
    def _register_detectors(self):
        """Register all detectors in priority order."""
        detectors = [
            ForexDetector(),
            CryptoDetector(),
            IndexDetector(),
            FuturesDetector(),
            CommodityDetector(),
            StockDetector(),
        ]
        self._detectors = sorted(detectors, key=lambda d: d.priority(), reverse=True)
        self._registered = True
        self.logger.debug(f"Registered {len(self._detectors)} detectors")
    
    def register_detector(self, detector: BaseDetector) -> None:
        """
        Register a custom detector.
        
        Args:
            detector: Detector instance
        """
        self._detectors.append(detector)
        self._detectors = sorted(self._detectors, key=lambda d: d.priority(), reverse=True)
        self.logger.debug(f"Registered detector: {detector.__class__.__name__}")
    
    def discover(self, symbol_obj: Any) -> DiscoveredMarket:
        """
        Discover market type for a single symbol.
        
        Args:
            symbol_obj: MT5 symbol object
            
        Returns:
            DiscoveredMarket object
            
        Raises:
            DiscoveryError: If discovery fails
        """
        name = getattr(symbol_obj, 'name', '')
        
        if not name:
            raise DiscoveryError("Symbol name is empty")
        
        # Check cache
        if name in self._cache:
            self.logger.debug(f"Cache hit: {name}")
            return self._cache[name]
        
        self.logger.debug(f"Discovering market: {name}")
        
        try:
            normalized = self._normalize_symbol(name)
            
            # Run detectors
            best_type = MarketType.UNKNOWN
            best_confidence = 0.0
            all_signals = []
            metadata = {}
            
            for detector in self._detectors:
                try:
                    detected_type, confidence, signals = detector.detect(symbol_obj, normalized)
                    if detected_type and confidence > best_confidence:
                        best_type = detected_type
                        best_confidence = confidence
                        all_signals = signals
                        metadata['detector'] = detector.__class__.__name__
                except Exception as e:
                    self.logger.warning(
                        f"⚠️ Detector {detector.__class__.__name__} failed for {name}: {e}"
                    )
            
            # Extract currencies
            base, quote = self._extract_currencies(symbol_obj, normalized)
            
            # Get description
            description = getattr(symbol_obj, 'description', name)
            
            discovered = DiscoveredMarket(
                symbol=name,
                normalized_symbol=normalized,
                market_type=best_type,
                confidence=best_confidence,
                base_currency=base,
                quote_currency=quote,
                description=description,
                signals=all_signals,
                metadata=metadata,
            )
            
            # Cache result
            self._cache[name] = discovered
            
            self.logger.debug(
                f"✅ Discovered {name}: {best_type.value} (confidence={best_confidence:.2f})"
            )
            return discovered
            
        except Exception as e:
            raise DiscoveryError(f"Failed to discover market {name}: {e}")
    
    def discover_many(self, symbols: List[Any]) -> List[DiscoveredMarket]:
        """
        Discover market types for multiple symbols.
        
        Args:
            symbols: List of MT5 symbol objects
            
        Returns:
            List of DiscoveredMarket objects
        """
        if not symbols:
            return []
        
        self.logger.debug(f"Discovering {len(symbols)} markets")
        
        results = []
        errors = 0
        
        for symbol in symbols:
            try:
                results.append(self.discover(symbol))
            except Exception as e:
                errors += 1
                self.logger.warning(f"⚠️ Failed to discover symbol: {e}")
        
        if errors > 0:
            self.logger.warning(f"⚠️ {errors} symbols failed discovery out of {len(symbols)}")
        
        return results
    
    def filter_tradable(self, markets: List[DiscoveredMarket]) -> List[DiscoveredMarket]:
        """
        Filter only tradable markets.
        
        Args:
            markets: List of DiscoveredMarket objects
            
        Returns:
            Filtered list of tradable markets
        """
        return [m for m in markets if m.is_tradable()]
    
    def get_by_type(self, markets: List[DiscoveredMarket], market_type: MarketType) -> List[DiscoveredMarket]:
        """
        Filter markets by type.
        
        Args:
            markets: List of DiscoveredMarket objects
            market_type: Market type to filter by
            
        Returns:
            Filtered list of markets of the specified type
        """
        return [m for m in markets if m.market_type == market_type]
    
    def get_by_confidence(self, markets: List[DiscoveredMarket], min_confidence: float = None) -> List[DiscoveredMarket]:
        """
        Filter markets by minimum confidence.
        
        Args:
            markets: List of DiscoveredMarket objects
            min_confidence: Minimum confidence threshold (default: from config)
            
        Returns:
            Filtered list of markets meeting confidence threshold
        """
        threshold = min_confidence if min_confidence is not None else self.min_confidence
        return [m for m in markets if m.confidence >= threshold]
    
    def get_statistics(self, discovered: List[DiscoveredMarket]) -> Dict[str, Any]:
        """
        Get statistics from discovery results.
        
        Args:
            discovered: List of DiscoveredMarket objects
            
        Returns:
            Dictionary with statistics
        """
        stats = {
            'total': len(discovered),
            'by_type': {},
            'by_confidence': {
                'high': 0,    # >= 0.8
                'medium': 0,  # >= 0.5
                'low': 0,     # >= 0.3
                'none': 0,    # < 0.3
            },
            'with_currency': 0,
            'tradable': 0,
            'unknown': 0,
        }
        
        for market in discovered:
            # By type
            type_name = market.market_type.value
            stats['by_type'][type_name] = stats['by_type'].get(type_name, 0) + 1
            
            # By confidence
            if market.confidence >= self.high_threshold:
                stats['by_confidence']['high'] += 1
            elif market.confidence >= self.medium_threshold:
                stats['by_confidence']['medium'] += 1
            elif market.confidence >= self.min_confidence:
                stats['by_confidence']['low'] += 1
            else:
                stats['by_confidence']['none'] += 1
            
            # With currency
            if market.base_currency and market.quote_currency:
                stats['with_currency'] += 1
            
            # Tradable
            if market.is_tradable():
                stats['tradable'] += 1
            
            # Unknown
            if market.market_type == MarketType.UNKNOWN:
                stats['unknown'] += 1
        
        return stats
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics.
        
        Returns:
            Dictionary with cache statistics
        """
        return {
            'cache_size': len(self._cache),
            'cached_symbols': list(self._cache.keys()),
        }
    
    def clear_cache(self) -> None:
        """Clear the discovery cache."""
        self._cache.clear()
        self.logger.debug("Discovery cache cleared")
    
    def _normalize_symbol(self, symbol: str) -> str:
        """
        Normalize symbol name.
        
        Args:
            symbol: Original symbol name
            
        Returns:
            Normalized symbol name
        """
        # Remove common suffixes
        normalized = re.sub(r'\.(cash|pro|mini|ecn|raw|a|b|c|d|e|f|g|h|i|j|k|l|m|n|o|p|q|r|s|t|u|v|w|x|y|z)$', '', symbol)
        normalized = re.sub(r'_(swapfree|islamic|demo|live|test|practice|real|sim)$', '', normalized)
        return normalized
    
    def _extract_currencies(self, symbol_obj: Any, normalized: str) -> Tuple[Optional[str], Optional[str]]:
        """
        Extract base and quote currencies.
        
        Args:
            symbol_obj: MT5 symbol object
            normalized: Normalized symbol name
            
        Returns:
            Tuple of (base_currency, quote_currency)
        """
        # Try MT5 fields first
        base = getattr(symbol_obj, 'currency_base', None)
        quote = getattr(symbol_obj, 'currency_profit', None)
        
        if base and quote:
            return base, quote
        
        # Parse from name
        if len(normalized) == 6:
            base = normalized[:3]
            quote = normalized[3:6]
            if base in FIAT_CURRENCIES and quote in FIAT_CURRENCIES:
                return base, quote
        
        # Check crypto
        for crypto in CRYPTO_CURRENCIES:
            if normalized.startswith(crypto):
                quote_part = normalized[len(crypto):]
                if quote_part in FIAT_CURRENCIES or quote_part in CRYPTO_CURRENCIES:
                    return crypto, quote_part
        
        return None, None


# ==============================================================================
# FACTORY FUNCTION
# ==============================================================================

def create_market_discovery(config: Config) -> MarketDiscovery:
    """
    Factory function for MarketDiscovery creation.
    
    Args:
        config: Application configuration
        
    Returns:
        MarketDiscovery instance
    """
    return MarketDiscovery(config)