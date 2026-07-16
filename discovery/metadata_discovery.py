"""
discovery/metadata_discovery.py - Metadata Discovery Engine

RESPONSIBILITY:
Discover and analyze market metadata from various sources.

ARCHITECTURAL PRINCIPLES:
1. Pure discovery - No data storage, no I/O, no business logic
2. Metadata extraction from symbol information
3. Cross-reference from multiple sources
4. Type-safe results with statistical analysis

WHAT IT NEVER DOES:
- ❌ Connect to MT5
- ❌ Store data
- ❌ Download data
- ❌ Make trading decisions

VERSION: 1.0.0
"""

import logging
import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional, Dict, Any, Tuple, Set, Union
from enum import Enum

from core.config import Config
from core.exceptions import DiscoveryError, DataValidationError
from core.constants import (
    FIAT_CURRENCIES,
    CRYPTO_CURRENCIES,
    COMMODITIES,
    INDICES,
)


# ==============================================================================
# EXPORTS
# ==============================================================================

__all__ = [
    'MetadataSource',
    'MetadataField',
    'SymbolMetadata',
    'MarketMetadata',
    'MetadataDiscovery',
    'create_metadata_discovery',
]


# ==============================================================================
# ENUMS
# ==============================================================================

class MetadataSource(Enum):
    """Source of metadata information."""
    MT5_SYMBOL = "mt5_symbol"
    MT5_CURRENCY = "mt5_currency"
    MT5_ACCOUNT = "mt5_account"
    INTERNAL = "internal"
    USER = "user"
    DISCOVERED = "discovered"


class MetadataField(Enum):
    """Metadata field types."""
    DESCRIPTION = "description"
    CATEGORY = "category"
    SECTOR = "sector"
    INDUSTRY = "industry"
    COUNTRY = "country"
    REGION = "region"
    CURRENCY = "currency"
    EXCHANGE = "exchange"
    LISTING_DATE = "listing_date"
    ISSUER = "issuer"
    TICKER = "ticker"
    ISIN = "isin"
    CUSIP = "cusip"
    SEDOL = "sedol"
    EXCHANGE_RATE = "exchange_rate"
    LAST_UPDATED = "last_updated"
    CUSTOM = "custom"


# ==============================================================================
# DATA MODELS
# ==============================================================================

@dataclass(frozen=True)
class SymbolMetadata:
    """Metadata for a symbol."""
    symbol: str
    source: MetadataSource
    fields: Dict[MetadataField, Any]
    confidence: float
    timestamp: datetime
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class MarketMetadata:
    """Complete metadata discovery result."""
    symbol: str
    timestamp: datetime
    metadata_list: List[SymbolMetadata]
    aggregated: Dict[str, Any]
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def get_field(self, field: MetadataField) -> Optional[Any]:
        """Get a specific metadata field from aggregated data."""
        return self.aggregated.get(field.value)
    
    def get_by_source(self, source: MetadataSource) -> List[SymbolMetadata]:
        """Get metadata from a specific source."""
        return [m for m in self.metadata_list if m.source == source]
    
    def get_summary(self) -> Dict[str, Any]:
        """Get summary of discovery results."""
        return {
            'symbol': self.symbol,
            'total_sources': len(set(m.source for m in self.metadata_list)),
            'total_fields': len(self.aggregated),
            'fields': list(self.aggregated.keys()),
            'sources': [s.value for s in set(m.source for m in self.metadata_list)],
        }


# ==============================================================================
# METADATA DISCOVERY
# ==============================================================================

class MetadataDiscovery:
    """
    Metadata discovery engine.
    
    Discovers and analyzes market metadata from various sources.
    """
    
    def __init__(self, config: Config):
        """
        Initialize the metadata discovery engine.
        
        Args:
            config: Application configuration
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        self._cache: Dict[str, MarketMetadata] = {}
        
        # Metadata patterns
        self._patterns = self._build_patterns()
        
        # Known market categories
        self._categories = self._build_categories()
    
    def _build_patterns(self) -> Dict[MetadataField, re.Pattern]:
        """Build regex patterns for metadata extraction."""
        return {
            MetadataField.ISIN: re.compile(r'^[A-Z]{2}[A-Z0-9]{9}[0-9]$'),
            MetadataField.CUSIP: re.compile(r'^[A-Z0-9]{9}$'),
            MetadataField.SEDOL: re.compile(r'^[A-Z0-9]{7}$'),
            MetadataField.TICKER: re.compile(r'^[A-Z]{1,5}$'),
        }
    
    def _build_categories(self) -> Dict[str, str]:
        """Build market categories."""
        return {
            'forex': 'Currency',
            'index': 'Index',
            'commodity': 'Commodity',
            'crypto': 'Cryptocurrency',
            'stock': 'Equity',
            'etf': 'ETF',
            'bond': 'Fixed Income',
            'futures': 'Derivative',
            'option': 'Derivative',
            'cfd': 'Derivative',
        }
    
    # ==========================================================================
    # PUBLIC METHODS
    # ==========================================================================
    
    def discover_from_symbol(self, symbol_obj: Any) -> MarketMetadata:
        """
        Discover metadata from a symbol object.
        
        Args:
            symbol_obj: MT5 symbol object, dict, or string
            
        Returns:
            MarketMetadata object
        """
        symbol_name = self._get_symbol_name(symbol_obj)
        
        # Check cache
        if symbol_name in self._cache:
            self.logger.debug(f"Cache hit: {symbol_name}")
            return self._cache[symbol_name]
        
        self.logger.debug(f"Discovering metadata for: {symbol_name}")
        
        try:
            metadata_list = []
            aggregated = {}
            
            # 1. Extract from MT5 symbol info
            if self._is_mt5_symbol(symbol_obj):
                mt5_metadata = self._extract_mt5_metadata(symbol_obj)
                metadata_list.append(mt5_metadata)
                self._merge_aggregated(aggregated, mt5_metadata)
            
            # 2. Extract from symbol name
            name_metadata = self._extract_from_name(symbol_name)
            metadata_list.append(name_metadata)
            self._merge_aggregated(aggregated, name_metadata)
            
            # 3. Extract from description
            desc = getattr(symbol_obj, 'description', '')
            if desc:
                desc_metadata = self._extract_from_description(desc)
                metadata_list.append(desc_metadata)
                self._merge_aggregated(aggregated, desc_metadata)
            
            # 4. Extract from path
            path = getattr(symbol_obj, 'path', '')
            if path:
                path_metadata = self._extract_from_path(path)
                metadata_list.append(path_metadata)
                self._merge_aggregated(aggregated, path_metadata)
            
            # 5. Cross-reference with known categories
            cross_ref = self._cross_reference(symbol_name)
            metadata_list.append(cross_ref)
            self._merge_aggregated(aggregated, cross_ref)
            
            result = MarketMetadata(
                symbol=symbol_name,
                timestamp=datetime.now(),
                metadata_list=metadata_list,
                aggregated=aggregated,
                metadata={
                    'sources_count': len(metadata_list),
                    'fields_count': len(aggregated),
                },
            )
            
            self._cache[symbol_name] = result
            return result
            
        except Exception as e:
            raise DiscoveryError(f"Failed to discover metadata for {symbol_name}: {e}")
    
    def discover_many(self, symbols: List[Any]) -> List[MarketMetadata]:
        """
        Discover metadata for multiple symbols.
        
        Args:
            symbols: List of MT5 symbol objects
            
        Returns:
            List of MarketMetadata objects
        """
        if not symbols:
            return []
        
        self.logger.debug(f"Discovering metadata for {len(symbols)} symbols")
        
        results = []
        errors = 0
        
        for symbol in symbols:
            try:
                results.append(self.discover_from_symbol(symbol))
            except Exception as e:
                errors += 1
                self.logger.warning(f"⚠️ Failed to discover metadata: {e}")
        
        if errors > 0:
            self.logger.warning(f"⚠️ {errors} symbols failed discovery out of {len(symbols)}")
        
        return results
    
    def get_cached(self, symbol: str) -> Optional[MarketMetadata]:
        """Get cached discovery result."""
        return self._cache.get(symbol)
    
    def clear_cache(self) -> None:
        """Clear the discovery cache."""
        self._cache.clear()
        self.logger.debug("Metadata discovery cache cleared")
    
    def get_statistics(self, results: List[MarketMetadata]) -> Dict[str, Any]:
        """
        Get statistics from discovery results.
        
        Args:
            results: List of MarketMetadata objects
            
        Returns:
            Dictionary with statistics
        """
        stats = {
            'total_symbols': len(results),
            'total_fields': 0,
            'field_counts': {},
            'source_counts': {},
            'unique_categories': set(),
        }
        
        for result in results:
            stats['total_fields'] += len(result.aggregated)
            
            for field in result.aggregated.keys():
                stats['field_counts'][field] = stats['field_counts'].get(field, 0) + 1
            
            for metadata in result.metadata_list:
                source = metadata.source.value
                stats['source_counts'][source] = stats['source_counts'].get(source, 0) + 1
            
            if 'category' in result.aggregated:
                stats['unique_categories'].add(result.aggregated['category'])
        
        stats['unique_categories'] = list(stats['unique_categories'])
        
        return stats
    
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
    
    def _is_mt5_symbol(self, obj: Any) -> bool:
        """Check if object is an MT5 symbol."""
        return hasattr(obj, 'name') and hasattr(obj, 'digits')
    
    def _extract_mt5_metadata(self, symbol_obj: Any) -> SymbolMetadata:
        """Extract metadata from MT5 symbol info."""
        fields = {}
        metadata = {}
        
        # Basic fields
        fields[MetadataField.DESCRIPTION] = getattr(symbol_obj, 'description', '')
        fields[MetadataField.CURRENCY] = getattr(symbol_obj, 'currency_base', '')
        
        # Additional fields
        if hasattr(symbol_obj, 'trade_mode'):
            fields[MetadataField.CATEGORY] = self._get_category_from_trade_mode(
                symbol_obj.trade_mode
            )
        
        if hasattr(symbol_obj, 'path'):
            metadata['path'] = symbol_obj.path
        
        if hasattr(symbol_obj, 'digits'):
            metadata['digits'] = symbol_obj.digits
        
        if hasattr(symbol_obj, 'point'):
            metadata['point'] = symbol_obj.point
        
        return SymbolMetadata(
            symbol=getattr(symbol_obj, 'name', ''),
            source=MetadataSource.MT5_SYMBOL,
            fields=fields,
            confidence=0.9,
            timestamp=datetime.now(),
            metadata=metadata,
        )
    
    def _extract_from_name(self, symbol: str) -> SymbolMetadata:
        """Extract metadata from symbol name."""
        fields = {}
        metadata = {}
        
        # Normalize symbol
        normalized = self._normalize_symbol(symbol)
        metadata['normalized'] = normalized
        
        # Try to detect category from name
        category = self._detect_category(normalized)
        if category:
            fields[MetadataField.CATEGORY] = category
            metadata['detected_by'] = 'name_pattern'
        
        # Try to detect currency pair
        base, quote = self._extract_currencies(normalized)
        if base and quote:
            metadata['base_currency'] = base
            metadata['quote_currency'] = quote
            fields[MetadataField.CURRENCY] = f"{base}/{quote}"
        
        # Try to detect index
        if normalized in INDICES:
            fields[MetadataField.DESCRIPTION] = INDICES[normalized]
            fields[MetadataField.CATEGORY] = 'index'
            metadata['detected_by'] = 'index_list'
        
        # Try to detect commodity
        for commodity in COMMODITIES:
            if commodity in normalized:
                fields[MetadataField.DESCRIPTION] = commodity
                fields[MetadataField.CATEGORY] = 'commodity'
                metadata['detected_by'] = 'commodity_list'
                break
        
        # Try to detect crypto
        for crypto in CRYPTO_CURRENCIES:
            if normalized.startswith(crypto):
                fields[MetadataField.CATEGORY] = 'crypto'
                metadata['detected_by'] = 'crypto_list'
                metadata['crypto_code'] = crypto
                break
        
        return SymbolMetadata(
            symbol=symbol,
            source=MetadataSource.INTERNAL,
            fields=fields,
            confidence=0.7,
            timestamp=datetime.now(),
            metadata=metadata,
        )
    
    def _extract_from_description(self, description: str) -> SymbolMetadata:
        """Extract metadata from description."""
        fields = {}
        metadata = {}
        
        if not description:
            return SymbolMetadata(
                symbol='',
                source=MetadataSource.INTERNAL,
                fields={},
                confidence=0.0,
                timestamp=datetime.now(),
            )
        
        # Try to detect exchange
        exchanges = ['NYSE', 'NASDAQ', 'LSE', 'TSE', 'HKEX', 'SGX']
        for exchange in exchanges:
            if exchange in description:
                fields[MetadataField.EXCHANGE] = exchange
                metadata['detected_by'] = 'exchange_list'
                break
        
        # Try to detect sector
        sectors = ['Technology', 'Finance', 'Healthcare', 'Energy', 'Consumer', 'Industrial']
        for sector in sectors:
            if sector in description:
                fields[MetadataField.SECTOR] = sector
                metadata['detected_by'] = 'sector_list'
                break
        
        # Try to detect country
        countries = ['United States', 'UK', 'Japan', 'China', 'Germany', 'France']
        for country in countries:
            if country in description:
                fields[MetadataField.COUNTRY] = country
                metadata['detected_by'] = 'country_list'
                break
        
        # Try to detect ISIN
        isin_match = self._patterns.get(MetadataField.ISIN)
        if isin_match:
            isin_found = isin_match.search(description)
            if isin_found:
                fields[MetadataField.ISIN] = isin_found.group()
                metadata['detected_by'] = 'isin_pattern'
        
        return SymbolMetadata(
            symbol='',
            source=MetadataSource.INTERNAL,
            fields=fields,
            confidence=0.5,
            timestamp=datetime.now(),
            metadata=metadata,
        )
    
    def _extract_from_path(self, path: str) -> SymbolMetadata:
        """Extract metadata from path."""
        fields = {}
        metadata = {}
        
        if not path:
            return SymbolMetadata(
                symbol='',
                source=MetadataSource.INTERNAL,
                fields={},
                confidence=0.0,
                timestamp=datetime.now(),
            )
        
        # Category from path
        path_lower = path.lower()
        for cat, name in self._categories.items():
            if cat in path_lower:
                fields[MetadataField.CATEGORY] = name
                metadata['detected_by'] = 'path'
                metadata['path_category'] = cat
                break
        
        # Region from path
        regions = ['asia', 'europe', 'americas']
        for region in regions:
            if region in path_lower:
                fields[MetadataField.REGION] = region.capitalize()
                metadata['detected_by'] = 'path_region'
                break
        
        return SymbolMetadata(
            symbol='',
            source=MetadataSource.INTERNAL,
            fields=fields,
            confidence=0.6,
            timestamp=datetime.now(),
            metadata=metadata,
        )
    
    def _cross_reference(self, symbol: str) -> SymbolMetadata:
        """Cross-reference symbol with known data."""
        fields = {}
        metadata = {}
        
        normalized = self._normalize_symbol(symbol)
        
        # Check if it's a known forex pair
        if len(normalized) == 6:
            base = normalized[:3]
            quote = normalized[3:6]
            if base in FIAT_CURRENCIES and quote in FIAT_CURRENCIES:
                fields[MetadataField.CATEGORY] = 'Currency'
                fields[MetadataField.CURRENCY] = f"{base}/{quote}"
                metadata['cross_reference'] = 'forex_pair'
        
        # Check if it's a known crypto
        for crypto in CRYPTO_CURRENCIES:
            if normalized.startswith(crypto):
                fields[MetadataField.CATEGORY] = 'Cryptocurrency'
                metadata['cross_reference'] = 'crypto'
                break
        
        # Check if it's a known index
        if normalized in INDICES:
            fields[MetadataField.CATEGORY] = 'Index'
            fields[MetadataField.DESCRIPTION] = INDICES[normalized]
            metadata['cross_reference'] = 'index'
        
        return SymbolMetadata(
            symbol=symbol,
            source=MetadataSource.DISCOVERED,
            fields=fields,
            confidence=0.8,
            timestamp=datetime.now(),
            metadata=metadata,
        )
    
    def _normalize_symbol(self, symbol: str) -> str:
        """Normalize symbol name."""
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
        
        return normalized
    
    def _extract_currencies(self, normalized: str) -> Tuple[Optional[str], Optional[str]]:
        """Extract base and quote currencies from normalized symbol."""
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
    
    def _detect_category(self, normalized: str) -> Optional[str]:
        """Detect category from normalized symbol."""
        # Check forex
        if len(normalized) == 6:
            base = normalized[:3]
            quote = normalized[3:6]
            if base in FIAT_CURRENCIES and quote in FIAT_CURRENCIES:
                return 'Forex'
        
        # Check crypto
        for crypto in CRYPTO_CURRENCIES:
            if normalized.startswith(crypto):
                return 'Cryptocurrency'
        
        # Check commodity
        for commodity in COMMODITIES:
            if commodity in normalized:
                return 'Commodity'
        
        # Check index
        if normalized in INDICES:
            return 'Index'
        
        return None
    
    def _get_category_from_trade_mode(self, trade_mode: int) -> str:
        """Get category from MT5 trade_mode."""
        categories = {
            0: 'Standard',
            1: 'Margin',
            2: 'Exchange',
            3: 'OTC',
            4: 'Futures',
            5: 'Options',
            6: 'Spread',
        }
        return categories.get(trade_mode, 'Unknown')
    
    def _merge_aggregated(
        self,
        aggregated: Dict[str, Any],
        metadata: SymbolMetadata
    ) -> None:
        """Merge metadata into aggregated dict."""
        for field, value in metadata.fields.items():
            if value and field.value not in aggregated:
                aggregated[field.value] = value
            elif value and aggregated.get(field.value) is None:
                aggregated[field.value] = value


# ==============================================================================
# FACTORY FUNCTION
# ==============================================================================

def create_metadata_discovery(config: Config) -> MetadataDiscovery:
    """
    Factory function for MetadataDiscovery creation.
    
    Args:
        config: Application configuration
        
    Returns:
        MetadataDiscovery instance
    """
    return MetadataDiscovery(config)