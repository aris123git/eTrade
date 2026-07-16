"""
discovery/dependency_discovery.py - Dependency Discovery Engine

RESPONSIBILITY:
Discover and analyze dependencies between markets.

ARCHITECTURAL PRINCIPLES:
1. Pure discovery - No data storage, no I/O, no business logic
2. Dependency detection from market relationships
3. Correlation-based dependency analysis with statistical significance
4. Type-safe results with proper validation
5. Pure Python implementation - no numpy/scipy required

WHAT IT NEVER DOES:
- ❌ Connect to MT5
- ❌ Store data
- ❌ Download data
- ❌ Make trading decisions

VERSION: 2.2.0
"""

import logging
import math
import hashlib
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional, Dict, Any, Tuple, Set, Union
from enum import Enum
from collections import defaultdict

from core.config import Config
from core.exceptions import DiscoveryError, DataValidationError
from core.utils import calculate_percentage


# ==============================================================================
# EXPORTS
# ==============================================================================

__all__ = [
    'DependencyType',
    'DependencyStrength',
    'DependencyDirection',
    'MarketDependency',
    'DependencyResult',
    'DependencyDiscovery',
    'create_dependency_discovery',
]


# ==============================================================================
# ENUMS
# ==============================================================================

class DependencyType(Enum):
    """Type of dependency between markets."""
    CORRELATION = "correlation"
    CAUSALITY = "causality"
    LEADING = "leading"
    LAGGING = "lagging"
    SYNCHRONOUS = "synchronous"
    INVERSE = "inverse"
    SPILLOVER = "spillover"
    CONTAGION = "contagion"
    STRUCTURAL = "structural"
    UNKNOWN = "unknown"


class DependencyStrength(Enum):
    """Strength of a dependency."""
    VERY_STRONG = 1.0
    STRONG = 0.8
    MODERATE = 0.6
    WEAK = 0.4
    VERY_WEAK = 0.2
    NONE = 0.0


class DependencyDirection(Enum):
    """Direction of a dependency."""
    POSITIVE = "positive"
    NEGATIVE = "negative"
    BIDIRECTIONAL = "bidirectional"
    UNIDIRECTIONAL = "unidirectional"
    UNKNOWN = "unknown"


# ==============================================================================
# DATA MODELS
# ==============================================================================

@dataclass(frozen=True)
class MarketDependency:
    """A dependency between two markets."""
    source_symbol: str
    target_symbol: str
    dependency_type: DependencyType
    direction: DependencyDirection
    strength: DependencyStrength
    correlation: float
    confidence: float
    lag: Optional[int] = None
    p_value: Optional[float] = None
    sample_size: int = 0
    signals: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def is_positive(self) -> bool:
        return self.direction == DependencyDirection.POSITIVE
    
    def is_negative(self) -> bool:
        return self.direction == DependencyDirection.NEGATIVE
    
    def is_strong(self) -> bool:
        return self.strength in (DependencyStrength.STRONG, DependencyStrength.VERY_STRONG)
    
    def is_very_strong(self) -> bool:
        return self.strength == DependencyStrength.VERY_STRONG
    
    def is_significant(self, alpha: float = 0.05) -> bool:
        """Check if dependency is statistically significant."""
        if self.p_value is None:
            return self.confidence >= 0.8
        return self.p_value < alpha
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'source_symbol': self.source_symbol,
            'target_symbol': self.target_symbol,
            'dependency_type': self.dependency_type.value,
            'direction': self.direction.value,
            'strength': self.strength.value,
            'correlation': self.correlation,
            'confidence': self.confidence,
            'lag': self.lag,
            'p_value': self.p_value,
            'sample_size': self.sample_size,
            'signals': self.signals,
            'metadata': self.metadata,
        }


@dataclass
class DependencyResult:
    """Complete dependency discovery result."""
    symbol: str
    timestamp: datetime
    dependencies: List[MarketDependency]
    market_count: int
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def get_dependencies_for(self, symbol: str) -> List[MarketDependency]:
        """Get all dependencies for a specific symbol."""
        return [
            d for d in self.dependencies
            if d.source_symbol == symbol or d.target_symbol == symbol
        ]
    
    def get_by_type(self, dep_type: DependencyType) -> List[MarketDependency]:
        """Get dependencies by type."""
        return [d for d in self.dependencies if d.dependency_type == dep_type]
    
    def get_by_strength(self, min_strength: DependencyStrength) -> List[MarketDependency]:
        """Get dependencies with at least the given strength."""
        return [d for d in self.dependencies if d.strength.value >= min_strength.value]
    
    def get_strongest(self, limit: int = 10) -> List[MarketDependency]:
        """Get strongest dependencies."""
        sorted_deps = sorted(
            self.dependencies,
            key=lambda d: d.strength.value,
            reverse=True
        )
        return sorted_deps[:limit]
    
    def get_positive(self) -> List[MarketDependency]:
        """Get positive dependencies."""
        return [d for d in self.dependencies if d.is_positive()]
    
    def get_negative(self) -> List[MarketDependency]:
        """Get negative dependencies."""
        return [d for d in self.dependencies if d.is_negative()]
    
    def get_significant(self, alpha: float = 0.05) -> List[MarketDependency]:
        """Get statistically significant dependencies."""
        return [d for d in self.dependencies if d.is_significant(alpha)]
    
    def to_dataframe(self) -> Optional[Any]:
        """
        Convert to pandas DataFrame for analysis.
        
        Returns:
            pandas DataFrame if pandas is available, None otherwise
        """
        try:
            import pandas as pd
            return pd.DataFrame([d.to_dict() for d in self.dependencies])
        except ImportError:
            return None
    
    def get_summary(self) -> Dict[str, Any]:
        """Get summary of discovery results."""
        return {
            'symbol': self.symbol,
            'total_dependencies': len(self.dependencies),
            'market_count': self.market_count,
            'by_type': {
                t.value: sum(1 for d in self.dependencies if d.dependency_type == t)
                for t in DependencyType
            },
            'by_strength': {
                s.value: sum(1 for d in self.dependencies if d.strength == s)
                for s in DependencyStrength
            },
            'positive_count': len(self.get_positive()),
            'negative_count': len(self.get_negative()),
            'significant_count': len(self.get_significant()),
            'avg_correlation': sum(d.correlation for d in self.dependencies) / len(self.dependencies) if self.dependencies else 0.0,
        }


# ==============================================================================
# DEPENDENCY DISCOVERY
# ==============================================================================

class DependencyDiscovery:
    """
    Dependency discovery engine.
    
    Discovers and analyzes dependencies between markets with statistical significance.
    Pure Python implementation - no numpy/scipy required.
    """
    
    # Correlation thresholds
    DEFAULT_VERY_STRONG_THRESHOLD = 0.9
    DEFAULT_STRONG_THRESHOLD = 0.7
    DEFAULT_MODERATE_THRESHOLD = 0.4
    DEFAULT_WEAK_THRESHOLD = 0.2
    DEFAULT_SIGNIFICANCE_LEVEL = 0.05
    MIN_SAMPLE_SIZE = 30
    MAX_LAG = 10
    
    def __init__(self, config: Config):
        """
        Initialize the dependency discovery engine.
        
        Args:
            config: Application configuration
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        self._cache: Dict[str, DependencyResult] = {}
        
        # Correlation thresholds
        self._very_strong_threshold = getattr(config, 'DEPENDENCY_VERY_STRONG_THRESHOLD', self.DEFAULT_VERY_STRONG_THRESHOLD)
        self._strong_threshold = getattr(config, 'DEPENDENCY_STRONG_THRESHOLD', self.DEFAULT_STRONG_THRESHOLD)
        self._moderate_threshold = getattr(config, 'DEPENDENCY_MODERATE_THRESHOLD', self.DEFAULT_MODERATE_THRESHOLD)
        self._weak_threshold = getattr(config, 'DEPENDENCY_WEAK_THRESHOLD', self.DEFAULT_WEAK_THRESHOLD)
        self._significance_level = getattr(config, 'DEPENDENCY_SIGNIFICANCE_LEVEL', self.DEFAULT_SIGNIFICANCE_LEVEL)
        self._min_sample_size = getattr(config, 'DEPENDENCY_MIN_SAMPLE_SIZE', self.MIN_SAMPLE_SIZE)
        
        self.logger.info(
            f"✅ DependencyDiscovery initialized: "
            f"very_strong={self._very_strong_threshold}, "
            f"strong={self._strong_threshold}, "
            f"moderate={self._moderate_threshold}, "
            f"weak={self._weak_threshold}"
        )
    
    # ==========================================================================
    # PUBLIC METHODS
    # ==========================================================================
    
    def discover(
        self,
        market_data: Dict[str, List[float]],
        symbol: str,
        correlation_type: str = 'returns',
    ) -> DependencyResult:
        """
        Discover dependencies between markets.
        
        Args:
            market_data: Dictionary mapping symbol to price data list
            symbol: Primary symbol to analyze
            correlation_type: 'returns' or 'prices'
            
        Returns:
            DependencyResult object
            
        Raises:
            DataValidationError: If data is invalid
            DiscoveryError: If discovery fails
        """
        # Validate inputs
        if not market_data:
            raise DataValidationError("Market data is empty")
        
        if symbol not in market_data:
            raise DataValidationError(f"Symbol {symbol} not found in market data")
        
        # Validate correlation_type
        if correlation_type not in ('returns', 'prices'):
            self.logger.warning(
                f"Invalid correlation_type '{correlation_type}', using 'returns'"
            )
            correlation_type = 'returns'
        
        # Check cache - use stable key with hashlib
        cache_key = self._build_cache_key(symbol, market_data, correlation_type)
        if cache_key in self._cache:
            self.logger.debug(f"Cache hit: {symbol}")
            return self._cache[cache_key]
        
        self.logger.debug(f"Discovering dependencies for: {symbol}")
        
        try:
            dependencies = []
            
            # Get data for the primary symbol
            primary_data = market_data[symbol]
            
            # Validate primary data
            if not self._validate_data(primary_data):
                raise DataValidationError(f"Invalid primary data for {symbol}: insufficient or invalid values")
            
            # Compare with all other symbols
            for other_symbol, other_data in market_data.items():
                if other_symbol == symbol:
                    continue
                
                # Validate other data
                if not self._validate_data(other_data):
                    self.logger.debug(f"Skipping {other_symbol}: invalid data")
                    continue
                
                # Calculate dependency
                dependency = self._calculate_dependency(
                    primary_data=primary_data,
                    other_data=other_data,
                    source_symbol=other_symbol,
                    target_symbol=symbol,
                    correlation_type=correlation_type,
                )
                
                if dependency:
                    dependencies.append(dependency)
            
            # Sort by strength
            dependencies.sort(key=lambda d: d.strength.value, reverse=True)
            
            result = DependencyResult(
                symbol=symbol,
                timestamp=datetime.now(),
                dependencies=dependencies,
                market_count=len(market_data),
                metadata={
                    'correlation_type': correlation_type,
                    'data_points': len(primary_data),
                    'symbols_analyzed': len(market_data),
                    'min_sample_size': self._min_sample_size,
                    'significance_level': self._significance_level,
                },
            )
            
            self._cache[cache_key] = result
            return result
            
        except DataValidationError:
            raise
        except Exception as e:
            raise DiscoveryError(f"Failed to discover dependencies for {symbol}: {e}")
    
    def discover_from_candles(
        self,
        candles_data: Dict[str, List[Dict[str, Any]]],
        symbol: str,
        price_field: str = 'close',
        correlation_type: str = 'returns',
    ) -> DependencyResult:
        """
        Discover dependencies from candle data.
        
        Args:
            candles_data: Dictionary mapping symbol to candle list
            symbol: Primary symbol to analyze
            price_field: Field to use for price data
            correlation_type: 'returns' or 'prices'
            
        Returns:
            DependencyResult object
        """
        # Validate candles
        if not candles_data:
            raise DataValidationError("Candles data is empty")
        
        if symbol not in candles_data:
            raise DataValidationError(f"Symbol {symbol} not found in candles data")
        
        # Extract and validate price data
        market_data = {}
        for sym, candles in candles_data.items():
            if not candles:
                continue
            
            prices = []
            for c in candles:
                if price_field in c and c[price_field] is not None:
                    try:
                        prices.append(float(c[price_field]))
                    except (ValueError, TypeError):
                        continue
            
            if len(prices) >= self._min_sample_size:
                market_data[sym] = prices
            else:
                self.logger.debug(f"Skipping {sym}: insufficient valid prices ({len(prices)})")
        
        if not market_data:
            raise DataValidationError("No valid price data extracted from candles")
        
        return self.discover(market_data, symbol, correlation_type)
    
    def get_dependencies_for_symbol(
        self,
        symbol: str,
        dependencies: List[MarketDependency],
    ) -> List[MarketDependency]:
        """Get dependencies for a specific symbol."""
        return [
            d for d in dependencies
            if d.source_symbol == symbol or d.target_symbol == symbol
        ]
    
    def get_positive(self, result: DependencyResult) -> List[MarketDependency]:
        """Get positive dependencies from a result."""
        return result.get_positive()
    
    def get_negative(self, result: DependencyResult) -> List[MarketDependency]:
        """Get negative dependencies from a result."""
        return result.get_negative()
    
    def get_cached(self, symbol: str) -> Optional[DependencyResult]:
        """Get cached discovery result by exact symbol match."""
        for cache_key, result in self._cache.items():
            # Extract symbol from cache key (format: symbol_keyshash_type)
            parts = cache_key.split('_')
            if parts and parts[0] == symbol:
                return result
        return None
    
    def clear_cache(self) -> None:
        """Clear the discovery cache."""
        self._cache.clear()
        self.logger.debug("Dependency discovery cache cleared")
    
    def get_statistics(self, results: List[DependencyResult]) -> Dict[str, Any]:
        """
        Get statistics from discovery results.
        
        Args:
            results: List of DependencyResult objects
            
        Returns:
            Dictionary with statistics
        """
        stats = {
            'total_results': len(results),
            'total_dependencies': 0,
            'by_type': {},
            'by_strength': {},
            'avg_correlation': 0.0,
            'avg_pvalue': 0.0,
            'very_strong_count': 0,
            'strong_count': 0,
            'positive_count': 0,
            'negative_count': 0,
            'significant_count': 0,
        }
        
        all_correlations = []
        all_pvalues = []
        
        for result in results:
            stats['total_dependencies'] += len(result.dependencies)
            
            for dep in result.dependencies:
                # By type
                type_key = dep.dependency_type.value
                stats['by_type'][type_key] = stats['by_type'].get(type_key, 0) + 1
                
                # By strength
                strength_key = dep.strength.value
                stats['by_strength'][strength_key] = stats['by_strength'].get(strength_key, 0) + 1
                
                # Very strong
                if dep.is_very_strong():
                    stats['very_strong_count'] += 1
                
                # Strong
                if dep.is_strong():
                    stats['strong_count'] += 1
                
                # Direction
                if dep.is_positive():
                    stats['positive_count'] += 1
                elif dep.is_negative():
                    stats['negative_count'] += 1
                
                # Significant
                if dep.is_significant(self._significance_level):
                    stats['significant_count'] += 1
                
                all_correlations.append(dep.correlation)
                if dep.p_value is not None:
                    all_pvalues.append(dep.p_value)
        
        if all_correlations:
            stats['avg_correlation'] = sum(all_correlations) / len(all_correlations)
        
        if all_pvalues:
            stats['avg_pvalue'] = sum(all_pvalues) / len(all_pvalues)
        
        return stats
    
    # ==========================================================================
    # PRIVATE METHODS
    # ==========================================================================
    
    def _validate_data(self, data: List[float]) -> bool:
        """
        Validate data for correlation analysis.
        
        Args:
            data: List of values
            
        Returns:
            True if data is valid, False otherwise
        """
        if not data or len(data) < self._min_sample_size:
            return False
        
        # Check for valid values (allow negative values for certain instruments)
        valid_count = 0
        for v in data:
            if v is not None and not math.isnan(v) and not math.isinf(v):
                valid_count += 1
        
        if valid_count < self._min_sample_size:
            return False
        
        return True
    
    def _calculate_dependency(
        self,
        primary_data: List[float],
        other_data: List[float],
        source_symbol: str,
        target_symbol: str,
        correlation_type: str = 'returns',
    ) -> Optional[MarketDependency]:
        """
        Calculate dependency between two data series.
        
        Returns:
            MarketDependency or None if calculation fails
        """
        try:
            # Align data by finding common length
            min_len = min(len(primary_data), len(other_data))
            d1 = primary_data[-min_len:]
            d2 = other_data[-min_len:]
            
            # Transform data based on correlation type
            if correlation_type == 'returns':
                series1 = self._calculate_returns(d1)
                series2 = self._calculate_returns(d2)
            else:
                series1 = d1[:]
                series2 = d2[:]
            
            # Validate transformed data
            if not self._validate_data(series1) or not self._validate_data(series2):
                return None
            
            # Calculate correlation with lag analysis
            best_corr = 0.0
            best_lag = 0
            best_pvalue = 1.0
            best_n = 0
            
            for lag in range(-self.MAX_LAG, self.MAX_LAG + 1):
                if lag < 0:
                    # primary leads other_data
                    r1 = series1[-lag:] if lag != 0 else series1
                    r2 = series2[:len(series2) + lag] if lag != 0 else series2
                elif lag > 0:
                    # other_data leads primary
                    r1 = series1[:len(series1) - lag]
                    r2 = series2[lag:]
                else:
                    r1 = series1
                    r2 = series2
                
                min_len = min(len(r1), len(r2))
                if min_len < self._min_sample_size:
                    continue
                
                r1 = r1[:min_len]
                r2 = r2[:min_len]
                
                # Calculate Pearson correlation with p-value (pure Python)
                corr, pvalue = self._pearson_correlation_with_pvalue(r1, r2)
                
                if corr is not None and abs(corr) > abs(best_corr):
                    best_corr = corr
                    best_lag = lag
                    best_pvalue = pvalue
                    best_n = min_len
            
            if best_n < self._min_sample_size:
                return None
            
            # Determine strength and type
            dep_type = self._determine_dependency_type(best_corr, best_lag)
            strength = self._determine_strength(abs(best_corr))
            direction = self._determine_direction(best_corr)
            
            # Calculate confidence
            confidence = self._calculate_confidence(best_corr, best_pvalue, best_n)
            
            signals = self._generate_signals(best_corr, dep_type, direction, best_pvalue, best_lag)
            
            return MarketDependency(
                source_symbol=source_symbol,
                target_symbol=target_symbol,
                dependency_type=dep_type,
                direction=direction,
                strength=strength,
                correlation=best_corr,
                confidence=confidence,
                lag=best_lag if best_lag != 0 else None,
                p_value=best_pvalue,
                sample_size=best_n,
                signals=signals,
                metadata={
                    'abs_correlation': abs(best_corr),
                    'correlation_type': 'positive' if best_corr > 0 else 'negative',
                    'lag': best_lag,
                    'sample_size': best_n,
                    'correlation_method': 'pearson_pure_python',
                    'data_type': correlation_type,
                },
            )
            
        except Exception as e:
            self.logger.warning(f"Error calculating dependency: {e}")
            return None
    
    def _calculate_returns(self, prices: List[float]) -> List[float]:
        """Calculate returns from price data."""
        returns = []
        for i in range(1, len(prices)):
            prev = prices[i-1]
            if prev is not None and prev != 0 and not math.isnan(prev) and not math.isinf(prev):
                curr = prices[i]
                if curr is not None and not math.isnan(curr) and not math.isinf(curr):
                    ret = (curr - prev) / prev
                    if not math.isnan(ret) and not math.isinf(ret):
                        returns.append(ret)
        return returns
    
    def _pearson_correlation_with_pvalue(self, x: List[float], y: List[float]) -> Tuple[Optional[float], float]:
        """
        Calculate Pearson correlation coefficient with p-value.
        Pure Python implementation - no numpy/scipy.
        
        Returns:
            Tuple of (correlation, p-value)
        """
        if len(x) != len(y) or len(x) < 3:
            return None, 1.0
        
        n = len(x)
        
        # Calculate means
        mean_x = sum(x) / n
        mean_y = sum(y) / n
        
        # Calculate correlation components
        cov_xy = 0.0
        var_x = 0.0
        var_y = 0.0
        
        for i in range(n):
            dx = x[i] - mean_x
            dy = y[i] - mean_y
            cov_xy += dx * dy
            var_x += dx * dx
            var_y += dy * dy
        
        if var_x == 0 or var_y == 0:
            return None, 1.0
        
        corr = cov_xy / (math.sqrt(var_x) * math.sqrt(var_y))
        
        if math.isnan(corr) or math.isinf(corr):
            return None, 1.0
        
        if abs(corr) >= 1.0:
            return corr, 0.0
        
        # Calculate t-statistic for p-value
        t = corr * math.sqrt((n - 2) / (1 - corr * corr))
        
        # Calculate p-value using Student's t-distribution approximation
        # Two-tailed test: 2 * P(T > |t|)
        pvalue = self._t_distribution_pvalue(abs(t), n - 2)
        
        return corr, pvalue
    
    def _t_distribution_pvalue(self, t: float, df: int) -> float:
        """
        Approximate p-value from t-distribution.
        
        Args:
            t: t-statistic (positive)
            df: degrees of freedom
            
        Returns:
            p-value (two-tailed)
        """
        # For large df, use normal approximation
        if df > 30:
            from math import erf, sqrt
            # Standard normal CDF approximation
            z = t  # For large df, t ≈ normal
            p = 0.5 * (1 + erf(z / sqrt(2)))
            # Two-tailed: 2 * (1 - cdf)
            return 2 * (1 - p)
        
        # For smaller df, use approximation via incomplete beta
        # Using the relationship: P(T ≤ t) = 0.5 * I_x(df/2, 1/2)
        # where x = df / (df + t^2)
        x = df / (df + t * t)
        
        # Incomplete beta function approximation (continued fraction)
        beta_cdf = self._incomplete_beta_approx(x, df / 2, 0.5)
        
        # Two-tailed p-value
        return 2 * (1 - beta_cdf)
    
    def _incomplete_beta_approx(self, x: float, a: float, b: float) -> float:
        """
        Approximate incomplete beta function using continued fraction.
        
        Args:
            x: Value between 0 and 1
            a: Alpha parameter
            b: Beta parameter
            
        Returns:
            Incomplete beta function value
        """
        if x == 0:
            return 0.0
        if x == 1:
            return 1.0
        
        # For small x, use series expansion (faster convergence)
        if x < 0.5:
            result = 0.0
            term = 1.0
            for k in range(50):
                if k > 0:
                    term *= (a + k - 1) * x / k
                result += term / (a + k)
            # Multiply by x^a / a
            result = result * (x ** a) / a
            # Normalize by beta(a, b)
            from math import gamma
            beta_ab = gamma(a) * gamma(b) / gamma(a + b)
            return result / beta_ab
        
        # For x > 0.5, use symmetry: I_x(a,b) = 1 - I_{1-x}(b,a)
        return 1 - self._incomplete_beta_approx(1 - x, b, a)
    
    def _determine_dependency_type(self, correlation: float, lag: int) -> DependencyType:
        """Determine dependency type from correlation and lag."""
        abs_corr = abs(correlation)
        
        if abs_corr >= self._very_strong_threshold:
            if lag != 0:
                return DependencyType.LEADING if lag < 0 else DependencyType.LAGGING
            return DependencyType.SYNCHRONOUS
        
        if abs_corr >= self._strong_threshold:
            if lag != 0:
                return DependencyType.CAUSALITY
            return DependencyType.CORRELATION
        
        if abs_corr >= self._moderate_threshold:
            if lag != 0:
                return DependencyType.SPILLOVER
            return DependencyType.CORRELATION
        
        if correlation < 0 and abs_corr >= self._weak_threshold:
            return DependencyType.INVERSE
        
        return DependencyType.UNKNOWN
    
    def _determine_strength(self, abs_corr: float) -> DependencyStrength:
        """Determine strength from absolute correlation."""
        if abs_corr >= self._very_strong_threshold:
            return DependencyStrength.VERY_STRONG
        elif abs_corr >= self._strong_threshold:
            return DependencyStrength.STRONG
        elif abs_corr >= self._moderate_threshold:
            return DependencyStrength.MODERATE
        elif abs_corr >= self._weak_threshold:
            return DependencyStrength.WEAK
        elif abs_corr > 0:
            return DependencyStrength.VERY_WEAK
        else:
            return DependencyStrength.NONE
    
    def _determine_direction(self, correlation: float) -> DependencyDirection:
        """Determine direction from correlation."""
        if correlation > 0.05:
            return DependencyDirection.POSITIVE
        elif correlation < -0.05:
            return DependencyDirection.NEGATIVE
        else:
            return DependencyDirection.UNKNOWN
    
    def _calculate_confidence(self, correlation: float, pvalue: float, n: int) -> float:
        """Calculate confidence score (0-1)."""
        # Base from correlation strength
        abs_corr = abs(correlation)
        confidence = abs_corr
        
        # Adjust for significance
        if pvalue <= self._significance_level:
            confidence = min(confidence + 0.15, 1.0)
        
        # Adjust for sample size
        sample_ratio = min(n / (self._min_sample_size * 2), 1.0)
        confidence = min(confidence + 0.05 * sample_ratio, 1.0)
        
        return max(min(confidence, 1.0), 0.0)
    
    def _generate_signals(
        self,
        correlation: float,
        dep_type: DependencyType,
        direction: DependencyDirection,
        pvalue: float,
        lag: int,
    ) -> List[str]:
        """Generate signals for dependency discovery."""
        signals = []
        
        if correlation > 0:
            signals.append(f"Positive correlation: {correlation:.3f}")
        else:
            signals.append(f"Negative correlation: {correlation:.3f}")
        
        if dep_type != DependencyType.UNKNOWN:
            signals.append(f"Type: {dep_type.value}")
        
        if direction != DependencyDirection.UNKNOWN:
            signals.append(f"Direction: {direction.value}")
        
        if pvalue <= self._significance_level:
            signals.append(f"Statistically significant (p={pvalue:.4f})")
        else:
            signals.append(f"Not statistically significant (p={pvalue:.4f})")
        
        if lag != 0:
            signals.append(f"Lag: {lag} periods")
        
        return signals
    
    def _build_cache_key(self, symbol: str, market_data: Dict, correlation_type: str) -> str:
        """Build stable cache key using hashlib."""
        # Sort keys for stability
        sorted_keys = tuple(sorted(market_data.keys()))
        # Use hashlib for stable hash
        key_string = ",".join(sorted_keys)
        keys_hash = hashlib.md5(key_string.encode()).hexdigest()[:16]
        return f"{symbol}_{keys_hash}_{correlation_type}"


# ==============================================================================
# FACTORY FUNCTION
# ==============================================================================

def create_dependency_discovery(config: Config) -> DependencyDiscovery:
    """
    Factory function for DependencyDiscovery creation.
    
    Args:
        config: Application configuration
        
    Returns:
        DependencyDiscovery instance
    """
    return DependencyDiscovery(config)