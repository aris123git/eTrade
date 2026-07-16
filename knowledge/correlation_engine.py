"""
discovery/correlation_engine.py - Correlation Discovery Engine

RESPONSIBILITY:
Discover and analyze correlations between markets and patterns.

ARCHITECTURAL PRINCIPLES:
1. Pure correlation discovery - No data storage, no I/O, no business logic
2. Statistical correlation analysis between time series
3. Type-safe results with validation
4. Multiple correlation methods (Pearson, Spearman, Kendall)

WHAT IT NEVER DOES:
- ❌ Connect to MT5
- ❌ Store data
- ❌ Download data
- ❌ Make trading decisions

VERSION: 1.0.0
"""

import logging
import math
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional, Dict, Any, Tuple, Set, Union, Callable
from enum import Enum
from collections import defaultdict

from core.config import Config
from core.exceptions import DiscoveryError, DataValidationError
from core.utils import to_datetime, format_datetime


# ==============================================================================
# EXPORTS
# ==============================================================================

__all__ = [
    'CorrelationMethod',
    'CorrelationStrength',
    'CorrelationDirection',
    'DiscoveredCorrelation',
    'CorrelationResult',
    'CorrelationEngine',
    'create_correlation_engine',
]


# ==============================================================================
# ENUMS
# ==============================================================================

class CorrelationMethod(Enum):
    """Method for calculating correlation."""
    PEARSON = "pearson"         # Pearson correlation coefficient
    SPEARMAN = "spearman"       # Spearman rank correlation
    KENDALL = "kendall"         # Kendall's tau
    DISTANCE = "distance"       # Distance correlation
    MUTUAL_INFO = "mutual_info" # Mutual information


class CorrelationStrength(Enum):
    """Strength of correlation."""
    VERY_STRONG = 1.0
    STRONG = 0.8
    MODERATE = 0.6
    WEAK = 0.4
    VERY_WEAK = 0.2
    NONE = 0.0


class CorrelationDirection(Enum):
    """Direction of correlation."""
    POSITIVE = "positive"
    NEGATIVE = "negative"
    NONE = "none"


# ==============================================================================
# DATA MODELS
# ==============================================================================

@dataclass
class DiscoveredCorrelation:
    """A discovered correlation between two entities."""
    source_symbol: str
    target_symbol: str
    correlation: float
    method: CorrelationMethod
    direction: CorrelationDirection
    strength: CorrelationStrength
    confidence: float
    lag: Optional[int] = None
    p_value: Optional[float] = None
    sample_size: int = 0
    signals: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def is_positive(self) -> bool:
        return self.direction == CorrelationDirection.POSITIVE
    
    def is_negative(self) -> bool:
        return self.direction == CorrelationDirection.NEGATIVE
    
    def is_strong(self) -> bool:
        return self.strength in (CorrelationStrength.STRONG, CorrelationStrength.VERY_STRONG)
    
    def is_significant(self, alpha: float = 0.05) -> bool:
        """Check if correlation is statistically significant."""
        if self.p_value is None:
            return self.confidence >= 0.8
        return self.p_value < alpha
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'source_symbol': self.source_symbol,
            'target_symbol': self.target_symbol,
            'correlation': self.correlation,
            'method': self.method.value,
            'direction': self.direction.value,
            'strength': self.strength.value,
            'confidence': self.confidence,
            'lag': self.lag,
            'p_value': self.p_value,
            'sample_size': self.sample_size,
            'signals': self.signals,
            'metadata': self.metadata,
        }


@dataclass
class CorrelationResult:
    """Result of correlation analysis."""
    symbol: str
    timestamp: datetime
    correlations: List[DiscoveredCorrelation]
    market_count: int
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def get_by_strength(self, min_strength: CorrelationStrength) -> List[DiscoveredCorrelation]:
        """Get correlations with at least the given strength."""
        return [c for c in self.correlations if c.strength.value >= min_strength.value]
    
    def get_positive(self) -> List[DiscoveredCorrelation]:
        """Get positive correlations."""
        return [c for c in self.correlations if c.is_positive()]
    
    def get_negative(self) -> List[DiscoveredCorrelation]:
        """Get negative correlations."""
        return [c for c in self.correlations if c.is_negative()]
    
    def get_strongest(self, limit: int = 10) -> List[DiscoveredCorrelation]:
        """Get strongest correlations."""
        sorted_corrs = sorted(
            self.correlations,
            key=lambda c: abs(c.correlation),
            reverse=True
        )
        return sorted_corrs[:limit]
    
    def get_significant(self, alpha: float = 0.05) -> List[DiscoveredCorrelation]:
        """Get statistically significant correlations."""
        return [c for c in self.correlations if c.is_significant(alpha)]
    
    def get_summary(self) -> Dict[str, Any]:
        """Get summary of correlation analysis."""
        return {
            'symbol': self.symbol,
            'total_correlations': len(self.correlations),
            'market_count': self.market_count,
            'by_strength': {
                s.value: sum(1 for c in self.correlations if c.strength == s)
                for s in CorrelationStrength
            },
            'positive_count': len(self.get_positive()),
            'negative_count': len(self.get_negative()),
            'significant_count': len(self.get_significant()),
            'avg_correlation': sum(c.correlation for c in self.correlations) / len(self.correlations) if self.correlations else 0.0,
        }


# ==============================================================================
# CORRELATION ENGINE
# ==============================================================================

class CorrelationEngine:
    """
    Correlation discovery engine.
    
    Discovers and analyzes correlations between markets.
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
        Initialize the correlation engine.
        
        Args:
            config: Application configuration
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Correlation thresholds
        self._very_strong_threshold = getattr(config, 'CORRELATION_VERY_STRONG_THRESHOLD', self.DEFAULT_VERY_STRONG_THRESHOLD)
        self._strong_threshold = getattr(config, 'CORRELATION_STRONG_THRESHOLD', self.DEFAULT_STRONG_THRESHOLD)
        self._moderate_threshold = getattr(config, 'CORRELATION_MODERATE_THRESHOLD', self.DEFAULT_MODERATE_THRESHOLD)
        self._weak_threshold = getattr(config, 'CORRELATION_WEAK_THRESHOLD', self.DEFAULT_WEAK_THRESHOLD)
        self._significance_level = getattr(config, 'CORRELATION_SIGNIFICANCE_LEVEL', self.DEFAULT_SIGNIFICANCE_LEVEL)
        self._min_sample_size = getattr(config, 'CORRELATION_MIN_SAMPLE_SIZE', self.MIN_SAMPLE_SIZE)
        self._max_lag = getattr(config, 'CORRELATION_MAX_LAG', self.MAX_LAG)
        self._default_method = CorrelationMethod.PEARSON
        
        self.logger.info(
            f"✅ CorrelationEngine initialized: "
            f"strong={self._strong_threshold}, "
            f"moderate={self._moderate_threshold}, "
            f"weak={self._weak_threshold}"
        )
    
    # ==========================================================================
    # PUBLIC METHODS
    # ==========================================================================
    
    def analyze_correlations(
        self,
        market_data: Dict[str, List[float]],
        symbol: str,
        method: Union[CorrelationMethod, str] = CorrelationMethod.PEARSON,
        correlation_type: str = 'returns',
        max_lag: Optional[int] = None,
    ) -> CorrelationResult:
        """
        Analyze correlations between markets.
        
        Args:
            market_data: Dictionary mapping symbol to price data list
            symbol: Primary symbol to analyze
            method: Correlation method
            correlation_type: 'returns' or 'prices'
            max_lag: Maximum lag to consider
            
        Returns:
            CorrelationResult object
        """
        if not market_data:
            raise DataValidationError("Market data is empty")
        
        if symbol not in market_data:
            raise DataValidationError(f"Symbol {symbol} not found in market data")
        
        method = self._parse_method(method)
        max_lag = max_lag or self._max_lag
        
        self.logger.debug(
            f"Analyzing correlations for {symbol} using {method.value}"
        )
        
        try:
            correlations = []
            
            # Get primary data
            primary_data = market_data[symbol]
            
            # Validate primary data
            if len(primary_data) < self._min_sample_size:
                raise DataValidationError(
                    f"Insufficient data for {symbol}: {len(primary_data)} < {self._min_sample_size}"
                )
            
            # Analyze correlations with all other symbols
            for other_symbol, other_data in market_data.items():
                if other_symbol == symbol:
                    continue
                
                if len(other_data) < self._min_sample_size:
                    self.logger.debug(f"Skipping {other_symbol}: insufficient data")
                    continue
                
                correlation = self._calculate_correlation(
                    primary_data=primary_data,
                    other_data=other_data,
                    source_symbol=symbol,
                    target_symbol=other_symbol,
                    method=method,
                    correlation_type=correlation_type,
                    max_lag=max_lag,
                )
                
                if correlation:
                    correlations.append(correlation)
            
            # Sort by absolute correlation
            correlations.sort(key=lambda c: abs(c.correlation), reverse=True)
            
            result = CorrelationResult(
                symbol=symbol,
                timestamp=datetime.now(),
                correlations=correlations,
                market_count=len(market_data),
                metadata={
                    'method': method.value,
                    'correlation_type': correlation_type,
                    'max_lag': max_lag,
                    'min_sample_size': self._min_sample_size,
                    'significance_level': self._significance_level,
                },
            )
            
            self.logger.debug(
                f"Correlation analysis complete: {len(correlations)} correlations found"
            )
            
            return result
            
        except Exception as e:
            self.logger.error(f"Correlation analysis failed: {e}")
            raise DiscoveryError(f"Failed to analyze correlations: {e}")
    
    def analyze_correlations_from_candles(
        self,
        candles_data: Dict[str, List[Dict[str, Any]]],
        symbol: str,
        price_field: str = 'close',
        method: Union[CorrelationMethod, str] = CorrelationMethod.PEARSON,
        correlation_type: str = 'returns',
    ) -> CorrelationResult:
        """
        Analyze correlations from candle data.
        
        Args:
            candles_data: Dictionary mapping symbol to candle list
            symbol: Primary symbol to analyze
            price_field: Field to use for price data
            method: Correlation method
            correlation_type: 'returns' or 'prices'
            
        Returns:
            CorrelationResult object
        """
        if not candles_data:
            raise DataValidationError("Candles data is empty")
        
        if symbol not in candles_data:
            raise DataValidationError(f"Symbol {symbol} not found in candles data")
        
        # Extract price data
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
                self.logger.debug(f"Skipping {sym}: insufficient prices ({len(prices)})")
        
        if not market_data:
            raise DataValidationError("No valid price data extracted from candles")
        
        return self.analyze_correlations(market_data, symbol, method, correlation_type)
    
    def calculate_correlation_matrix(
        self,
        market_data: Dict[str, List[float]],
        method: Union[CorrelationMethod, str] = CorrelationMethod.PEARSON,
        correlation_type: str = 'returns',
    ) -> Dict[str, Dict[str, float]]:
        """
        Calculate correlation matrix for all symbols.
        
        Args:
            market_data: Dictionary mapping symbol to price data list
            method: Correlation method
            correlation_type: 'returns' or 'prices'
            
        Returns:
            Correlation matrix as dict of dicts
        """
        method = self._parse_method(method)
        
        symbols = list(market_data.keys())
        if len(symbols) < 2:
            return {}
        
        matrix = {}
        
        for i, symbol1 in enumerate(symbols):
            matrix[symbol1] = {}
            data1 = market_data[symbol1]
            
            for symbol2 in symbols[i:]:
                if symbol1 == symbol2:
                    matrix[symbol1][symbol2] = 1.0
                    continue
                
                data2 = market_data[symbol2]
                correlation = self._calculate_raw_correlation(
                    data1, data2, method, correlation_type
                )
                
                matrix[symbol1][symbol2] = correlation
                if symbol2 not in matrix:
                    matrix[symbol2] = {}
                matrix[symbol2][symbol1] = correlation
        
        return matrix
    
    def find_leading_indicators(
        self,
        market_data: Dict[str, List[float]],
        symbol: str,
        max_lag: Optional[int] = None,
    ) -> List[DiscoveredCorrelation]:
        """
        Find leading indicators for a symbol.
        
        Args:
            market_data: Dictionary mapping symbol to price data list
            symbol: Target symbol
            max_lag: Maximum lag to consider
            
        Returns:
            List of leading indicator correlations
        """
        max_lag = max_lag or self._max_lag
        
        if symbol not in market_data:
            return []
        
        result = self.analyze_correlations(
            market_data=market_data,
            symbol=symbol,
            max_lag=max_lag,
        )
        
        # Filter for leading indicators (where other symbol leads)
        leaders = []
        for corr in result.correlations:
            if corr.lag is not None and corr.lag > 0:
                leaders.append(corr)
        
        return leaders
    
    # ==========================================================================
    # PRIVATE METHODS
    # ==========================================================================
    
    def _parse_method(self, method: Union[CorrelationMethod, str]) -> CorrelationMethod:
        """Parse correlation method from string or enum."""
        if isinstance(method, CorrelationMethod):
            return method
        if isinstance(method, str):
            try:
                return CorrelationMethod(method.lower())
            except ValueError:
                self.logger.warning(f"Unknown method '{method}', using PEARSON")
                return CorrelationMethod.PEARSON
        return self._default_method
    
    def _calculate_correlation(
        self,
        primary_data: List[float],
        other_data: List[float],
        source_symbol: str,
        target_symbol: str,
        method: CorrelationMethod,
        correlation_type: str,
        max_lag: int,
    ) -> Optional[DiscoveredCorrelation]:
        """Calculate correlation between two data series."""
        try:
            # Align data
            min_len = min(len(primary_data), len(other_data))
            d1 = primary_data[-min_len:]
            d2 = other_data[-min_len:]
            
            # Transform data
            if correlation_type == 'returns':
                series1 = self._calculate_returns(d1)
                series2 = self._calculate_returns(d2)
            else:
                series1 = d1
                series2 = d2
            
            if len(series1) < self._min_sample_size or len(series2) < self._min_sample_size:
                return None
            
            # Find best correlation with lag
            best_corr = 0.0
            best_lag = 0
            best_pvalue = 1.0
            best_n = 0
            
            for lag in range(-max_lag, max_lag + 1):
                if lag < 0:
                    r1 = series1[-lag:] if lag != 0 else series1
                    r2 = series2[:len(series2) + lag] if lag != 0 else series2
                elif lag > 0:
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
                
                corr, pvalue = self._calculate_correlation_with_pvalue(
                    r1, r2, method
                )
                
                if corr is not None and abs(corr) > abs(best_corr):
                    best_corr = corr
                    best_lag = lag
                    best_pvalue = pvalue
                    best_n = min_len
            
            if best_n < self._min_sample_size:
                return None
            
            # Determine direction and strength
            direction = self._determine_direction(best_corr)
            strength = self._determine_strength(abs(best_corr))
            confidence = self._calculate_confidence(best_corr, best_pvalue, best_n)
            
            signals = self._generate_signals(best_corr, direction, best_pvalue, best_lag)
            
            return DiscoveredCorrelation(
                source_symbol=source_symbol,
                target_symbol=target_symbol,
                correlation=best_corr,
                method=method,
                direction=direction,
                strength=strength,
                confidence=confidence,
                lag=best_lag if best_lag != 0 else None,
                p_value=best_pvalue,
                sample_size=best_n,
                signals=signals,
                metadata={
                    'abs_correlation': abs(best_corr),
                    'correlation_type': correlation_type,
                    'lag': best_lag,
                },
            )
            
        except Exception as e:
            self.logger.warning(f"Error calculating correlation: {e}")
            return None
    
    def _calculate_raw_correlation(
        self,
        data1: List[float],
        data2: List[float],
        method: CorrelationMethod,
        correlation_type: str,
    ) -> float:
        """Calculate raw correlation between two data series."""
        try:
            min_len = min(len(data1), len(data2))
            d1 = data1[-min_len:]
            d2 = data2[-min_len:]
            
            if correlation_type == 'returns':
                series1 = self._calculate_returns(d1)
                series2 = self._calculate_returns(d2)
            else:
                series1 = d1
                series2 = d2
            
            if len(series1) < self._min_sample_size:
                return 0.0
            
            corr, _ = self._calculate_correlation_with_pvalue(
                series1, series2, method
            )
            
            return corr if corr is not None else 0.0
            
        except Exception:
            return 0.0
    
    def _calculate_correlation_with_pvalue(
        self,
        x: List[float],
        y: List[float],
        method: CorrelationMethod,
    ) -> Tuple[Optional[float], float]:
        """Calculate correlation with p-value."""
        if len(x) != len(y) or len(x) < 3:
            return None, 1.0
        
        n = len(x)
        
        if method == CorrelationMethod.PEARSON:
            return self._pearson_correlation(x, y)
        elif method == CorrelationMethod.SPEARMAN:
            return self._spearman_correlation(x, y)
        elif method == CorrelationMethod.KENDALL:
            return self._kendall_correlation(x, y)
        else:
            # Default to Pearson
            return self._pearson_correlation(x, y)
    
    def _pearson_correlation(self, x: List[float], y: List[float]) -> Tuple[Optional[float], float]:
        """Calculate Pearson correlation coefficient."""
        n = len(x)
        if n < 3:
            return None, 1.0
        
        mean_x = sum(x) / n
        mean_y = sum(y) / n
        
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
        
        # Calculate p-value
        if abs(corr) >= 1.0:
            return corr, 0.0
        
        t = corr * math.sqrt((n - 2) / (1 - corr * corr))
        pvalue = self._t_distribution_pvalue(abs(t), n - 2)
        
        return corr, pvalue
    
    def _spearman_correlation(self, x: List[float], y: List[float]) -> Tuple[Optional[float], float]:
        """Calculate Spearman rank correlation coefficient."""
        n = len(x)
        if n < 3:
            return None, 1.0
        
        # Rank the data
        x_ranks = self._rank_data(x)
        y_ranks = self._rank_data(y)
        
        return self._pearson_correlation(x_ranks, y_ranks)
    
    def _kendall_correlation(self, x: List[float], y: List[float]) -> Tuple[Optional[float], float]:
        """Calculate Kendall's tau correlation coefficient."""
        n = len(x)
        if n < 3:
            return None, 1.0
        
        # Count concordant and discordant pairs
        concordant = 0
        discordant = 0
        
        for i in range(n):
            for j in range(i + 1, n):
                if x[i] == x[j] or y[i] == y[j]:
                    continue
                
                if (x[i] - x[j]) * (y[i] - y[j]) > 0:
                    concordant += 1
                else:
                    discordant += 1
        
        total = concordant + discordant
        if total == 0:
            return None, 1.0
        
        tau = (concordant - discordant) / total
        
        # Approximate p-value for Kendall's tau
        # Using normal approximation
        var = 2 * (2 * n + 5) / (9 * n * (n - 1))
        z = tau / math.sqrt(var)
        pvalue = 2 * (1 - self._normal_cdf(abs(z)))
        
        return tau, pvalue
    
    def _rank_data(self, data: List[float]) -> List[float]:
        """Rank data (average for ties)."""
        n = len(data)
        sorted_indices = sorted(range(n), key=lambda i: data[i])
        
        ranks = [0.0] * n
        i = 0
        while i < n:
            j = i
            while j < n and data[sorted_indices[j]] == data[sorted_indices[i]]:
                j += 1
            
            rank = (i + j - 1) / 2 + 1
            for k in range(i, j):
                ranks[sorted_indices[k]] = rank
            
            i = j
        
        return ranks
    
    def _calculate_returns(self, prices: List[float]) -> List[float]:
        """Calculate returns from price data."""
        returns = []
        for i in range(1, len(prices)):
            if prices[i-1] != 0:
                ret = (prices[i] - prices[i-1]) / prices[i-1]
                if not math.isnan(ret) and not math.isinf(ret):
                    returns.append(ret)
        return returns
    
    def _t_distribution_pvalue(self, t: float, df: int) -> float:
        """Calculate p-value from t-distribution."""
        # For large df, use normal approximation
        if df > 30:
            return 2 * (1 - self._normal_cdf(t))
        
        # For smaller df, use approximation
        from math import gamma, pi
        
        def incomplete_beta(x: float, a: float, b: float) -> float:
            # Continued fraction approximation for incomplete beta
            if x == 0:
                return 0.0
            if x == 1:
                return 1.0
            
            # Use series expansion for x < 0.5
            if x < 0.5:
                result = 0.0
                term = 1.0
                for k in range(50):
                    if k > 0:
                        term *= (a + k - 1) * x / k
                    result += term / (a + k)
                result = result * (x ** a) / a
                beta_ab = gamma(a) * gamma(b) / gamma(a + b)
                return result / beta_ab
            
            # Use symmetry for x > 0.5
            return 1 - incomplete_beta(1 - x, b, a)
        
        x = df / (df + t * t)
        cdf = 1 - 0.5 * incomplete_beta(x, df / 2, 0.5)
        return 2 * (1 - cdf)
    
    def _normal_cdf(self, z: float) -> float:
        """Approximate normal CDF."""
        from math import erf, sqrt
        return 0.5 * (1 + erf(z / sqrt(2)))
    
    def _determine_direction(self, correlation: float) -> CorrelationDirection:
        """Determine direction from correlation."""
        if correlation > 0.05:
            return CorrelationDirection.POSITIVE
        elif correlation < -0.05:
            return CorrelationDirection.NEGATIVE
        else:
            return CorrelationDirection.NONE
    
    def _determine_strength(self, abs_corr: float) -> CorrelationStrength:
        """Determine strength from absolute correlation."""
        if abs_corr >= self._very_strong_threshold:
            return CorrelationStrength.VERY_STRONG
        elif abs_corr >= self._strong_threshold:
            return CorrelationStrength.STRONG
        elif abs_corr >= self._moderate_threshold:
            return CorrelationStrength.MODERATE
        elif abs_corr >= self._weak_threshold:
            return CorrelationStrength.WEAK
        elif abs_corr > 0:
            return CorrelationStrength.VERY_WEAK
        else:
            return CorrelationStrength.NONE
    
    def _calculate_confidence(self, correlation: float, pvalue: float, n: int) -> float:
        """Calculate confidence score (0-1)."""
        abs_corr = abs(correlation)
        confidence = abs_corr
        
        if pvalue <= self._significance_level:
            confidence = min(confidence + 0.15, 1.0)
        
        sample_ratio = min(n / (self._min_sample_size * 2), 1.0)
        confidence = min(confidence + 0.05 * sample_ratio, 1.0)
        
        return max(min(confidence, 1.0), 0.0)
    
    def _generate_signals(
        self,
        correlation: float,
        direction: CorrelationDirection,
        pvalue: float,
        lag: int,
    ) -> List[str]:
        """Generate signals for correlation discovery."""
        signals = []
        
        if correlation > 0:
            signals.append(f"Positive correlation: {correlation:.3f}")
        else:
            signals.append(f"Negative correlation: {correlation:.3f}")
        
        if direction != CorrelationDirection.NONE:
            signals.append(f"Direction: {direction.value}")
        
        if pvalue <= self._significance_level:
            signals.append(f"Statistically significant (p={pvalue:.4f})")
        else:
            signals.append(f"Not statistically significant (p={pvalue:.4f})")
        
        if lag != 0:
            signals.append(f"Lag: {lag} periods")
        
        return signals


# ==============================================================================
# FACTORY FUNCTION
# ==============================================================================

def create_correlation_engine(config: Config) -> CorrelationEngine:
    """
    Factory function for CorrelationEngine creation.
    
    Args:
        config: Application configuration
        
    Returns:
        CorrelationEngine instance
    """
    return CorrelationEngine(config)