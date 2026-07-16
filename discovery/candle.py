"""
eTrade Discovery Engine - Candle Module

Production-quality immutable Candle dataclass with mathematical methods.

Optimized for:
- Millions of candles in memory
- Fast calculations with caching
- Pattern analysis and feature extraction
- Cross-market comparisons

Author: eTrade Development Team
Version: 1.0.0
"""

from dataclasses import dataclass
from functools import cached_property
from typing import Optional, Dict, List, Tuple
import numpy as np
import math


@dataclass(frozen=True)
class Candle:
    """
    Immutable market candle with mathematical analysis methods.
    
    Represents a single OHLCV candle for a specific market and timeframe.
    All calculations are cached to optimize performance with large datasets.
    
    Attributes:
        time: Unix timestamp (seconds)
        open: Opening price
        high: Highest price
        low: Lowest price
        close: Closing price
        tick_volume: Volume in ticks
        real_volume: Real volume (if available)
        spread: Bid-ask spread in points
        market_id: Market identifier (1=EURUSD, 2=GBPUSD, etc.)
        timeframe_id: Timeframe identifier (1=M1, 3=M5, etc.)
    """
    
    time: int
    open: float
    high: float
    low: float
    close: float
    tick_volume: int
    real_volume: int = 0
    spread: float = 0.0
    market_id: int = 0
    timeframe_id: int = 0
    
    # ==================================================================
    # CORE PRICE MOVEMENTS
    # ==================================================================
    
    @cached_property
    def body(self) -> float:
        """
        Candle body size (close - open).
        
        Positive = bullish candle
        Negative = bearish candle
        Zero = doji candle
        
        Returns:
            float: Signed body size
        """
        return self.close - self.open
    
    @cached_property
    def body_size(self) -> float:
        """
        Absolute candle body size |close - open|.
        
        Always positive. Represents magnitude of price change.
        
        Returns:
            float: Absolute body size
        """
        return abs(self.body)
    
    @cached_property
    def range(self) -> float:
        """
        Total price range (high - low).
        
        Represents the complete price movement during the candle period.
        Always positive.
        
        Returns:
            float: High - low
        """
        return self.high - self.low
    
    @cached_property
    def upper_wick(self) -> float:
        """
        Distance from max(open, close) to high.
        
        Represents upside rejection. Large upper wicks suggest
        bulls tried to push price higher but were rejected.
        
        Returns:
            float: Distance from candle top to high
        """
        return self.high - max(self.open, self.close)
    
    @cached_property
    def lower_wick(self) -> float:
        """
        Distance from min(open, close) to low.
        
        Represents downside testing. Large lower wicks suggest
        bears tried to push price lower but were rejected.
        
        Returns:
            float: Distance from candle bottom to low
        """
        return min(self.open, self.close) - self.low
    
    # ==================================================================
    # RATIOS & NORMALIZED METRICS
    # ==================================================================
    
    @cached_property
    def body_ratio(self) -> float:
        """
        Body as percentage of range: body / range.
        
        Range: [-1.0, 1.0]
        - 1.0 = strong bullish candle (close at high)
        - 0.0 = doji (open == close)
        - -1.0 = strong bearish candle (close at low)
        
        Returns:
            float: Normalized body ratio
        """
        if self.range == 0:
            return 0.0
        return self.body / self.range
    
    @cached_property
    def upper_wick_ratio(self) -> float:
        """
        Upper wick as percentage of range: upper_wick / range.
        
        Range: [0.0, 1.0]
        - 0.0 = no upper wick (close or open at high)
        - 1.0 = upper wick is entire range (rare)
        
        Interpretation: Higher values = more upside rejection
        
        Returns:
            float: Normalized upper wick ratio [0-1]
        """
        if self.range == 0:
            return 0.0
        return self.upper_wick / self.range
    
    @cached_property
    def lower_wick_ratio(self) -> float:
        """
        Lower wick as percentage of range: lower_wick / range.
        
        Range: [0.0, 1.0]
        - 0.0 = no lower wick (close or open at low)
        - 1.0 = lower wick is entire range (rare)
        
        Interpretation: Higher values = more downside testing
        
        Returns:
            float: Normalized lower wick ratio [0-1]
        """
        if self.range == 0:
            return 0.0
        return self.lower_wick / self.range
    
    # ==================================================================
    # DIRECTION & CLASSIFICATION
    # ==================================================================
    
    @cached_property
    def direction(self) -> float:
        """
        Candle direction as normalized value.
        
        Returns:
            float: 1.0 = bullish, -1.0 = bearish, 0.0 = doji
        """
        if self.body > 0:
            return 1.0
        elif self.body < 0:
            return -1.0
        else:
            return 0.0
    
    def is_bullish(self) -> bool:
        """
        Check if candle is bullish (close > open).
        
        Returns:
            bool: True if bullish, False otherwise
        """
        return self.close > self.open
    
    def is_bearish(self) -> bool:
        """
        Check if candle is bearish (close < open).
        
        Returns:
            bool: True if bearish, False otherwise
        """
        return self.close < self.open
    
    def is_doji(self, threshold: float = 0.01) -> bool:
        """
        Check if candle is a doji (open ≈ close).
        
        Args:
            threshold: Maximum body size as % of range to consider doji.
                      Default 0.01 = 1% of range.
        
        Returns:
            bool: True if doji, False otherwise
        """
        if self.range == 0:
            return True
        return (self.body_size / self.range) < threshold
    
    # ==================================================================
    # PRICE AGGREGATES
    # ==================================================================
    
    @cached_property
    def mid_price(self) -> float:
        """
        Middle price (high + low) / 2.
        
        Represents the midpoint of the price range.
        
        Returns:
            float: (high + low) / 2
        """
        return (self.high + self.low) / 2.0
    
    @cached_property
    def typical_price(self) -> float:
        """
        Typical price (high + low + close) / 3.
        
        Weighted average emphasizing the closing price.
        Common in technical analysis.
        
        Returns:
            float: (high + low + close) / 3
        """
        return (self.high + self.low + self.close) / 3.0
    
    @cached_property
    def weighted_price(self) -> float:
        """
        Weighted price (high + low + 2*close) / 4.
        
        Gives double weight to closing price.
        More reflective of closing bias in markets.
        
        Returns:
            float: (high + low + 2*close) / 4
        """
        return (self.high + self.low + 2.0 * self.close) / 4.0
    
    @cached_property
    def ohlc4(self) -> float:
        """
        OHLC4 price (open + high + low + close) / 4.
        
        Equal-weighted average of all four prices.
        
        Returns:
            float: (open + high + low + close) / 4
        """
        return (self.open + self.high + self.low + self.close) / 4.0
    
    # ==================================================================
    # RETURNS & VOLATILITY
    # ==================================================================
    
    def returns(self, previous_candle: Optional['Candle'] = None) -> float:
        """
        Simple return from previous close to current close.
        
        Formula: (current_close - previous_close) / previous_close
        
        Args:
            previous_candle: Previous candle. If None, uses current open.
        
        Returns:
            float: Return as decimal (e.g., 0.01 = 1% gain)
        
        Raises:
            ValueError: If previous close is zero
        """
        if previous_candle is None:
            previous_price = self.open
        else:
            previous_price = previous_candle.close
        
        if previous_price == 0:
            raise ValueError("Cannot calculate return: previous price is zero")
        
        return (self.close - previous_price) / previous_price
    
    def log_return(self, previous_candle: Optional['Candle'] = None) -> float:
        """
        Logarithmic return from previous close to current close.
        
        Formula: ln(current_close / previous_close)
        
        Better properties than simple returns for statistical analysis:
        - Additive across time periods
        - Better for volatility estimation
        
        Args:
            previous_candle: Previous candle. If None, uses current open.
        
        Returns:
            float: Log return as decimal
        
        Raises:
            ValueError: If previous price is zero or negative
        """
        if previous_candle is None:
            previous_price = self.open
        else:
            previous_price = previous_candle.close
        
        if previous_price <= 0:
            raise ValueError("Cannot calculate log return: previous price must be positive")
        
        if self.close <= 0:
            raise ValueError("Cannot calculate log return: current price must be positive")
        
        return math.log(self.close / previous_price)
    
    def true_range(self, previous_candle: Optional['Candle'] = None) -> float:
        """
        True Range (Wilder's definition).
        
        Maximum of:
        1. Current high - current low
        2. Current high - previous close
        3. Current low - previous close
        
        Used in ATR (Average True Range) volatility calculation.
        
        Args:
            previous_candle: Previous candle (required for full calculation)
        
        Returns:
            float: True range value
        """
        if previous_candle is None:
            return self.range
        
        tr1 = self.range
        tr2 = abs(self.high - previous_candle.close)
        tr3 = abs(self.low - previous_candle.close)
        
        return max(tr1, tr2, tr3)
    
    def gap(self, previous_candle: Optional['Candle'] = None) -> float:
        """
        Gap from previous close to current open.
        
        Formula: (current_open - previous_close) / previous_close
        
        Positive gap = gap up (bullish)
        Negative gap = gap down (bearish)
        
        Args:
            previous_candle: Previous candle (required for gap calculation)
        
        Returns:
            float: Gap as decimal (e.g., 0.002 = 0.2% gap)
        
        Raises:
            ValueError: If previous_candle is None or previous close is zero
        """
        if previous_candle is None:
            raise ValueError("gap() requires previous_candle parameter")
        
        if previous_candle.close == 0:
            raise ValueError("Cannot calculate gap: previous close is zero")
        
        return (self.open - previous_candle.close) / previous_candle.close
    
    # ==================================================================
    # FEATURE EXTRACTION FOR ML
    # ==================================================================
    
    @cached_property
    def feature_vector(self) -> np.ndarray:
        """
        Create normalized feature vector for machine learning.
        
        Returns 8 scale-invariant features normalized to [0, 1]:
        [body_ratio, body_size_norm, upper_wick_ratio,
         lower_wick_ratio, direction_norm, range_norm,
         volume_norm, spread_norm]
        
        These features are scale-invariant and can be compared
        across different assets and price levels.
        
        Returns:
            np.ndarray: Feature vector of shape (8,)
        """
        # Normalize direction: -1 → 0, 0 → 0.5, 1 → 1
        direction_norm = (self.direction + 1.0) / 2.0
        
        # Normalize body size relative to range
        if self.range > 0:
            body_size_norm = self.body_size / self.range
        else:
            body_size_norm = 0.0
        
        # Log normalize range and volume
        range_norm = np.log(self.range + 1) / 10.0
        volume_norm = np.log(self.tick_volume + 1) / 20.0
        
        # Normalize spread
        spread_norm = min(1.0, self.spread / 100.0) if self.spread > 0 else 0.0
        
        # Clip all to [0, 1]
        features = np.array([
            (self.body_ratio + 1.0) / 2.0,      # [-1,1] → [0,1]
            min(1.0, body_size_norm),
            min(1.0, self.upper_wick_ratio),
            min(1.0, self.lower_wick_ratio),
            direction_norm,
            min(1.0, range_norm),
            min(1.0, volume_norm),
            min(1.0, spread_norm),
        ], dtype=np.float32)
        
        return np.clip(features, 0.0, 1.0)
    
    # ==================================================================
    # CONVERSION & SERIALIZATION
    # ==================================================================
    
    def to_numpy(self) -> np.ndarray:
        """
        Convert candle to numpy array.
        
        Order: [time, open, high, low, close, tick_volume,
                real_volume, spread, market_id, timeframe_id]
        
        Returns:
            np.ndarray: Array of shape (10,) with dtype float64
        """
        return np.array([
            self.time,
            self.open,
            self.high,
            self.low,
            self.close,
            self.tick_volume,
            self.real_volume,
            self.spread,
            self.market_id,
            self.timeframe_id,
        ], dtype=np.float64)
    
    def to_dict(self) -> Dict[str, float]:
        """
        Convert candle to dictionary.
        
        Returns:
            Dict: All fields plus calculated properties
        """
        return {
            "time": self.time,
            "open": self.open,
            "high": self.high,
            "low": self.low,
            "close": self.close,
            "tick_volume": self.tick_volume,
            "real_volume": self.real_volume,
            "spread": self.spread,
            "market_id": self.market_id,
            "timeframe_id": self.timeframe_id,
            # Calculated properties
            "body": self.body,
            "body_size": self.body_size,
            "range": self.range,
            "upper_wick": self.upper_wick,
            "lower_wick": self.lower_wick,
            "body_ratio": self.body_ratio,
            "upper_wick_ratio": self.upper_wick_ratio,
            "lower_wick_ratio": self.lower_wick_ratio,
            "direction": self.direction,
            "mid_price": self.mid_price,
            "typical_price": self.typical_price,
            "weighted_price": self.weighted_price,
            "ohlc4": self.ohlc4,
        }
    
    # ==================================================================
    # UTILITY METHODS
    # ==================================================================
    
    def __str__(self) -> str:
        """String representation."""
        direction_str = "↑" if self.is_bullish() else ("↓" if self.is_bearish() else "→")
        return (f"Candle({direction_str} O:{self.open:.5f} H:{self.high:.5f} "
                f"L:{self.low:.5f} C:{self.close:.5f} V:{self.tick_volume})")
    
    def __repr__(self) -> str:
        """Detailed representation."""
        return (f"Candle(time={self.time}, open={self.open}, high={self.high}, "
                f"low={self.low}, close={self.close}, tick_volume={self.tick_volume}, "
                f"market_id={self.market_id}, timeframe_id={self.timeframe_id})")
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'Candle':
        """
        Create Candle from dictionary.
        
        Args:
            data: Dictionary with candle data
        
        Returns:
            Candle: New candle instance
        """
        return cls(
            time=int(data['time']),
            open=float(data['open']),
            high=float(data['high']),
            low=float(data['low']),
            close=float(data['close']),
            tick_volume=int(data.get('tick_volume', 0)),
            real_volume=int(data.get('real_volume', 0)),
            spread=float(data.get('spread', 0.0)),
            market_id=int(data.get('market_id', 0)),
            timeframe_id=int(data.get('timeframe_id', 0)),
        )
    
    @classmethod
    def from_numpy(cls, arr: np.ndarray) -> 'Candle':
        """
        Create Candle from numpy array.
        
        Array format: [time, open, high, low, close, tick_volume,
                      real_volume, spread, market_id, timeframe_id]
        
        Args:
            arr: Numpy array of shape (10,)
        
        Returns:
            Candle: New candle instance
        """
        return cls(
            time=int(arr[0]),
            open=float(arr[1]),
            high=float(arr[2]),
            low=float(arr[3]),
            close=float(arr[4]),
            tick_volume=int(arr[5]),
            real_volume=int(arr[6]),
            spread=float(arr[7]),
            market_id=int(arr[8]),
            timeframe_id=int(arr[9]),
        )


# ==============================================================================
# CANDLE BATCH UTILITIES
# ==============================================================================

def candles_to_numpy(candles: List[Candle]) -> np.ndarray:
    """
    Convert list of candles to 2D numpy array.
    
    Args:
        candles: List of Candle objects
    
    Returns:
        np.ndarray: Shape (n_candles, 10) with dtype float64
    """
    return np.array([c.to_numpy() for c in candles], dtype=np.float64)


def numpy_to_candles(arr: np.ndarray) -> List[Candle]:
    """
    Convert 2D numpy array to list of candles.
    
    Args:
        arr: Numpy array of shape (n_candles, 10)
    
    Returns:
        List[Candle]: List of Candle objects
    """
    return [Candle.from_numpy(row) for row in arr]


def validate_candle_sequence(candles: List[Candle]) -> bool:
    """
    Validate a sequence of candles for consistency.
    
    Checks:
    - Times are increasing
    - OHLC relationships (high >= max(open,close), low <= min(open,close))
    - No NaN or infinite values
    
    Args:
        candles: List of Candle objects
    
    Returns:
        bool: True if valid, False otherwise
    """
    if not candles:
        return True
    
    # Check time ordering
    for i in range(1, len(candles)):
        if candles[i].time <= candles[i-1].time:
            return False
    
    # Check OHLC relationships
    for candle in candles:
        if not (candle.high >= max(candle.open, candle.close)):
            return False
        if not (candle.low <= min(candle.open, candle.close)):
            return False
        if candle.high < candle.low:
            return False
        if not (0 <= candle.range < float('inf')):
            return False
    
    return True