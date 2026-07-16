"""
preprocessing/feature_builder.py - Feature Builder Module

RESPONSIBILITY:
Build technical features from raw market data for AI/ML models.

ARCHITECTURAL PRINCIPLES:
1. Pure feature engineering - No data storage, no I/O, no business logic
2. Extract meaningful features from price/volume data
3. Type-safe results with validation
4. Multiple feature types (technical, statistical, derived)

WHAT IT NEVER DOES:
- ❌ Connect to MT5
- ❌ Store data
- ❌ Download data
- ❌ Make trading decisions
- ❌ Analyze patterns (only builds features)

VERSION: 1.0.1
"""

import logging
import math
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional, Dict, Any, Tuple, Set, Union, Callable
from enum import Enum

from core.config import Config
from core.exceptions import DataValidationError


# ==============================================================================
# EXPORTS
# ==============================================================================

__all__ = [
    'FeatureType',
    'FeatureSet',
    'FeatureResult',
    'FeatureBuilder',
    'create_feature_builder',
]


# ==============================================================================
# ENUMS
# ==============================================================================

class FeatureType(Enum):
    """Types of features that can be built."""
    # Price features
    PRICE = "price"
    RETURN = "return"
    LOG_RETURN = "log_return"
    
    # Momentum features
    MOMENTUM = "momentum"
    ROC = "roc"
    RSI = "rsi"
    MACD = "macd"
    
    # Volatility features
    VOLATILITY = "volatility"
    ATR = "atr"
    BANDWIDTH = "bandwidth"
    
    # Volume features
    VOLUME = "volume"
    VOLUME_RATIO = "volume_ratio"
    VWAP = "vwap"
    
    # Statistical features
    MEAN = "mean"
    MEDIAN = "median"
    STD = "std"
    SKEW = "skew"
    KURTOSIS = "kurtosis"
    
    # Derived features
    SPREAD = "spread"
    RANGE = "range"
    BODY = "body"
    WICK = "wick"
    
    # Pattern features
    DOJI = "doji"
    HAMMER = "hammer"
    ENGULFING = "engulfing"


class FeatureSet(Enum):
    """Predefined feature sets."""
    BASIC = "basic"                 # OHLC + volume
    TECHNICAL = "technical"         # RSI, MACD, ATR, etc.
    STATISTICAL = "statistical"     # Mean, std, skew, kurtosis
    MOMENTUM = "momentum"           # Returns, ROC, momentum
    VOLUME = "volume"               # Volume, VWAP, volume ratio
    PATTERN = "pattern"             # Doji, hammer, engulfing
    ALL = "all"                     # All features


# ==============================================================================
# DATA MODELS
# ==============================================================================

@dataclass
class FeatureResult:
    """Result of feature building operation."""
    features: List[Dict[str, float]]
    feature_names: List[str]
    feature_count: int
    window_size: int
    original_count: int
    feature_set: FeatureSet
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def success(self) -> bool:
        return len(self.features) > 0
    
    def get_features_dataframe(self) -> Any:
        """
        Convert features to pandas DataFrame.
        
        Returns:
            pandas DataFrame if pandas is available, None otherwise
        """
        try:
            import pandas as pd
            return pd.DataFrame(self.features)
        except ImportError:
            return None
    
    def get_summary(self) -> Dict[str, Any]:
        """Get summary of feature building operation."""
        return {
            'feature_count': self.feature_count,
            'window_size': self.window_size,
            'original_count': self.original_count,
            'feature_set': self.feature_set.value,
            'feature_names': self.feature_names,
            'success': self.success,
        }


# ==============================================================================
# FEATURE BUILDER
# ==============================================================================

class FeatureBuilder:
    """
    Feature building engine.
    
    Builds technical features from raw market data.
    """
    
    # RSI settings
    RSI_PERIOD = 14
    RSI_OVERBOUGHT = 70
    RSI_OVERSOLD = 30
    
    # MACD settings
    MACD_FAST = 12
    MACD_SLOW = 26
    MACD_SIGNAL = 9
    
    # ATR settings
    ATR_PERIOD = 14
    
    # Bollinger Bands settings
    BB_PERIOD = 20
    BB_STD = 2.0
    
    def __init__(self, config: Config):
        """
        Initialize the feature builder.
        
        Args:
            config: Application configuration
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Feature defaults
        self._default_window = getattr(config, 'FEATURE_WINDOW', 20)
        self._default_feature_set = FeatureSet.TECHNICAL
        
        self.logger.info("✅ FeatureBuilder initialized")
    
    # ==========================================================================
    # PUBLIC METHODS
    # ==========================================================================
    
    def build_features(
        self,
        candles: List[Dict[str, Any]],
        feature_set: Union[FeatureSet, str] = FeatureSet.TECHNICAL,
        window_size: Optional[int] = None,
        fields: Optional[List[str]] = None,
        include_price: bool = True,
    ) -> FeatureResult:
        """
        Build features from candles.
        
        Args:
            candles: List of candle dictionaries
            feature_set: Feature set to build
            window_size: Rolling window size
            fields: Specific fields to include
            include_price: Include price fields
            
        Returns:
            FeatureResult object
        """
        if not candles:
            return FeatureResult(
                features=[],
                feature_names=[],
                feature_count=0,
                window_size=window_size or self._default_window,
                original_count=0,
                feature_set=self._parse_feature_set(feature_set),
                metadata={'error': 'No candles provided'},
            )
        
        # Parse feature set
        feature_set = self._parse_feature_set(feature_set)
        window_size = window_size or self._default_window
        
        self.logger.debug(
            f"Building features: {len(candles)} candles, "
            f"feature_set={feature_set.value}, window={window_size}"
        )
        
        try:
            # Validate candles
            if not self._validate_candles(candles):
                raise DataValidationError("Invalid candles provided")
            
            # Extract data
            n = len(candles)
            close = [c['close'] for c in candles]
            high = [c['high'] for c in candles]
            low = [c['low'] for c in candles]
            open_price = [c['open'] for c in candles]
            volume = [c.get('volume', 0) for c in candles]
            timestamp = [c.get('timestamp') for c in candles]
            
            # Initialize features
            features = []
            feature_names = []
            
            # Always include price if requested
            if include_price:
                price_features, price_names = self._build_price_features(
                    open_price, high, low, close
                )
                features.extend(price_features)
                feature_names.extend(price_names)
            
            # Build features by set
            if feature_set in (FeatureSet.BASIC, FeatureSet.ALL):
                basic_features, basic_names = self._build_basic_features(
                    open_price, high, low, close, volume
                )
                features.extend(basic_features)
                feature_names.extend(basic_names)
            
            if feature_set in (FeatureSet.TECHNICAL, FeatureSet.ALL):
                tech_features, tech_names = self._build_technical_features(
                    close, high, low, volume, window_size
                )
                features.extend(tech_features)
                feature_names.extend(tech_names)
            
            if feature_set in (FeatureSet.STATISTICAL, FeatureSet.ALL):
                stat_features, stat_names = self._build_statistical_features(
                    close, window_size
                )
                features.extend(stat_features)
                feature_names.extend(stat_names)
            
            if feature_set in (FeatureSet.MOMENTUM, FeatureSet.ALL):
                mom_features, mom_names = self._build_momentum_features(
                    close, window_size
                )
                features.extend(mom_features)
                feature_names.extend(mom_names)
            
            if feature_set in (FeatureSet.VOLUME, FeatureSet.ALL):
                vol_features, vol_names = self._build_volume_features(
                    volume, close, window_size
                )
                features.extend(vol_features)
                feature_names.extend(vol_names)
            
            if feature_set in (FeatureSet.PATTERN, FeatureSet.ALL):
                pattern_features, pattern_names = self._build_pattern_features(
                    open_price, high, low, close
                )
                features.extend(pattern_features)
                feature_names.extend(pattern_names)
            
            # Filter features if fields specified
            if fields:
                filtered = []
                filtered_names = []
                for i, name in enumerate(feature_names):
                    if name in fields:
                        filtered.append(features[i])
                        filtered_names.append(name)
                features = filtered
                feature_names = filtered_names
            
            result = FeatureResult(
                features=features,
                feature_names=feature_names,
                feature_count=len(feature_names),
                window_size=window_size,
                original_count=len(candles),
                feature_set=feature_set,
                metadata={
                    'timestamp_range': (
                        timestamp[0] if timestamp else None,
                        timestamp[-1] if timestamp else None
                    ),
                },
            )
            
            self.logger.debug(
                f"Feature build complete: {result.feature_count} features"
            )
            
            return result
            
        except Exception as e:
            self.logger.error(f"Feature build failed: {e}")
            raise DataValidationError(f"Failed to build features: {e}")
    
    def build_features_for_symbol(
        self,
        candles: List[Dict[str, Any]],
        symbol: str,
        feature_set: Union[FeatureSet, str] = FeatureSet.TECHNICAL,
        window_size: Optional[int] = None,
    ) -> FeatureResult:
        """
        Build features for a specific symbol.
        
        Args:
            candles: List of candle dictionaries
            symbol: Symbol name
            feature_set: Feature set to build
            window_size: Rolling window size
            
        Returns:
            FeatureResult object with symbol metadata
        """
        result = self.build_features(candles, feature_set, window_size)
        result.metadata['symbol'] = symbol
        return result
    
    def get_feature_names(self, feature_set: Union[FeatureSet, str]) -> List[str]:
        """
        Get feature names for a feature set.
        
        Args:
            feature_set: Feature set
            
        Returns:
            List of feature names
        """
        feature_set = self._parse_feature_set(feature_set)
        
        # Return predefined feature names
        if feature_set == FeatureSet.BASIC:
            return ['body', 'range', 'spread', 'volume']
        elif feature_set == FeatureSet.TECHNICAL:
            return ['rsi', 'macd', 'macd_signal', 'atr', 'bb_upper', 'bb_lower', 'bb_middle']
        elif feature_set == FeatureSet.STATISTICAL:
            return ['mean', 'median', 'std', 'skew', 'kurtosis']
        elif feature_set == FeatureSet.MOMENTUM:
            return ['return_1', 'return_5', 'return_20', 'momentum', 'roc']
        elif feature_set == FeatureSet.VOLUME:
            return ['volume', 'volume_ratio', 'vwap']
        elif feature_set == FeatureSet.PATTERN:
            return ['is_doji', 'is_hammer', 'is_bullish', 'is_bearish']
        elif feature_set == FeatureSet.ALL:
            return ['open', 'high', 'low', 'close', 'body', 'range', 'spread', 'volume',
                   'rsi', 'macd', 'macd_signal', 'atr', 'bb_upper', 'bb_lower', 'bb_middle',
                   'mean', 'median', 'std', 'skew', 'kurtosis',
                   'return_1', 'return_5', 'return_20', 'momentum', 'roc',
                   'volume_ratio', 'vwap',
                   'is_doji', 'is_hammer', 'is_bullish', 'is_bearish']
        else:
            return []
    
    # ==========================================================================
    # PRIVATE METHODS
    # ==========================================================================
    
    def _parse_feature_set(self, feature_set: Union[FeatureSet, str]) -> FeatureSet:
        """Parse feature set from string or enum."""
        if isinstance(feature_set, FeatureSet):
            return feature_set
        if isinstance(feature_set, str):
            try:
                return FeatureSet(feature_set.lower())
            except ValueError:
                self.logger.warning(f"Unknown feature set '{feature_set}', using TECHNICAL")
                return FeatureSet.TECHNICAL
        return self._default_feature_set
    
    def _validate_candles(self, candles: List[Dict[str, Any]]) -> bool:
        """Validate candles for feature building."""
        if not candles:
            return False
        
        required_fields = {'open', 'high', 'low', 'close'}
        
        for i, candle in enumerate(candles):
            if not all(field in candle for field in required_fields):
                self.logger.debug(f"Candle {i} missing required fields")
                return False
        
        return True
    
    def _build_price_features(
        self,
        open_price: List[float],
        high: List[float],
        low: List[float],
        close: List[float],
    ) -> Tuple[List[Dict[str, float]], List[str]]:
        """Build price features."""
        features = []
        names = ['open', 'high', 'low', 'close']
        
        n = len(close)
        for i in range(n):
            feat = {}
            feat['open'] = open_price[i] if i < len(open_price) else close[i]
            feat['high'] = high[i] if i < len(high) else close[i]
            feat['low'] = low[i] if i < len(low) else close[i]
            feat['close'] = close[i]
            
            features.append(feat)
        
        return features, names
    
    def _build_basic_features(
        self,
        open_price: List[float],
        high: List[float],
        low: List[float],
        close: List[float],
        volume: List[float],
    ) -> Tuple[List[Dict[str, float]], List[str]]:
        """Build basic OHLCV features."""
        features = []
        names = ['body', 'range', 'spread', 'volume']
        
        n = len(close)
        for i in range(n):
            feat = {}
            
            # Body
            open_val = open_price[i] if i < len(open_price) else close[i]
            feat['body'] = abs(close[i] - open_val)
            
            # Range
            high_val = high[i] if i < len(high) else close[i]
            low_val = low[i] if i < len(low) else close[i]
            feat['range'] = high_val - low_val
            
            # Spread (high-low ratio)
            if close[i] > 0:
                feat['spread'] = (high_val - low_val) / close[i]
            else:
                feat['spread'] = 0.0
            
            # Volume
            feat['volume'] = volume[i] if i < len(volume) else 0.0
            
            features.append(feat)
        
        return features, names
    
    def _build_technical_features(
        self,
        close: List[float],
        high: List[float],
        low: List[float],
        volume: List[float],
        window: int,
    ) -> Tuple[List[Dict[str, float]], List[str]]:
        """Build technical indicator features."""
        features = []
        names = ['rsi', 'macd', 'macd_signal', 'atr', 'bb_upper', 'bb_lower', 'bb_middle']
        
        n = len(close)
        
        # Calculate indicators
        rsi = self._calculate_rsi(close)
        macd, macd_signal = self._calculate_macd(close)
        atr = self._calculate_atr(high, low, close)
        bb_middle, bb_upper, bb_lower = self._calculate_bollinger_bands(close)
        
        for i in range(n):
            feat = {}
            
            feat['rsi'] = rsi[i] if i < len(rsi) else 50.0
            feat['macd'] = macd[i] if i < len(macd) else 0.0
            feat['macd_signal'] = macd_signal[i] if i < len(macd_signal) else 0.0
            feat['atr'] = atr[i] if i < len(atr) else 0.0
            feat['bb_upper'] = bb_upper[i] if i < len(bb_upper) else close[i]
            feat['bb_lower'] = bb_lower[i] if i < len(bb_lower) else close[i]
            feat['bb_middle'] = bb_middle[i] if i < len(bb_middle) else close[i]
            
            features.append(feat)
        
        return features, names
    
    def _build_statistical_features(
        self,
        close: List[float],
        window: int,
    ) -> Tuple[List[Dict[str, float]], List[str]]:
        """Build statistical features."""
        features = []
        names = ['mean', 'median', 'std', 'skew', 'kurtosis']
        
        n = len(close)
        for i in range(n):
            feat = {}
            
            # Use window for statistics
            start = max(0, i - window + 1)
            window_data = close[start:i+1]
            window_len = len(window_data)
            
            if window_len >= 2:
                mean = sum(window_data) / window_len
                feat['mean'] = mean
                
                sorted_data = sorted(window_data)
                if window_len % 2 == 1:
                    feat['median'] = sorted_data[window_len // 2]
                else:
                    feat['median'] = (sorted_data[window_len // 2 - 1] + sorted_data[window_len // 2]) / 2
                
                variance = sum((x - mean) ** 2 for x in window_data) / (window_len - 1)
                std = math.sqrt(variance) if variance > 0 else 0.0
                feat['std'] = std
                
                # Skewness
                if std > 0:
                    skewness = sum(((x - mean) / std) ** 3 for x in window_data) / window_len
                    feat['skew'] = skewness
                else:
                    feat['skew'] = 0.0
                
                # Kurtosis
                if std > 0:
                    kurtosis = sum(((x - mean) / std) ** 4 for x in window_data) / window_len - 3
                    feat['kurtosis'] = kurtosis
                else:
                    feat['kurtosis'] = 0.0
            else:
                feat['mean'] = close[i]
                feat['median'] = close[i]
                feat['std'] = 0.0
                feat['skew'] = 0.0
                feat['kurtosis'] = 0.0
            
            features.append(feat)
        
        return features, names
    
    def _build_momentum_features(
        self,
        close: List[float],
        window: int,
    ) -> Tuple[List[Dict[str, float]], List[str]]:
        """Build momentum features."""
        features = []
        names = ['return_1', 'return_5', 'return_20', 'momentum', 'roc']
        
        n = len(close)
        for i in range(n):
            feat = {}
            
            # Returns over different periods
            feat['return_1'] = self._safe_return(close, i, 1)
            feat['return_5'] = self._safe_return(close, i, 5)
            feat['return_20'] = self._safe_return(close, i, 20)
            
            # Momentum (close - close[period])
            if i >= window:
                feat['momentum'] = close[i] - close[i - window]
            else:
                feat['momentum'] = 0.0
            
            # ROC (Rate of Change)
            if i >= window:
                prev = close[i - window]
                if prev != 0:
                    feat['roc'] = (close[i] - prev) / prev * 100
                else:
                    feat['roc'] = 0.0
            else:
                feat['roc'] = 0.0
            
            features.append(feat)
        
        return features, names
    
    def _build_volume_features(
        self,
        volume: List[float],
        close: List[float],
        window: int,
    ) -> Tuple[List[Dict[str, float]], List[str]]:
        """Build volume features."""
        features = []
        names = ['volume', 'volume_ratio', 'vwap']
        
        n = len(volume)
        for i in range(n):
            feat = {}
            
            vol = volume[i] if i < len(volume) else 0.0
            feat['volume'] = vol
            
            # Volume ratio (current / average)
            start = max(0, i - window + 1)
            window_len = i - start + 1
            avg_volume = sum(volume[start:i+1]) / window_len if window_len > 0 else 1.0
            if avg_volume > 0:
                feat['volume_ratio'] = vol / avg_volume
            else:
                feat['volume_ratio'] = 1.0
            
            # VWAP (Volume Weighted Average Price) - simplified
            # Accumulate volume and price*volume over window
            total_volume = sum(volume[start:i+1]) if window_len > 0 else 1.0
            total_vwap = sum(close[j] * volume[j] for j in range(start, i+1) if j < len(close))
            if total_volume > 0:
                feat['vwap'] = total_vwap / total_volume
            else:
                feat['vwap'] = close[i] if i < len(close) else 0.0
            
            features.append(feat)
        
        return features, names
    
    def _build_pattern_features(
        self,
        open_price: List[float],
        high: List[float],
        low: List[float],
        close: List[float],
    ) -> Tuple[List[Dict[str, float]], List[str]]:
        """Build pattern features."""
        features = []
        names = ['is_doji', 'is_hammer', 'is_bullish', 'is_bearish']
        
        n = len(close)
        for i in range(n):
            feat = {}
            
            open_val = open_price[i] if i < len(open_price) else close[i]
            high_val = high[i] if i < len(high) else close[i]
            low_val = low[i] if i < len(low) else close[i]
            close_val = close[i]
            
            total_range = high_val - low_val
            
            # Doji detection
            if total_range > 0 and close_val != 0:
                body = abs(close_val - open_val)
                feat['is_doji'] = 1.0 if body / total_range < 0.1 else 0.0
            else:
                feat['is_doji'] = 0.0
            
            # Hammer detection
            if total_range > 0 and close_val != 0:
                body = abs(close_val - open_val)
                lower_wick = min(open_val, close_val) - low_val
                upper_wick = high_val - max(open_val, close_val)
                is_hammer = (
                    body / total_range < 0.3 and
                    lower_wick / total_range > 0.6 and
                    upper_wick / total_range < 0.1
                )
                feat['is_hammer'] = 1.0 if is_hammer else 0.0
            else:
                feat['is_hammer'] = 0.0
            
            # Bullish/Bearish
            if close_val > open_val:
                feat['is_bullish'] = 1.0
                feat['is_bearish'] = 0.0
            elif close_val < open_val:
                feat['is_bullish'] = 0.0
                feat['is_bearish'] = 1.0
            else:
                feat['is_bullish'] = 0.0
                feat['is_bearish'] = 0.0
            
            features.append(feat)
        
        return features, names
    
    # ==========================================================================
    # INDICATOR CALCULATIONS
    # ==========================================================================
    
    def _calculate_rsi(self, prices: List[float]) -> List[float]:
        """Calculate RSI."""
        n = len(prices)
        if n < self.RSI_PERIOD + 1:
            return [50.0] * n
        
        rsi = [50.0] * n
        
        # Calculate gains and losses
        gains = [0.0] * n
        losses = [0.0] * n
        
        for i in range(1, n):
            change = prices[i] - prices[i-1]
            if change >= 0:
                gains[i] = change
                losses[i] = 0.0
            else:
                gains[i] = 0.0
                losses[i] = abs(change)
        
        # Calculate average gains and losses
        avg_gain = sum(gains[1:self.RSI_PERIOD+1]) / self.RSI_PERIOD
        avg_loss = sum(losses[1:self.RSI_PERIOD+1]) / self.RSI_PERIOD
        
        if avg_loss == 0:
            rsi[self.RSI_PERIOD] = 100.0
        else:
            rs = avg_gain / avg_loss
            rsi[self.RSI_PERIOD] = 100 - (100 / (1 + rs))
        
        # Calculate subsequent RSI values
        for i in range(self.RSI_PERIOD + 1, n):
            avg_gain = (avg_gain * (self.RSI_PERIOD - 1) + gains[i]) / self.RSI_PERIOD
            avg_loss = (avg_loss * (self.RSI_PERIOD - 1) + losses[i]) / self.RSI_PERIOD
            
            if avg_loss == 0:
                rsi[i] = 100.0
            else:
                rs = avg_gain / avg_loss
                rsi[i] = 100 - (100 / (1 + rs))
        
        return rsi
    
    def _calculate_macd(self, prices: List[float]) -> Tuple[List[float], List[float]]:
        """Calculate MACD and signal line."""
        n = len(prices)
        if n < self.MACD_SLOW:
            return [0.0] * n, [0.0] * n
        
        # Calculate EMAs
        fast_ema = self._calculate_ema(prices, self.MACD_FAST)
        slow_ema = self._calculate_ema(prices, self.MACD_SLOW)
        
        # MACD line
        macd = [0.0] * n
        for i in range(max(self.MACD_FAST, self.MACD_SLOW), n):
            macd[i] = fast_ema[i] - slow_ema[i]
        
        # Signal line (EMA of MACD)
        signal = self._calculate_ema(macd, self.MACD_SIGNAL)
        
        return macd, signal
    
    def _calculate_atr(self, high: List[float], low: List[float], close: List[float]) -> List[float]:
        """Calculate ATR."""
        n = len(high)
        if n < self.ATR_PERIOD + 1:
            return [0.0] * n
        
        tr = [0.0] * n
        
        for i in range(n):
            if i == 0:
                tr[i] = high[i] - low[i]
            else:
                hl = high[i] - low[i]
                hc = abs(high[i] - close[i-1])
                lc = abs(low[i] - close[i-1])
                tr[i] = max(hl, hc, lc)
        
        # Initial ATR is average of first N TR
        atr = [0.0] * n
        atr[self.ATR_PERIOD] = sum(tr[1:self.ATR_PERIOD+1]) / self.ATR_PERIOD
        
        # Subsequent ATR with smoothing
        for i in range(self.ATR_PERIOD + 1, n):
            atr[i] = (atr[i-1] * (self.ATR_PERIOD - 1) + tr[i]) / self.ATR_PERIOD
        
        return atr
    
    def _calculate_bollinger_bands(
        self,
        prices: List[float]
    ) -> Tuple[List[float], List[float], List[float]]:
        """Calculate Bollinger Bands."""
        n = len(prices)
        if n < self.BB_PERIOD:
            return prices.copy(), prices.copy(), prices.copy()
        
        middle = [0.0] * n
        upper = [0.0] * n
        lower = [0.0] * n
        
        for i in range(n):
            if i < self.BB_PERIOD - 1:
                middle[i] = prices[i]
                upper[i] = prices[i]
                lower[i] = prices[i]
            else:
                start = i - self.BB_PERIOD + 1
                window = prices[start:i+1]
                mean = sum(window) / self.BB_PERIOD
                variance = sum((x - mean) ** 2 for x in window) / self.BB_PERIOD
                std = math.sqrt(variance)
                
                middle[i] = mean
                upper[i] = mean + self.BB_STD * std
                lower[i] = mean - self.BB_STD * std
        
        return middle, upper, lower
    
    def _calculate_ema(self, prices: List[float], period: int) -> List[float]:
        """Calculate EMA."""
        n = len(prices)
        if n == 0:
            return []
        
        if n < period:
            return [prices[0]] * n
        
        ema = [0.0] * n
        multiplier = 2 / (period + 1)
        
        # Initial EMA is SMA
        ema[period - 1] = sum(prices[:period]) / period
        
        # Fill early values
        for i in range(period - 1):
            ema[i] = prices[i]
        
        # Subsequent EMAs
        for i in range(period, n):
            ema[i] = (prices[i] - ema[i-1]) * multiplier + ema[i-1]
        
        return ema
    
    def _safe_return(self, prices: List[float], idx: int, period: int) -> float:
        """Calculate return safely."""
        if idx >= period and prices[idx - period] != 0:
            return (prices[idx] - prices[idx - period]) / prices[idx - period]
        return 0.0


# ==============================================================================
# FACTORY FUNCTION
# ==============================================================================

def create_feature_builder(config: Config) -> FeatureBuilder:
    """
    Factory function for FeatureBuilder creation.
    
    Args:
        config: Application configuration
        
    Returns:
        FeatureBuilder instance
    """
    return FeatureBuilder(config)