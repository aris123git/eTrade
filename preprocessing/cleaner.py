"""
preprocessing/cleaner.py - Data Cleaning Module

RESPONSIBILITY:
Clean and prepare raw market data for analysis.

ARCHITECTURAL PRINCIPLES:
1. Pure data cleaning - No data storage, no I/O, no business logic
2. Remove duplicates, outliers, and invalid data
3. Handle missing values with proper interpolation
4. Type-safe results with validation

WHAT IT NEVER DOES:
- ❌ Connect to MT5
- ❌ Store data
- ❌ Download data
- ❌ Make trading decisions
- ❌ Analyze patterns

VERSION: 1.0.2
"""

import logging
import math
import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional, Dict, Any, Tuple, Set, Union, Callable
from enum import Enum
from collections import defaultdict
from copy import deepcopy

from core.config import Config
from core.exceptions import DataValidationError, DataFormatError
from core.utils import is_valid_symbol, is_valid_timeframe


# ==============================================================================
# EXPORTS
# ==============================================================================

__all__ = [
    'CleanMethod',
    'OutlierMethod',
    'CleanResult',
    'DataCleaner',
    'create_data_cleaner',
]


# ==============================================================================
# ENUMS
# ==============================================================================

class CleanMethod(Enum):
    """Method for handling missing values."""
    DROP = "drop"
    FILL_ZERO = "fill_zero"
    FILL_MEAN = "fill_mean"
    FILL_MEDIAN = "fill_median"
    FILL_FORWARD = "fill_forward"
    FILL_BACKWARD = "fill_backward"
    INTERPOLATE_LINEAR = "interpolate_linear"
    INTERPOLATE_CUBIC = "interpolate_cubic"


class OutlierMethod(Enum):
    """Method for handling outliers."""
    DROP = "drop"
    CLIP = "clip"
    WINSORIZE = "winsorize"
    MEDIAN_REPLACE = "median_replace"


# ==============================================================================
# DATA MODELS
# ==============================================================================

@dataclass
class CleanResult:
    """Result of data cleaning operation."""
    original_count: int
    cleaned_count: int
    removed_count: int
    cleaned_data: List[Dict[str, Any]]
    duplicates_removed: int
    outliers_removed: int
    missing_filled: int
    missing_detected: int
    dropped_count: int
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def success(self) -> bool:
        return self.cleaned_count > 0
    
    @property
    def removal_rate(self) -> float:
        if self.original_count == 0:
            return 0.0
        return self.removed_count / self.original_count
    
    def get_summary(self) -> Dict[str, Any]:
        """Get summary of cleaning operation."""
        return {
            'original_count': self.original_count,
            'cleaned_count': self.cleaned_count,
            'removed_count': self.removed_count,
            'removal_rate': self.removal_rate,
            'duplicates_removed': self.duplicates_removed,
            'outliers_removed': self.outliers_removed,
            'missing_filled': self.missing_filled,
            'missing_detected': self.missing_detected,
            'dropped_count': self.dropped_count,
            'success': self.success,
        }


# ==============================================================================
# DATA CLEANER
# ==============================================================================

class DataCleaner:
    """
    Data cleaning engine.
    
    Cleans and prepares raw market data for analysis.
    """
    
    # Standard candle fields
    STANDARD_FIELDS = {'open', 'high', 'low', 'close', 'volume', 'timestamp'}
    TIME_FIELDS = {'timestamp', 'time', 'open_time'}
    
    def __init__(self, config: Config):
        """
        Initialize the data cleaner.
        
        Args:
            config: Application configuration
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Cleaning defaults
        self._default_clean_method = CleanMethod.INTERPOLATE_LINEAR
        self._default_outlier_method = OutlierMethod.CLIP
        self._default_zscore_threshold = 3.0
        self._default_missing_threshold = 0.5
        self._window_size = 20  # For rolling calculations
        self._min_window_size = 5  # Minimum valid window size
        
        # Configure from config
        if hasattr(config, 'CLEAN_METHOD'):
            try:
                self._default_clean_method = CleanMethod(config.CLEAN_METHOD)
            except ValueError:
                pass
        
        if hasattr(config, 'OUTLIER_METHOD'):
            try:
                self._default_outlier_method = OutlierMethod(config.OUTLIER_METHOD)
            except ValueError:
                pass
        
        if hasattr(config, 'ZSCORE_THRESHOLD'):
            self._default_zscore_threshold = config.ZSCORE_THRESHOLD
        
        if hasattr(config, 'MISSING_THRESHOLD'):
            self._default_missing_threshold = config.MISSING_THRESHOLD
        
        if hasattr(config, 'CLEAN_WINDOW_SIZE'):
            self._window_size = config.CLEAN_WINDOW_SIZE
        
        self.logger.info(
            f"✅ DataCleaner initialized: "
            f"clean_method={self._default_clean_method.value}, "
            f"outlier_method={self._default_outlier_method.value}"
        )
    
    # ==========================================================================
    # PUBLIC METHODS
    # ==========================================================================
    
    def clean_candles(
        self,
        candles: List[Dict[str, Any]],
        clean_method: Optional[Union[CleanMethod, str]] = None,
        outlier_method: Optional[Union[OutlierMethod, str]] = None,
        zscore_threshold: Optional[float] = None,
        missing_threshold: Optional[float] = None,
        fields: Optional[List[str]] = None,
    ) -> CleanResult:
        """
        Clean a list of candles.
        
        Args:
            candles: List of candle dictionaries
            clean_method: Method for handling missing values
            outlier_method: Method for handling outliers
            zscore_threshold: Z-score threshold for outlier detection
            missing_threshold: Maximum allowed missing data per candle
            fields: Fields to clean (default: all numeric fields)
            
        Returns:
            CleanResult object
            
        Raises:
            DataValidationError: If candles are invalid
        """
        if not candles:
            return CleanResult(
                original_count=0,
                cleaned_count=0,
                removed_count=0,
                cleaned_data=[],
                duplicates_removed=0,
                outliers_removed=0,
                missing_filled=0,
                missing_detected=0,
                dropped_count=0,
                metadata={'error': 'No candles provided'},
            )
        
        # Parse methods
        clean_method = self._parse_clean_method(clean_method)
        outlier_method = self._parse_outlier_method(outlier_method)
        zscore_threshold = zscore_threshold or self._default_zscore_threshold
        missing_threshold = missing_threshold or self._default_missing_threshold
        
        # Determine fields to clean
        if fields is None:
            fields = ['open', 'high', 'low', 'close', 'volume']
        
        self.logger.debug(
            f"Cleaning {len(candles)} candles "
            f"(clean={clean_method.value}, outlier={outlier_method.value})"
        )
        
        try:
            # Validate candles
            if not self._validate_candles(candles):
                raise DataValidationError("Invalid candles provided")
            
            # Create deep copy to avoid side effects
            working_candles = deepcopy(candles)
            
            # Remove duplicates
            unique_candles, duplicates_removed = self._remove_duplicates(working_candles)
            
            # Detect and handle missing data
            cleaned_candles, missing_filled, missing_detected, dropped = self._handle_missing(
                unique_candles, fields, clean_method, missing_threshold
            )
            
            # Remove outliers
            cleaned_candles, outliers_removed = self._handle_outliers(
                cleaned_candles, fields, outlier_method, zscore_threshold
            )
            
            result = CleanResult(
                original_count=len(candles),
                cleaned_count=len(cleaned_candles),
                removed_count=len(candles) - len(cleaned_candles),
                cleaned_data=cleaned_candles,
                duplicates_removed=duplicates_removed,
                outliers_removed=outliers_removed,
                missing_filled=missing_filled,
                missing_detected=missing_detected,
                dropped_count=dropped,
                metadata={
                    'fields_cleaned': fields,
                    'clean_method': clean_method.value,
                    'outlier_method': outlier_method.value,
                    'zscore_threshold': zscore_threshold,
                    'missing_threshold': missing_threshold,
                },
            )
            
            self.logger.debug(
                f"Cleaned {len(candles)} candles: "
                f"{result.cleaned_count} kept, {result.removed_count} removed"
            )
            
            return result
            
        except DataValidationError:
            raise
        except Exception as e:
            self.logger.error(f"Cleaning failed: {e}")
            raise DataValidationError(f"Failed to clean candles: {e}")
    
    def clean_prices(
        self,
        prices: List[float],
        clean_method: Optional[Union[CleanMethod, str]] = None,
        outlier_method: Optional[Union[OutlierMethod, str]] = None,
        zscore_threshold: Optional[float] = None,
    ) -> CleanResult:
        """
        Clean a list of prices.
        
        Args:
            prices: List of price values
            clean_method: Method for handling missing values
            outlier_method: Method for handling outliers
            zscore_threshold: Z-score threshold for outlier detection
            
        Returns:
            CleanResult object
        """
        if not prices:
            return CleanResult(
                original_count=0,
                cleaned_count=0,
                removed_count=0,
                cleaned_data=[],
                duplicates_removed=0,
                outliers_removed=0,
                missing_filled=0,
                missing_detected=0,
                dropped_count=0,
                metadata={'error': 'No prices provided'},
            )
        
        try:
            # Convert to dicts with standard fields
            data = []
            for i, p in enumerate(prices):
                data.append({
                    'open': float(p),
                    'high': float(p),
                    'low': float(p),
                    'close': float(p),
                    'volume': 0,
                    'timestamp': int(i),
                })
            
            # Clean
            result = self.clean_candles(
                candles=data,
                clean_method=clean_method,
                outlier_method=outlier_method,
                zscore_threshold=zscore_threshold,
                fields=['open', 'high', 'low', 'close'],
            )
            
            # Extract cleaned prices (use close as primary)
            result.cleaned_data = [c['close'] for c in result.cleaned_data]
            
            return result
            
        except Exception as e:
            self.logger.error(f"Price cleaning failed: {e}")
            raise DataValidationError(f"Failed to clean prices: {e}")
    
    def clean_volume(
        self,
        volumes: List[int],
        clean_method: Optional[Union[CleanMethod, str]] = None,
        outlier_method: Optional[Union[OutlierMethod, str]] = None,
        zscore_threshold: Optional[float] = None,
    ) -> CleanResult:
        """
        Clean a list of volumes.
        
        Args:
            volumes: List of volume values
            clean_method: Method for handling missing values
            outlier_method: Method for handling outliers
            zscore_threshold: Z-score threshold for outlier detection
            
        Returns:
            CleanResult object
        """
        if not volumes:
            return CleanResult(
                original_count=0,
                cleaned_count=0,
                removed_count=0,
                cleaned_data=[],
                duplicates_removed=0,
                outliers_removed=0,
                missing_filled=0,
                missing_detected=0,
                dropped_count=0,
                metadata={'error': 'No volumes provided'},
            )
        
        try:
            # Convert to dicts with standard fields
            data = []
            for i, v in enumerate(volumes):
                data.append({
                    'open': 0,
                    'high': 0,
                    'low': 0,
                    'close': 0,
                    'volume': int(v),
                    'timestamp': int(i),
                })
            
            # Clean
            result = self.clean_candles(
                candles=data,
                clean_method=clean_method,
                outlier_method=outlier_method,
                zscore_threshold=zscore_threshold,
                fields=['volume'],
            )
            
            # Extract cleaned volumes
            result.cleaned_data = [c['volume'] for c in result.cleaned_data]
            
            return result
            
        except Exception as e:
            self.logger.error(f"Volume cleaning failed: {e}")
            raise DataValidationError(f"Failed to clean volumes: {e}")
    
    def normalize_symbol(self, symbol: str) -> str:
        """
        Normalize a symbol name.
        
        Args:
            symbol: Symbol name to normalize
            
        Returns:
            Normalized symbol name
        """
        if not symbol:
            return symbol
        
        # Remove common suffixes
        normalized = symbol
        normalized = re.sub(r'\.(cash|pro|mini|ecn|raw|a|b|c|d|e|f|g|h|i|j|k|l|m|n|o|p|q|r|s|t|u|v|w|x|y|z)$', '', normalized)
        normalized = re.sub(r'_(swapfree|islamic|demo|live|test|practice|real|sim)$', '', normalized)
        
        return normalized
    
    def detect_missing(self, candles: List[Dict[str, Any]], fields: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Detect missing values in candles.
        
        Args:
            candles: List of candle dictionaries
            fields: Fields to check (default: all standard fields)
            
        Returns:
            Dictionary with missing value statistics
        """
        if not candles:
            return {'total_candles': 0, 'missing': {}, 'total_missing': 0}
        
        if fields is None:
            fields = ['open', 'high', 'low', 'close', 'volume']
        
        missing_count = {f: 0 for f in fields}
        total_missing = 0
        candles_with_missing = 0
        
        for candle in candles:
            has_missing = False
            for field in fields:
                if self._is_missing(candle, field):
                    missing_count[field] += 1
                    total_missing += 1
                    has_missing = True
            if has_missing:
                candles_with_missing += 1
        
        return {
            'total_candles': len(candles),
            'candles_with_missing': candles_with_missing,
            'missing': missing_count,
            'total_missing': total_missing,
            'missing_rate': total_missing / (len(candles) * len(fields)) if candles else 0,
        }
    
    # ==========================================================================
    # PRIVATE METHODS
    # ==========================================================================
    
    def _parse_clean_method(self, method: Optional[Union[CleanMethod, str]]) -> CleanMethod:
        """Parse clean method from string or enum."""
        if method is None:
            return self._default_clean_method
        if isinstance(method, CleanMethod):
            return method
        if isinstance(method, str):
            try:
                return CleanMethod(method.lower())
            except ValueError:
                self.logger.warning(f"Unknown clean method '{method}', using default")
                return self._default_clean_method
        return self._default_clean_method
    
    def _parse_outlier_method(self, method: Optional[Union[OutlierMethod, str]]) -> OutlierMethod:
        """Parse outlier method from string or enum."""
        if method is None:
            return self._default_outlier_method
        if isinstance(method, OutlierMethod):
            return method
        if isinstance(method, str):
            try:
                return OutlierMethod(method.lower())
            except ValueError:
                self.logger.warning(f"Unknown outlier method '{method}', using default")
                return self._default_outlier_method
        return self._default_outlier_method
    
    def _is_missing(self, candle: Dict[str, Any], field: str) -> bool:
        """Check if a field is missing in a candle."""
        if field not in candle:
            return True
        value = candle[field]
        if value is None:
            return True
        if isinstance(value, float) and math.isnan(value):
            return True
        if isinstance(value, str) and value.strip() == '':
            return True
        return False
    
    def _validate_candles(self, candles: List[Dict[str, Any]]) -> bool:
        """
        Validate candles for cleaning.
        
        Args:
            candles: List of candle dictionaries
            
        Returns:
            True if valid, False otherwise
        """
        if not candles:
            return False
        
        required_fields = {'open', 'high', 'low', 'close'}
        
        for i, candle in enumerate(candles):
            # Check required fields
            if not all(field in candle for field in required_fields):
                self.logger.debug(f"Candle {i} missing required fields: {candle.keys()}")
                return False
            
            # Check values
            try:
                open_price = float(candle['open'])
                high = float(candle['high'])
                low = float(candle['low'])
                close = float(candle['close'])
                
                # Basic validation
                if high < low:
                    self.logger.debug(f"Candle {i} has high < low: {high} < {low}")
                    return False
                if open_price <= 0 or high <= 0 or low <= 0 or close <= 0:
                    self.logger.debug(f"Candle {i} has invalid prices")
                    return False
                
                # Validate volume if present
                if 'volume' in candle:
                    volume = candle['volume']
                    if volume is not None:
                        try:
                            vol = float(volume)
                            if vol < 0:
                                self.logger.debug(f"Candle {i} has negative volume: {vol}")
                                return False
                        except (ValueError, TypeError):
                            self.logger.debug(f"Candle {i} has invalid volume")
                            return False
                    
            except (ValueError, TypeError):
                self.logger.debug(f"Candle {i} has invalid numeric values")
                return False
        
        return True
    
    def _remove_duplicates(self, candles: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], int]:
        """
        Remove duplicate candles based on timestamp.
        
        Args:
            candles: List of candle dictionaries
            
        Returns:
            Tuple of (unique_candles, duplicates_removed)
        """
        if not candles:
            return [], 0
        
        # Track seen timestamps
        seen = set()
        unique = []
        duplicates = 0
        
        for candle in candles:
            # Try different time fields
            ts = None
            for time_field in self.TIME_FIELDS:
                if time_field in candle:
                    ts = candle[time_field]
                    # Convert to hashable type if needed
                    if isinstance(ts, dict):
                        ts = str(ts)
                    elif isinstance(ts, datetime):
                        ts = ts.isoformat()
                    break
            
            # If no timestamp, use a combination of fields as fallback
            if ts is None:
                ts = f"{candle.get('open')}_{candle.get('close')}_{candle.get('high')}_{candle.get('low')}"
            
            # Check if seen
            if ts in seen:
                duplicates += 1
                continue
            
            seen.add(ts)
            unique.append(candle)
        
        return unique, duplicates
    
    def _handle_missing(
        self,
        candles: List[Dict[str, Any]],
        fields: List[str],
        clean_method: CleanMethod,
        missing_threshold: float,
    ) -> Tuple[List[Dict[str, Any]], int, int, int]:
        """
        Handle missing values in candles.
        
        Args:
            candles: List of candle dictionaries
            fields: Fields to check for missing values
            clean_method: Method for handling missing values
            missing_threshold: Maximum allowed missing data per candle
            
        Returns:
            Tuple of (cleaned_candles, missing_filled, missing_detected, dropped)
        """
        if not candles:
            return [], 0, 0, 0
        
        cleaned = []
        missing_filled = 0
        missing_detected = 0
        dropped = 0
        
        for i, candle in enumerate(candles):
            # Detect missing values
            missing_fields = []
            for field in fields:
                if self._is_missing(candle, field):
                    missing_fields.append(field)
            
            missing_detected += len(missing_fields)
            
            # Check if too many missing values
            if len(missing_fields) / len(fields) > missing_threshold:
                dropped += 1
                continue
            
            # Create a copy to avoid modifying original
            cleaned_candle = candle.copy()
            
            # Handle DROP method
            if clean_method == CleanMethod.DROP and missing_fields:
                dropped += 1
                continue
            
            # Fill missing values
            for field in missing_fields:
                if clean_method == CleanMethod.FILL_ZERO:
                    cleaned_candle[field] = 0
                    missing_filled += 1
                elif clean_method == CleanMethod.FILL_MEAN:
                    mean_value = self._calculate_mean(candles, field)
                    cleaned_candle[field] = mean_value if mean_value is not None else 0
                    missing_filled += 1
                elif clean_method == CleanMethod.FILL_MEDIAN:
                    median_value = self._calculate_median(candles, field)
                    cleaned_candle[field] = median_value if median_value is not None else 0
                    missing_filled += 1
                elif clean_method == CleanMethod.FILL_FORWARD:
                    # Use previous candle value
                    if len(cleaned) > 0 and field in cleaned[-1]:
                        cleaned_candle[field] = cleaned[-1][field]
                        missing_filled += 1
                    else:
                        # Use mean if no previous value
                        mean_value = self._calculate_mean(candles, field)
                        cleaned_candle[field] = mean_value if mean_value is not None else 0
                        missing_filled += 1
                elif clean_method == CleanMethod.FILL_BACKWARD:
                    # Use next candle value
                    next_value = self._find_next_value(candles, i, field)
                    if next_value is not None:
                        cleaned_candle[field] = next_value
                        missing_filled += 1
                    else:
                        # Use mean if no next value
                        mean_value = self._calculate_mean(candles, field)
                        cleaned_candle[field] = mean_value if mean_value is not None else 0
                        missing_filled += 1
                elif clean_method in (CleanMethod.INTERPOLATE_LINEAR, CleanMethod.INTERPOLATE_CUBIC):
                    # Linear interpolation with proper neighbor finding
                    interpolated = self._interpolate_value(candles, i, field)
                    if interpolated is not None:
                        cleaned_candle[field] = interpolated
                        missing_filled += 1
                    else:
                        # Fallback to mean
                        mean_value = self._calculate_mean(candles, field)
                        cleaned_candle[field] = mean_value if mean_value is not None else 0
                        missing_filled += 1
                else:
                    # Default: fill with 0
                    cleaned_candle[field] = 0
                    missing_filled += 1
            
            cleaned.append(cleaned_candle)
        
        return cleaned, missing_filled, missing_detected, dropped
    
    def _handle_outliers(
        self,
        candles: List[Dict[str, Any]],
        fields: List[str],
        outlier_method: OutlierMethod,
        zscore_threshold: float,
    ) -> Tuple[List[Dict[str, Any]], int]:
        """
        Handle outliers in candles.
        
        Args:
            candles: List of candle dictionaries
            fields: Fields to check for outliers
            outlier_method: Method for handling outliers
            zscore_threshold: Z-score threshold for outlier detection
            
        Returns:
            Tuple of (cleaned_candles, outliers_removed)
        """
        if not candles:
            return [], 0
        
        cleaned = []
        outliers_removed = 0
        
        for i, candle in enumerate(candles):
            cleaned_candle = candle.copy()
            is_outlier = False
            
            for field in fields:
                if self._is_missing(candle, field):
                    continue
                
                value = candle[field]
                if not isinstance(value, (int, float)) or math.isnan(value):
                    continue
                
                # Calculate rolling z-score (window-based)
                zscore = self._calculate_rolling_zscore(candles, i, field, value)
                
                if abs(zscore) > zscore_threshold:
                    is_outlier = True
                    
                    if outlier_method == OutlierMethod.DROP:
                        break
                    elif outlier_method == OutlierMethod.CLIP:
                        # Clip to threshold
                        if zscore > zscore_threshold:
                            cleaned_candle[field] = self._get_rolling_percentile(candles, i, field, 0.95)
                        else:
                            cleaned_candle[field] = self._get_rolling_percentile(candles, i, field, 0.05)
                    elif outlier_method == OutlierMethod.MEDIAN_REPLACE:
                        median_value = self._calculate_rolling_median(candles, i, field)
                        if median_value is not None:
                            cleaned_candle[field] = median_value
                    elif outlier_method == OutlierMethod.WINSORIZE:
                        # Replace with nearest non-outlier from window
                        cleaned_candle[field] = self._winsorize_value(candles, i, field, value, zscore_threshold)
            
            if is_outlier and outlier_method == OutlierMethod.DROP:
                outliers_removed += 1
                continue
            
            cleaned.append(cleaned_candle)
        
        return cleaned, outliers_removed
    
    def _calculate_mean(self, candles: List[Dict[str, Any]], field: str) -> Optional[float]:
        """Calculate mean of a field across candles."""
        values = []
        for c in candles:
            if field in c and not self._is_missing(c, field):
                val = c[field]
                if isinstance(val, (int, float)) and not math.isnan(val):
                    values.append(float(val))
        if not values:
            return None
        return sum(values) / len(values)
    
    def _calculate_median(self, candles: List[Dict[str, Any]], field: str) -> Optional[float]:
        """Calculate median of a field across candles."""
        values = []
        for c in candles:
            if field in c and not self._is_missing(c, field):
                val = c[field]
                if isinstance(val, (int, float)) and not math.isnan(val):
                    values.append(float(val))
        if not values:
            return None
        sorted_values = sorted(values)
        n = len(sorted_values)
        if n % 2 == 1:
            return sorted_values[n // 2]
        return (sorted_values[n // 2 - 1] + sorted_values[n // 2]) / 2
    
    def _calculate_std(self, candles: List[Dict[str, Any]], field: str, mean: float) -> float:
        """Calculate standard deviation (sample) of a field."""
        values = []
        for c in candles:
            if field in c and not self._is_missing(c, field):
                val = c[field]
                if isinstance(val, (int, float)) and not math.isnan(val):
                    values.append(float(val))
        if len(values) < 2:
            return 0.0
        
        variance = sum((v - mean) ** 2 for v in values) / (len(values) - 1)
        return math.sqrt(variance) if variance > 0 else 0.0
    
    def _calculate_rolling_zscore(self, candles: List[Dict[str, Any]], idx: int, field: str, value: float) -> float:
        """
        Calculate rolling z-score for a value at position idx.
        
        Uses a window around idx for mean and std.
        """
        half_window = self._window_size // 2
        start = max(0, idx - half_window)
        end = min(len(candles), idx + half_window + 1)
        
        window_candles = candles[start:end]
        
        mean = self._calculate_mean(window_candles, field)
        if mean is None:
            return 0.0
        
        std = self._calculate_std(window_candles, field, mean)
        if std == 0:
            return 0.0
        
        return (value - mean) / std
    
    def _calculate_rolling_median(self, candles: List[Dict[str, Any]], idx: int, field: str) -> Optional[float]:
        """Calculate rolling median for a field at position idx."""
        half_window = self._window_size // 2
        start = max(0, idx - half_window)
        end = min(len(candles), idx + half_window + 1)
        
        window_candles = candles[start:end]
        return self._calculate_median(window_candles, field)
    
    def _get_rolling_percentile(self, candles: List[Dict[str, Any]], idx: int, field: str, percentile: float) -> float:
        """Get rolling percentile for a field at position idx."""
        half_window = self._window_size // 2
        start = max(0, idx - half_window)
        end = min(len(candles), idx + half_window + 1)
        
        if end - start < self._min_window_size:
            # Fallback to global percentile
            return self._get_percentile(candles, field, percentile)
        
        window_candles = candles[start:end]
        values = []
        for c in window_candles:
            if field in c and not self._is_missing(c, field):
                val = c[field]
                if isinstance(val, (int, float)) and not math.isnan(val):
                    values.append(float(val))
        
        if not values:
            return 0.0
        
        sorted_values = sorted(values)
        idx_perc = int(len(sorted_values) * percentile)
        if idx_perc >= len(sorted_values):
            return sorted_values[-1]
        return sorted_values[idx_perc]
    
    def _get_percentile(self, candles: List[Dict[str, Any]], field: str, percentile: float) -> float:
        """Get global percentile for a field."""
        values = []
        for c in candles:
            if field in c and not self._is_missing(c, field):
                val = c[field]
                if isinstance(val, (int, float)) and not math.isnan(val):
                    values.append(float(val))
        
        if not values:
            return 0.0
        
        sorted_values = sorted(values)
        idx_perc = int(len(sorted_values) * percentile)
        if idx_perc >= len(sorted_values):
            return sorted_values[-1]
        return sorted_values[idx_perc]
    
    def _winsorize_value(
        self,
        candles: List[Dict[str, Any]],
        idx: int,
        field: str,
        value: float,
        zscore_threshold: float,
    ) -> float:
        """Winsorize a value using rolling window."""
        half_window = self._window_size // 2
        start = max(0, idx - half_window)
        end = min(len(candles), idx + half_window + 1)
        
        if end - start < self._min_window_size:
            return value
        
        window_candles = candles[start:end]
        values = []
        for c in window_candles:
            if field in c and not self._is_missing(c, field):
                val = c[field]
                if isinstance(val, (int, float)) and not math.isnan(val):
                    values.append(float(val))
        
        if len(values) < self._min_window_size:
            return value
        
        sorted_values = sorted(values)
        p5 = sorted_values[int(len(sorted_values) * 0.05)]
        p95 = sorted_values[int(len(sorted_values) * 0.95)]
        
        if value < p5:
            return p5
        if value > p95:
            return p95
        return value
    
    def _find_next_value(self, candles: List[Dict[str, Any]], start_idx: int, field: str) -> Optional[float]:
        """Find next non-null value after start_idx."""
        for i in range(start_idx + 1, len(candles)):
            if not self._is_missing(candles[i], field):
                val = candles[i][field]
                if isinstance(val, (int, float)) and not math.isnan(val):
                    return float(val)
        return None
    
    def _find_prev_value(self, candles: List[Dict[str, Any]], start_idx: int, field: str) -> Optional[float]:
        """Find previous non-null value before start_idx."""
        for i in range(start_idx - 1, -1, -1):
            if not self._is_missing(candles[i], field):
                val = candles[i][field]
                if isinstance(val, (int, float)) and not math.isnan(val):
                    return float(val)
        return None
    
    def _find_prev_index(self, candles: List[Dict[str, Any]], start_idx: int, field: str) -> int:
        """Find index of previous non-null value before start_idx."""
        for i in range(start_idx - 1, -1, -1):
            if not self._is_missing(candles[i], field):
                return i
        return -1
    
    def _find_next_index(self, candles: List[Dict[str, Any]], start_idx: int, field: str) -> int:
        """Find index of next non-null value after start_idx."""
        for i in range(start_idx + 1, len(candles)):
            if not self._is_missing(candles[i], field):
                return i
        return -1
    
    def _interpolate_value(self, candles: List[Dict[str, Any]], idx: int, field: str) -> Optional[float]:
        """
        Interpolate missing value using linear interpolation between nearest valid neighbors.
        """
        # Find previous and next valid values with indices
        prev_idx = self._find_prev_index(candles, idx, field)
        next_idx = self._find_next_index(candles, idx, field)
        
        if prev_idx == -1 and next_idx == -1:
            return None
        
        if prev_idx == -1:
            return float(candles[next_idx][field])
        
        if next_idx == -1:
            return float(candles[prev_idx][field])
        
        prev_val = float(candles[prev_idx][field])
        next_val = float(candles[next_idx][field])
        
        # Linear interpolation
        if next_idx > prev_idx:
            weight = (idx - prev_idx) / (next_idx - prev_idx)
            return prev_val + weight * (next_val - prev_val)
        
        # Fallback
        return (prev_val + next_val) / 2


# ==============================================================================
# FACTORY FUNCTION
# ==============================================================================

def create_data_cleaner(config: Config) -> DataCleaner:
    """
    Factory function for DataCleaner creation.
    
    Args:
        config: Application configuration
        
    Returns:
        DataCleaner instance
    """
    return DataCleaner(config)