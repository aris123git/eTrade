"""
preprocessing/resampler.py - Data Resampler Module

RESPONSIBILITY:
Resample market data to different timeframes.

ARCHITECTURAL PRINCIPLES:
1. Pure data resampling - No data storage, no I/O, no business logic
2. Convert between timeframes (M1 → M5, M5 → H1, etc.)
3. Type-safe results with validation
4. Multiple aggregation methods (OHLC, volume, etc.)

WHAT IT NEVER DOES:
- ❌ Connect to MT5
- ❌ Store data
- ❌ Download data
- ❌ Make trading decisions
- ❌ Analyze patterns

VERSION: 1.0.0
"""

import logging
import math
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Any, Tuple, Set, Union, Callable
from enum import Enum
from collections import defaultdict

from core.config import Config
from core.exceptions import DataValidationError
from core.utils import to_datetime, format_datetime


# ==============================================================================
# EXPORTS
# ==============================================================================

__all__ = [
    'ResampleMethod',
    'OHLCMethod',
    'ResampleResult',
    'DataResampler',
    'create_data_resampler',
]


# ==============================================================================
# ENUMS
# ==============================================================================

class ResampleMethod(Enum):
    """Method for resampling data."""
    OHLC = "ohlc"               # Open, High, Low, Close
    AGGREGATE = "aggregate"     # Sum/Average aggregation
    SAMPLE = "sample"           # Sample at regular intervals
    DECIMATE = "decimate"       # Decimate (keep every Nth)
    CUSTOM = "custom"           # Custom aggregation


class OHLCMethod(Enum):
    """Method for OHLC calculation."""
    STANDARD = "standard"       # Standard OHLC from candles
    TICK = "tick"               # OHLC from tick data
    VOLUME = "volume"           # Volume-weighted OHLC
    TIME = "time"               # Time-weighted OHLC


# ==============================================================================
# DATA MODELS
# ==============================================================================

@dataclass
class ResampleResult:
    """Result of resample operation."""
    resampled_data: List[Dict[str, Any]]
    original_count: int
    resampled_count: int
    dropped_count: int
    method: ResampleMethod
    timeframe_from: str
    timeframe_to: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def success(self) -> bool:
        return self.resampled_count > 0
    
    @property
    def reduction_ratio(self) -> float:
        if self.original_count == 0:
            return 0.0
        return self.resampled_count / self.original_count
    
    def get_summary(self) -> Dict[str, Any]:
        """Get summary of resample operation."""
        return {
            'original_count': self.original_count,
            'resampled_count': self.resampled_count,
            'dropped_count': self.dropped_count,
            'reduction_ratio': self.reduction_ratio,
            'method': self.method.value,
            'timeframe_from': self.timeframe_from,
            'timeframe_to': self.timeframe_to,
            'success': self.success,
        }


# ==============================================================================
# DATA RESAMPLER
# ==============================================================================

class DataResampler:
    """
    Data resampling engine.
    
    Resamples market data to different timeframes.
    """
    
    # Timeframe mappings (in seconds)
    TIMEFRAME_SECONDS = {
        'M1': 60,
        'M5': 300,
        'M15': 900,
        'M30': 1800,
        'H1': 3600,
        'H4': 14400,
        'D1': 86400,
        'W1': 604800,
        'MN1': 2592000,
    }
    
    # Timeframe hierarchy (for reduction ratio calculation)
    TIMEFRAME_ORDER = ['M1', 'M5', 'M15', 'M30', 'H1', 'H4', 'D1', 'W1', 'MN1']
    
    def __init__(self, config: Config):
        """
        Initialize the data resampler.
        
        Args:
            config: Application configuration
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Resample defaults
        self._default_ohlc_method = OHLCMethod.STANDARD
        self._default_resample_method = ResampleMethod.OHLC
        self._time_tolerance = getattr(config, 'RESAMPLE_TIME_TOLERANCE', 30)  # seconds
        
        self.logger.info("✅ DataResampler initialized")
    
    # ==========================================================================
    # PUBLIC METHODS
    # ==========================================================================
    
    def resample(
        self,
        candles: List[Dict[str, Any]],
        from_timeframe: str,
        to_timeframe: str,
        method: Union[ResampleMethod, str] = ResampleMethod.OHLC,
        ohlc_method: Union[OHLCMethod, str] = OHLCMethod.STANDARD,
        timestamp_field: str = 'timestamp',
        volume_field: str = 'volume',
    ) -> ResampleResult:
        """
        Resample candles to a different timeframe.
        
        Args:
            candles: List of candle dictionaries
            from_timeframe: Source timeframe (e.g., 'M5')
            to_timeframe: Target timeframe (e.g., 'H1')
            method: Resample method
            ohlc_method: OHLC calculation method
            timestamp_field: Field name for timestamp
            volume_field: Field name for volume
            
        Returns:
            ResampleResult object
        """
        if not candles:
            return ResampleResult(
                resampled_data=[],
                original_count=0,
                resampled_count=0,
                dropped_count=0,
                method=self._parse_method(method),
                timeframe_from=from_timeframe,
                timeframe_to=to_timeframe,
                metadata={'error': 'No candles provided'},
            )
        
        # Parse method
        method = self._parse_method(method)
        ohlc_method = self._parse_ohlc_method(ohlc_method)
        
        # Validate timeframes
        if not self._is_valid_timeframe(from_timeframe):
            raise DataValidationError(f"Invalid from_timeframe: {from_timeframe}")
        
        if not self._is_valid_timeframe(to_timeframe):
            raise DataValidationError(f"Invalid to_timeframe: {to_timeframe}")
        
        # Check if target is higher timeframe
        from_seconds = self.TIMEFRAME_SECONDS[from_timeframe]
        to_seconds = self.TIMEFRAME_SECONDS[to_timeframe]
        
        if to_seconds <= from_seconds:
            self.logger.warning(
                f"Target timeframe {to_timeframe} is not higher than {from_timeframe}"
            )
            # Return original data
            return ResampleResult(
                resampled_data=candles.copy(),
                original_count=len(candles),
                resampled_count=len(candles),
                dropped_count=0,
                method=method,
                timeframe_from=from_timeframe,
                timeframe_to=to_timeframe,
                metadata={'note': 'Target timeframe not higher, returned original'},
            )
        
        self.logger.debug(
            f"Resampling {len(candles)} candles from {from_timeframe} to {to_timeframe}"
        )
        
        try:
            # Validate candles
            if not self._validate_candles(candles):
                raise DataValidationError("Invalid candles provided")
            
            # Calculate interval ratio
            ratio = to_seconds / from_seconds
            self.logger.debug(f"Interval ratio: {ratio:.2f}x")
            
            # Group candles by target timeframe
            grouped = self._group_by_timeframe(
                candles, timestamp_field, to_timeframe, from_timeframe
            )
            
            # Resample each group
            resampled = self._apply_resample_method(
                grouped, method, ohlc_method, timestamp_field, volume_field
            )
            
            # Count dropped (incomplete groups)
            dropped = len(candles) - sum(len(g) for g in grouped.values())
            
            result = ResampleResult(
                resampled_data=resampled,
                original_count=len(candles),
                resampled_count=len(resampled),
                dropped_count=dropped,
                method=method,
                timeframe_from=from_timeframe,
                timeframe_to=to_timeframe,
                metadata={
                    'ratio': ratio,
                    'groups': len(grouped),
                    'ohlc_method': ohlc_method.value,
                    'from_seconds': from_seconds,
                    'to_seconds': to_seconds,
                },
            )
            
            self.logger.debug(
                f"Resample complete: {result.resampled_count} candles, "
                f"{result.dropped_count} dropped (ratio: {result.reduction_ratio:.2f})"
            )
            
            return result
            
        except Exception as e:
            self.logger.error(f"Resample failed: {e}")
            raise DataValidationError(f"Failed to resample data: {e}")
    
    def upscale(
        self,
        candles: List[Dict[str, Any]],
        from_timeframe: str,
        to_timeframe: str,
        method: str = 'interpolate',
    ) -> ResampleResult:
        """
        Upscale data to a higher timeframe.
        
        Args:
            candles: List of candle dictionaries
            from_timeframe: Source timeframe
            to_timeframe: Target timeframe (must be lower than from)
            method: Interpolation method ('interpolate', 'forward', 'backward')
            
        Returns:
            ResampleResult object
        """
        # This is a stub for future implementation
        # Upscaling (lower timeframe) requires interpolation
        raise NotImplementedError("Upscaling not yet implemented")
    
    def downsample_to(
        self,
        candles: List[Dict[str, Any]],
        target_timeframe: str,
    ) -> ResampleResult:
        """
        Downsample to a specific timeframe.
        
        Args:
            candles: List of candle dictionaries
            target_timeframe: Target timeframe
            
        Returns:
            ResampleResult object
        """
        if not candles:
            return ResampleResult(
                resampled_data=[],
                original_count=0,
                resampled_count=0,
                dropped_count=0,
                method=self._default_resample_method,
                timeframe_from='unknown',
                timeframe_to=target_timeframe,
                metadata={'error': 'No candles provided'},
            )
        
        # Detect source timeframe
        from_timeframe = self._detect_timeframe(candles)
        if not from_timeframe:
            from_timeframe = 'M5'  # Default fallback
        
        return self.resample(candles, from_timeframe, target_timeframe)
    
    def aggregate_by_volume(
        self,
        candles: List[Dict[str, Any]],
        volume_threshold: float,
        timestamp_field: str = 'timestamp',
    ) -> ResampleResult:
        """
        Aggregate candles by volume threshold.
        
        Args:
            candles: List of candle dictionaries
            volume_threshold: Minimum volume per aggregated candle
            timestamp_field: Field name for timestamp
            
        Returns:
            ResampleResult object
        """
        if not candles or volume_threshold <= 0:
            return ResampleResult(
                resampled_data=candles.copy() if candles else [],
                original_count=len(candles) if candles else 0,
                resampled_count=len(candles) if candles else 0,
                dropped_count=0,
                method=self._default_resample_method,
                timeframe_from='unknown',
                timeframe_to='volume',
                metadata={'error': 'Invalid volume threshold'},
            )
        
        self.logger.debug(f"Aggregating by volume threshold: {volume_threshold}")
        
        resampled = []
        current_group = []
        current_volume = 0
        
        for candle in candles:
            volume = candle.get('volume', 0)
            current_group.append(candle)
            current_volume += volume
            
            if current_volume >= volume_threshold:
                # Create aggregated candle
                agg = self._aggregate_group(current_group, timestamp_field)
                if agg:
                    resampled.append(agg)
                current_group = []
                current_volume = 0
        
        # Add remaining group if any
        if current_group:
            agg = self._aggregate_group(current_group, timestamp_field)
            if agg:
                resampled.append(agg)
        
        return ResampleResult(
            resampled_data=resampled,
            original_count=len(candles),
            resampled_count=len(resampled),
            dropped_count=len(candles) - len(resampled),
            method=ResampleMethod.AGGREGATE,
            timeframe_from='unknown',
            timeframe_to='volume',
            metadata={
                'volume_threshold': volume_threshold,
                'groups': len(resampled),
            },
        )
    
    def detect_timeframe(self, candles: List[Dict[str, Any]]) -> Optional[str]:
        """
        Detect the timeframe of candle data.
        
        Args:
            candles: List of candle dictionaries
            
        Returns:
            Detected timeframe string or None
        """
        if len(candles) < 2:
            return None
        
        try:
            timestamps = []
            for candle in candles:
                ts = candle.get('timestamp')
                if ts is not None:
                    dt = to_datetime(ts)
                    if dt:
                        timestamps.append(dt)
            
            if len(timestamps) < 2:
                return None
            
            # Calculate average gap
            gaps = []
            for i in range(1, min(len(timestamps), 100)):
                gap = (timestamps[i] - timestamps[i-1]).total_seconds()
                if gap > 0:
                    gaps.append(gap)
            
            if not gaps:
                return None
            
            avg_gap = sum(gaps) / len(gaps)
            
            # Find closest timeframe
            best_match = None
            best_diff = float('inf')
            
            for tf, seconds in self.TIMEFRAME_SECONDS.items():
                diff = abs(avg_gap - seconds)
                # Allow 30% tolerance
                if diff / seconds < 0.3:
                    if diff < best_diff:
                        best_diff = diff
                        best_match = tf
            
            return best_match
            
        except Exception:
            return None
    
    # ==========================================================================
    # PRIVATE METHODS
    # ==========================================================================
    
    def _parse_method(self, method: Union[ResampleMethod, str]) -> ResampleMethod:
        """Parse resample method from string or enum."""
        if isinstance(method, ResampleMethod):
            return method
        if isinstance(method, str):
            try:
                return ResampleMethod(method.lower())
            except ValueError:
                self.logger.warning(f"Unknown method '{method}', using OHLC")
                return ResampleMethod.OHLC
        return self._default_resample_method
    
    def _parse_ohlc_method(self, method: Union[OHLCMethod, str]) -> OHLCMethod:
        """Parse OHLC method from string or enum."""
        if isinstance(method, OHLCMethod):
            return method
        if isinstance(method, str):
            try:
                return OHLCMethod(method.lower())
            except ValueError:
                self.logger.warning(f"Unknown OHLC method '{method}', using STANDARD")
                return OHLCMethod.STANDARD
        return self._default_ohlc_method
    
    def _is_valid_timeframe(self, timeframe: str) -> bool:
        """Check if timeframe is valid."""
        return timeframe in self.TIMEFRAME_SECONDS
    
    def _validate_candles(self, candles: List[Dict[str, Any]]) -> bool:
        """Validate candles for resampling."""
        if not candles:
            return False
        
        required_fields = {'open', 'high', 'low', 'close', 'timestamp'}
        
        for i, candle in enumerate(candles):
            if not all(field in candle for field in required_fields):
                self.logger.debug(f"Candle {i} missing required fields")
                return False
        
        return True
    
    def _detect_timeframe(self, candles: List[Dict[str, Any]]) -> Optional[str]:
        """Detect timeframe from candles."""
        return self.detect_timeframe(candles)
    
    def _group_by_timeframe(
        self,
        candles: List[Dict[str, Any]],
        timestamp_field: str,
        target_timeframe: str,
        source_timeframe: str,
    ) -> Dict[datetime, List[Dict[str, Any]]]:
        """
        Group candles by target timeframe.
        
        Returns:
            Dictionary mapping period start to list of candles
        """
        target_seconds = self.TIMEFRAME_SECONDS[target_timeframe]
        source_seconds = self.TIMEFRAME_SECONDS[source_timeframe]
        
        # Calculate target period size in source candles
        period_size = max(1, int(target_seconds / source_seconds))
        
        grouped = defaultdict(list)
        
        for i, candle in enumerate(candles):
            # Calculate period index
            period_idx = i // period_size
            period_start = i - (i % period_size)
            
            # Use the timestamp of the first candle in the period
            if period_start < len(candles):
                key = candles[period_start].get(timestamp_field)
                if key is not None:
                    dt = to_datetime(key)
                    if dt:
                        grouped[dt].append(candle)
        
        return grouped
    
    def _apply_resample_method(
        self,
        grouped: Dict[datetime, List[Dict[str, Any]]],
        method: ResampleMethod,
        ohlc_method: OHLCMethod,
        timestamp_field: str,
        volume_field: str,
    ) -> List[Dict[str, Any]]:
        """
        Apply resample method to groups.
        
        Returns:
            List of resampled candles
        """
        resampled = []
        
        for period_start, group in grouped.items():
            if method == ResampleMethod.OHLC:
                candle = self._create_ohlc(group, ohlc_method, period_start, volume_field)
            elif method == ResampleMethod.AGGREGATE:
                candle = self._aggregate_group(group, timestamp_field)
            elif method == ResampleMethod.SAMPLE:
                candle = self._sample_group(group, period_start)
            elif method == ResampleMethod.DECIMATE:
                candle = self._decimate_group(group, period_start)
            else:
                candle = self._create_ohlc(group, ohlc_method, period_start, volume_field)
            
            if candle:
                resampled.append(candle)
        
        return resampled
    
    def _create_ohlc(
        self,
        group: List[Dict[str, Any]],
        ohlc_method: OHLCMethod,
        period_start: datetime,
        volume_field: str,
    ) -> Optional[Dict[str, Any]]:
        """Create OHLC candle from group."""
        if not group:
            return None
        
        # Get first and last candles
        first = group[0]
        last = group[-1]
        
        # Calculate OHLC
        if ohlc_method == OHLCMethod.STANDARD:
            open_price = first.get('open', 0)
            high = max(c.get('high', 0) for c in group)
            low = min(c.get('low', 0) for c in group)
            close = last.get('close', 0)
        elif ohlc_method == OHLCMethod.TICK:
            # Use first tick as open, last as close
            open_price = first.get('open', 0)
            high = max(c.get('high', 0) for c in group)
            low = min(c.get('low', 0) for c in group)
            close = last.get('close', 0)
        elif ohlc_method == OHLCMethod.VOLUME:
            # Volume-weighted OHLC
            total_volume = sum(c.get(volume_field, 0) for c in group)
            if total_volume > 0:
                open_price = sum(c.get('open', 0) * c.get(volume_field, 0) for c in group) / total_volume
                high = max(c.get('high', 0) for c in group)
                low = min(c.get('low', 0) for c in group)
                close = sum(c.get('close', 0) * c.get(volume_field, 0) for c in group) / total_volume
            else:
                open_price = first.get('open', 0)
                high = max(c.get('high', 0) for c in group)
                low = min(c.get('low', 0) for c in group)
                close = last.get('close', 0)
        else:
            open_price = first.get('open', 0)
            high = max(c.get('high', 0) for c in group)
            low = min(c.get('low', 0) for c in group)
            close = last.get('close', 0)
        
        # Calculate volume
        volume = sum(c.get(volume_field, 0) for c in group)
        
        return {
            'timestamp': period_start,
            'open': open_price,
            'high': high,
            'low': low,
            'close': close,
            'volume': volume,
        }
    
    def _aggregate_group(
        self,
        group: List[Dict[str, Any]],
        timestamp_field: str,
    ) -> Optional[Dict[str, Any]]:
        """Aggregate group (sum/average)."""
        if not group:
            return None
        
        # Aggregate numeric fields
        aggregated = {}
        numeric_fields = {'open', 'high', 'low', 'close', 'volume'}
        
        for field in numeric_fields:
            values = [c.get(field, 0) for c in group if field in c]
            if values:
                if field == 'volume':
                    aggregated[field] = sum(values)
                else:
                    aggregated[field] = sum(values) / len(values)
            else:
                aggregated[field] = 0
        
        # Add timestamp
        if group:
            ts = group[0].get(timestamp_field)
            if ts is not None:
                aggregated[timestamp_field] = ts
        
        return aggregated
    
    def _sample_group(
        self,
        group: List[Dict[str, Any]],
        period_start: datetime,
    ) -> Optional[Dict[str, Any]]:
        """Sample group (take first element)."""
        if not group:
            return None
        
        candle = group[0].copy()
        candle['timestamp'] = period_start
        return candle
    
    def _decimate_group(
        self,
        group: List[Dict[str, Any]],
        period_start: datetime,
    ) -> Optional[Dict[str, Any]]:
        """Decimate group (take average of all elements)."""
        return self._aggregate_group(group, 'timestamp')


# ==============================================================================
# FACTORY FUNCTION
# ==============================================================================

def create_data_resampler(config: Config) -> DataResampler:
    """
    Factory function for DataResampler creation.
    
    Args:
        config: Application configuration
        
    Returns:
        DataResampler instance
    """
    return DataResampler(config)