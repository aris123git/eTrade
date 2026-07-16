"""
preprocessing/merger.py - Data Merger Module

RESPONSIBILITY:
Merge multiple data sources into unified datasets for analysis.

ARCHITECTURAL PRINCIPLES:
1. Pure data merging - No data storage, no I/O, no business logic
2. Handle different timeframes and symbols
3. Type-safe results with validation
4. Multiple merge strategies (inner, outer, left, right)

WHAT IT NEVER DOES:
- ❌ Connect to MT5
- ❌ Store data
- ❌ Download data
- ❌ Make trading decisions
- ❌ Analyze patterns

VERSION: 1.0.3
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


# ==============================================================================
# EXPORTS
# ==============================================================================

__all__ = [
    'MergeStrategy',
    'AlignMethod',
    'MergeResult',
    'DataMerger',
    'create_data_merger',
]


# ==============================================================================
# ENUMS
# ==============================================================================

class MergeStrategy(Enum):
    """Strategy for merging datasets."""
    INNER = "inner"     # Only keep matching timestamps
    OUTER = "outer"     # Keep all timestamps
    LEFT = "left"       # Keep all timestamps from left dataset
    RIGHT = "right"     # Keep all timestamps from right dataset
    CROSS = "cross"     # Cartesian product (careful!)


class AlignMethod(Enum):
    """Method for aligning timestamps."""
    EXACT = "exact"             # Exact match only
    NEAREST = "nearest"         # Nearest timestamp
    FORWARD = "forward"         # Forward fill (previous value)
    BACKWARD = "backward"       # Backward fill (next value)
    INTERPOLATE = "interpolate" # Linear interpolation
    NONE = "none"               # No alignment


# ==============================================================================
# DATA MODELS
# ==============================================================================

@dataclass
class MergeResult:
    """Result of merge operation."""
    merged_data: List[Dict[str, Any]]
    original_count: int
    merged_count: int
    dropped_count: int
    duplicate_count: int
    aligned_count: int
    strategy: MergeStrategy
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def success(self) -> bool:
        return self.merged_count > 0
    
    @property
    def retention_rate(self) -> float:
        if self.original_count == 0:
            return 0.0
        return self.merged_count / self.original_count
    
    def get_summary(self) -> Dict[str, Any]:
        """Get summary of merge operation."""
        return {
            'original_count': self.original_count,
            'merged_count': self.merged_count,
            'dropped_count': self.dropped_count,
            'duplicate_count': self.duplicate_count,
            'aligned_count': self.aligned_count,
            'retention_rate': self.retention_rate,
            'strategy': self.strategy.value,
            'success': self.success,
        }


# ==============================================================================
# DATA MERGER
# ==============================================================================

class DataMerger:
    """
    Data merging engine.
    
    Merges multiple data sources into unified datasets.
    """
    
    def __init__(self, config: Config):
        """
        Initialize the data merger.
        
        Args:
            config: Application configuration
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Merge defaults
        self._default_strategy = MergeStrategy.INNER
        self._default_align = AlignMethod.NEAREST
        self._time_tolerance = getattr(config, 'MERGE_TIME_TOLERANCE', 60)  # seconds
        
        self.logger.info("✅ DataMerger initialized")
    
    # ==========================================================================
    # PUBLIC METHODS
    # ==========================================================================
    
    def merge_datasets(
        self,
        left_data: List[Dict[str, Any]],
        right_data: List[Dict[str, Any]],
        left_key: str = 'timestamp',
        right_key: str = 'timestamp',
        strategy: Union[MergeStrategy, str] = MergeStrategy.INNER,
        align_method: Union[AlignMethod, str] = AlignMethod.NEAREST,
        suffix: Tuple[str, str] = ('_left', '_right'),
        fields: Optional[List[str]] = None,
    ) -> MergeResult:
        """
        Merge two datasets.
        
        Args:
            left_data: Left dataset (list of dicts)
            right_data: Right dataset (list of dicts)
            left_key: Key for matching in left dataset
            right_key: Key for matching in right dataset
            strategy: Merge strategy
            align_method: Method for aligning timestamps
            suffix: Suffix for overlapping fields
            fields: Fields to include from each dataset
            
        Returns:
            MergeResult object
        """
        if not left_data or not right_data:
            return MergeResult(
                merged_data=[],
                original_count=len(left_data) + len(right_data),
                merged_count=0,
                dropped_count=0,
                duplicate_count=0,
                aligned_count=0,
                strategy=self._parse_strategy(strategy),
                metadata={'error': 'Empty dataset provided'},
            )
        
        # Parse strategy and align method
        strategy = self._parse_strategy(strategy)
        align_method = self._parse_align_method(align_method)
        
        # Determine fields
        if fields is None:
            # Use all fields except keys
            left_fields = [k for k in left_data[0].keys() if k != left_key]
            right_fields = [k for k in right_data[0].keys() if k != right_key]
        else:
            left_fields = fields
            right_fields = fields
        
        self.logger.debug(
            f"Merging datasets: left={len(left_data)}, right={len(right_data)}"
        )
        
        try:
            # Validate data
            self._validate_data(left_data, left_key)
            self._validate_data(right_data, right_key)
            
            # Extract timestamps
            left_timestamps = self._extract_timestamps(left_data, left_key)
            right_timestamps = self._extract_timestamps(right_data, right_key)
            
            # Handle alignment
            if align_method != AlignMethod.EXACT:
                aligned_right = self._align_data(
                    right_data, right_key, left_timestamps, align_method
                )
            else:
                aligned_right = right_data
            
            # Merge based on strategy
            merged, dropped, aligned_count = self._merge_by_strategy(
                left_data, aligned_right, left_key, right_key,
                strategy, suffix, left_fields, right_fields
            )
            
            # Count duplicates
            duplicate_count = self._count_duplicates(merged, left_key)
            
            result = MergeResult(
                merged_data=merged,
                original_count=len(left_data) + len(right_data),
                merged_count=len(merged),
                dropped_count=dropped,
                duplicate_count=duplicate_count,
                aligned_count=aligned_count,
                strategy=strategy,
                metadata={
                    'left_count': len(left_data),
                    'right_count': len(right_data),
                    'left_key': left_key,
                    'right_key': right_key,
                    'align_method': align_method.value,
                },
            )
            
            self.logger.debug(
                f"Merge complete: {result.merged_count} rows, "
                f"{result.dropped_count} dropped, {result.aligned_count} aligned"
            )
            
            return result
            
        except Exception as e:
            self.logger.error(f"Merge failed: {e}")
            raise DataValidationError(f"Failed to merge datasets: {e}")
    
    def merge_multiple(
        self,
        datasets: List[List[Dict[str, Any]]],
        keys: List[str],
        strategy: Union[MergeStrategy, str] = MergeStrategy.INNER,
        align_method: Union[AlignMethod, str] = AlignMethod.NEAREST,
    ) -> MergeResult:
        """
        Merge multiple datasets.
        
        Args:
            datasets: List of datasets to merge
            keys: Keys for each dataset
            strategy: Merge strategy
            align_method: Method for aligning timestamps
            
        Returns:
            MergeResult object
        """
        if not datasets or len(datasets) < 2:
            return MergeResult(
                merged_data=datasets[0] if datasets else [],
                original_count=sum(len(d) for d in datasets),
                merged_count=len(datasets[0]) if datasets else 0,
                dropped_count=0,
                duplicate_count=0,
                aligned_count=0,
                strategy=self._parse_strategy(strategy),
                metadata={'error': 'Less than 2 datasets provided'},
            )
        
        # Filter out empty datasets
        non_empty_datasets = [d for d in datasets if d]
        if len(non_empty_datasets) < 2:
            return MergeResult(
                merged_data=non_empty_datasets[0] if non_empty_datasets else [],
                original_count=sum(len(d) for d in datasets),
                merged_count=len(non_empty_datasets[0]) if non_empty_datasets else 0,
                dropped_count=0,
                duplicate_count=0,
                aligned_count=0,
                strategy=self._parse_strategy(strategy),
                metadata={'error': 'Less than 2 non-empty datasets'},
            )
        
        # Ensure keys list is long enough
        if len(keys) < len(non_empty_datasets):
            # Extend with the last key
            last_key = keys[-1] if keys else 'timestamp'
            keys.extend([last_key] * (len(non_empty_datasets) - len(keys)))
        
        # Start with first dataset
        result = self.merge_datasets(
            non_empty_datasets[0], non_empty_datasets[1],
            keys[0], keys[1],
            strategy, align_method
        )
        
        if not result.success:
            return result
        
        merged = result.merged_data
        
        # Merge remaining datasets
        for i in range(2, len(non_empty_datasets)):
            result = self.merge_datasets(
                merged, non_empty_datasets[i],
                keys[0], keys[i],
                strategy, align_method
            )
            
            if not result.success:
                return result
            
            merged = result.merged_data
        
        return result
    
    def merge_by_symbol(
        self,
        symbol_data: Dict[str, List[Dict[str, Any]]],
        key: str = 'timestamp',
        strategy: Union[MergeStrategy, str] = MergeStrategy.INNER,
        align_method: Union[AlignMethod, str] = AlignMethod.NEAREST,
    ) -> MergeResult:
        """
        Merge data for multiple symbols.
        
        Args:
            symbol_data: Dictionary mapping symbol to data list
            key: Key for matching
            strategy: Merge strategy
            align_method: Method for aligning timestamps
            
        Returns:
            MergeResult object
        """
        if not symbol_data:
            return MergeResult(
                merged_data=[],
                original_count=0,
                merged_count=0,
                dropped_count=0,
                duplicate_count=0,
                aligned_count=0,
                strategy=self._parse_strategy(strategy),
                metadata={'error': 'No symbol data provided'},
            )
        
        # Filter out empty symbol data
        non_empty_symbols = {k: v for k, v in symbol_data.items() if v}
        if not non_empty_symbols:
            return MergeResult(
                merged_data=[],
                original_count=sum(len(d) for d in symbol_data.values()),
                merged_count=0,
                dropped_count=0,
                duplicate_count=0,
                aligned_count=0,
                strategy=self._parse_strategy(strategy),
                metadata={'error': 'All symbol data is empty'},
            )
        
        symbols = list(non_empty_symbols.keys())
        if len(symbols) < 2:
            return MergeResult(
                merged_data=non_empty_symbols[symbols[0]] if symbols else [],
                original_count=sum(len(d) for d in symbol_data.values()),
                merged_count=len(non_empty_symbols[symbols[0]]) if symbols else 0,
                dropped_count=0,
                duplicate_count=0,
                aligned_count=0,
                strategy=self._parse_strategy(strategy),
                metadata={'error': 'Only one non-empty symbol provided'},
            )
        
        # Start with first symbol
        result = self.merge_datasets(
            non_empty_symbols[symbols[0]], non_empty_symbols[symbols[1]],
            key, key, strategy, align_method
        )
        
        if not result.success:
            return result
        
        merged = result.merged_data
        
        # Merge remaining symbols
        for i in range(2, len(symbols)):
            result = self.merge_datasets(
                merged, non_empty_symbols[symbols[i]],
                key, key, strategy, align_method
            )
            
            if not result.success:
                return result
            
            merged = result.merged_data
        
        return result
    
    def validate_merge(
        self,
        left_data: List[Dict[str, Any]],
        right_data: List[Dict[str, Any]],
        left_key: str = 'timestamp',
        right_key: str = 'timestamp',
    ) -> Dict[str, Any]:
        """
        Validate merge compatibility before merging.
        
        Args:
            left_data: Left dataset
            right_data: Right dataset
            left_key: Key for left dataset
            right_key: Key for right dataset
            
        Returns:
            Dictionary with validation results
        """
        result = {
            'valid': True,
            'issues': [],
            'left_count': len(left_data),
            'right_count': len(right_data),
            'left_keys': set(),
            'right_keys': set(),
            'common_keys': set(),
            'left_only': set(),
            'right_only': set(),
        }
        
        if not left_data:
            result['valid'] = False
            result['issues'].append('Left dataset is empty')
            return result
        
        if not right_data:
            result['valid'] = False
            result['issues'].append('Right dataset is empty')
            return result
        
        # Check keys exist
        if left_key not in left_data[0]:
            result['valid'] = False
            result['issues'].append(f"Left key '{left_key}' not found in left data")
            return result
        
        if right_key not in right_data[0]:
            result['valid'] = False
            result['issues'].append(f"Right key '{right_key}' not found in right data")
            return result
        
        # Extract keys
        left_keys = set(self._extract_timestamps(left_data, left_key))
        right_keys = set(self._extract_timestamps(right_data, right_key))
        
        # Check key types are comparable
        if left_keys and right_keys:
            left_type = type(next(iter(left_keys)))
            right_type = type(next(iter(right_keys)))
            if left_type != right_type:
                result['valid'] = False
                result['issues'].append(
                    f"Key types mismatch: left={left_type.__name__}, right={right_type.__name__}"
                )
        
        result['left_keys'] = left_keys
        result['right_keys'] = right_keys
        result['common_keys'] = left_keys & right_keys
        result['left_only'] = left_keys - right_keys
        result['right_only'] = right_keys - left_keys
        
        # Check for empty intersection
        if not result['common_keys']:
            result['valid'] = False
            result['issues'].append('No common keys between datasets')
        
        return result
    
    # ==========================================================================
    # PRIVATE METHODS
    # ==========================================================================
    
    def _parse_strategy(self, strategy: Union[MergeStrategy, str]) -> MergeStrategy:
        """Parse merge strategy from string or enum."""
        if isinstance(strategy, MergeStrategy):
            return strategy
        if isinstance(strategy, str):
            try:
                return MergeStrategy(strategy.lower())
            except ValueError:
                self.logger.warning(f"Unknown strategy '{strategy}', using INNER")
                return MergeStrategy.INNER
        return MergeStrategy.INNER
    
    def _parse_align_method(self, method: Union[AlignMethod, str]) -> AlignMethod:
        """Parse align method from string or enum."""
        if isinstance(method, AlignMethod):
            return method
        if isinstance(method, str):
            try:
                return AlignMethod(method.lower())
            except ValueError:
                self.logger.warning(f"Unknown align method '{method}', using NEAREST")
                return AlignMethod.NEAREST
        return AlignMethod.NEAREST
    
    def _validate_data(self, data: List[Dict[str, Any]], key: str) -> None:
        """Validate data for merging."""
        if not data:
            return
        
        # Check all items have the key
        for i, item in enumerate(data):
            if key not in item:
                raise DataValidationError(
                    f"Item at index {i} missing key '{key}'"
                )
    
    def _extract_timestamps(self, data: List[Dict[str, Any]], key: str) -> List[Any]:
        """Extract timestamps from data."""
        return [item.get(key) for item in data if key in item]
    
    def _align_data(
        self,
        data: List[Dict[str, Any]],
        key: str,
        target_timestamps: List[Any],
        align_method: AlignMethod,
    ) -> List[Dict[str, Any]]:
        """
        Align data to target timestamps.
        
        Args:
            data: Data to align
            key: Key field
            target_timestamps: Target timestamps
            align_method: Alignment method
            
        Returns:
            Aligned data
        """
        if align_method == AlignMethod.EXACT:
            return data
        
        if align_method == AlignMethod.NONE:
            return data
        
        # Sort data by timestamp
        sorted_data = sorted(data, key=lambda x: self._to_datetime(x.get(key)))
        sorted_timestamps = [self._to_datetime(item.get(key)) for item in sorted_data]
        
        aligned = []
        
        for target in target_timestamps:
            target_dt = self._to_datetime(target)
            
            if align_method == AlignMethod.NEAREST:
                # Find nearest timestamp
                nearest_idx = self._find_nearest(sorted_timestamps, target_dt)
                if nearest_idx is not None:
                    aligned_item = sorted_data[nearest_idx].copy()
                    aligned_item[key] = target
                    aligned.append(aligned_item)
            
            elif align_method == AlignMethod.FORWARD:
                # Forward fill (previous value)
                idx = self._find_previous(sorted_timestamps, target_dt)
                if idx is not None:
                    aligned_item = sorted_data[idx].copy()
                    aligned_item[key] = target
                    aligned.append(aligned_item)
            
            elif align_method == AlignMethod.BACKWARD:
                # Backward fill (next value)
                idx = self._find_next(sorted_timestamps, target_dt)
                if idx is not None:
                    aligned_item = sorted_data[idx].copy()
                    aligned_item[key] = target
                    aligned.append(aligned_item)
            
            elif align_method == AlignMethod.INTERPOLATE:
                # Linear interpolation
                prev_idx, next_idx = self._find_neighbors(sorted_timestamps, target_dt)
                if prev_idx is not None and next_idx is not None:
                    prev_item = sorted_data[prev_idx]
                    next_item = sorted_data[next_idx]
                    aligned_item = self._interpolate_item(
                        prev_item, next_item, key, target_dt,
                        sorted_timestamps[prev_idx], sorted_timestamps[next_idx]
                    )
                    aligned.append(aligned_item)
                elif prev_idx is not None:
                    aligned_item = sorted_data[prev_idx].copy()
                    aligned_item[key] = target
                    aligned.append(aligned_item)
                elif next_idx is not None:
                    aligned_item = sorted_data[next_idx].copy()
                    aligned_item[key] = target
                    aligned.append(aligned_item)
        
        return aligned
    
    def _to_datetime(self, value: Any) -> Optional[datetime]:
        """
        Convert value to datetime.
        
        Returns:
            datetime object, or None if conversion fails
        """
        if value is None:
            return None
        if isinstance(value, datetime):
            return value
        if isinstance(value, (int, float)):
            return datetime.fromtimestamp(value)
        if isinstance(value, str):
            try:
                return datetime.fromisoformat(value)
            except ValueError:
                try:
                    return datetime.strptime(value, "%Y-%m-%d %H:%M:%S")
                except ValueError:
                    pass
        # Return None for non-convertible types
        return None
    
    def _find_nearest(self, timestamps: List[datetime], target: datetime) -> Optional[int]:
        """Find index of nearest timestamp."""
        if not timestamps:
            return None
        
        min_dist = float('inf')
        min_idx = None
        
        for i, ts in enumerate(timestamps):
            dist = abs((ts - target).total_seconds())
            if dist < min_dist:
                min_dist = dist
                min_idx = i
        
        # Check if within tolerance
        if min_dist <= self._time_tolerance:
            return min_idx
        
        # Return nearest even if outside tolerance (warn)
        self.logger.debug(
            f"Nearest timestamp outside tolerance: dist={min_dist:.2f}s, "
            f"tolerance={self._time_tolerance}s"
        )
        return min_idx
    
    def _find_previous(self, timestamps: List[datetime], target: datetime) -> Optional[int]:
        """Find index of previous timestamp."""
        if not timestamps:
            return None
        
        prev_idx = None
        for i, ts in enumerate(timestamps):
            if ts <= target:
                prev_idx = i
            else:
                break
        
        return prev_idx
    
    def _find_next(self, timestamps: List[datetime], target: datetime) -> Optional[int]:
        """Find index of next timestamp."""
        if not timestamps:
            return None
        
        for i, ts in enumerate(timestamps):
            if ts >= target:
                return i
        
        return None
    
    def _find_neighbors(
        self,
        timestamps: List[datetime],
        target: datetime
    ) -> Tuple[Optional[int], Optional[int]]:
        """Find previous and next indices."""
        prev_idx = self._find_previous(timestamps, target)
        next_idx = self._find_next(timestamps, target)
        
        # If prev and next are the same or adjacent, adjust
        if prev_idx is not None and next_idx is not None and prev_idx >= next_idx:
            if prev_idx > 0:
                prev_idx -= 1
            else:
                next_idx += 1
        
        return prev_idx, next_idx
    
    def _interpolate_item(
        self,
        prev_item: Dict[str, Any],
        next_item: Dict[str, Any],
        key: str,
        target: datetime,
        prev_ts: datetime,
        next_ts: datetime,
    ) -> Dict[str, Any]:
        """Interpolate between two items."""
        if prev_ts == next_ts:
            return prev_item.copy()
        
        # Calculate weight
        weight = (target - prev_ts).total_seconds() / (next_ts - prev_ts).total_seconds()
        weight = max(0.0, min(1.0, weight))
        
        # Interpolate numeric fields
        result = {}
        
        for field, value in prev_item.items():
            if field == key:
                result[field] = target
            elif isinstance(value, (int, float)) and field in next_item:
                next_val = next_item[field]
                if isinstance(next_val, (int, float)):
                    result[field] = value + weight * (next_val - value)
                else:
                    result[field] = value
            else:
                result[field] = value
        
        # Add any fields from next_item that are missing
        for field, value in next_item.items():
            if field not in result:
                result[field] = value
        
        return result
    
    def _merge_by_strategy(
        self,
        left: List[Dict[str, Any]],
        right: List[Dict[str, Any]],
        left_key: str,
        right_key: str,
        strategy: MergeStrategy,
        suffix: Tuple[str, str],
        left_fields: List[str],
        right_fields: List[str],
    ) -> Tuple[List[Dict[str, Any]], int, int]:
        """
        Merge data based on strategy.
        
        Returns:
            Tuple of (merged_data, dropped_count, aligned_count)
        """
        if not left or not right:
            return [], len(left) + len(right), 0
        
        # Build index for right data
        right_index = {}
        for item in right:
            key_val = item.get(right_key)
            if key_val not in right_index:
                right_index[key_val] = []
            right_index[key_val].append(item)
        
        # Handle CROSS strategy
        if strategy == MergeStrategy.CROSS:
            merged = []
            for left_item in left:
                for right_item in right:
                    merged_item = self._combine_items(
                        left_item, right_item, left_key, right_key,
                        suffix, left_fields, right_fields
                    )
                    merged.append(merged_item)
            return merged, 0, len(left) * len(right)
        
        merged = []
        dropped = 0
        aligned = 0
        seen_keys = set()
        
        # Create merged items
        for left_item in left:
            left_key_val = left_item.get(left_key)
            right_matches = right_index.get(left_key_val, [])
            
            if not right_matches:
                if strategy in (MergeStrategy.OUTER, MergeStrategy.LEFT):
                    # Keep left item with empty right fields
                    merged_item = self._combine_items(
                        left_item, {}, left_key, right_key, suffix, left_fields, right_fields
                    )
                    merged.append(merged_item)
                else:
                    dropped += 1
                continue
            
            for right_item in right_matches:
                merged_item = self._combine_items(
                    left_item, right_item, left_key, right_key,
                    suffix, left_fields, right_fields
                )
                merged.append(merged_item)
                seen_keys.add(left_key_val)
                aligned += 1
        
        # Handle RIGHT and OUTER strategy for right-only items
        if strategy in (MergeStrategy.OUTER, MergeStrategy.RIGHT):
            # Track right items that were matched
            right_matched_keys = seen_keys.copy()
            
            for right_item in right:
                right_key_val = right_item.get(right_key)
                if right_key_val not in right_matched_keys:
                    merged_item = self._combine_items(
                        {}, right_item, left_key, right_key,
                        suffix, left_fields, right_fields
                    )
                    merged.append(merged_item)
        
        return merged, dropped, aligned
    
    def _combine_items(
        self,
        left_item: Dict[str, Any],
        right_item: Dict[str, Any],
        left_key: str,
        right_key: str,
        suffix: Tuple[str, str],
        left_fields: List[str],
        right_fields: List[str],
    ) -> Dict[str, Any]:
        """Combine two items into one."""
        result = {}
        left_suffix, right_suffix = suffix
        
        # Add left fields
        if left_item:
            for field in left_fields:
                if field in left_item:
                    result[field] = left_item[field]
        
        # Add right fields with suffix if needed
        if right_item:
            for field in right_fields:
                if field in right_item:
                    # Check if field already exists
                    if field in result:
                        # Use suffix for right field
                        result[f"{field}{right_suffix}"] = right_item[field]
                        # Keep left field as is
                    else:
                        result[field] = right_item[field]
        
        # Add key
        left_val = left_item.get(left_key) if left_item else None
        right_val = right_item.get(right_key) if right_item else None
        
        if left_val is not None:
            result[left_key] = left_val
        elif right_val is not None:
            result[left_key] = right_val
        else:
            result[left_key] = None
        
        return result
    
    def _count_duplicates(self, data: List[Dict[str, Any]], key: str = 'timestamp') -> int:
        """Count duplicate entries based on key."""
        seen = set()
        duplicates = 0
        
        for item in data:
            val = item.get(key)
            if val is not None:
                # Use hash of value if possible
                try:
                    if isinstance(val, dict):
                        val_key = str(sorted(val.items()))
                    elif isinstance(val, list):
                        val_key = str(sorted(val))
                    else:
                        val_key = str(val)
                except Exception:
                    val_key = str(val)
                
                if val_key in seen:
                    duplicates += 1
                else:
                    seen.add(val_key)
        
        return duplicates


# ==============================================================================
# FACTORY FUNCTION
# ==============================================================================

def create_data_merger(config: Config) -> DataMerger:
    """
    Factory function for DataMerger creation.
    
    Args:
        config: Application configuration
        
    Returns:
        DataMerger instance
    """
    return DataMerger(config)