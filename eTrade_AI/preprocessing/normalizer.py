"""
preprocessing/normalizer.py - Data Normalizer Module

RESPONSIBILITY:
Normalize and scale market data for AI/ML models.

ARCHITECTURAL PRINCIPLES:
1. Pure data normalization - No data storage, no I/O, no business logic
2. Scale data to consistent ranges
3. Type-safe results with validation
4. Multiple normalization methods (min-max, z-score, robust, etc.)

WHAT IT NEVER DOES:
- ❌ Connect to MT5
- ❌ Store data
- ❌ Download data
- ❌ Make trading decisions
- ❌ Analyze patterns (only normalizes)

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
from core.exceptions import DataValidationError


# ==============================================================================
# EXPORTS
# ==============================================================================

__all__ = [
    'NormalizeMethod',
    'NormalizeResult',
    'DataNormalizer',
    'NormalizerState',
    'create_data_normalizer',
]


# ==============================================================================
# ENUMS
# ==============================================================================

class NormalizeMethod(Enum):
    """Method for normalizing data."""
    MIN_MAX = "min_max"             # Min-Max scaling to [0, 1]
    Z_SCORE = "z_score"             # Z-score standardization
    ROBUST = "robust"               # Robust scaling using percentiles
    MAX_ABS = "max_abs"             # Max absolute scaling to [-1, 1]
    MEAN = "mean"                   # Mean normalization
    UNIT_VECTOR = "unit_vector"     # Unit vector normalization
    LOG = "log"                     # Log transformation
    SQUARE_ROOT = "square_root"     # Square root transformation


# ==============================================================================
# DATA MODELS
# ==============================================================================

@dataclass
class NormalizerState:
    """State of a normalizer for a specific field."""
    method: NormalizeMethod
    min_val: Optional[float] = None
    max_val: Optional[float] = None
    mean: Optional[float] = None
    std: Optional[float] = None
    q25: Optional[float] = None
    q75: Optional[float] = None
    max_abs: Optional[float] = None
    sum_val: Optional[float] = None
    count: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'method': self.method.value,
            'min_val': self.min_val,
            'max_val': self.max_val,
            'mean': self.mean,
            'std': self.std,
            'q25': self.q25,
            'q75': self.q75,
            'max_abs': self.max_abs,
            'sum_val': self.sum_val,
            'count': self.count,
        }


@dataclass
class NormalizeResult:
    """Result of normalization operation."""
    normalized_data: List[Dict[str, float]]
    field_names: List[str]
    state: Dict[str, NormalizerState]
    original_count: int
    method: NormalizeMethod
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def success(self) -> bool:
        return len(self.normalized_data) > 0
    
    def get_summary(self) -> Dict[str, Any]:
        """Get summary of normalization operation."""
        return {
            'original_count': self.original_count,
            'normalized_count': len(self.normalized_data),
            'method': self.method.value,
            'fields': list(self.state.keys()),
            'success': self.success,
        }


# ==============================================================================
# DATA NORMALIZER
# ==============================================================================

class DataNormalizer:
    """
    Data normalizer engine.
    
    Normalizes and scales market data for AI/ML models.
    """
    
    def __init__(self, config: Config):
        """
        Initialize the data normalizer.
        
        Args:
            config: Application configuration
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Normalization defaults
        self._default_method = NormalizeMethod.MIN_MAX
        self._eps = 1e-10  # Small epsilon to avoid division by zero
        
        self.logger.info("✅ DataNormalizer initialized")
    
    # ==========================================================================
    # PUBLIC METHODS
    # ==========================================================================
    
    def normalize(
        self,
        data: List[Dict[str, float]],
        method: Union[NormalizeMethod, str] = NormalizeMethod.MIN_MAX,
        fields: Optional[List[str]] = None,
        state: Optional[Dict[str, NormalizerState]] = None,
        fit: bool = True,
    ) -> NormalizeResult:
        """
        Normalize data using specified method.
        
        Args:
            data: List of dictionaries with numeric values
            method: Normalization method
            fields: Fields to normalize (default: all numeric fields)
            state: Existing normalizer state (for transform only)
            fit: If True, fit the normalizer; if False, use provided state
            
        Returns:
            NormalizeResult object
        """
        if not data:
            return NormalizeResult(
                normalized_data=[],
                field_names=[],
                state={},
                original_count=0,
                method=self._parse_method(method),
                metadata={'error': 'No data provided'},
            )
        
        # Parse method
        method = self._parse_method(method)
        
        # Determine fields
        if fields is None:
            # Use all numeric fields from first item
            fields = [k for k, v in data[0].items() if isinstance(v, (int, float))]
        
        self.logger.debug(
            f"Normalizing {len(data)} records, {len(fields)} fields, "
            f"method={method.value}, fit={fit}"
        )
        
        try:
            # Validate data
            if not self._validate_data(data, fields):
                raise DataValidationError("Invalid data provided")
            
            # If not fitting, state must be provided
            if not fit and state is None:
                raise DataValidationError("State required for transform without fit")
            
            # Initialize or use provided state
            if fit:
                normalizer_state = self._fit(data, fields, method)
            else:
                normalizer_state = state
            
            # Transform data
            normalized, field_names = self._transform(data, fields, normalizer_state, method)
            
            result = NormalizeResult(
                normalized_data=normalized,
                field_names=field_names,
                state=normalizer_state,
                original_count=len(data),
                method=method,
                metadata={
                    'fields_normalized': fields,
                    'fit': fit,
                },
            )
            
            self.logger.debug(
                f"Normalization complete: {len(normalized)} records, "
                f"{len(field_names)} fields"
            )
            
            return result
            
        except Exception as e:
            self.logger.error(f"Normalization failed: {e}")
            raise DataValidationError(f"Failed to normalize data: {e}")
    
    def fit_transform(
        self,
        data: List[Dict[str, float]],
        method: Union[NormalizeMethod, str] = NormalizeMethod.MIN_MAX,
        fields: Optional[List[str]] = None,
    ) -> NormalizeResult:
        """
        Fit and transform data in one step.
        
        Args:
            data: List of dictionaries with numeric values
            method: Normalization method
            fields: Fields to normalize
            
        Returns:
            NormalizeResult object
        """
        return self.normalize(data, method, fields, fit=True)
    
    def transform(
        self,
        data: List[Dict[str, float]],
        state: Dict[str, NormalizerState],
        fields: Optional[List[str]] = None,
    ) -> NormalizeResult:
        """
        Transform data using existing state.
        
        Args:
            data: List of dictionaries with numeric values
            state: Normalizer state from previous fit
            fields: Fields to normalize
            
        Returns:
            NormalizeResult object
        """
        return self.normalize(data, state=state, fields=fields, fit=False)
    
    def inverse_transform(
        self,
        normalized_data: List[Dict[str, float]],
        state: Dict[str, NormalizerState],
        fields: Optional[List[str]] = None,
    ) -> NormalizeResult:
        """
        Inverse transform normalized data back to original scale.
        
        Args:
            normalized_data: List of normalized dictionaries
            state: Normalizer state from previous fit
            fields: Fields to inverse transform
            
        Returns:
            NormalizeResult object
        """
        if not normalized_data:
            return NormalizeResult(
                normalized_data=[],
                field_names=[],
                state={},
                original_count=0,
                method=NormalizeMethod.MIN_MAX,
                metadata={'error': 'No data provided'},
            )
        
        if not state:
            raise DataValidationError("State required for inverse transform")
        
        # Determine fields
        if fields is None:
            fields = list(state.keys())
        
        self.logger.debug(f"Inverse transforming {len(normalized_data)} records")
        
        try:
            inversed = []
            field_names = []
            
            for record in normalized_data:
                inv_record = {}
                for field in fields:
                    if field in record:
                        val = record[field]
                        field_state = state.get(field)
                        if field_state:
                            inv_val = self._inverse_transform_value(val, field_state)
                            inv_record[field] = inv_val
                        else:
                            inv_record[field] = val
                        if field not in field_names:
                            field_names.append(field)
                inversed.append(inv_record)
            
            result = NormalizeResult(
                normalized_data=inversed,
                field_names=field_names,
                state=state,
                original_count=len(normalized_data),
                method=NormalizeMethod.MIN_MAX,
                metadata={
                    'inverse': True,
                },
            )
            
            return result
            
        except Exception as e:
            self.logger.error(f"Inverse transform failed: {e}")
            raise DataValidationError(f"Failed to inverse transform data: {e}")
    
    def save_state(self, state: Dict[str, NormalizerState], path: str) -> bool:
        """
        Save normalizer state to file.
        
        Args:
            state: Normalizer state
            path: File path
            
        Returns:
            True if saved successfully
        """
        import json
        
        try:
            serialized = {}
            for field, field_state in state.items():
                serialized[field] = field_state.to_dict()
            
            with open(path, 'w') as f:
                json.dump(serialized, f, indent=2)
            
            self.logger.info(f"✅ State saved to {path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to save state: {e}")
            return False
    
    def load_state(self, path: str) -> Dict[str, NormalizerState]:
        """
        Load normalizer state from file.
        
        Args:
            path: File path
            
        Returns:
            Normalizer state dictionary
        """
        import json
        
        try:
            with open(path, 'r') as f:
                data = json.load(f)
            
            state = {}
            for field, field_data in data.items():
                method = NormalizeMethod(field_data['method'])
                state[field] = NormalizerState(
                    method=method,
                    min_val=field_data.get('min_val'),
                    max_val=field_data.get('max_val'),
                    mean=field_data.get('mean'),
                    std=field_data.get('std'),
                    q25=field_data.get('q25'),
                    q75=field_data.get('q75'),
                    max_abs=field_data.get('max_abs'),
                    sum_val=field_data.get('sum_val'),
                    count=field_data.get('count', 0),
                )
            
            self.logger.info(f"✅ State loaded from {path}")
            return state
            
        except Exception as e:
            self.logger.error(f"Failed to load state: {e}")
            return {}
    
    # ==========================================================================
    # PRIVATE METHODS
    # ==========================================================================
    
    def _parse_method(self, method: Union[NormalizeMethod, str]) -> NormalizeMethod:
        """Parse normalization method from string or enum."""
        if isinstance(method, NormalizeMethod):
            return method
        if isinstance(method, str):
            try:
                return NormalizeMethod(method.lower())
            except ValueError:
                self.logger.warning(f"Unknown method '{method}', using MIN_MAX")
                return NormalizeMethod.MIN_MAX
        return self._default_method
    
    def _validate_data(self, data: List[Dict[str, float]], fields: List[str]) -> bool:
        """Validate data for normalization."""
        if not data:
            return False
        
        for i, record in enumerate(data):
            if not isinstance(record, dict):
                return False
            for field in fields:
                if field not in record:
                    return False
                val = record[field]
                if not isinstance(val, (int, float)):
                    return False
                if math.isnan(val) or math.isinf(val):
                    return False
        
        return True
    
    def _fit(
        self,
        data: List[Dict[str, float]],
        fields: List[str],
        method: NormalizeMethod,
    ) -> Dict[str, NormalizerState]:
        """Fit normalizer to data."""
        state = {}
        
        for field in fields:
            values = [record[field] for record in data]
            field_state = self._fit_field(values, method)
            state[field] = field_state
        
        return state
    
    def _fit_field(self, values: List[float], method: NormalizeMethod) -> NormalizerState:
        """Fit normalizer to a single field."""
        n = len(values)
        if n == 0:
            return NormalizerState(method=method)
        
        state = NormalizerState(method=method, count=n)
        
        if method == NormalizeMethod.MIN_MAX:
            state.min_val = min(values)
            state.max_val = max(values)
        
        elif method == NormalizeMethod.Z_SCORE:
            state.mean = sum(values) / n
            variance = sum((v - state.mean) ** 2 for v in values) / (n - 1)
            state.std = math.sqrt(variance) if variance > 0 else 1.0
        
        elif method == NormalizeMethod.ROBUST:
            sorted_values = sorted(values)
            state.q25 = self._percentile(sorted_values, 25)
            state.q75 = self._percentile(sorted_values, 75)
            state.min_val = min(values)
            state.max_val = max(values)
        
        elif method == NormalizeMethod.MAX_ABS:
            state.max_abs = max(abs(v) for v in values)
            if state.max_abs == 0:
                state.max_abs = 1.0
        
        elif method == NormalizeMethod.MEAN:
            state.mean = sum(values) / n
        
        elif method == NormalizeMethod.UNIT_VECTOR:
            sum_sq = sum(v ** 2 for v in values)
            state.max_abs = math.sqrt(sum_sq) if sum_sq > 0 else 1.0
        
        elif method == NormalizeMethod.LOG:
            # For log transformation, values must be > 0
            positive_values = [v for v in values if v > 0]
            if positive_values:
                state.mean = sum(math.log(v) for v in positive_values) / len(positive_values)
                state.min_val = min(positive_values)
                state.max_val = max(positive_values)
            else:
                state.mean = 0.0
                state.min_val = 0.001
                state.max_val = 1.0
        
        elif method == NormalizeMethod.SQUARE_ROOT:
            state.min_val = min(v for v in values if v >= 0)
            state.max_val = max(v for v in values if v >= 0)
        
        return state
    
    def _transform(
        self,
        data: List[Dict[str, float]],
        fields: List[str],
        state: Dict[str, NormalizerState],
        method: NormalizeMethod,
    ) -> Tuple[List[Dict[str, float]], List[str]]:
        """Transform data using fitted normalizer."""
        transformed = []
        field_names = []
        
        for record in data:
            trans_record = {}
            for field in fields:
                if field in record:
                    val = record[field]
                    field_state = state.get(field)
                    if field_state:
                        trans_val = self._transform_value(val, field_state)
                        trans_record[field] = trans_val
                    else:
                        trans_record[field] = val
                    if field not in field_names:
                        field_names.append(field)
            transformed.append(trans_record)
        
        return transformed, field_names
    
    def _transform_value(self, value: float, state: NormalizerState) -> float:
        """Transform a single value using state."""
        method = state.method
        eps = self._eps
        
        if method == NormalizeMethod.MIN_MAX:
            min_val = state.min_val or 0
            max_val = state.max_val or 1
            denom = max_val - min_val
            if abs(denom) < eps:
                return 0.5
            return (value - min_val) / denom
        
        elif method == NormalizeMethod.Z_SCORE:
            mean = state.mean or 0
            std = state.std or 1
            if abs(std) < eps:
                return 0
            return (value - mean) / std
        
        elif method == NormalizeMethod.ROBUST:
            q25 = state.q25 or 0
            q75 = state.q75 or 1
            iqr = q75 - q25
            if abs(iqr) < eps:
                return 0
            return (value - q25) / iqr
        
        elif method == NormalizeMethod.MAX_ABS:
            max_abs = state.max_abs or 1
            if abs(max_abs) < eps:
                return 0
            return value / max_abs
        
        elif method == NormalizeMethod.MEAN:
            mean = state.mean or 0
            if abs(mean) < eps:
                return value
            return value / mean
        
        elif method == NormalizeMethod.UNIT_VECTOR:
            norm = state.max_abs or 1
            if abs(norm) < eps:
                return 0
            return value / norm
        
        elif method == NormalizeMethod.LOG:
            if value <= 0:
                return 0.0
            return math.log(value)
        
        elif method == NormalizeMethod.SQUARE_ROOT:
            if value < 0:
                return 0.0
            return math.sqrt(value)
        
        return value
    
    def _inverse_transform_value(self, value: float, state: NormalizerState) -> float:
        """Inverse transform a single value using state."""
        method = state.method
        eps = self._eps
        
        if method == NormalizeMethod.MIN_MAX:
            min_val = state.min_val or 0
            max_val = state.max_val or 1
            return value * (max_val - min_val) + min_val
        
        elif method == NormalizeMethod.Z_SCORE:
            mean = state.mean or 0
            std = state.std or 1
            return value * std + mean
        
        elif method == NormalizeMethod.ROBUST:
            q25 = state.q25 or 0
            q75 = state.q75 or 1
            iqr = q75 - q25
            return value * iqr + q25
        
        elif method == NormalizeMethod.MAX_ABS:
            max_abs = state.max_abs or 1
            return value * max_abs
        
        elif method == NormalizeMethod.MEAN:
            mean = state.mean or 1
            return value * mean
        
        elif method == NormalizeMethod.UNIT_VECTOR:
            norm = state.max_abs or 1
            return value * norm
        
        elif method == NormalizeMethod.LOG:
            return math.exp(value)
        
        elif method == NormalizeMethod.SQUARE_ROOT:
            return value ** 2
        
        return value
    
    def _percentile(self, sorted_values: List[float], p: float) -> float:
        """Calculate percentile from sorted values."""
        n = len(sorted_values)
        if n == 0:
            return 0.0
        
        idx = p / 100 * (n - 1)
        lower = int(math.floor(idx))
        upper = int(math.ceil(idx))
        
        if lower == upper:
            return sorted_values[lower]
        
        weight = idx - lower
        return sorted_values[lower] * (1 - weight) + sorted_values[upper] * weight


# ==============================================================================
# FACTORY FUNCTION
# ==============================================================================

def create_data_normalizer(config: Config) -> DataNormalizer:
    """
    Factory function for DataNormalizer creation.
    
    Args:
        config: Application configuration
        
    Returns:
        DataNormalizer instance
    """
    return DataNormalizer(config)