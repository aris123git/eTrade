"""
preprocessing/validator.py - Data Validation Module

RESPONSIBILITY:
Validate market data quality and integrity before processing.

ARCHITECTURAL PRINCIPLES:
1. Pure validation - No data storage, no I/O, no business logic
2. Check data quality, completeness, and consistency
3. Type-safe results with detailed error reporting
4. Multiple validation levels (basic, strict, comprehensive)

WHAT IT NEVER DOES:
- ❌ Connect to MT5
- ❌ Store data
- ❌ Download data
- ❌ Make trading decisions
- ❌ Modify data (only validates)

VERSION: 1.0.4
"""

import logging
import math
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional, Dict, Any, Tuple, Set, Union
from enum import Enum

from core.config import Config
from core.exceptions import DataValidationError


# ==============================================================================
# EXPORTS
# ==============================================================================

__all__ = [
    'ValidationLevel',
    'ValidationSeverity',
    'ValidationIssue',
    'ValidationResult',
    'DataValidator',
    'create_data_validator',
]


# ==============================================================================
# ENUMS
# ==============================================================================

class ValidationLevel(Enum):
    """Level of validation to perform."""
    BASIC = "basic"          # Only check structure
    STANDARD = "standard"    # Structure + basic quality
    STRICT = "strict"        # Full quality checks
    COMPREHENSIVE = "comprehensive"  # Everything including statistics


class ValidationSeverity(Enum):
    """Severity of validation issues."""
    INFO = "info"           # Informational only
    WARNING = "warning"     # Potential issue, should be reviewed
    ERROR = "error"         # Invalid data, must be fixed
    CRITICAL = "critical"   # Data cannot be used


# ==============================================================================
# DATA MODELS
# ==============================================================================

@dataclass
class ValidationIssue:
    """A single validation issue."""
    rule: str
    severity: ValidationSeverity
    message: str
    field: Optional[str] = None
    index: Optional[int] = None
    value: Optional[Any] = None
    expected: Optional[Any] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'rule': self.rule,
            'severity': self.severity.value,
            'message': self.message,
            'field': self.field,
            'index': self.index,
            'value': self.value,
            'expected': self.expected,
        }


@dataclass
class ValidationResult:
    """Result of validation operation."""
    valid: bool
    total_items: int
    issues: List[ValidationIssue]
    field_stats: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def errors(self) -> List[ValidationIssue]:
        """Get all ERROR issues."""
        return [i for i in self.issues if i.severity == ValidationSeverity.ERROR]
    
    @property
    def warnings(self) -> List[ValidationIssue]:
        """Get all WARNING issues."""
        return [i for i in self.issues if i.severity == ValidationSeverity.WARNING]
    
    @property
    def critical(self) -> List[ValidationIssue]:
        """Get all CRITICAL issues."""
        return [i for i in self.issues if i.severity == ValidationSeverity.CRITICAL]
    
    @property
    def error_count(self) -> int:
        return len(self.errors)
    
    @property
    def warning_count(self) -> int:
        return len(self.warnings)
    
    @property
    def critical_count(self) -> int:
        return len(self.critical)
    
    def get_summary(self) -> Dict[str, Any]:
        """Get summary of validation."""
        return {
            'valid': self.valid,
            'total_items': self.total_items,
            'error_count': self.error_count,
            'warning_count': self.warning_count,
            'critical_count': self.critical_count,
            'total_issues': len(self.issues),
            'fields': self.field_stats,
        }


# ==============================================================================
# DATA VALIDATOR
# ==============================================================================

class DataValidator:
    """
    Data validation engine.
    
    Validates market data quality and integrity.
    """
    
    # Standard candle fields
    STANDARD_FIELDS = {'open', 'high', 'low', 'close', 'volume', 'timestamp'}
    REQUIRED_FIELDS = {'open', 'high', 'low', 'close'}
    
    def __init__(self, config: Config):
        """
        Initialize the data validator.
        
        Args:
            config: Application configuration
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Validation thresholds
        self._max_missing_percent = getattr(config, 'MAX_MISSING_PERCENT', 0.05)
        self._min_price_value = getattr(config, 'MIN_PRICE_VALUE', 0.0001)
        self._max_price_value = getattr(config, 'MAX_PRICE_VALUE', 1e12)
        self._max_spread_ratio = getattr(config, 'MAX_SPREAD_RATIO', 0.05)
        self._max_volume_spike = getattr(config, 'MAX_VOLUME_SPIKE', 10.0)
        self._flat_price_threshold = getattr(config, 'FLAT_PRICE_THRESHOLD', 0.0005)
        
        self.logger.info("✅ DataValidator initialized")
    
    # ==========================================================================
    # PUBLIC METHODS
    # ==========================================================================
    
    def validate_candles(
        self,
        candles: List[Dict[str, Any]],
        level: Union[ValidationLevel, str] = ValidationLevel.STANDARD,
        fields: Optional[List[str]] = None,
    ) -> ValidationResult:
        """
        Validate a list of candles.
        
        Args:
            candles: List of candle dictionaries
            level: Validation level
            fields: Fields to validate (default: all standard fields)
            
        Returns:
            ValidationResult object
        """
        if not candles:
            return ValidationResult(
                valid=False,
                total_items=0,
                issues=[ValidationIssue(
                    rule='EMPTY_DATA',
                    severity=ValidationSeverity.CRITICAL,
                    message='No candles provided for validation',
                )],
                field_stats={},
                metadata={'level': str(level)},
            )
        
        # Parse validation level
        level = self._parse_validation_level(level)
        
        # Determine fields
        if fields is None:
            fields = ['open', 'high', 'low', 'close', 'volume']
        
        self.logger.debug(
            f"Validating {len(candles)} candles at {level.value} level"
        )
        
        issues = []
        field_stats = self._init_field_stats(fields, len(candles))
        
        # Run validation rules based on level
        if level == ValidationLevel.BASIC:
            basic_issues = self._validate_basic(candles)
            issues.extend(basic_issues)
            self._update_field_stats_with_issues(field_stats, basic_issues)
            self._update_field_stats_with_values(field_stats, candles, fields)
        
        if level in (ValidationLevel.STANDARD, ValidationLevel.STRICT, ValidationLevel.COMPREHENSIVE):
            struct_issues = self._validate_structure(candles, fields)
            issues.extend(struct_issues)
            self._update_field_stats_with_issues(field_stats, struct_issues)
            
            value_issues = self._validate_values(candles, fields)
            issues.extend(value_issues)
            self._update_field_stats_with_issues(field_stats, value_issues)
            
            missing_issues = self._validate_missing(candles, fields)
            issues.extend(missing_issues)
            self._update_field_stats_with_issues(field_stats, missing_issues)
            
            self._update_field_stats_with_values(field_stats, candles, fields)
        
        if level in (ValidationLevel.STRICT, ValidationLevel.COMPREHENSIVE):
            ohlc_issues = self._validate_ohlc_logic(candles)
            issues.extend(ohlc_issues)
            self._update_field_stats_with_issues(field_stats, ohlc_issues)
            
            volume_issues = self._validate_volume(candles)
            issues.extend(volume_issues)
            self._update_field_stats_with_issues(field_stats, volume_issues)
            
            consistency_issues = self._validate_consistency(candles)
            issues.extend(consistency_issues)
            self._update_field_stats_with_issues(field_stats, consistency_issues)
        
        if level == ValidationLevel.COMPREHENSIVE:
            stat_issues = self._validate_statistics(candles, fields)
            issues.extend(stat_issues)
            self._update_field_stats_with_issues(field_stats, stat_issues)
            
            trend_issues = self._validate_trends(candles)
            issues.extend(trend_issues)
            self._update_field_stats_with_issues(field_stats, trend_issues)
            
            self._update_field_stats_with_statistics(field_stats, candles, fields)
        
        # Determine validity
        critical_errors = [i for i in issues if i.severity in (ValidationSeverity.CRITICAL, ValidationSeverity.ERROR)]
        valid = len(critical_errors) == 0
        
        result = ValidationResult(
            valid=valid,
            total_items=len(candles),
            issues=issues,
            field_stats=field_stats,
            metadata={
                'level': level.value,
                'fields_validated': fields,
            },
        )
        
        self.logger.debug(
            f"Validation complete: {'PASSED' if valid else 'FAILED'} "
            f"(errors={result.error_count}, warnings={result.warning_count})"
        )
        
        return result
    
    def validate_price(self, price: float) -> bool:
        """Validate a single price value."""
        if not isinstance(price, (int, float)):
            return False
        if math.isnan(price) or math.isinf(price):
            return False
        if price <= 0:
            return False
        return True
    
    def validate_volume(self, volume: int) -> bool:
        """Validate a single volume value."""
        if not isinstance(volume, (int, float)):
            return False
        if math.isnan(volume) or math.isinf(volume):
            return False
        if volume < 0:
            return False
        return True
    
    def is_valid_candle(self, candle: Dict[str, Any]) -> bool:
        """
        Check if a single candle is valid.
        
        Args:
            candle: Candle dictionary
            
        Returns:
            True if valid, False otherwise
        """
        result = self.validate_candles([candle], ValidationLevel.BASIC)
        return result.valid
    
    # ==========================================================================
    # PRIVATE METHODS
    # ==========================================================================
    
    def _parse_validation_level(self, level: Union[ValidationLevel, str]) -> ValidationLevel:
        """Parse validation level from string or enum."""
        if isinstance(level, ValidationLevel):
            return level
        if isinstance(level, str):
            try:
                return ValidationLevel(level.lower())
            except ValueError:
                self.logger.warning(f"Unknown validation level '{level}', using STANDARD")
                return ValidationLevel.STANDARD
        return ValidationLevel.STANDARD
    
    def _init_field_stats(self, fields: List[str], total_items: int) -> Dict[str, Dict[str, Any]]:
        """Initialize field statistics."""
        stats: Dict[str, Dict[str, Any]] = {}
        for field in fields:
            stats[field] = {
                'present': 0,
                'missing': 0,
                'null': 0,
                'valid': 0,
                'invalid': 0,
                'errors': 0,
                'warnings': 0,
                'min': None,
                'max': None,
                'mean': None,
                'std': None,
            }
        stats['_total_items'] = total_items
        return stats
    
    def _update_field_stats_with_issues(self, stats: Dict[str, Dict[str, Any]], issues: List[ValidationIssue]) -> None:
        """Update field statistics with issues."""
        for issue in issues:
            if issue.field and issue.field in stats:
                if issue.severity == ValidationSeverity.ERROR:
                    stats[issue.field]['errors'] += 1
                elif issue.severity == ValidationSeverity.WARNING:
                    stats[issue.field]['warnings'] += 1
    
    def _update_field_stats_with_values(self, stats: Dict[str, Dict[str, Any]], candles: List[Dict[str, Any]], fields: List[str]) -> None:
        """Update field statistics with actual values."""
        for field in fields:
            if field not in stats:
                continue
            
            values = []
            present = 0
            missing = 0
            null_count = 0
            valid = 0
            invalid = 0
            
            for candle in candles:
                if field not in candle:
                    missing += 1
                    continue
                
                value = candle[field]
                if value is None:
                    null_count += 1
                    missing += 1
                    continue
                
                present += 1
                
                try:
                    val = float(value)
                    if math.isnan(val) or math.isinf(val):
                        invalid += 1
                    else:
                        valid += 1
                        values.append(val)
                except (ValueError, TypeError):
                    invalid += 1
            
            stats[field]['present'] = present
            stats[field]['missing'] = missing
            stats[field]['null'] = null_count
            stats[field]['valid'] = valid
            stats[field]['invalid'] = invalid
            
            if values:
                stats[field]['min'] = min(values)
                stats[field]['max'] = max(values)
            else:
                stats[field]['min'] = None
                stats[field]['max'] = None
    
    def _update_field_stats_with_statistics(self, stats: Dict[str, Dict[str, Any]], candles: List[Dict[str, Any]], fields: List[str]) -> None:
        """Update field statistics with mean and std."""
        for field in fields:
            if field not in stats:
                continue
            
            values = []
            for candle in candles:
                if field in candle and candle[field] is not None:
                    try:
                        val = float(candle[field])
                        if not math.isnan(val) and not math.isinf(val):
                            values.append(val)
                    except (ValueError, TypeError):
                        continue
            
            if len(values) >= 2:
                mean = sum(values) / len(values)
                variance = sum((v - mean) ** 2 for v in values) / (len(values) - 1)
                std = math.sqrt(variance) if variance > 0 else 0.0
                stats[field]['mean'] = mean
                stats[field]['std'] = std
    
    def _validate_basic(self, candles: List[Dict[str, Any]]) -> List[ValidationIssue]:
        """Basic structure validation."""
        issues = []
        
        for i, candle in enumerate(candles):
            # Check if candle is a dict
            if not isinstance(candle, dict):
                issues.append(ValidationIssue(
                    rule='INVALID_TYPE',
                    severity=ValidationSeverity.ERROR,
                    message=f'Candle at index {i} is not a dictionary',
                    index=i,
                ))
                continue
            
            # Check required fields
            missing_fields = [f for f in self.REQUIRED_FIELDS if f not in candle]
            if missing_fields:
                for missing_field in missing_fields:
                    issues.append(ValidationIssue(
                        rule='MISSING_FIELD',
                        severity=ValidationSeverity.CRITICAL,
                        message=f'Candle at index {i} missing required field: {missing_field}',
                        index=i,
                        field=missing_field,
                    ))
                continue
            
            # Basic price validation (non-zero)
            try:
                open_price = float(candle['open'])
                high = float(candle['high'])
                low = float(candle['low'])
                close = float(candle['close'])
                
                if open_price <= 0:
                    issues.append(ValidationIssue(
                        rule='ZERO_OPEN',
                        severity=ValidationSeverity.WARNING,
                        message=f'Open price at index {i} is zero or negative: {open_price}',
                        index=i,
                        field='open',
                        value=open_price,
                    ))
                if high <= 0:
                    issues.append(ValidationIssue(
                        rule='ZERO_HIGH',
                        severity=ValidationSeverity.WARNING,
                        message=f'High price at index {i} is zero or negative: {high}',
                        index=i,
                        field='high',
                        value=high,
                    ))
                if low <= 0:
                    issues.append(ValidationIssue(
                        rule='ZERO_LOW',
                        severity=ValidationSeverity.WARNING,
                        message=f'Low price at index {i} is zero or negative: {low}',
                        index=i,
                        field='low',
                        value=low,
                    ))
                if close <= 0:
                    issues.append(ValidationIssue(
                        rule='ZERO_CLOSE',
                        severity=ValidationSeverity.WARNING,
                        message=f'Close price at index {i} is zero or negative: {close}',
                        index=i,
                        field='close',
                        value=close,
                    ))
                    
            except (ValueError, TypeError):
                issues.append(ValidationIssue(
                    rule='INVALID_PRICE',
                    severity=ValidationSeverity.ERROR,
                    message=f'Price values at index {i} are invalid',
                    index=i,
                ))
        
        return issues
    
    def _validate_structure(
        self,
        candles: List[Dict[str, Any]],
        fields: List[str],
    ) -> List[ValidationIssue]:
        """Validate data structure."""
        issues = []
        
        for i, candle in enumerate(candles):
            # Check all fields are present
            for field in fields:
                if field not in candle:
                    issues.append(ValidationIssue(
                        rule='FIELD_MISSING',
                        severity=ValidationSeverity.WARNING,
                        message=f'Field "{field}" missing at index {i}',
                        index=i,
                        field=field,
                    ))
                else:
                    # Check field type
                    value = candle[field]
                    if field in ('open', 'high', 'low', 'close'):
                        if not isinstance(value, (int, float)):
                            issues.append(ValidationIssue(
                                rule='INVALID_TYPE',
                                severity=ValidationSeverity.ERROR,
                                message=f'Field "{field}" at index {i} is not numeric: {type(value)}',
                                index=i,
                                field=field,
                                value=value,
                            ))
                    elif field == 'volume':
                        if not isinstance(value, (int, float)):
                            issues.append(ValidationIssue(
                                rule='INVALID_TYPE',
                                severity=ValidationSeverity.ERROR,
                                message=f'Field "{field}" at index {i} is not numeric: {type(value)}',
                                index=i,
                                field=field,
                                value=value,
                            ))
        
        return issues
    
    def _validate_values(
        self,
        candles: List[Dict[str, Any]],
        fields: List[str],
    ) -> List[ValidationIssue]:
        """Validate field values."""
        issues = []
        
        for i, candle in enumerate(candles):
            for field in fields:
                if field not in candle:
                    continue
                
                value = candle[field]
                if value is None:
                    issues.append(ValidationIssue(
                        rule='NULL_VALUE',
                        severity=ValidationSeverity.ERROR,
                        message=f'Field "{field}" at index {i} is None',
                        index=i,
                        field=field,
                    ))
                    continue
                
                if field in ('open', 'high', 'low', 'close'):
                    # Check for NaN/Inf
                    if isinstance(value, float):
                        if math.isnan(value):
                            issues.append(ValidationIssue(
                                rule='NAN_VALUE',
                                severity=ValidationSeverity.ERROR,
                                message=f'Field "{field}" at index {i} is NaN',
                                index=i,
                                field=field,
                            ))
                            continue
                        if math.isinf(value):
                            issues.append(ValidationIssue(
                                rule='INF_VALUE',
                                severity=ValidationSeverity.ERROR,
                                message=f'Field "{field}" at index {i} is infinite',
                                index=i,
                                field=field,
                            ))
                            continue
                    
                    # Check if price is positive and within range
                    try:
                        val = float(value)
                        if val <= 0:
                            issues.append(ValidationIssue(
                                rule='NON_POSITIVE_PRICE',
                                severity=ValidationSeverity.ERROR,
                                message=f'Field "{field}" at index {i} is non-positive: {val}',
                                index=i,
                                field=field,
                                value=val,
                            ))
                        elif val > self._max_price_value:
                            issues.append(ValidationIssue(
                                rule='EXCESSIVE_PRICE',
                                severity=ValidationSeverity.WARNING,
                                message=f'Field "{field}" at index {i} is excessive: {val}',
                                index=i,
                                field=field,
                                value=val,
                                expected=f'<= {self._max_price_value}',
                            ))
                        elif val < self._min_price_value:
                            issues.append(ValidationIssue(
                                rule='MIN_PRICE',
                                severity=ValidationSeverity.WARNING,
                                message=f'Field "{field}" at index {i} is below minimum: {val}',
                                index=i,
                                field=field,
                                value=val,
                                expected=f'>= {self._min_price_value}',
                            ))
                    except (ValueError, TypeError):
                        issues.append(ValidationIssue(
                            rule='INVALID_PRICE',
                            severity=ValidationSeverity.ERROR,
                            message=f'Field "{field}" at index {i} cannot be converted to float: {value}',
                            index=i,
                            field=field,
                            value=value,
                        ))
                
                elif field == 'volume':
                    # Check for NaN/Inf in volume
                    if isinstance(value, float):
                        if math.isnan(value):
                            issues.append(ValidationIssue(
                                rule='NAN_VOLUME',
                                severity=ValidationSeverity.ERROR,
                                message=f'Volume at index {i} is NaN',
                                index=i,
                                field='volume',
                            ))
                            continue
                        if math.isinf(value):
                            issues.append(ValidationIssue(
                                rule='INF_VOLUME',
                                severity=ValidationSeverity.ERROR,
                                message=f'Volume at index {i} is infinite',
                                index=i,
                                field='volume',
                            ))
                            continue
                    
                    try:
                        if float(value) < 0:
                            issues.append(ValidationIssue(
                                rule='NEGATIVE_VOLUME',
                                severity=ValidationSeverity.ERROR,
                                message=f'Volume at index {i} is negative: {value}',
                                index=i,
                                field='volume',
                                value=value,
                            ))
                    except (ValueError, TypeError):
                        issues.append(ValidationIssue(
                            rule='INVALID_VOLUME',
                            severity=ValidationSeverity.WARNING,
                            message=f'Volume at index {i} cannot be converted to float: {value}',
                            index=i,
                            field='volume',
                            value=value,
                        ))
        
        return issues
    
    def _validate_missing(
        self,
        candles: List[Dict[str, Any]],
        fields: List[str],
    ) -> List[ValidationIssue]:
        """Validate missing data."""
        issues = []
        
        total_candles = len(candles)
        if total_candles == 0:
            return issues
        
        # Count missing values per field
        missing_counts = {f: 0 for f in fields}
        
        for candle in candles:
            for field in fields:
                if field not in candle or candle[field] is None:
                    missing_counts[field] += 1
        
        # Check if missing rate exceeds threshold
        for field, count in missing_counts.items():
            rate = count / total_candles
            if rate > self._max_missing_percent:
                issues.append(ValidationIssue(
                    rule='HIGH_MISSING_RATE',
                    severity=ValidationSeverity.WARNING,
                    message=f'Field "{field}" has {count}/{total_candles} missing values ({rate:.1%})',
                    field=field,
                    value=rate,
                    expected=f'< {self._max_missing_percent:.1%}',
                ))
        
        return issues
    
    def _validate_ohlc_logic(self, candles: List[Dict[str, Any]]) -> List[ValidationIssue]:
        """Validate OHLC logic (high >= low, etc.)."""
        issues = []
        
        for i, candle in enumerate(candles):
            try:
                open_price = candle.get('open')
                high = candle.get('high')
                low = candle.get('low')
                close = candle.get('close')
                
                # Check if values exist
                if open_price is None or high is None or low is None or close is None:
                    issues.append(ValidationIssue(
                        rule='MISSING_OHLC',
                        severity=ValidationSeverity.ERROR,
                        message=f'Missing OHLC values at index {i}',
                        index=i,
                    ))
                    continue
                
                open_price = float(open_price)
                high = float(high)
                low = float(low)
                close = float(close)
                
                # Check if values are valid
                if open_price <= 0 or high <= 0 or low <= 0 or close <= 0:
                    issues.append(ValidationIssue(
                        rule='INVALID_OHLC',
                        severity=ValidationSeverity.ERROR,
                        message=f'Invalid OHLC values (non-positive) at index {i}',
                        index=i,
                        value={'open': open_price, 'high': high, 'low': low, 'close': close},
                    ))
                    continue
                
                # High >= Low
                if high < low:
                    issues.append(ValidationIssue(
                        rule='HIGH_LESS_THAN_LOW',
                        severity=ValidationSeverity.ERROR,
                        message=f'High ({high}) < Low ({low}) at index {i}',
                        index=i,
                        field='high',
                        value=high,
                        expected=f'>= {low}',
                    ))
                
                # High >= Open and High >= Close
                if high < open_price:
                    issues.append(ValidationIssue(
                        rule='HIGH_LESS_THAN_OPEN',
                        severity=ValidationSeverity.ERROR,
                        message=f'High ({high}) < Open ({open_price}) at index {i}',
                        index=i,
                        field='high',
                        value=high,
                        expected=f'>= {open_price}',
                    ))
                if high < close:
                    issues.append(ValidationIssue(
                        rule='HIGH_LESS_THAN_CLOSE',
                        severity=ValidationSeverity.ERROR,
                        message=f'High ({high}) < Close ({close}) at index {i}',
                        index=i,
                        field='high',
                        value=high,
                        expected=f'>= {close}',
                    ))
                
                # Low <= Open and Low <= Close
                if low > open_price:
                    issues.append(ValidationIssue(
                        rule='LOW_GREATER_THAN_OPEN',
                        severity=ValidationSeverity.ERROR,
                        message=f'Low ({low}) > Open ({open_price}) at index {i}',
                        index=i,
                        field='low',
                        value=low,
                        expected=f'<= {open_price}',
                    ))
                if low > close:
                    issues.append(ValidationIssue(
                        rule='LOW_GREATER_THAN_CLOSE',
                        severity=ValidationSeverity.ERROR,
                        message=f'Low ({low}) > Close ({close}) at index {i}',
                        index=i,
                        field='low',
                        value=low,
                        expected=f'<= {close}',
                    ))
                
                # Check spread (high - low)
                spread = high - low
                if spread <= 0:
                    issues.append(ValidationIssue(
                        rule='ZERO_SPREAD',
                        severity=ValidationSeverity.WARNING,
                        message=f'Zero spread at index {i}: high={high}, low={low}',
                        index=i,
                        field='spread',
                        value=spread,
                        expected='> 0',
                    ))
                
                # Check spread ratio
                if close > 0:
                    spread_ratio = spread / close
                    if spread_ratio > self._max_spread_ratio:
                        issues.append(ValidationIssue(
                            rule='EXCESSIVE_SPREAD',
                            severity=ValidationSeverity.WARNING,
                            message=f'Excessive spread at index {i}: {spread:.4f} ({spread_ratio:.2%} of close)',
                            index=i,
                            field='spread',
                            value=spread_ratio,
                            expected=f'<= {self._max_spread_ratio:.2%}',
                        ))
                
            except (ValueError, TypeError):
                issues.append(ValidationIssue(
                    rule='INVALID_OHLC_TYPE',
                    severity=ValidationSeverity.ERROR,
                    message=f'OHLC values at index {i} are not numeric',
                    index=i,
                ))
                continue
        
        return issues
    
    def _validate_volume(self, candles: List[Dict[str, Any]]) -> List[ValidationIssue]:
        """Validate volume data."""
        issues = []
        
        volume_data = []  # (index, volume)
        for i, candle in enumerate(candles):
            try:
                volume = candle.get('volume')
                if volume is None:
                    # Volume is missing - this is an issue
                    issues.append(ValidationIssue(
                        rule='MISSING_VOLUME',
                        severity=ValidationSeverity.WARNING,
                        message=f'Volume missing at index {i}',
                        index=i,
                        field='volume',
                    ))
                    continue
                
                vol = float(volume)
                if vol < 0:
                    issues.append(ValidationIssue(
                        rule='NEGATIVE_VOLUME',
                        severity=ValidationSeverity.ERROR,
                        message=f'Volume at index {i} is negative: {vol}',
                        index=i,
                        field='volume',
                        value=vol,
                    ))
                    continue
                
                if math.isnan(vol) or math.isinf(vol):
                    issues.append(ValidationIssue(
                        rule='INVALID_VOLUME',
                        severity=ValidationSeverity.ERROR,
                        message=f'Volume at index {i} is NaN or Inf: {vol}',
                        index=i,
                        field='volume',
                        value=vol,
                    ))
                    continue
                
                volume_data.append((i, vol))
                
            except (ValueError, TypeError):
                issues.append(ValidationIssue(
                    rule='INVALID_VOLUME_TYPE',
                    severity=ValidationSeverity.WARNING,
                    message=f'Volume at index {i} cannot be converted to float',
                    index=i,
                    field='volume',
                    value=candle.get('volume'),
                ))
                continue
        
        # Check for zero volume
        for i, vol in volume_data:
            if vol == 0:
                issues.append(ValidationIssue(
                    rule='ZERO_VOLUME',
                    severity=ValidationSeverity.INFO,
                    message=f'Zero volume at index {i}',
                    index=i,
                    field='volume',
                    value=0,
                ))
        
        # Check for volume spikes
        if len(volume_data) >= 10:
            volumes = [v for _, v in volume_data]
            sorted_volumes = sorted(volumes)
            n = len(sorted_volumes)
            if n % 2 == 1:
                median = sorted_volumes[n // 2]
            else:
                median = (sorted_volumes[n // 2 - 1] + sorted_volumes[n // 2]) / 2
            
            if median > 0:
                for i, vol in volume_data:
                    if vol > median * self._max_volume_spike:
                        issues.append(ValidationIssue(
                            rule='VOLUME_SPIKE',
                            severity=ValidationSeverity.WARNING,
                            message=f'Volume spike at index {i}: {vol} (median: {median})',
                            index=i,
                            field='volume',
                            value=vol,
                            expected=f'<= {median * self._max_volume_spike}',
                        ))
        
        return issues
    
    def _validate_consistency(self, candles: List[Dict[str, Any]]) -> List[ValidationIssue]:
        """Validate data consistency."""
        issues = []
        
        if len(candles) < 2:
            return issues
        
        # Check for duplicate timestamps
        timestamps = {}
        for i, candle in enumerate(candles):
            ts = candle.get('timestamp')
            if ts is not None:
                # Convert to hashable string
                if isinstance(ts, datetime):
                    ts_key = ts.isoformat()
                elif isinstance(ts, (int, float)):
                    ts_key = str(ts)
                else:
                    ts_key = str(ts)
                
                if ts_key in timestamps:
                    issues.append(ValidationIssue(
                        rule='DUPLICATE_TIMESTAMP',
                        severity=ValidationSeverity.WARNING,
                        message=f'Duplicate timestamp at index {i}: {ts} (also at index {timestamps[ts_key]})',
                        index=i,
                        field='timestamp',
                        value=ts,
                    ))
                else:
                    timestamps[ts_key] = i
        
        # Check for gaps in time (if timestamps exist and are numeric)
        numeric_ts = []
        for i, candle in enumerate(candles):
            ts = candle.get('timestamp')
            if ts is not None:
                if isinstance(ts, (int, float)):
                    numeric_ts.append((i, ts))
                elif isinstance(ts, datetime):
                    try:
                        numeric_ts.append((i, ts.timestamp()))
                    except (ValueError, TypeError):
                        pass
        
        if len(numeric_ts) >= 3:
            sorted_ts = sorted(numeric_ts, key=lambda x: x[1])
            avg_gap = sum(sorted_ts[i+1][1] - sorted_ts[i][1] for i in range(len(sorted_ts)-1)) / (len(sorted_ts)-1)
            
            for i in range(len(sorted_ts)-1):
                gap = sorted_ts[i+1][1] - sorted_ts[i][1]
                if gap > avg_gap * 3 and gap > 10:
                    issues.append(ValidationIssue(
                        rule='TIME_GAP',
                        severity=ValidationSeverity.WARNING,
                        message=f'Large time gap between index {sorted_ts[i][0]} and {sorted_ts[i+1][0]}: {gap:.1f}s (avg: {avg_gap:.1f}s)',
                        index=sorted_ts[i][0],
                        field='timestamp',
                        value=gap,
                        expected=f'<= {avg_gap * 3:.1f}s',
                    ))
        
        return issues
    
    def _validate_statistics(
        self,
        candles: List[Dict[str, Any]],
        fields: List[str],
    ) -> List[ValidationIssue]:
        """Validate statistical properties."""
        issues = []
        
        for field in fields:
            if field not in self.REQUIRED_FIELDS:
                continue
            
            values_with_indices = []  # (original_index, value)
            for i, candle in enumerate(candles):
                if field in candle and candle[field] is not None:
                    try:
                        val = float(candle[field])
                        if not math.isnan(val) and not math.isinf(val):
                            values_with_indices.append((i, val))
                    except (ValueError, TypeError):
                        continue
            
            if len(values_with_indices) < 10:
                continue
            
            values = [v for _, v in values_with_indices]
            n = len(values)
            mean = sum(values) / n
            
            if n > 1:
                variance = sum((v - mean) ** 2 for v in values) / (n - 1)
                std = math.sqrt(variance) if variance > 0 else 0.0
            else:
                std = 0.0
            
            # Check for extreme values (beyond 5 standard deviations)
            if std > 0:
                for idx, v in enumerate(values):
                    if abs(v - mean) > 5 * std:
                        orig_idx = values_with_indices[idx][0]
                        issues.append(ValidationIssue(
                            rule='EXTREME_VALUE',
                            severity=ValidationSeverity.WARNING,
                            message=f'Extreme value in "{field}" at index {orig_idx}: {v} (mean: {mean:.4f}, std: {std:.4f})',
                            index=orig_idx,
                            field=field,
                            value=v,
                            expected=f'Within {mean:.4f} ± 5*{std:.4f}',
                        ))
        
        # Validate volume statistics
        volume_with_indices = []  # (original_index, volume)
        for i, candle in enumerate(candles):
            if 'volume' in candle and candle['volume'] is not None:
                try:
                    vol = float(candle['volume'])
                    if not math.isnan(vol) and not math.isinf(vol):
                        volume_with_indices.append((i, vol))
                except (ValueError, TypeError):
                    continue
        
        if len(volume_with_indices) >= 10:
            vol_values = [v for _, v in volume_with_indices]
            vol_mean = sum(vol_values) / len(vol_values)
            if len(vol_values) > 1:
                vol_variance = sum((v - vol_mean) ** 2 for v in vol_values) / (len(vol_values) - 1)
                vol_std = math.sqrt(vol_variance) if vol_variance > 0 else 0.0
            else:
                vol_std = 0.0
            
            if vol_std > 0:
                for idx, v in enumerate(vol_values):
                    if abs(v - vol_mean) > 5 * vol_std:
                        orig_idx = volume_with_indices[idx][0]
                        issues.append(ValidationIssue(
                            rule='EXTREME_VOLUME',
                            severity=ValidationSeverity.WARNING,
                            message=f'Extreme volume at index {orig_idx}: {v} (mean: {vol_mean:.2f}, std: {vol_std:.2f})',
                            index=orig_idx,
                            field='volume',
                            value=v,
                            expected=f'Within {vol_mean:.2f} ± 5*{vol_std:.2f}',
                        ))
        
        return issues
    
    def _validate_trends(self, candles: List[Dict[str, Any]]) -> List[ValidationIssue]:
        """Validate trend patterns."""
        issues = []
        
        if len(candles) < 5:
            return issues
        
        closes = []  # (index, close)
        for i, candle in enumerate(candles):
            try:
                close = candle.get('close')
                if close is None:
                    continue
                
                close_val = float(close)
                if not math.isnan(close_val) and not math.isinf(close_val) and close_val > 0:
                    closes.append((i, close_val))
            except (ValueError, TypeError):
                continue
        
        if len(closes) < 5:
            return issues
        
        close_values = [c[1] for c in closes]
        
        # Check for flat price (no movement)
        price_range = max(close_values) - min(close_values)
        avg_price = sum(close_values) / len(close_values)
        
        if avg_price > 0 and price_range / avg_price < self._flat_price_threshold:
            issues.append(ValidationIssue(
                rule='FLAT_PRICE',
                severity=ValidationSeverity.WARNING,
                message=f'Price is almost flat: range={price_range:.4f}, avg={avg_price:.4f} (threshold: {self._flat_price_threshold})',
                field='close',
                value=price_range / avg_price,
                expected=f'>= {self._flat_price_threshold}',
            ))
        
        # Check for consecutive identical closes
        for i in range(1, len(closes)):
            if closes[i][1] == closes[i-1][1]:
                issues.append(ValidationIssue(
                    rule='DUPLICATE_CLOSE',
                    severity=ValidationSeverity.WARNING,
                    message=f'Duplicate close price at index {closes[i][0]}: {closes[i][1]}',
                    index=closes[i][0],
                    field='close',
                    value=closes[i][1],
                ))
        
        return issues


# ==============================================================================
# FACTORY FUNCTION
# ==============================================================================

def create_data_validator(config: Config) -> DataValidator:
    """
    Factory function for DataValidator creation.
    
    Args:
        config: Application configuration
        
    Returns:
        DataValidator instance
    """
    return DataValidator(config)