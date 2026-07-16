"""
discovery/candle_discovery.py - Candle Pattern Discovery Engine

RESPONSIBILITY:
Discover and analyze candle patterns from market data.

ARCHITECTURAL PRINCIPLES:
1. Pure pattern discovery - No data storage, no I/O, no business logic
2. Pattern detection based on price action and volume
3. Extensible pattern detector pattern
4. Type-safe results with statistical analysis

WHAT IT NEVER DOES:
- ❌ Connect to MT5
- ❌ Store data
- ❌ Download data
- ❌ Make trading decisions

VERSION: 1.0.2
"""

import logging
import math
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional, Dict, Any, Tuple, Callable, Set, Union
from enum import Enum

from core.config import Config
from core.exceptions import DiscoveryError, DataValidationError
from core.utils import calculate_percentage


# ==============================================================================
# EXPORTS
# ==============================================================================

__all__ = [
    # Enums
    'CandlePattern',
    'PatternStrength',
    'PatternDirection',
    # Data classes
    'CandleData',
    'DetectedPattern',
    'PatternResult',
    # Detectors
    'BasePatternDetector',
    'DojiDetector',
    'HammerDetector',
    'MarubozuDetector',
    'EngulfingDetector',
    'HaramiDetector',
    'ThreeWhiteSoldiersDetector',
    'ThreeBlackCrowsDetector',
    # Manager
    'PatternManager',
    # Factory
    'create_pattern_manager',
]


# ==============================================================================
# ENUMS
# ==============================================================================

class CandlePattern(Enum):
    """Candle pattern types."""
    # Single candle patterns
    DOJI = "doji"
    HAMMER = "hammer"
    SHOOTING_STAR = "shooting_star"
    MARUBOZU = "marubozu"
    SPINNING_TOP = "spinning_top"
    
    # Two candle patterns
    ENGULFING_BULLISH = "engulfing_bullish"
    ENGULFING_BEARISH = "engulfing_bearish"
    HARAMI_BULLISH = "harami_bullish"
    HARAMI_BEARISH = "harami_bearish"
    PIERCING = "piercing"
    DARK_CLOUD = "dark_cloud"
    
    # Three candle patterns
    MORNING_STAR = "morning_star"
    EVENING_STAR = "evening_star"
    THREE_WHITE_SOLDIERS = "three_white_soldiers"
    THREE_BLACK_CROWS = "three_black_crows"
    
    # Complex patterns
    HEAD_AND_SHOULDERS = "head_and_shoulders"
    INVERSE_HEAD_AND_SHOULDERS = "inverse_head_and_shoulders"
    DOUBLE_TOP = "double_top"
    DOUBLE_BOTTOM = "double_bottom"
    TRIANGLE = "triangle"
    FLAG = "flag"
    WEDGE = "wedge"


class PatternStrength(Enum):
    """Strength of a detected pattern."""
    VERY_STRONG = 1.0
    STRONG = 0.8
    MODERATE = 0.6
    WEAK = 0.4
    VERY_WEAK = 0.2


class PatternDirection(Enum):
    """Direction of a pattern."""
    BULLISH = "bullish"
    BEARISH = "bearish"
    NEUTRAL = "neutral"
    REVERSAL = "reversal"
    CONTINUATION = "continuation"


# ==============================================================================
# DATA MODELS
# ==============================================================================

@dataclass(frozen=True)
class CandleData:
    """Candle data for pattern detection."""
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: int
    spread: Optional[int] = None
    
    def __post_init__(self):
        """Validate candle data after initialization."""
        if self.high < self.low:
            raise DataValidationError(
                f"High ({self.high}) < Low ({self.low}) at {self.timestamp}"
            )
        if self.high < self.open or self.high < self.close:
            raise DataValidationError(
                f"High ({self.high}) must be >= Open ({self.open}) and Close ({self.close})"
            )
        if self.low > self.open or self.low > self.close:
            raise DataValidationError(
                f"Low ({self.low}) must be <= Open ({self.open}) and Close ({self.close})"
            )
        if self.open <= 0 or self.high <= 0 or self.low <= 0 or self.close <= 0:
            raise DataValidationError(f"All prices must be positive: {self}")
        if self.volume < 0:
            raise DataValidationError(f"Volume cannot be negative: {self.volume}")
    
    def is_valid(self) -> bool:
        """
        Validate OHLC logic.
        
        Returns:
            True if candle data is valid, False otherwise
        """
        try:
            self.__post_init__()
            return True
        except DataValidationError:
            return False
    
    @property
    def body(self) -> float:
        """Candle body size."""
        return abs(self.close - self.open)
    
    @property
    def body_range(self) -> float:
        """Candle body range as percentage of total range."""
        total_range = self.high - self.low
        if total_range == 0:
            return 0.0
        return self.body / total_range
    
    @property
    def upper_wick(self) -> float:
        """Upper wick size."""
        return self.high - max(self.open, self.close)
    
    @property
    def lower_wick(self) -> float:
        """Lower wick size."""
        return min(self.open, self.close) - self.low
    
    @property
    def total_range(self) -> float:
        """Total candle range."""
        return self.high - self.low
    
    @property
    def is_bullish(self) -> bool:
        """Check if candle is bullish."""
        return self.close > self.open
    
    @property
    def is_bearish(self) -> bool:
        """Check if candle is bearish."""
        return self.close < self.open
    
    @property
    def is_doji(self) -> bool:
        """Check if candle is a doji."""
        return self.body_range < 0.1


@dataclass
class DetectedPattern:
    """Result of pattern detection."""
    pattern: CandlePattern
    direction: PatternDirection
    strength: PatternStrength
    timestamp: datetime
    confidence: float
    candle_count: int
    price_range: Tuple[float, float]
    signals: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def is_bullish(self) -> bool:
        return self.direction in (PatternDirection.BULLISH, PatternDirection.REVERSAL)
    
    def is_bearish(self) -> bool:
        return self.direction in (PatternDirection.BEARISH, PatternDirection.REVERSAL)
    
    def is_reversal(self) -> bool:
        return self.direction == PatternDirection.REVERSAL
    
    def is_continuation(self) -> bool:
        return self.direction == PatternDirection.CONTINUATION
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'pattern': self.pattern.value,
            'direction': self.direction.value,
            'strength': self.strength.value,
            'timestamp': self.timestamp.isoformat(),
            'confidence': self.confidence,
            'candle_count': self.candle_count,
            'price_range': self.price_range,
            'signals': self.signals,
            'metadata': self.metadata,
        }


@dataclass
class PatternResult:
    """Complete pattern detection result."""
    symbol: str
    timeframe: str
    timestamp: datetime
    patterns: List[DetectedPattern]
    candles_analyzed: int
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def add_metadata(self, key: str, value: Any) -> None:
        """Add metadata after creation."""
        self.metadata[key] = value
    
    def get_patterns_by_type(self, pattern_type: CandlePattern) -> List[DetectedPattern]:
        """Filter patterns by type."""
        return [p for p in self.patterns if p.pattern == pattern_type]
    
    def get_patterns_by_direction(self, direction: PatternDirection) -> List[DetectedPattern]:
        """Filter patterns by direction."""
        return [p for p in self.patterns if p.direction == direction]
    
    def get_patterns_by_strength(self, min_strength: PatternStrength) -> List[DetectedPattern]:
        """Filter patterns by minimum strength."""
        return [p for p in self.patterns if p.strength.value >= min_strength.value]
    
    def get_strongest_patterns(self, limit: int = 5) -> List[DetectedPattern]:
        """Get patterns with highest confidence."""
        sorted_patterns = sorted(self.patterns, key=lambda p: p.confidence, reverse=True)
        return sorted_patterns[:limit]
    
    def get_pattern_summary(self) -> Dict[str, Any]:
        """Get statistical summary of detected patterns."""
        if not self.patterns:
            return {
                'total_patterns': 0,
                'by_type': {},
                'by_direction': {},
                'by_strength': {},
                'avg_confidence': 0.0,
                'max_confidence': 0.0,
                'min_confidence': 0.0,
            }
        
        by_type = {}
        by_direction = {}
        by_strength = {}
        total_confidence = 0.0
        
        for pattern in self.patterns:
            # By type
            type_key = pattern.pattern.value
            by_type[type_key] = by_type.get(type_key, 0) + 1
            
            # By direction
            dir_key = pattern.direction.value
            by_direction[dir_key] = by_direction.get(dir_key, 0) + 1
            
            # By strength
            strength_key = pattern.strength.name
            by_strength[strength_key] = by_strength.get(strength_key, 0) + 1
            
            total_confidence += pattern.confidence
        
        count = len(self.patterns)
        
        return {
            'total_patterns': count,
            'by_type': by_type,
            'by_direction': by_direction,
            'by_strength': by_strength,
            'avg_confidence': total_confidence / count,
            'max_confidence': max(p.confidence for p in self.patterns),
            'min_confidence': min(p.confidence for p in self.patterns),
        }


# ==============================================================================
# DETECTOR BASE CLASS
# ==============================================================================

class BasePatternDetector:
    """Base class for all pattern detectors."""
    
    def __init__(self, config: Config, min_confidence: float = 0.5):
        """
        Initialize detector.
        
        Args:
            config: Application configuration
            min_confidence: Minimum confidence threshold
        """
        self.config = config
        self.min_confidence = min_confidence
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self.enabled = True
        self._detection_count = 0
    
    def detect(self, candles: List[CandleData]) -> List[DetectedPattern]:
        """
        Detect patterns in candle data.
        
        Args:
            candles: List of CandleData objects
            
        Returns:
            List of DetectedPattern objects
        """
        raise NotImplementedError("Subclasses must implement detect()")
    
    def pattern_type(self) -> CandlePattern:
        """Return the pattern type this detector detects."""
        raise NotImplementedError("Subclasses must implement pattern_type()")
    
    def min_candles(self) -> int:
        """Minimum candles needed for detection."""
        return 1
    
    def get_info(self) -> Dict[str, Any]:
        """Get detector information."""
        return {
            'name': self.__class__.__name__,
            'pattern_type': self.pattern_type().value if hasattr(self, 'pattern_type') else None,
            'min_candles': self.min_candles() if hasattr(self, 'min_candles') else 1,
            'enabled': self.enabled,
            'min_confidence': self.min_confidence,
            'detection_count': self._detection_count,
        }


# ==============================================================================
# SINGLE CANDLE DETECTORS
# ==============================================================================

class DojiDetector(BasePatternDetector):
    """Detects Doji patterns."""
    
    def pattern_type(self) -> CandlePattern:
        return CandlePattern.DOJI
    
    def min_candles(self) -> int:
        return 1
    
    def detect(self, candles: List[CandleData]) -> List[DetectedPattern]:
        if len(candles) < 1:
            return []
        
        results = []
        candle = candles[-1]
        
        # Validate candle
        if not candle.is_valid():
            return results
        
        if candle.is_doji:
            strength = PatternStrength.MODERATE
            confidence = 0.6
            
            signals = [
                f"Doji detected with body range {candle.body_range:.2%}",
                f"Upper wick: {candle.upper_wick:.4f}, Lower wick: {candle.lower_wick:.4f}"
            ]
            
            # Determine direction based on context
            if len(candles) >= 2:
                prev = candles[-2]
                if prev.is_bearish and candle.close > prev.close:
                    direction = PatternDirection.REVERSAL
                    confidence = 0.7
                    signals.append("Potential reversal after bearish move")
                elif prev.is_bullish and candle.close < prev.close:
                    direction = PatternDirection.REVERSAL
                    confidence = 0.7
                    signals.append("Potential reversal after bullish move")
                else:
                    direction = PatternDirection.NEUTRAL
            else:
                direction = PatternDirection.NEUTRAL
            
            if confidence >= self.min_confidence:
                self._detection_count += 1
                results.append(DetectedPattern(
                    pattern=self.pattern_type(),
                    direction=direction,
                    strength=strength,
                    timestamp=candle.timestamp,
                    confidence=confidence,
                    candle_count=1,
                    price_range=(candle.low, candle.high),
                    signals=signals,
                ))
        
        return results


class HammerDetector(BasePatternDetector):
    """Detects Hammer and Shooting Star patterns."""
    
    def pattern_type(self) -> CandlePattern:
        return CandlePattern.HAMMER
    
    def min_candles(self) -> int:
        return 1
    
    def detect(self, candles: List[CandleData]) -> List[DetectedPattern]:
        if len(candles) < 1:
            return []
        
        results = []
        candle = candles[-1]
        
        # Validate candle
        if not candle.is_valid():
            return results
        
        total_range = candle.total_range
        if total_range == 0:
            return results
        
        body_ratio = candle.body / total_range
        lower_wick_ratio = candle.lower_wick / total_range
        upper_wick_ratio = candle.upper_wick / total_range
        
        # Hammer conditions
        is_hammer = (
            body_ratio < 0.3 and
            lower_wick_ratio > 0.6 and
            upper_wick_ratio < 0.1
        )
        
        if is_hammer:
            # Determine if it's a hammer (bullish reversal) or shooting star (bearish reversal)
            if candle.is_bullish:
                pattern = CandlePattern.HAMMER
                direction = PatternDirection.REVERSAL
                confidence = 0.7
            else:
                pattern = CandlePattern.SHOOTING_STAR
                direction = PatternDirection.REVERSAL
                confidence = 0.7
            
            # Check context for higher confidence
            if len(candles) >= 2:
                prev = candles[-2]
                if direction == PatternDirection.REVERSAL and prev.is_bearish:
                    confidence = 0.8
                    signals = [f"{pattern.value} detected after bearish move"]
                elif direction == PatternDirection.REVERSAL and prev.is_bullish:
                    confidence = 0.8
                    signals = [f"{pattern.value} detected after bullish move"]
                else:
                    signals = [f"{pattern.value} detected in neutral context"]
            else:
                signals = [f"{pattern.value} detected"]
            
            # Determine strength
            if confidence >= 0.8:
                strength = PatternStrength.STRONG
            elif confidence >= 0.6:
                strength = PatternStrength.MODERATE
            else:
                strength = PatternStrength.WEAK
            
            if confidence >= self.min_confidence:
                self._detection_count += 1
                results.append(DetectedPattern(
                    pattern=pattern,
                    direction=direction,
                    strength=strength,
                    timestamp=candle.timestamp,
                    confidence=confidence,
                    candle_count=1,
                    price_range=(candle.low, candle.high),
                    signals=signals,
                ))
        
        return results


class MarubozuDetector(BasePatternDetector):
    """Detects Marubozu patterns."""
    
    def pattern_type(self) -> CandlePattern:
        return CandlePattern.MARUBOZU
    
    def min_candles(self) -> int:
        return 1
    
    def detect(self, candles: List[CandleData]) -> List[DetectedPattern]:
        if len(candles) < 1:
            return []
        
        results = []
        candle = candles[-1]
        
        # Validate candle
        if not candle.is_valid():
            return results
        
        total_range = candle.total_range
        if total_range == 0:
            return results
        
        body_ratio = candle.body / total_range
        upper_wick_ratio = candle.upper_wick / total_range
        lower_wick_ratio = candle.lower_wick / total_range
        
        # Marubozu: very small wicks, large body
        is_marubozu = (
            body_ratio > 0.8 and
            upper_wick_ratio < 0.1 and
            lower_wick_ratio < 0.1
        )
        
        if is_marubozu:
            if candle.is_bullish:
                direction = PatternDirection.CONTINUATION
                direction_name = "bullish"
                confidence = 0.75
            else:
                direction = PatternDirection.CONTINUATION
                direction_name = "bearish"
                confidence = 0.75
            
            signals = [f"Marubozu detected ({direction_name})"]
            
            if confidence >= self.min_confidence:
                self._detection_count += 1
                results.append(DetectedPattern(
                    pattern=self.pattern_type(),
                    direction=direction,
                    strength=PatternStrength.STRONG,
                    timestamp=candle.timestamp,
                    confidence=confidence,
                    candle_count=1,
                    price_range=(candle.low, candle.high),
                    signals=signals,
                ))
        
        return results


# ==============================================================================
# TWO CANDLE DETECTORS
# ==============================================================================

class EngulfingDetector(BasePatternDetector):
    """Detects Engulfing patterns."""
    
    def pattern_type(self) -> CandlePattern:
        return CandlePattern.ENGULFING_BULLISH
    
    def min_candles(self) -> int:
        return 2
    
    def detect(self, candles: List[CandleData]) -> List[DetectedPattern]:
        if len(candles) < 2:
            return []
        
        results = []
        current = candles[-1]
        previous = candles[-2]
        
        # Validate candles
        if not current.is_valid() or not previous.is_valid():
            return results
        
        # Bullish Engulfing: current bullish, previous bearish, current body covers previous
        if (current.is_bullish and previous.is_bearish and
            current.open < previous.close and
            current.close > previous.open):
            
            signals = [
                "Bullish engulfing pattern detected",
                f"Previous bearish candle {previous.close:.4f} -> {previous.open:.4f}",
                f"Current bullish candle {current.open:.4f} -> {current.close:.4f}"
            ]
            
            confidence = 0.75
            if current.body > previous.body * 1.5:
                confidence = 0.85
                signals.append("Strong engulfing with large body ratio")
            
            if confidence >= self.min_confidence:
                self._detection_count += 1
                results.append(DetectedPattern(
                    pattern=CandlePattern.ENGULFING_BULLISH,
                    direction=PatternDirection.REVERSAL,
                    strength=PatternStrength.STRONG if confidence > 0.8 else PatternStrength.MODERATE,
                    timestamp=current.timestamp,
                    confidence=confidence,
                    candle_count=2,
                    price_range=(min(current.low, previous.low), max(current.high, previous.high)),
                    signals=signals,
                ))
        
        # Bearish Engulfing: current bearish, previous bullish, current body covers previous
        if (current.is_bearish and previous.is_bullish and
            current.open > previous.close and
            current.close < previous.open):
            
            signals = [
                "Bearish engulfing pattern detected",
                f"Previous bullish candle {previous.close:.4f} -> {previous.open:.4f}",
                f"Current bearish candle {current.open:.4f} -> {current.close:.4f}"
            ]
            
            confidence = 0.75
            if current.body > previous.body * 1.5:
                confidence = 0.85
                signals.append("Strong engulfing with large body ratio")
            
            if confidence >= self.min_confidence:
                self._detection_count += 1
                results.append(DetectedPattern(
                    pattern=CandlePattern.ENGULFING_BEARISH,
                    direction=PatternDirection.REVERSAL,
                    strength=PatternStrength.STRONG if confidence > 0.8 else PatternStrength.MODERATE,
                    timestamp=current.timestamp,
                    confidence=confidence,
                    candle_count=2,
                    price_range=(min(current.low, previous.low), max(current.high, previous.high)),
                    signals=signals,
                ))
        
        return results


class HaramiDetector(BasePatternDetector):
    """Detects Harami patterns."""
    
    def pattern_type(self) -> CandlePattern:
        return CandlePattern.HARAMI_BULLISH
    
    def min_candles(self) -> int:
        return 2
    
    def detect(self, candles: List[CandleData]) -> List[DetectedPattern]:
        if len(candles) < 2:
            return []
        
        results = []
        current = candles[-1]
        previous = candles[-2]
        
        # Validate candles
        if not current.is_valid() or not previous.is_valid():
            return results
        
        # Bullish Harami: current bullish, previous bearish, current body inside previous
        if (current.is_bullish and previous.is_bearish and
            current.open > previous.close and
            current.close < previous.open):
            
            signals = [
                "Bullish harami pattern detected",
                "Small bullish candle inside previous bearish body"
            ]
            
            confidence = 0.65
            
            if confidence >= self.min_confidence:
                self._detection_count += 1
                results.append(DetectedPattern(
                    pattern=CandlePattern.HARAMI_BULLISH,
                    direction=PatternDirection.REVERSAL,
                    strength=PatternStrength.MODERATE,
                    timestamp=current.timestamp,
                    confidence=confidence,
                    candle_count=2,
                    price_range=(previous.low, previous.high),
                    signals=signals,
                ))
        
        # Bearish Harami: current bearish, previous bullish, current body inside previous
        if (current.is_bearish and previous.is_bullish and
            current.open < previous.close and
            current.close > previous.open):
            
            signals = [
                "Bearish harami pattern detected",
                "Small bearish candle inside previous bullish body"
            ]
            
            confidence = 0.65
            
            if confidence >= self.min_confidence:
                self._detection_count += 1
                results.append(DetectedPattern(
                    pattern=CandlePattern.HARAMI_BEARISH,
                    direction=PatternDirection.REVERSAL,
                    strength=PatternStrength.MODERATE,
                    timestamp=current.timestamp,
                    confidence=confidence,
                    candle_count=2,
                    price_range=(previous.low, previous.high),
                    signals=signals,
                ))
        
        return results


# ==============================================================================
# THREE CANDLE DETECTORS
# ==============================================================================

class ThreeWhiteSoldiersDetector(BasePatternDetector):
    """Detects Three White Soldiers pattern."""
    
    def pattern_type(self) -> CandlePattern:
        return CandlePattern.THREE_WHITE_SOLDIERS
    
    def min_candles(self) -> int:
        return 3
    
    def detect(self, candles: List[CandleData]) -> List[DetectedPattern]:
        if len(candles) < 3:
            return []
        
        c1, c2, c3 = candles[-3], candles[-2], candles[-1]
        
        # Validate candles
        if not c1.is_valid() or not c2.is_valid() or not c3.is_valid():
            return []
        
        # Three White Soldiers: three consecutive bullish candles,
        # each closing higher than the previous, with small wicks
        if (c1.is_bullish and c2.is_bullish and c3.is_bullish and
            c2.close > c1.close and c3.close > c2.close and
            c1.body_range > 0.5 and c2.body_range > 0.5 and c3.body_range > 0.5):
            
            signals = [
                "Three White Soldiers pattern detected",
                "Three consecutive strong bullish candles",
                f"Candles: {c1.close:.4f} -> {c2.close:.4f} -> {c3.close:.4f}"
            ]
            
            confidence = 0.8
            
            if confidence >= self.min_confidence:
                self._detection_count += 1
                results.append(DetectedPattern(
                    pattern=self.pattern_type(),
                    direction=PatternDirection.CONTINUATION,
                    strength=PatternStrength.STRONG,
                    timestamp=c3.timestamp,
                    confidence=confidence,
                    candle_count=3,
                    price_range=(c1.low, c3.high),
                    signals=signals,
                ))
        
        return results


class ThreeBlackCrowsDetector(BasePatternDetector):
    """Detects Three Black Crows pattern."""
    
    def pattern_type(self) -> CandlePattern:
        return CandlePattern.THREE_BLACK_CROWS
    
    def min_candles(self) -> int:
        return 3
    
    def detect(self, candles: List[CandleData]) -> List[DetectedPattern]:
        if len(candles) < 3:
            return []
        
        c1, c2, c3 = candles[-3], candles[-2], candles[-1]
        
        # Validate candles
        if not c1.is_valid() or not c2.is_valid() or not c3.is_valid():
            return []
        
        # Three Black Crows: three consecutive bearish candles,
        # each closing lower than the previous, with small wicks
        if (c1.is_bearish and c2.is_bearish and c3.is_bearish and
            c2.close < c1.close and c3.close < c2.close and
            c1.body_range > 0.5 and c2.body_range > 0.5 and c3.body_range > 0.5):
            
            signals = [
                "Three Black Crows pattern detected",
                "Three consecutive strong bearish candles",
                f"Candles: {c1.close:.4f} -> {c2.close:.4f} -> {c3.close:.4f}"
            ]
            
            confidence = 0.8
            
            if confidence >= self.min_confidence:
                self._detection_count += 1
                results.append(DetectedPattern(
                    pattern=self.pattern_type(),
                    direction=PatternDirection.CONTINUATION,
                    strength=PatternStrength.STRONG,
                    timestamp=c3.timestamp,
                    confidence=confidence,
                    candle_count=3,
                    price_range=(c1.low, c3.high),
                    signals=signals,
                ))
        
        return results


# ==============================================================================
# PATTERN MANAGER
# ==============================================================================

class PatternManager:
    """
    Candle pattern discovery manager.
    
    Orchestrates all pattern detectors and provides unified interface.
    """
    
    def __init__(self, config: Config, min_confidence: float = 0.5):
        """
        Initialize the pattern manager.
        
        Args:
            config: Application configuration
            min_confidence: Minimum confidence threshold for patterns
        """
        self.config = config
        self.min_confidence = min_confidence
        self.logger = logging.getLogger(__name__)
        self._detectors: List[BasePatternDetector] = []
        self._detector_map: Dict[CandlePattern, BasePatternDetector] = {}
        self._cache: Dict[str, PatternResult] = {}
        self._registered = False
        
        # Register default detectors
        self._register_detectors()
    
    def _register_detectors(self):
        """Register all default detectors."""
        detectors = [
            DojiDetector(self.config, self.min_confidence),
            HammerDetector(self.config, self.min_confidence),
            MarubozuDetector(self.config, self.min_confidence),
            EngulfingDetector(self.config, self.min_confidence),
            HaramiDetector(self.config, self.min_confidence),
            ThreeWhiteSoldiersDetector(self.config, self.min_confidence),
            ThreeBlackCrowsDetector(self.config, self.min_confidence),
        ]
        
        self._detectors = detectors
        for detector in detectors:
            self._detector_map[detector.pattern_type()] = detector
        
        self._registered = True
        self.logger.debug(f"Registered {len(detectors)} pattern detectors")
    
    def register_detector(self, detector: BasePatternDetector) -> None:
        """
        Register a custom pattern detector.
        
        Args:
            detector: Detector instance
        """
        self._detectors.append(detector)
        self._detector_map[detector.pattern_type()] = detector
        self.logger.debug(f"Registered detector: {detector.__class__.__name__}")
    
    def enable_detector(self, pattern_type: CandlePattern) -> None:
        """Enable a detector by pattern type."""
        if pattern_type in self._detector_map:
            self._detector_map[pattern_type].enabled = True
            self.logger.debug(f"Enabled detector for {pattern_type.value}")
    
    def disable_detector(self, pattern_type: CandlePattern) -> None:
        """Disable a detector by pattern type."""
        if pattern_type in self._detector_map:
            self._detector_map[pattern_type].enabled = False
            self.logger.debug(f"Disabled detector for {pattern_type.value}")
    
    def get_detector_info(self) -> List[Dict[str, Any]]:
        """Get information about all registered detectors."""
        return [detector.get_info() for detector in self._detectors]
    
    def validate_candles(self, candles: List[Dict[str, Any]]) -> List[CandleData]:
        """
        Validate and convert candles to CandleData.
        
        Args:
            candles: List of candle dictionaries
            
        Returns:
            List of valid CandleData objects
            
        Raises:
            DataValidationError: If no valid candles
        """
        candle_data = []
        errors = 0
        
        for i, c in enumerate(candles):
            try:
                # Validate required fields
                required = {'open', 'high', 'low', 'close'}
                if not all(field in c for field in required):
                    self.logger.warning(f"Missing required fields at index {i}: {c.keys()}")
                    errors += 1
                    continue
                
                candle = CandleData(
                    timestamp=c.get('timestamp', datetime.now()),
                    open=float(c['open']),
                    high=float(c['high']),
                    low=float(c['low']),
                    close=float(c['close']),
                    volume=int(c.get('volume', 0)),
                    spread=c.get('spread'),
                )
                candle_data.append(candle)
            except Exception as e:
                errors += 1
                self.logger.warning(f"⚠️ Failed to convert candle at index {i}: {e}")
        
        if not candle_data:
            raise DataValidationError(
                f"No valid candles after validation ({errors} errors)"
            )
        
        if errors > 0:
            self.logger.warning(
                f"⚠️ {errors} candles failed validation out of {len(candles)}"
            )
        
        return candle_data
    
    def detect(self, candles: List[CandleData]) -> List[DetectedPattern]:
        """
        Detect patterns in candle data.
        
        Args:
            candles: List of CandleData objects
            
        Returns:
            List of DetectedPattern objects
        """
        if not candles:
            return []
        
        # Validate all candles
        valid_candles = []
        for i, candle in enumerate(candles):
            if candle.is_valid():
                valid_candles.append(candle)
            else:
                self.logger.warning(f"Invalid candle at index {i}: {candle.timestamp}")
        
        if not valid_candles:
            return []
        
        self.logger.debug(f"Detecting patterns on {len(valid_candles)} valid candles")
        
        all_patterns = []
        
        for detector in self._detectors:
            if not detector.enabled:
                continue
            
            if len(valid_candles) < detector.min_candles():
                continue
            
            try:
                patterns = detector.detect(valid_candles)
                # Filter by min_confidence
                filtered = [p for p in patterns if p.confidence >= self.min_confidence]
                all_patterns.extend(filtered)
            except Exception as e:
                self.logger.warning(
                    f"⚠️ Detector {detector.__class__.__name__} failed: {e}"
                )
        
        # Sort by confidence (highest first)
        all_patterns.sort(key=lambda p: p.confidence, reverse=True)
        
        return all_patterns
    
    def detect_on_symbol(
        self,
        candles: List[Dict[str, Any]],
        symbol: str,
        timeframe: str,
    ) -> PatternResult:
        """
        Detect patterns on a specific symbol.
        
        Args:
            candles: List of candle dictionaries
            symbol: Symbol name
            timeframe: Timeframe
            
        Returns:
            PatternResult object
        """
        # Check cache
        cache_key = f"{symbol}_{timeframe}"
        if cache_key in self._cache:
            self.logger.debug(f"Cache hit for {cache_key}")
            return self._cache[cache_key]
        
        try:
            candle_data = self.validate_candles(candles)
            patterns = self.detect(candle_data)
            
            result = PatternResult(
                symbol=symbol,
                timeframe=timeframe,
                timestamp=datetime.now(),
                patterns=patterns,
                candles_analyzed=len(candle_data),
                metadata={
                    'detectors_used': len(self._detectors),
                    'candle_range': (candle_data[0].timestamp, candle_data[-1].timestamp),
                    'min_confidence': self.min_confidence,
                },
            )
        except DataValidationError as e:
            result = PatternResult(
                symbol=symbol,
                timeframe=timeframe,
                timestamp=datetime.now(),
                patterns=[],
                candles_analyzed=0,
                metadata={'error': str(e)},
            )
        
        # Cache result
        self._cache[cache_key] = result
        
        return result
    
    def get_patterns_by_type(self, result: PatternResult, pattern_type: CandlePattern) -> List[DetectedPattern]:
        """Filter patterns by type."""
        return result.get_patterns_by_type(pattern_type)
    
    def get_patterns_by_direction(self, result: PatternResult, direction: PatternDirection) -> List[DetectedPattern]:
        """Filter patterns by direction."""
        return result.get_patterns_by_direction(direction)
    
    def get_patterns_by_strength(self, result: PatternResult, min_strength: PatternStrength) -> List[DetectedPattern]:
        """Filter patterns by minimum strength."""
        patterns = getattr(result, 'patterns', None) or getattr(result, 'detected_patterns', None) or []

        try:
            min_value = float(min_strength.value)
        except (AttributeError, TypeError, ValueError):
            return list(patterns)

        filtered = []
        for pattern in patterns:
            strength = getattr(pattern, 'strength', None)
            try:
                if float(strength.value) >= min_value:
                    filtered.append(pattern)
            except (AttributeError, TypeError, ValueError):
                continue
        return filtered

    def get_strongest_patterns(self, result: PatternResult, limit: int = 5) -> List[DetectedPattern]:
        """Get strongest patterns from a result."""
        return result.get_strongest_patterns(limit)

    def clear_cache(self) -> None:
        """Clear cached pattern results."""
        self._cache.clear()

    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        return {
            'cache_size': len(self._cache),
            'cached_keys': list(self._cache.keys()),
        }


# ==============================================================================
# FACTORY
# ==============================================================================

def create_pattern_manager(config: Optional[Config] = None, min_confidence: float = 0.5) -> PatternManager:
    """Create a PatternManager instance."""
    return PatternManager(config=config, min_confidence=min_confidence)