"""
discovery/pattern_engine.py - Pattern Discovery Engine

RESPONSIBILITY:
Discover and analyze patterns in market data.

ARCHITECTURAL PRINCIPLES:
1. Pure pattern discovery - No data storage, no I/O, no business logic
2. Statistical analysis of market patterns
3. Type-safe results with validation
4. Extensible pattern detection

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
    'PatternType',
    'PatternDirection',
    'PatternStrength',
    'DetectedPattern',
    'PatternResult',
    'PatternEngine',
    'create_pattern_engine',
]


# ==============================================================================
# ENUMS
# ==============================================================================

class PatternType(Enum):
    """Types of detectable patterns."""
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
    BREAKOUT = "breakout"
    RETRACEMENT = "retracement"
    
    # Statistical patterns
    TREND = "trend"
    CONSOLIDATION = "consolidation"
    EXPANSION = "expansion"
    COMPRESSION = "compression"
    ACCELERATION = "acceleration"
    DECELERATION = "deceleration"


class PatternDirection(Enum):
    """Direction of a pattern."""
    BULLISH = "bullish"
    BEARISH = "bearish"
    NEUTRAL = "neutral"
    REVERSAL = "reversal"
    CONTINUATION = "continuation"


class PatternStrength(Enum):
    """Strength of a detected pattern."""
    VERY_STRONG = 1.0
    STRONG = 0.8
    MODERATE = 0.6
    WEAK = 0.4
    VERY_WEAK = 0.2


# ==============================================================================
# DATA MODELS
# ==============================================================================

@dataclass
class DetectedPattern:
    """A detected pattern."""
    pattern_type: PatternType
    direction: PatternDirection
    strength: PatternStrength
    confidence: float
    start_index: int
    end_index: int
    timestamp: datetime
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
            'pattern_type': self.pattern_type.value,
            'direction': self.direction.value,
            'strength': self.strength.value,
            'confidence': self.confidence,
            'start_index': self.start_index,
            'end_index': self.end_index,
            'timestamp': self.timestamp.isoformat(),
            'candle_count': self.candle_count,
            'price_range': self.price_range,
            'signals': self.signals,
            'metadata': self.metadata,
        }


@dataclass
class PatternResult:
    """Result of pattern detection."""
    symbol: str
    timeframe: str
    timestamp: datetime
    patterns: List[DetectedPattern]
    candles_analyzed: int
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def get_patterns_by_type(self, pattern_type: PatternType) -> List[DetectedPattern]:
        """Filter patterns by type."""
        return [p for p in self.patterns if p.pattern_type == pattern_type]
    
    def get_patterns_by_direction(self, direction: PatternDirection) -> List[DetectedPattern]:
        """Filter patterns by direction."""
        return [p for p in self.patterns if p.direction == direction]
    
    def get_patterns_by_strength(self, min_strength: PatternStrength) -> List[DetectedPattern]:
        """Filter patterns by strength."""
        return [p for p in self.patterns if p.strength.value >= min_strength.value]
    
    def get_strongest(self, limit: int = 10) -> List[DetectedPattern]:
        """Get strongest patterns."""
        sorted_patterns = sorted(self.patterns, key=lambda p: p.confidence, reverse=True)
        return sorted_patterns[:limit]
    
    def get_summary(self) -> Dict[str, Any]:
        """Get summary of pattern detection."""
        return {
            'symbol': self.symbol,
            'timeframe': self.timeframe,
            'total_patterns': len(self.patterns),
            'candles_analyzed': self.candles_analyzed,
            'by_type': {
                t.value: sum(1 for p in self.patterns if p.pattern_type == t)
                for t in PatternType
            },
            'by_direction': {
                d.value: sum(1 for p in self.patterns if p.direction == d)
                for d in PatternDirection
            },
            'by_strength': {
                s.value: sum(1 for p in self.patterns if p.strength == s)
                for s in PatternStrength
            },
            'avg_confidence': sum(p.confidence for p in self.patterns) / len(self.patterns) if self.patterns else 0.0,
        }


# ==============================================================================
# PATTERN ENGINE
# ==============================================================================

class PatternEngine:
    """
    Pattern discovery engine.
    
    Discovers and analyzes patterns in market data.
    """
    
    # Pattern thresholds
    DOJI_THRESHOLD = 0.1          # Body < 10% of range
    HAMMER_THRESHOLD = 0.6        # Lower wick > 60% of range
    MARUBOZU_THRESHOLD = 0.8      # Body > 80% of range
    ENGULFING_THRESHOLD = 1.0     # Body covers previous body
    
    def __init__(self, config: Config):
        """
        Initialize the pattern engine.
        
        Args:
            config: Application configuration
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Pattern thresholds from config
        self.doji_threshold = getattr(config, 'DOJI_THRESHOLD', self.DOJI_THRESHOLD)
        self.hammer_threshold = getattr(config, 'HAMMER_THRESHOLD', self.HAMMER_THRESHOLD)
        self.marubozu_threshold = getattr(config, 'MARUBOZU_THRESHOLD', self.MARUBOZU_THRESHOLD)
        self.engulfing_threshold = getattr(config, 'ENGULFING_THRESHOLD', self.ENGULFING_THRESHOLD)
        
        # Minimum confidence threshold
        self.min_confidence = getattr(config, 'PATTERN_MIN_CONFIDENCE', 0.5)
        
        # Pattern detectors
        self._detectors = self._register_detectors()
        
        self.logger.info(
            f"✅ PatternEngine initialized with {len(self._detectors)} detectors"
        )
    
    # ==========================================================================
    # PUBLIC METHODS
    # ==========================================================================
    
    def detect_patterns(
        self,
        candles: List[Dict[str, Any]],
        symbol: str,
        timeframe: str,
    ) -> PatternResult:
        """
        Detect patterns in candle data.
        
        Args:
            candles: List of candle dictionaries
            symbol: Symbol name
            timeframe: Timeframe
            
        Returns:
            PatternResult object
        """
        if not candles:
            return PatternResult(
                symbol=symbol,
                timeframe=timeframe,
                timestamp=datetime.now(),
                patterns=[],
                candles_analyzed=0,
                metadata={'error': 'No candles provided'},
            )
        
        self.logger.debug(f"Detecting patterns for {symbol} {timeframe}")
        
        try:
            # Validate candles
            if not self._validate_candles(candles):
                raise DataValidationError("Invalid candles provided")
            
            # Extract candle data
            candle_data = self._extract_candle_data(candles)
            
            # Run all detectors
            patterns = []
            for detector in self._detectors:
                if detector['enabled']:
                    detected = detector['function'](candle_data)
                    patterns.extend(detected)
            
            # Filter by confidence
            patterns = [p for p in patterns if p.confidence >= self.min_confidence]
            
            # Sort by confidence
            patterns.sort(key=lambda p: p.confidence, reverse=True)
            
            result = PatternResult(
                symbol=symbol,
                timeframe=timeframe,
                timestamp=datetime.now(),
                patterns=patterns,
                candles_analyzed=len(candles),
                metadata={
                    'detectors_used': len(self._detectors),
                    'patterns_found': len(patterns),
                    'min_confidence': self.min_confidence,
                },
            )
            
            self.logger.debug(
                f"Pattern detection complete: {len(patterns)} patterns found"
            )
            
            return result
            
        except Exception as e:
            self.logger.error(f"Pattern detection failed: {e}")
            raise DiscoveryError(f"Failed to detect patterns: {e}")
    
    def register_detector(
        self,
        name: str,
        detector_func: Callable,
        enabled: bool = True,
    ) -> None:
        """
        Register a custom pattern detector.
        
        Args:
            name: Detector name
            detector_func: Function that takes candle data and returns patterns
            enabled: Whether the detector is enabled by default
        """
        self._detectors.append({
            'name': name,
            'function': detector_func,
            'enabled': enabled,
        })
        self.logger.debug(f"Registered detector: {name}")
    
    def enable_detector(self, name: str) -> None:
        """Enable a detector by name."""
        for detector in self._detectors:
            if detector['name'] == name:
                detector['enabled'] = True
                self.logger.debug(f"Enabled detector: {name}")
                return
    
    def disable_detector(self, name: str) -> None:
        """Disable a detector by name."""
        for detector in self._detectors:
            if detector['name'] == name:
                detector['enabled'] = False
                self.logger.debug(f"Disabled detector: {name}")
                return
    
    def get_detector_info(self) -> List[Dict[str, Any]]:
        """Get information about all detectors."""
        return [
            {'name': d['name'], 'enabled': d['enabled']}
            for d in self._detectors
        ]
    
    # ==========================================================================
    # PRIVATE METHODS
    # ==========================================================================
    
    def _register_detectors(self) -> List[Dict[str, Any]]:
        """Register all pattern detectors."""
        return [
            {'name': 'doji', 'function': self._detect_doji, 'enabled': True},
            {'name': 'hammer', 'function': self._detect_hammer, 'enabled': True},
            {'name': 'shooting_star', 'function': self._detect_shooting_star, 'enabled': True},
            {'name': 'marubozu', 'function': self._detect_marubozu, 'enabled': True},
            {'name': 'spinning_top', 'function': self._detect_spinning_top, 'enabled': True},
            {'name': 'engulfing', 'function': self._detect_engulfing, 'enabled': True},
            {'name': 'harami', 'function': self._detect_harami, 'enabled': True},
            {'name': 'three_white_soldiers', 'function': self._detect_three_white_soldiers, 'enabled': True},
            {'name': 'three_black_crows', 'function': self._detect_three_black_crows, 'enabled': True},
            {'name': 'trend', 'function': self._detect_trend, 'enabled': True},
        ]
    
    def _validate_candles(self, candles: List[Dict[str, Any]]) -> bool:
        """Validate candles for pattern detection."""
        if not candles:
            return False
        
        required_fields = {'open', 'high', 'low', 'close', 'timestamp'}
        
        for i, candle in enumerate(candles):
            if not all(field in candle for field in required_fields):
                self.logger.debug(f"Candle {i} missing required fields")
                return False
        
        return True
    
    def _extract_candle_data(self, candles: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Extract candle data for pattern detection."""
        candle_data = []
        for candle in candles:
            candle_data.append({
                'open': candle['open'],
                'high': candle['high'],
                'low': candle['low'],
                'close': candle['close'],
                'volume': candle.get('volume', 0),
                'timestamp': candle['timestamp'],
            })
        return candle_data
    
    # ==========================================================================
    # PATTERN DETECTORS
    # ==========================================================================
    
    def _detect_doji(self, candles: List[Dict[str, Any]]) -> List[DetectedPattern]:
        """Detect Doji patterns."""
        patterns = []
        n = len(candles)
        
        for i in range(n):
            candle = candles[i]
            total_range = candle['high'] - candle['low']
            body = abs(candle['close'] - candle['open'])
            
            if total_range == 0:
                continue
            
            body_ratio = body / total_range
            
            if body_ratio < self.doji_threshold:
                # Determine direction
                if i > 0:
                    prev_close = candles[i-1]['close']
                    if candle['close'] > prev_close:
                        direction = PatternDirection.REVERSAL
                        confidence = 0.65
                    else:
                        direction = PatternDirection.REVERSAL
                        confidence = 0.55
                else:
                    direction = PatternDirection.NEUTRAL
                    confidence = 0.5
                
                patterns.append(DetectedPattern(
                    pattern_type=PatternType.DOJI,
                    direction=direction,
                    strength=PatternStrength.MODERATE,
                    confidence=confidence,
                    start_index=i,
                    end_index=i,
                    timestamp=to_datetime(candle['timestamp']),
                    candle_count=1,
                    price_range=(candle['low'], candle['high']),
                    signals=[f"Doji detected with body ratio {body_ratio:.2%}"],
                    metadata={'body_ratio': body_ratio},
                ))
        
        return patterns
    
    def _detect_hammer(self, candles: List[Dict[str, Any]]) -> List[DetectedPattern]:
        """Detect Hammer patterns."""
        patterns = []
        n = len(candles)
        
        for i in range(n):
            candle = candles[i]
            total_range = candle['high'] - candle['low']
            body = abs(candle['close'] - candle['open'])
            lower_wick = min(candle['open'], candle['close']) - candle['low']
            upper_wick = candle['high'] - max(candle['open'], candle['close'])
            
            if total_range == 0:
                continue
            
            body_ratio = body / total_range
            lower_wick_ratio = lower_wick / total_range
            upper_wick_ratio = upper_wick / total_range
            
            # Hammer: small body, long lower wick
            if (body_ratio < 0.3 and lower_wick_ratio > self.hammer_threshold and
                upper_wick_ratio < 0.1):
                
                # Check if in downtrend
                if i >= 5:
                    trend_direction = self._detect_trend_direction(
                        candles[i-5:i], 'close'
                    )
                    if trend_direction == 'bearish':
                        direction = PatternDirection.REVERSAL
                        confidence = 0.75
                    else:
                        direction = PatternDirection.REVERSAL
                        confidence = 0.55
                else:
                    direction = PatternDirection.REVERSAL
                    confidence = 0.6
                
                patterns.append(DetectedPattern(
                    pattern_type=PatternType.HAMMER,
                    direction=direction,
                    strength=PatternStrength.MODERATE,
                    confidence=confidence,
                    start_index=i,
                    end_index=i,
                    timestamp=to_datetime(candle['timestamp']),
                    candle_count=1,
                    price_range=(candle['low'], candle['high']),
                    signals=[
                        f"Hammer detected: lower wick {lower_wick_ratio:.2%} of range"
                    ],
                    metadata={
                        'body_ratio': body_ratio,
                        'lower_wick_ratio': lower_wick_ratio,
                        'upper_wick_ratio': upper_wick_ratio,
                    },
                ))
        
        return patterns
    
    def _detect_shooting_star(self, candles: List[Dict[str, Any]]) -> List[DetectedPattern]:
        """Detect Shooting Star patterns."""
        patterns = []
        n = len(candles)
        
        for i in range(n):
            candle = candles[i]
            total_range = candle['high'] - candle['low']
            body = abs(candle['close'] - candle['open'])
            upper_wick = candle['high'] - max(candle['open'], candle['close'])
            lower_wick = min(candle['open'], candle['close']) - candle['low']
            
            if total_range == 0:
                continue
            
            body_ratio = body / total_range
            upper_wick_ratio = upper_wick / total_range
            lower_wick_ratio = lower_wick / total_range
            
            # Shooting Star: small body, long upper wick
            if (body_ratio < 0.3 and upper_wick_ratio > self.hammer_threshold and
                lower_wick_ratio < 0.1):
                
                # Check if in uptrend
                if i >= 5:
                    trend_direction = self._detect_trend_direction(
                        candles[i-5:i], 'close'
                    )
                    if trend_direction == 'bullish':
                        direction = PatternDirection.REVERSAL
                        confidence = 0.75
                    else:
                        direction = PatternDirection.REVERSAL
                        confidence = 0.55
                else:
                    direction = PatternDirection.REVERSAL
                    confidence = 0.6
                
                patterns.append(DetectedPattern(
                    pattern_type=PatternType.SHOOTING_STAR,
                    direction=direction,
                    strength=PatternStrength.MODERATE,
                    confidence=confidence,
                    start_index=i,
                    end_index=i,
                    timestamp=to_datetime(candle['timestamp']),
                    candle_count=1,
                    price_range=(candle['low'], candle['high']),
                    signals=[
                        f"Shooting star detected: upper wick {upper_wick_ratio:.2%} of range"
                    ],
                    metadata={
                        'body_ratio': body_ratio,
                        'upper_wick_ratio': upper_wick_ratio,
                        'lower_wick_ratio': lower_wick_ratio,
                    },
                ))
        
        return patterns
    
    def _detect_marubozu(self, candles: List[Dict[str, Any]]) -> List[DetectedPattern]:
        """Detect Marubozu patterns."""
        patterns = []
        n = len(candles)
        
        for i in range(n):
            candle = candles[i]
            total_range = candle['high'] - candle['low']
            body = abs(candle['close'] - candle['open'])
            upper_wick = candle['high'] - max(candle['open'], candle['close'])
            lower_wick = min(candle['open'], candle['close']) - candle['low']
            
            if total_range == 0:
                continue
            
            body_ratio = body / total_range
            upper_wick_ratio = upper_wick / total_range
            lower_wick_ratio = lower_wick / total_range
            
            # Marubozu: large body, small wicks
            if (body_ratio > self.marubozu_threshold and
                upper_wick_ratio < 0.05 and lower_wick_ratio < 0.05):
                
                if candle['close'] > candle['open']:
                    direction = PatternDirection.CONTINUATION
                    direction_name = 'bullish'
                else:
                    direction = PatternDirection.CONTINUATION
                    direction_name = 'bearish'
                
                patterns.append(DetectedPattern(
                    pattern_type=PatternType.MARUBOZU,
                    direction=direction,
                    strength=PatternStrength.STRONG,
                    confidence=0.8,
                    start_index=i,
                    end_index=i,
                    timestamp=to_datetime(candle['timestamp']),
                    candle_count=1,
                    price_range=(candle['low'], candle['high']),
                    signals=[f"Marubozu detected ({direction_name})"],
                    metadata={
                        'body_ratio': body_ratio,
                        'direction': direction_name,
                    },
                ))
        
        return patterns
    
    def _detect_spinning_top(self, candles: List[Dict[str, Any]]) -> List[DetectedPattern]:
        """Detect Spinning Top patterns."""
        patterns = []
        n = len(candles)
        
        for i in range(n):
            candle = candles[i]
            total_range = candle['high'] - candle['low']
            body = abs(candle['close'] - candle['open'])
            
            if total_range == 0:
                continue
            
            body_ratio = body / total_range
            
            # Spinning Top: medium body, medium wicks
            if 0.3 < body_ratio < 0.6:
                patterns.append(DetectedPattern(
                    pattern_type=PatternType.SPINNING_TOP,
                    direction=PatternDirection.NEUTRAL,
                    strength=PatternStrength.WEAK,
                    confidence=0.5,
                    start_index=i,
                    end_index=i,
                    timestamp=to_datetime(candle['timestamp']),
                    candle_count=1,
                    price_range=(candle['low'], candle['high']),
                    signals=["Spinning top detected"],
                    metadata={'body_ratio': body_ratio},
                ))
        
        return patterns
    
    def _detect_engulfing(self, candles: List[Dict[str, Any]]) -> List[DetectedPattern]:
        """Detect Engulfing patterns."""
        patterns = []
        n = len(candles)
        
        if n < 2:
            return patterns
        
        for i in range(1, n):
            current = candles[i]
            previous = candles[i-1]
            
            current_body = abs(current['close'] - current['open'])
            previous_body = abs(previous['close'] - previous['open'])
            
            # Bullish Engulfing
            if (current['close'] > current['open'] and
                previous['close'] < previous['open'] and
                current['open'] < previous['close'] and
                current['close'] > previous['open']):
                
                confidence = 0.7
                if current_body > previous_body * 1.5:
                    confidence = 0.85
                
                patterns.append(DetectedPattern(
                    pattern_type=PatternType.ENGULFING_BULLISH,
                    direction=PatternDirection.REVERSAL,
                    strength=PatternStrength.MODERATE if confidence < 0.8 else PatternStrength.STRONG,
                    confidence=confidence,
                    start_index=i-1,
                    end_index=i,
                    timestamp=to_datetime(current['timestamp']),
                    candle_count=2,
                    price_range=(
                        min(current['low'], previous['low']),
                        max(current['high'], previous['high'])
                    ),
                    signals=["Bullish engulfing pattern detected"],
                    metadata={
                        'current_body': current_body,
                        'previous_body': previous_body,
                        'ratio': current_body / previous_body if previous_body > 0 else 0,
                    },
                ))
            
            # Bearish Engulfing
            if (current['close'] < current['open'] and
                previous['close'] > previous['open'] and
                current['open'] > previous['close'] and
                current['close'] < previous['open']):
                
                confidence = 0.7
                if current_body > previous_body * 1.5:
                    confidence = 0.85
                
                patterns.append(DetectedPattern(
                    pattern_type=PatternType.ENGULFING_BEARISH,
                    direction=PatternDirection.REVERSAL,
                    strength=PatternStrength.MODERATE if confidence < 0.8 else PatternStrength.STRONG,
                    confidence=confidence,
                    start_index=i-1,
                    end_index=i,
                    timestamp=to_datetime(current['timestamp']),
                    candle_count=2,
                    price_range=(
                        min(current['low'], previous['low']),
                        max(current['high'], previous['high'])
                    ),
                    signals=["Bearish engulfing pattern detected"],
                    metadata={
                        'current_body': current_body,
                        'previous_body': previous_body,
                        'ratio': current_body / previous_body if previous_body > 0 else 0,
                    },
                ))
        
        return patterns
    
    def _detect_harami(self, candles: List[Dict[str, Any]]) -> List[DetectedPattern]:
        """Detect Harami patterns."""
        patterns = []
        n = len(candles)
        
        if n < 2:
            return patterns
        
        for i in range(1, n):
            current = candles[i]
            previous = candles[i-1]
            
            current_body = abs(current['close'] - current['open'])
            previous_body = abs(previous['close'] - previous['open'])
            
            # Bullish Harami
            if (current['close'] > current['open'] and
                previous['close'] < previous['open'] and
                current['open'] > previous['close'] and
                current['close'] < previous['open']):
                
                patterns.append(DetectedPattern(
                    pattern_type=PatternType.HARAMI_BULLISH,
                    direction=PatternDirection.REVERSAL,
                    strength=PatternStrength.MODERATE,
                    confidence=0.65,
                    start_index=i-1,
                    end_index=i,
                    timestamp=to_datetime(current['timestamp']),
                    candle_count=2,
                    price_range=(previous['low'], previous['high']),
                    signals=["Bullish harami pattern detected"],
                ))
            
            # Bearish Harami
            if (current['close'] < current['open'] and
                previous['close'] > previous['open'] and
                current['open'] < previous['close'] and
                current['close'] > previous['open']):
                
                patterns.append(DetectedPattern(
                    pattern_type=PatternType.HARAMI_BEARISH,
                    direction=PatternDirection.REVERSAL,
                    strength=PatternStrength.MODERATE,
                    confidence=0.65,
                    start_index=i-1,
                    end_index=i,
                    timestamp=to_datetime(current['timestamp']),
                    candle_count=2,
                    price_range=(previous['low'], previous['high']),
                    signals=["Bearish harami pattern detected"],
                ))
        
        return patterns
    
    def _detect_three_white_soldiers(self, candles: List[Dict[str, Any]]) -> List[DetectedPattern]:
        """Detect Three White Soldiers pattern."""
        patterns = []
        n = len(candles)
        
        if n < 3:
            return patterns
        
        for i in range(2, n):
            c1, c2, c3 = candles[i-2], candles[i-1], candles[i]
            
            # Three consecutive bullish candles with higher closes
            if (c1['close'] > c1['open'] and c2['close'] > c2['open'] and
                c3['close'] > c3['open'] and
                c2['close'] > c1['close'] and c3['close'] > c2['close']):
                
                # Check body sizes
                body1 = abs(c1['close'] - c1['open'])
                body2 = abs(c2['close'] - c2['open'])
                body3 = abs(c3['close'] - c3['open'])
                
                total_range1 = c1['high'] - c1['low']
                total_range2 = c2['high'] - c2['low']
                total_range3 = c3['high'] - c3['low']
                
                if (total_range1 > 0 and total_range2 > 0 and total_range3 > 0 and
                    body1 / total_range1 > 0.5 and
                    body2 / total_range2 > 0.5 and
                    body3 / total_range3 > 0.5):
                    
                    patterns.append(DetectedPattern(
                        pattern_type=PatternType.THREE_WHITE_SOLDIERS,
                        direction=PatternDirection.CONTINUATION,
                        strength=PatternStrength.STRONG,
                        confidence=0.8,
                        start_index=i-2,
                        end_index=i,
                        timestamp=to_datetime(c3['timestamp']),
                        candle_count=3,
                        price_range=(c1['low'], c3['high']),
                        signals=["Three white soldiers pattern detected"],
                    ))
        
        return patterns
    
    def _detect_three_black_crows(self, candles: List[Dict[str, Any]]) -> List[DetectedPattern]:
        """Detect Three Black Crows pattern."""
        patterns = []
        n = len(candles)
        
        if n < 3:
            return patterns
        
        for i in range(2, n):
            c1, c2, c3 = candles[i-2], candles[i-1], candles[i]
            
            # Three consecutive bearish candles with lower closes
            if (c1['close'] < c1['open'] and c2['close'] < c2['open'] and
                c3['close'] < c3['open'] and
                c2['close'] < c1['close'] and c3['close'] < c2['close']):
                
                # Check body sizes
                body1 = abs(c1['close'] - c1['open'])
                body2 = abs(c2['close'] - c2['open'])
                body3 = abs(c3['close'] - c3['open'])
                
                total_range1 = c1['high'] - c1['low']
                total_range2 = c2['high'] - c2['low']
                total_range3 = c3['high'] - c3['low']
                
                if (total_range1 > 0 and total_range2 > 0 and total_range3 > 0 and
                    body1 / total_range1 > 0.5 and
                    body2 / total_range2 > 0.5 and
                    body3 / total_range3 > 0.5):
                    
                    patterns.append(DetectedPattern(
                        pattern_type=PatternType.THREE_BLACK_CROWS,
                        direction=PatternDirection.CONTINUATION,
                        strength=PatternStrength.STRONG,
                        confidence=0.8,
                        start_index=i-2,
                        end_index=i,
                        timestamp=to_datetime(c3['timestamp']),
                        candle_count=3,
                        price_range=(c3['low'], c1['high']),
                        signals=["Three black crows pattern detected"],
                    ))
        
        return patterns
    
    def _detect_trend(self, candles: List[Dict[str, Any]]) -> List[DetectedPattern]:
        """Detect trend patterns."""
        patterns = []
        n = len(candles)
        
        if n < 5:
            return patterns
        
        # Check for trend using linear regression or simple slope
        for i in range(4, n):
            window = candles[i-4:i+1]
            prices = [c['close'] for c in window]
            
            # Simple slope calculation
            x = list(range(len(prices)))
            n_win = len(prices)
            sum_x = sum(x)
            sum_y = sum(prices)
            sum_xy = sum(x[i] * prices[i] for i in range(n_win))
            sum_xx = sum(x[i] * x[i] for i in range(n_win))
            
            denominator = n_win * sum_xx - sum_x * sum_x
            if denominator == 0:
                continue
            
            slope = (n_win * sum_xy - sum_x * sum_y) / denominator
            
            # Determine trend type
            if slope > 0.5:
                # Uptrend
                # Check for acceleration
                if i >= 2:
                    prev_window = candles[i-4:i]
                    prev_prices = [c['close'] for c in prev_window]
                    prev_slope = self._calculate_slope(prev_prices)
                    
                    if slope > prev_slope * 1.5:
                        pattern_type = PatternType.ACCELERATION
                        direction = PatternDirection.CONTINUATION
                        confidence = 0.7
                        signals = ["Acceleration detected in uptrend"]
                    else:
                        pattern_type = PatternType.TREND
                        direction = PatternDirection.BULLISH
                        confidence = 0.6
                        signals = ["Uptrend detected"]
                else:
                    pattern_type = PatternType.TREND
                    direction = PatternDirection.BULLISH
                    confidence = 0.6
                    signals = ["Uptrend detected"]
                
                patterns.append(DetectedPattern(
                    pattern_type=pattern_type,
                    direction=direction,
                    strength=PatternStrength.MODERATE,
                    confidence=confidence,
                    start_index=i-4,
                    end_index=i,
                    timestamp=to_datetime(candles[i]['timestamp']),
                    candle_count=5,
                    price_range=(
                        min(c['low'] for c in window),
                        max(c['high'] for c in window)
                    ),
                    signals=signals,
                    metadata={'slope': slope},
                ))
            
            elif slope < -0.5:
                # Downtrend
                if i >= 2:
                    prev_window = candles[i-4:i]
                    prev_prices = [c['close'] for c in prev_window]
                    prev_slope = self._calculate_slope(prev_prices)
                    
                    if slope < prev_slope * 1.5:
                        pattern_type = PatternType.ACCELERATION
                        direction = PatternDirection.CONTINUATION
                        confidence = 0.7
                        signals = ["Acceleration detected in downtrend"]
                    else:
                        pattern_type = PatternType.TREND
                        direction = PatternDirection.BEARISH
                        confidence = 0.6
                        signals = ["Downtrend detected"]
                else:
                    pattern_type = PatternType.TREND
                    direction = PatternDirection.BEARISH
                    confidence = 0.6
                    signals = ["Downtrend detected"]
                
                patterns.append(DetectedPattern(
                    pattern_type=pattern_type,
                    direction=direction,
                    strength=PatternStrength.MODERATE,
                    confidence=confidence,
                    start_index=i-4,
                    end_index=i,
                    timestamp=to_datetime(candles[i]['timestamp']),
                    candle_count=5,
                    price_range=(
                        min(c['low'] for c in window),
                        max(c['high'] for c in window)
                    ),
                    signals=signals,
                    metadata={'slope': slope},
                ))
        
        return patterns
    
    # ==========================================================================
    # HELPER METHODS
    # ==========================================================================
    
    def _detect_trend_direction(self, candles: List[Dict[str, Any]], field: str) -> str:
        """Detect trend direction from candles."""
        if len(candles) < 2:
            return 'neutral'
        
        start_price = candles[0][field]
        end_price = candles[-1][field]
        
        if end_price > start_price * 1.005:
            return 'bullish'
        elif end_price < start_price * 0.995:
            return 'bearish'
        else:
            return 'neutral'
    
    def _calculate_slope(self, values: List[float]) -> float:
        """Calculate slope of values using linear regression."""
        n = len(values)
        if n < 2:
            return 0.0
        
        x = list(range(n))
        sum_x = sum(x)
        sum_y = sum(values)
        sum_xy = sum(x[i] * values[i] for i in range(n))
        sum_xx = sum(x[i] * x[i] for i in range(n))
        
        denominator = n * sum_xx - sum_x * sum_x
        if denominator == 0:
            return 0.0
        
        return (n * sum_xy - sum_x * sum_y) / denominator
    
    def _to_datetime(self, value: Any) -> datetime:
        """Convert value to datetime."""
        return to_datetime(value)


# ==============================================================================
# FACTORY FUNCTION
# ==============================================================================

def create_pattern_engine(config: Config) -> PatternEngine:
    """
    Factory function for PatternEngine creation.
    
    Args:
        config: Application configuration
        
    Returns:
        PatternEngine instance
    """
    return PatternEngine(config)