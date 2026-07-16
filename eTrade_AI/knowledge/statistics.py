"""
knowledge/statistics.py - Long-Term Statistical Knowledge Base

RESPONSIBILITY:
Discover, store, and update long-term statistical knowledge learned by the AI
from historical market data. This is the memory of market behaviour.

This is NOT for:
- Model evaluation
- Backtesting metrics
- Prediction accuracy

This IS for:
- Market behaviour discovery
- Statistical profiles of symbols
- Pattern reliability tracking
- Session/seasonal patterns
- Reusable market knowledge

ARCHITECTURAL PRINCIPLES:
1. Repository-based - SQLite for persistence
2. Incremental updates - Never recompute everything
3. Extensible - Add new statistics without breaking existing
4. Type-safe - Dataclasses for all data structures
5. Production-ready - Scale to millions of candles

VERSION: 1.0.0
"""

import json
import logging
import math
import sqlite3
import statistics as stats
from dataclasses import dataclass, field
from datetime import datetime, time, timedelta
from enum import Enum
from typing import List, Optional, Dict, Any, Tuple, Set, Union, NamedTuple
from collections import defaultdict
from contextlib import contextmanager

from core.config import Config
from core.exceptions import DatabaseError, DataValidationError
from core.utils import to_datetime, format_datetime


# ==============================================================================
# EXPORTS
# ==============================================================================

__all__ = [
    # Data classes
    'VolatilityProfile',
    'SessionProfile',
    'SeasonalProfile',
    'CandleProfile',
    'PatternProfile',
    'SymbolProfile',
    'MarketProfile',
    'TrendProfile',
    'PullbackProfile',
    'BreakoutProfile',
    # Statistics
    'VolatilityStatistics',
    'SessionStatistics',
    'SeasonalStatistics',
    'CandleStatistics',
    'PatternStatistics',
    'SymbolStatistics',
    'MarketStatistics',
    'TrendStatistics',
    'PullbackStatistics',
    'BreakoutStatistics',
    # Manager
    'KnowledgeStatistics',
    'create_knowledge_statistics',
]


# ==============================================================================
# ENUMS
# ==============================================================================

class SessionType(Enum):
    """Trading session types."""
    ASIAN = "asian"
    EUROPEAN = "european"
    US = "us"
    PACIFIC = "pacific"
    OVERLAP = "overlap"
    HOLIDAY = "holiday"
    WEEKEND = "weekend"
    UNKNOWN = "unknown"


class TrendDirection(Enum):
    """Trend direction."""
    UP = "up"
    DOWN = "down"
    SIDEWAYS = "sideways"
    UNKNOWN = "unknown"


class PatternCategory(Enum):
    """Pattern categories."""
    SINGLE_CANDLE = "single_candle"
    TWO_CANDLE = "two_candle"
    THREE_CANDLE = "three_candle"
    COMPLEX = "complex"
    STATISTICAL = "statistical"


# ==============================================================================
# DATA CLASSES - PROFILES
# ==============================================================================

@dataclass
class VolatilityProfile:
    """Volatility profile for a market/timeframe."""
    symbol: str
    timeframe: str
    mean_volatility: float
    median_volatility: float
    std_volatility: float
    min_volatility: float
    max_volatility: float
    p25_volatility: float
    p75_volatility: float
    sample_size: int
    updated_at: datetime
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def volatility_range(self) -> Tuple[float, float]:
        return (self.min_volatility, self.max_volatility)
    
    @property
    def iqr_volatility(self) -> float:
        return self.p75_volatility - self.p25_volatility
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'symbol': self.symbol,
            'timeframe': self.timeframe,
            'mean': self.mean_volatility,
            'median': self.median_volatility,
            'std': self.std_volatility,
            'min': self.min_volatility,
            'max': self.max_volatility,
            'p25': self.p25_volatility,
            'p75': self.p75_volatility,
            'sample_size': self.sample_size,
            'updated_at': self.updated_at.isoformat(),
        }


@dataclass
class SessionProfile:
    """Profile for a specific trading session."""
    symbol: str
    session_type: SessionType
    timeframe: str
    avg_volatility: float
    avg_volume: float
    avg_range: float
    avg_spread: float
    avg_candle_body: float
    candle_count: int
    sample_size: int
    updated_at: datetime
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'symbol': self.symbol,
            'session_type': self.session_type.value,
            'timeframe': self.timeframe,
            'avg_volatility': self.avg_volatility,
            'avg_volume': self.avg_volume,
            'avg_range': self.avg_range,
            'avg_spread': self.avg_spread,
            'avg_candle_body': self.avg_candle_body,
            'candle_count': self.candle_count,
            'sample_size': self.sample_size,
        }


@dataclass
class SeasonalProfile:
    """Seasonal behaviour profile."""
    symbol: str
    season_type: str  # 'month', 'weekday', 'hour'
    season_value: Union[int, str]
    avg_return: float
    avg_volatility: float
    avg_volume: float
    direction_bias: float  # -1 to 1 (bearish to bullish)
    sample_size: int
    updated_at: datetime
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'symbol': self.symbol,
            'season_type': self.season_type,
            'season_value': self.season_value,
            'avg_return': self.avg_return,
            'avg_volatility': self.avg_volatility,
            'avg_volume': self.avg_volume,
            'direction_bias': self.direction_bias,
            'sample_size': self.sample_size,
        }


@dataclass
class CandleProfile:
    """Profile for candle characteristics."""
    symbol: str
    timeframe: str
    avg_body: float
    avg_upper_wick: float
    avg_lower_wick: float
    avg_range: float
    bullish_ratio: float
    doji_frequency: float
    marubozu_frequency: float
    sample_size: int
    updated_at: datetime
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'symbol': self.symbol,
            'timeframe': self.timeframe,
            'avg_body': self.avg_body,
            'avg_upper_wick': self.avg_upper_wick,
            'avg_lower_wick': self.avg_lower_wick,
            'avg_range': self.avg_range,
            'bullish_ratio': self.bullish_ratio,
            'doji_frequency': self.doji_frequency,
            'marubozu_frequency': self.marubozu_frequency,
            'sample_size': self.sample_size,
        }


@dataclass
class PatternProfile:
    """Profile for a specific pattern."""
    pattern_name: str
    pattern_category: PatternCategory
    symbol: str
    timeframe: str
    occurrence_count: int
    success_rate: float
    avg_win: float
    avg_loss: float
    avg_hold_time: float  # in candles
    confidence: float
    updated_at: datetime
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def win_loss_ratio(self) -> float:
        return abs(self.avg_win / self.avg_loss) if self.avg_loss != 0 else 0.0
    
    @property
    def expectancy(self) -> float:
        return self.success_rate * self.avg_win - (1 - self.success_rate) * abs(self.avg_loss)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'pattern_name': self.pattern_name,
            'category': self.pattern_category.value,
            'symbol': self.symbol,
            'timeframe': self.timeframe,
            'occurrence_count': self.occurrence_count,
            'success_rate': self.success_rate,
            'avg_win': self.avg_win,
            'avg_loss': self.avg_loss,
            'avg_hold_time': self.avg_hold_time,
            'confidence': self.confidence,
            'win_loss_ratio': self.win_loss_ratio,
            'expectancy': self.expectancy,
        }


@dataclass
class SymbolProfile:
    """Complete profile for a symbol."""
    symbol: str
    avg_volatility: float
    avg_volume: float
    avg_spread: float
    avg_candle_body: float
    avg_range: float
    bullish_ratio: float
    sample_size: int
    updated_at: datetime
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'symbol': self.symbol,
            'avg_volatility': self.avg_volatility,
            'avg_volume': self.avg_volume,
            'avg_spread': self.avg_spread,
            'avg_candle_body': self.avg_candle_body,
            'avg_range': self.avg_range,
            'bullish_ratio': self.bullish_ratio,
            'sample_size': self.sample_size,
        }


@dataclass
class MarketProfile:
    """Aggregate market profile."""
    market_id: str
    avg_volatility: float
    avg_volume: float
    avg_correlation: float
    symbol_count: int
    sample_size: int
    updated_at: datetime
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'market_id': self.market_id,
            'avg_volatility': self.avg_volatility,
            'avg_volume': self.avg_volume,
            'avg_correlation': self.avg_correlation,
            'symbol_count': self.symbol_count,
            'sample_size': self.sample_size,
        }


@dataclass
class TrendProfile:
    """Profile for trend behaviour."""
    symbol: str
    timeframe: str
    direction: TrendDirection
    avg_duration: float  # in candles
    avg_strength: float
    avg_pullback_depth: float
    continuation_rate: float
    reversal_rate: float
    sample_size: int
    updated_at: datetime
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'symbol': self.symbol,
            'timeframe': self.timeframe,
            'direction': self.direction.value,
            'avg_duration': self.avg_duration,
            'avg_strength': self.avg_strength,
            'avg_pullback_depth': self.avg_pullback_depth,
            'continuation_rate': self.continuation_rate,
            'reversal_rate': self.reversal_rate,
            'sample_size': self.sample_size,
        }


@dataclass
class PullbackProfile:
    """Profile for pullback behaviour."""
    symbol: str
    timeframe: str
    avg_depth: float
    avg_duration: float  # in candles
    recovery_rate: float
    continuation_rate: float
    reversal_rate: float
    sample_size: int
    updated_at: datetime
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'symbol': self.symbol,
            'timeframe': self.timeframe,
            'avg_depth': self.avg_depth,
            'avg_duration': self.avg_duration,
            'recovery_rate': self.recovery_rate,
            'continuation_rate': self.continuation_rate,
            'reversal_rate': self.reversal_rate,
            'sample_size': self.sample_size,
        }


@dataclass
class BreakoutProfile:
    """Profile for breakout behaviour."""
    symbol: str
    timeframe: str
    breakout_type: str  # 'resistance', 'support', 'range'
    success_rate: float
    avg_move: float
    avg_failure_retrace: float
    avg_follow_through: float
    avg_hold_time: float
    sample_size: int
    updated_at: datetime
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'symbol': self.symbol,
            'timeframe': self.timeframe,
            'breakout_type': self.breakout_type,
            'success_rate': self.success_rate,
            'avg_move': self.avg_move,
            'avg_failure_retrace': self.avg_failure_retrace,
            'avg_follow_through': self.avg_follow_through,
            'avg_hold_time': self.avg_hold_time,
            'sample_size': self.sample_size,
        }


# ==============================================================================
# STATISTICS COLLECTORS
# ==============================================================================

class VolatilityStatistics:
    """Collector for volatility statistics."""
    
    def __init__(self):
        self._values: List[float] = []
        self._by_timeframe: Dict[str, List[float]] = defaultdict(list)
        self._by_symbol: Dict[str, List[float]] = defaultdict(list)
        self._by_weekday: Dict[int, List[float]] = defaultdict(list)
        self._by_session: Dict[str, List[float]] = defaultdict(list)
    
    def add(self, value: float, symbol: str, timeframe: str, timestamp: datetime):
        """Add a volatility value."""
        self._values.append(value)
        self._by_timeframe[timeframe].append(value)
        self._by_symbol[symbol].append(value)
        self._by_weekday[timestamp.weekday()].append(value)
        
        # Session detection (simplified)
        hour = timestamp.hour
        if 0 <= hour < 7:
            session = 'asian'
        elif 7 <= hour < 13:
            session = 'european'
        elif 13 <= hour < 22:
            session = 'us'
        else:
            session = 'pacific'
        self._by_session[session].append(value)
    
    def get_profile(self, symbol: str, timeframe: str) -> VolatilityProfile:
        """Get volatility profile for a symbol and timeframe."""
        values = self._by_symbol.get(symbol, [])
        if not values:
            return None
        
        sorted_values = sorted(values)
        n = len(sorted_values)
        
        return VolatilityProfile(
            symbol=symbol,
            timeframe=timeframe,
            mean_volatility=stats.mean(values),
            median_volatility=stats.median(values),
            std_volatility=stats.stdev(values) if n > 1 else 0.0,
            min_volatility=sorted_values[0],
            max_volatility=sorted_values[-1],
            p25_volatility=self._percentile(sorted_values, 0.25),
            p75_volatility=self._percentile(sorted_values, 0.75),
            sample_size=n,
            updated_at=datetime.now(),
        )
    
    def _percentile(self, sorted_values: List[float], p: float) -> float:
        """Calculate percentile from sorted values."""
        n = len(sorted_values)
        if n == 0:
            return 0.0
        idx = p * (n - 1)
        lower = int(math.floor(idx))
        upper = int(math.ceil(idx))
        if lower == upper:
            return sorted_values[lower]
        weight = idx - lower
        return sorted_values[lower] * (1 - weight) + sorted_values[upper] * weight


class SessionStatistics:
    """Collector for session statistics."""
    
    def __init__(self):
        self._sessions: Dict[str, Dict[str, List[Dict[str, float]]]] = defaultdict(
            lambda: defaultdict(list)
        )
    
    def add(
        self,
        symbol: str,
        session_type: SessionType,
        timeframe: str,
        volatility: float,
        volume: float,
        price_range: float,
        spread: float,
        candle_body: float,
    ):
        """Add a session data point."""
        key = f"{symbol}|{timeframe}"
        self._sessions[key][session_type.value].append({
            'volatility': volatility,
            'volume': volume,
            'price_range': price_range,
            'spread': spread,
            'candle_body': candle_body,
        })
    
    def get_profile(self, symbol: str, session_type: SessionType, timeframe: str) -> SessionProfile:
        """Get session profile."""
        key = f"{symbol}|{timeframe}"
        session_data = self._sessions.get(key, {}).get(session_type.value, [])
        
        if not session_data:
            return None
        
        n = len(session_data)
        
        return SessionProfile(
            symbol=symbol,
            session_type=session_type,
            timeframe=timeframe,
            avg_volatility=sum(d['volatility'] for d in session_data) / n,
            avg_volume=sum(d['volume'] for d in session_data) / n,
            avg_range=sum(d['price_range'] for d in session_data) / n,
            avg_spread=sum(d['spread'] for d in session_data) / n,
            avg_candle_body=sum(d['candle_body'] for d in session_data) / n,
            candle_count=n,
            sample_size=n,
            updated_at=datetime.now(),
        )


class SeasonalStatistics:
    """Collector for seasonal statistics."""
    
    def __init__(self):
        self._monthly: Dict[int, List[float]] = defaultdict(list)
        self._weekly: Dict[int, List[float]] = defaultdict(list)
        self._hourly: Dict[int, List[float]] = defaultdict(list)
        self._symbol: str = None
        self._timeframe: str = None
    
    def set_context(self, symbol: str, timeframe: str):
        """Set the current symbol and timeframe."""
        self._symbol = symbol
        self._timeframe = timeframe
    
    def add(self, timestamp: datetime, return_value: float, volatility: float, volume: float):
        """Add a seasonal data point."""
        self._monthly[timestamp.month].append(return_value)
        self._weekly[timestamp.weekday()].append(return_value)
        self._hourly[timestamp.hour].append(return_value)
    
    def get_monthly_profile(self, month: int) -> SeasonalProfile:
        """Get monthly profile."""
        values = self._monthly.get(month, [])
        if not values or not self._symbol:
            return None
        
        avg_return = sum(values) / len(values)
        
        return SeasonalProfile(
            symbol=self._symbol,
            season_type='month',
            season_value=month,
            avg_return=avg_return,
            avg_volatility=0.0,
            avg_volume=0.0,
            direction_bias=max(-1.0, min(1.0, avg_return * 10)),
            sample_size=len(values),
            updated_at=datetime.now(),
        )


class PatternStatistics:
    """Collector for pattern statistics."""
    
    def __init__(self):
        self._patterns: Dict[str, Dict[str, List[Dict[str, Any]]]] = defaultdict(
            lambda: defaultdict(list)
        )
    
    def add(
        self,
        pattern_name: str,
        pattern_category: PatternCategory,
        symbol: str,
        timeframe: str,
        success: bool,
        win: float,
        loss: float,
        hold_time: float,
    ):
        """Add a pattern occurrence."""
        key = f"{symbol}|{timeframe}"
        self._patterns[pattern_name][key].append({
            'success': success,
            'win': win,
            'loss': loss,
            'hold_time': hold_time,
        })
    
    def get_profile(self, pattern_name: str, symbol: str, timeframe: str) -> PatternProfile:
        """Get pattern profile."""
        key = f"{symbol}|{timeframe}"
        occurrences = self._patterns.get(pattern_name, {}).get(key, [])
        
        if not occurrences:
            return None
        
        n = len(occurrences)
        successes = sum(1 for o in occurrences if o['success'])
        success_rate = successes / n if n > 0 else 0.0
        
        wins = [o['win'] for o in occurrences if o['success']]
        losses = [o['loss'] for o in occurrences if not o['success']]
        
        avg_win = sum(wins) / len(wins) if wins else 0.0
        avg_loss = sum(losses) / len(losses) if losses else 0.0
        avg_hold_time = sum(o['hold_time'] for o in occurrences) / n if n > 0 else 0.0
        
        # Confidence: based on sample size and success rate
        confidence = min(1.0, (n / 100) * success_rate)
        
        return PatternProfile(
            pattern_name=pattern_name,
            pattern_category=PatternCategory(pattern_name) if pattern_name in PatternCategory.__members__ else PatternCategory.STATISTICAL,
            symbol=symbol,
            timeframe=timeframe,
            occurrence_count=n,
            success_rate=success_rate,
            avg_win=avg_win,
            avg_loss=avg_loss,
            avg_hold_time=avg_hold_time,
            confidence=confidence,
            updated_at=datetime.now(),
        )


# ==============================================================================
# MAIN KNOWLEDGE STATISTICS CLASS
# ==============================================================================

class KnowledgeStatistics:
    """
    Long-term statistical knowledge base.
    
    Discovers, stores, and updates statistical knowledge about market behaviour.
    This is the AI's memory of how markets behave.
    
    Architecture:
    - Repository-based with SQLite persistence
    - Incremental updates (never recompute everything)
    - Extensible (add new statistics without breaking existing)
    - Scales to millions of candles
    """
    
    # Database schema version
    SCHEMA_VERSION = 1
    
    def __init__(self, config: Config):
        """
        Initialize the knowledge statistics engine.
        
        Args:
            config: Application configuration
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.db_path = getattr(config, 'DB_PATH', 'market_ai.db')
        
        # Initialize collectors
        self._volatility_stats = VolatilityStatistics()
        self._session_stats = SessionStatistics()
        self._seasonal_stats = SeasonalStatistics()
        self._pattern_stats = PatternStatistics()
        
        # Caches
        self._volatility_cache: Dict[str, VolatilityProfile] = {}
        self._session_cache: Dict[str, SessionProfile] = {}
        self._seasonal_cache: Dict[str, SeasonalProfile] = {}
        self._pattern_cache: Dict[str, PatternProfile] = {}
        self._symbol_cache: Dict[str, SymbolProfile] = {}
        
        # Initialize database
        self._init_database()
        
        self.logger.info("✅ KnowledgeStatistics initialized")
    
    # ==========================================================================
    # DATABASE METHODS
    # ==========================================================================
    
    @contextmanager
    def _get_connection(self):
        """Get database connection with context manager."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()
    
    def _init_database(self):
        """Initialize database schema."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            # Volatility profiles
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS knowledge_volatility (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    timeframe TEXT NOT NULL,
                    mean REAL,
                    median REAL,
                    std REAL,
                    min_val REAL,
                    max_val REAL,
                    p25 REAL,
                    p75 REAL,
                    sample_size INTEGER,
                    metadata TEXT,
                    updated_at TIMESTAMP,
                    UNIQUE(symbol, timeframe)
                )
            """)
            
            # Session profiles
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS knowledge_sessions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    session_type TEXT NOT NULL,
                    timeframe TEXT NOT NULL,
                    avg_volatility REAL,
                    avg_volume REAL,
                    avg_range REAL,
                    avg_spread REAL,
                    avg_candle_body REAL,
                    candle_count INTEGER,
                    sample_size INTEGER,
                    metadata TEXT,
                    updated_at TIMESTAMP,
                    UNIQUE(symbol, session_type, timeframe)
                )
            """)
            
            # Seasonal profiles
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS knowledge_seasonal (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    season_type TEXT NOT NULL,
                    season_value TEXT NOT NULL,
                    avg_return REAL,
                    avg_volatility REAL,
                    avg_volume REAL,
                    direction_bias REAL,
                    sample_size INTEGER,
                    metadata TEXT,
                    updated_at TIMESTAMP,
                    UNIQUE(symbol, season_type, season_value)
                )
            """)
            
            # Pattern profiles
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS knowledge_patterns (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    pattern_name TEXT NOT NULL,
                    pattern_category TEXT NOT NULL,
                    symbol TEXT NOT NULL,
                    timeframe TEXT NOT NULL,
                    occurrence_count INTEGER,
                    success_rate REAL,
                    avg_win REAL,
                    avg_loss REAL,
                    avg_hold_time REAL,
                    confidence REAL,
                    metadata TEXT,
                    updated_at TIMESTAMP,
                    UNIQUE(pattern_name, symbol, timeframe)
                )
            """)
            
            # Symbol profiles
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS knowledge_symbols (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT UNIQUE NOT NULL,
                    avg_volatility REAL,
                    avg_volume REAL,
                    avg_spread REAL,
                    avg_candle_body REAL,
                    avg_range REAL,
                    bullish_ratio REAL,
                    sample_size INTEGER,
                    metadata TEXT,
                    updated_at TIMESTAMP
                )
            """)
            
            # Market profiles
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS knowledge_markets (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    market_id TEXT UNIQUE NOT NULL,
                    avg_volatility REAL,
                    avg_volume REAL,
                    avg_correlation REAL,
                    symbol_count INTEGER,
                    sample_size INTEGER,
                    metadata TEXT,
                    updated_at TIMESTAMP
                )
            """)
            
            # Trend profiles
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS knowledge_trends (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    timeframe TEXT NOT NULL,
                    direction TEXT NOT NULL,
                    avg_duration REAL,
                    avg_strength REAL,
                    avg_pullback_depth REAL,
                    continuation_rate REAL,
                    reversal_rate REAL,
                    sample_size INTEGER,
                    metadata TEXT,
                    updated_at TIMESTAMP,
                    UNIQUE(symbol, timeframe, direction)
                )
            """)
            
            # Pullback profiles
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS knowledge_pullbacks (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    timeframe TEXT NOT NULL,
                    avg_depth REAL,
                    avg_duration REAL,
                    recovery_rate REAL,
                    continuation_rate REAL,
                    reversal_rate REAL,
                    sample_size INTEGER,
                    metadata TEXT,
                    updated_at TIMESTAMP,
                    UNIQUE(symbol, timeframe)
                )
            """)
            
            # Breakout profiles
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS knowledge_breakouts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    timeframe TEXT NOT NULL,
                    breakout_type TEXT NOT NULL,
                    success_rate REAL,
                    avg_move REAL,
                    avg_failure_retrace REAL,
                    avg_follow_through REAL,
                    avg_hold_time REAL,
                    sample_size INTEGER,
                    metadata TEXT,
                    updated_at TIMESTAMP,
                    UNIQUE(symbol, timeframe, breakout_type)
                )
            """)
            
            # Create indexes
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_knowledge_volatility_symbol ON knowledge_volatility(symbol)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_knowledge_volatility_timeframe ON knowledge_volatility(timeframe)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_knowledge_sessions_symbol ON knowledge_sessions(symbol)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_knowledge_patterns_symbol ON knowledge_patterns(symbol)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_knowledge_symbols_symbol ON knowledge_symbols(symbol)")
            
            self.logger.info("✅ Database schema initialized")
    
    # ==========================================================================
    # UPDATE METHODS
    # ==========================================================================
    
    def update_volatility(
        self,
        symbol: str,
        timeframe: str,
        candles: List[Dict[str, Any]],
        incremental: bool = True,
    ) -> VolatilityProfile:
        """
        Update volatility statistics for a symbol and timeframe.
        
        Args:
            symbol: Symbol name
            timeframe: Timeframe
            candles: List of candle dictionaries
            incremental: If True, only update with new data
            
        Returns:
            VolatilityProfile
        """
        self.logger.debug(f"Updating volatility for {symbol} {timeframe}")
        
        # Calculate volatility from candles
        for candle in candles:
            high = candle['high']
            low = candle['low']
            close = candle['close']
            timestamp = candle.get('timestamp', datetime.now())
            
            # ATR-like volatility (using true range)
            true_range = high - low
            if 'close_prev' in candle:
                true_range = max(high - low, abs(high - candle['close_prev']), abs(low - candle['close_prev']))
            
            # Normalize by price level
            normalized_vol = true_range / close if close > 0 else 0.0
            
            self._volatility_stats.add(normalized_vol, symbol, timeframe, timestamp)
        
        # Get profile
        profile = self._volatility_stats.get_profile(symbol, timeframe)
        if profile:
            self._save_volatility(profile)
            self._volatility_cache[f"{symbol}|{timeframe}"] = profile
        
        return profile
    
    def update_session(
        self,
        symbol: str,
        timeframe: str,
        candles: List[Dict[str, Any]],
    ) -> List[SessionProfile]:
        """
        Update session statistics for a symbol and timeframe.
        
        Args:
            symbol: Symbol name
            timeframe: Timeframe
            candles: List of candle dictionaries
            
        Returns:
            List of SessionProfile objects
        """
        self.logger.debug(f"Updating session statistics for {symbol} {timeframe}")
        
        # Group candles by session
        session_groups = defaultdict(list)
        for candle in candles:
            timestamp = candle.get('timestamp', datetime.now())
            hour = timestamp.hour
            
            if 0 <= hour < 7:
                session_type = SessionType.ASIAN
            elif 7 <= hour < 13:
                session_type = SessionType.EUROPEAN
            elif 13 <= hour < 22:
                session_type = SessionType.US
            else:
                session_type = SessionType.PACIFIC
            
            session_groups[session_type].append(candle)
        
        profiles = []
        for session_type, session_candles in session_groups.items():
            for candle in session_candles:
                high = candle['high']
                low = candle['low']
                close = candle['close']
                open_price = candle['open']
                volume = candle.get('volume', 0)
                spread = candle.get('spread', 0)
                
                volatility = (high - low) / close if close > 0 else 0.0
                price_range = high - low
                candle_body = abs(close - open_price)
                
                self._session_stats.add(
                    symbol, session_type, timeframe,
                    volatility, volume, price_range, spread, candle_body
                )
            
            profile = self._session_stats.get_profile(symbol, session_type, timeframe)
            if profile:
                self._save_session(profile)
                self._session_cache[f"{symbol}|{session_type.value}|{timeframe}"] = profile
                profiles.append(profile)
        
        return profiles
    
    def update_seasonal(
        self,
        symbol: str,
        timeframe: str,
        candles: List[Dict[str, Any]],
    ) -> List[SeasonalProfile]:
        """
        Update seasonal statistics for a symbol and timeframe.
        
        Args:
            symbol: Symbol name
            timeframe: Timeframe
            candles: List of candle dictionaries
            
        Returns:
            List of SeasonalProfile objects
        """
        self.logger.debug(f"Updating seasonal statistics for {symbol} {timeframe}")
        
        self._seasonal_stats.set_context(symbol, timeframe)
        
        for i in range(1, len(candles)):
            prev = candles[i-1]
            current = candles[i]
            
            timestamp = current.get('timestamp', datetime.now())
            return_val = (current['close'] - prev['close']) / prev['close'] if prev['close'] > 0 else 0.0
            volatility = (current['high'] - current['low']) / current['close'] if current['close'] > 0 else 0.0
            volume = current.get('volume', 0)
            
            self._seasonal_stats.add(timestamp, return_val, volatility, volume)
        
        profiles = []
        for month in range(1, 13):
            profile = self._seasonal_stats.get_monthly_profile(month)
            if profile:
                self._save_seasonal(profile)
                self._seasonal_cache[f"{symbol}|month|{month}"] = profile
                profiles.append(profile)
        
        return profiles
    
    def update_pattern(
        self,
        pattern_name: str,
        pattern_category: PatternCategory,
        symbol: str,
        timeframe: str,
        occurrences: List[Dict[str, Any]],
    ) -> PatternProfile:
        """
        Update pattern statistics.
        
        Args:
            pattern_name: Name of the pattern
            pattern_category: Category of the pattern
            symbol: Symbol name
            timeframe: Timeframe
            occurrences: List of occurrence dictionaries with success, win, loss, hold_time
            
        Returns:
            PatternProfile
        """
        self.logger.debug(f"Updating pattern statistics for {pattern_name} {symbol} {timeframe}")
        
        for occ in occurrences:
            self._pattern_stats.add(
                pattern_name,
                pattern_category,
                symbol,
                timeframe,
                occ.get('success', False),
                occ.get('win', 0.0),
                occ.get('loss', 0.0),
                occ.get('hold_time', 0.0),
            )
        
        profile = self._pattern_stats.get_profile(pattern_name, symbol, timeframe)
        if profile:
            self._save_pattern(profile)
            self._pattern_cache[f"{pattern_name}|{symbol}|{timeframe}"] = profile
        
        return profile
    
    # ==========================================================================
    # GET METHODS
    # ==========================================================================
    
    def get_volatility_profile(self, symbol: str, timeframe: str) -> Optional[VolatilityProfile]:
        """Get volatility profile for a symbol and timeframe."""
        cache_key = f"{symbol}|{timeframe}"
        if cache_key in self._volatility_cache:
            return self._volatility_cache[cache_key]
        
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT * FROM knowledge_volatility
                WHERE symbol = ? AND timeframe = ?
            """, (symbol, timeframe))
            row = cursor.fetchone()
            
            if row:
                profile = VolatilityProfile(
                    symbol=row['symbol'],
                    timeframe=row['timeframe'],
                    mean_volatility=row['mean'],
                    median_volatility=row['median'],
                    std_volatility=row['std'],
                    min_volatility=row['min_val'],
                    max_volatility=row['max_val'],
                    p25_volatility=row['p25'],
                    p75_volatility=row['p75'],
                    sample_size=row['sample_size'],
                    updated_at=to_datetime(row['updated_at']),
                    metadata=json.loads(row['metadata']) if row['metadata'] else {},
                )
                self._volatility_cache[cache_key] = profile
                return profile
        
        return None
    
    def get_session_profile(self, symbol: str, session_type: SessionType, timeframe: str) -> Optional[SessionProfile]:
        """Get session profile."""
        cache_key = f"{symbol}|{session_type.value}|{timeframe}"
        if cache_key in self._session_cache:
            return self._session_cache[cache_key]
        
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT * FROM knowledge_sessions
                WHERE symbol = ? AND session_type = ? AND timeframe = ?
            """, (symbol, session_type.value, timeframe))
            row = cursor.fetchone()
            
            if row:
                profile = SessionProfile(
                    symbol=row['symbol'],
                    session_type=SessionType(row['session_type']),
                    timeframe=row['timeframe'],
                    avg_volatility=row['avg_volatility'],
                    avg_volume=row['avg_volume'],
                    avg_range=row['avg_range'],
                    avg_spread=row['avg_spread'],
                    avg_candle_body=row['avg_candle_body'],
                    candle_count=row['candle_count'],
                    sample_size=row['sample_size'],
                    updated_at=to_datetime(row['updated_at']),
                    metadata=json.loads(row['metadata']) if row['metadata'] else {},
                )
                self._session_cache[cache_key] = profile
                return profile
        
        return None
    
    def get_pattern_profile(self, pattern_name: str, symbol: str, timeframe: str) -> Optional[PatternProfile]:
        """Get pattern profile."""
        cache_key = f"{pattern_name}|{symbol}|{timeframe}"
        if cache_key in self._pattern_cache:
            return self._pattern_cache[cache_key]
        
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT * FROM knowledge_patterns
                WHERE pattern_name = ? AND symbol = ? AND timeframe = ?
            """, (pattern_name, symbol, timeframe))
            row = cursor.fetchone()
            
            if row:
                profile = PatternProfile(
                    pattern_name=row['pattern_name'],
                    pattern_category=PatternCategory(row['pattern_category']),
                    symbol=row['symbol'],
                    timeframe=row['timeframe'],
                    occurrence_count=row['occurrence_count'],
                    success_rate=row['success_rate'],
                    avg_win=row['avg_win'],
                    avg_loss=row['avg_loss'],
                    avg_hold_time=row['avg_hold_time'],
                    confidence=row['confidence'],
                    updated_at=to_datetime(row['updated_at']),
                    metadata=json.loads(row['metadata']) if row['metadata'] else {},
                )
                self._pattern_cache[cache_key] = profile
                return profile
        
        return None
    
    def get_symbol_profile(self, symbol: str) -> Optional[SymbolProfile]:
        """Get symbol profile."""
        if symbol in self._symbol_cache:
            return self._symbol_cache[symbol]
        
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT * FROM knowledge_symbols WHERE symbol = ?
            """, (symbol,))
            row = cursor.fetchone()
            
            if row:
                profile = SymbolProfile(
                    symbol=row['symbol'],
                    avg_volatility=row['avg_volatility'],
                    avg_volume=row['avg_volume'],
                    avg_spread=row['avg_spread'],
                    avg_candle_body=row['avg_candle_body'],
                    avg_range=row['avg_range'],
                    bullish_ratio=row['bullish_ratio'],
                    sample_size=row['sample_size'],
                    updated_at=to_datetime(row['updated_at']),
                    metadata=json.loads(row['metadata']) if row['metadata'] else {},
                )
                self._symbol_cache[symbol] = profile
                return profile
        
        return None
    
    def get_all_statistics(self, symbol: str) -> Dict[str, Any]:
        """Get all statistics for a symbol."""
        return {
            'volatility': self.get_volatility_profile(symbol, 'H1'),
            'symbol': self.get_symbol_profile(symbol),
            'sessions': {
                'asian': self.get_session_profile(symbol, SessionType.ASIAN, 'H1'),
                'european': self.get_session_profile(symbol, SessionType.EUROPEAN, 'H1'),
                'us': self.get_session_profile(symbol, SessionType.US, 'H1'),
            },
        }
    
    # ==========================================================================
    # SAVE METHODS
    # ==========================================================================
    
    def _save_volatility(self, profile: VolatilityProfile):
        """Save volatility profile to database."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT OR REPLACE INTO knowledge_volatility
                (symbol, timeframe, mean, median, std, min_val, max_val, p25, p75,
                 sample_size, metadata, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                profile.symbol, profile.timeframe,
                profile.mean_volatility, profile.median_volatility, profile.std_volatility,
                profile.min_volatility, profile.max_volatility,
                profile.p25_volatility, profile.p75_volatility,
                profile.sample_size,
                json.dumps(profile.metadata),
                profile.updated_at.isoformat(),
            ))
    
    def _save_session(self, profile: SessionProfile):
        """Save session profile to database."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT OR REPLACE INTO knowledge_sessions
                (symbol, session_type, timeframe, avg_volatility, avg_volume,
                 avg_range, avg_spread, avg_candle_body, candle_count,
                 sample_size, metadata, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                profile.symbol, profile.session_type.value, profile.timeframe,
                profile.avg_volatility, profile.avg_volume,
                profile.avg_range, profile.avg_spread, profile.avg_candle_body,
                profile.candle_count, profile.sample_size,
                json.dumps(profile.metadata),
                profile.updated_at.isoformat(),
            ))
    
    def _save_seasonal(self, profile: SeasonalProfile):
        """Save seasonal profile to database."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT OR REPLACE INTO knowledge_seasonal
                (symbol, season_type, season_value, avg_return, avg_volatility,
                 avg_volume, direction_bias, sample_size, metadata, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                profile.symbol, profile.season_type, str(profile.season_value),
                profile.avg_return, profile.avg_volatility, profile.avg_volume,
                profile.direction_bias, profile.sample_size,
                json.dumps(profile.metadata),
                profile.updated_at.isoformat(),
            ))
    
    def _save_pattern(self, profile: PatternProfile):
        """Save pattern profile to database."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT OR REPLACE INTO knowledge_patterns
                (pattern_name, pattern_category, symbol, timeframe,
                 occurrence_count, success_rate, avg_win, avg_loss,
                 avg_hold_time, confidence, metadata, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                profile.pattern_name, profile.pattern_category.value,
                profile.symbol, profile.timeframe,
                profile.occurrence_count, profile.success_rate,
                profile.avg_win, profile.avg_loss,
                profile.avg_hold_time, profile.confidence,
                json.dumps(profile.metadata),
                profile.updated_at.isoformat(),
            ))
    
    # ==========================================================================
    # UTILITY METHODS
    # ==========================================================================
    
    def clear_cache(self):
        """Clear all caches."""
        self._volatility_cache.clear()
        self._session_cache.clear()
        self._seasonal_cache.clear()
        self._pattern_cache.clear()
        self._symbol_cache.clear()
        self.logger.debug("All caches cleared")
    
    def get_summary(self) -> Dict[str, Any]:
        """Get summary of stored statistics."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            # Count records
            cursor.execute("SELECT COUNT(*) FROM knowledge_volatility")
            volatility_count = cursor.fetchone()[0]
            
            cursor.execute("SELECT COUNT(*) FROM knowledge_sessions")
            session_count = cursor.fetchone()[0]
            
            cursor.execute("SELECT COUNT(*) FROM knowledge_patterns")
            pattern_count = cursor.fetchone()[0]
            
            cursor.execute("SELECT COUNT(*) FROM knowledge_symbols")
            symbol_count = cursor.fetchone()[0]
            
            return {
                'volatility_profiles': volatility_count,
                'session_profiles': session_count,
                'pattern_profiles': pattern_count,
                'symbol_profiles': symbol_count,
                'total_records': volatility_count + session_count + pattern_count + symbol_count,
                'cache_size': {
                    'volatility': len(self._volatility_cache),
                    'session': len(self._session_cache),
                    'seasonal': len(self._seasonal_cache),
                    'pattern': len(self._pattern_cache),
                    'symbol': len(self._symbol_cache),
                },
            }


# ==============================================================================
# FACTORY FUNCTION
# ==============================================================================

def create_knowledge_statistics(config: Config) -> KnowledgeStatistics:
    """
    Factory function for KnowledgeStatistics creation.
    
    Args:
        config: Application configuration
        
    Returns:
        KnowledgeStatistics instance
    """
    return KnowledgeStatistics(config)