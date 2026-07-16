"""
discovery/session_discovery.py - Trading Session Discovery Engine

RESPONSIBILITY:
Discover and analyze trading sessions from market data.

ARCHITECTURAL PRINCIPLES:
1. Pure discovery - No data storage, no I/O, no business logic
2. Session detection based on market activity patterns
3. Time-based session identification
4. Type-safe results with statistical analysis

WHAT IT NEVER DOES:
- ❌ Connect to MT5
- ❌ Store data
- ❌ Download data
- ❌ Make trading decisions

VERSION: 1.0.0
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime, time, timedelta
from typing import List, Optional, Dict, Any, Tuple, Set
from enum import Enum

from core.config import Config
from core.exceptions import DiscoveryError, DataValidationError


# ==============================================================================
# EXPORTS
# ==============================================================================

__all__ = [
    'SessionType',
    'SessionState',
    'SessionInfo',
    'DetectedSession',
    'SessionResult',
    'SessionDiscovery',
    'create_session_discovery',
]


# ==============================================================================
# ENUMS
# ==============================================================================

class SessionType(Enum):
    """Type of trading session."""
    ASIAN = "asian"
    EUROPEAN = "european"
    US = "us"
    PACIFIC = "pacific"
    OVERLAP = "overlap"
    HOLIDAY = "holiday"
    WEEKEND = "weekend"
    UNKNOWN = "unknown"


class SessionState(Enum):
    """State of a trading session."""
    PRE_OPEN = "pre_open"
    OPEN = "open"
    ACTIVE = "active"
    QUIET = "quiet"
    CLOSING = "closing"
    CLOSED = "closed"


# ==============================================================================
# DATA MODELS
# ==============================================================================

@dataclass(frozen=True)
class SessionInfo:
    """Information about a trading session."""
    session_type: SessionType
    name: str
    timezone: str
    open_time: time
    close_time: time
    overlap: Optional['SessionInfo'] = None
    active_hours: Optional[List[time]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class DetectedSession:
    """Detected trading session."""
    session_type: SessionType
    start_time: datetime
    end_time: datetime
    state: SessionState
    volatility: float
    volume: float
    candle_count: int
    price_range: Tuple[float, float]
    confidence: float
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SessionResult:
    """Complete session discovery result."""
    symbol: str
    timeframe: str
    timestamp: datetime
    sessions: List[DetectedSession]
    current_session: Optional[DetectedSession] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def get_session_by_type(self, session_type: SessionType) -> Optional[DetectedSession]:
        """Get the first session of a specific type."""
        for session in self.sessions:
            if session.session_type == session_type:
                return session
        return None
    
    def get_sessions_by_state(self, state: SessionState) -> List[DetectedSession]:
        """Get all sessions in a specific state."""
        return [s for s in self.sessions if s.state == state]
    
    def get_active_sessions(self) -> List[DetectedSession]:
        """Get all active sessions."""
        return self.get_sessions_by_state(SessionState.ACTIVE)
    
    def get_summary(self) -> Dict[str, Any]:
        """Get summary of discovery results."""
        return {
            'symbol': self.symbol,
            'timeframe': self.timeframe,
            'total_sessions': len(self.sessions),
            'active_sessions': len(self.get_active_sessions()),
            'current_session': self.current_session.session_type.value if self.current_session else None,
            'by_type': {
                t.value: sum(1 for s in self.sessions if s.session_type == t)
                for t in SessionType
            },
            'by_state': {
                t.value: sum(1 for s in self.sessions if s.state == t)
                for t in SessionState
            },
        }


# ==============================================================================
# SESSION DISCOVERY
# ==============================================================================

class SessionDiscovery:
    """
    Trading session discovery engine.
    
    Discovers and analyzes trading sessions from market data.
    """
    
    # Session time definitions (UTC)
    ASIAN_OPEN = time(0, 0)      # 00:00 UTC
    ASIAN_CLOSE = time(9, 0)     # 09:00 UTC
    EUROPEAN_OPEN = time(7, 0)   # 07:00 UTC
    EUROPEAN_CLOSE = time(16, 0) # 16:00 UTC
    US_OPEN = time(13, 0)        # 13:00 UTC
    US_CLOSE = time(22, 0)       # 22:00 UTC
    PACIFIC_OPEN = time(22, 0)   # 22:00 UTC
    PACIFIC_CLOSE = time(6, 0)   # 06:00 UTC
    
    # Session names
    SESSION_NAMES = {
        SessionType.ASIAN: "Asian Session",
        SessionType.EUROPEAN: "European Session",
        SessionType.US: "US Session",
        SessionType.PACIFIC: "Pacific Session",
        SessionType.OVERLAP: "Session Overlap",
        SessionType.HOLIDAY: "Holiday",
        SessionType.WEEKEND: "Weekend",
    }
    
    def __init__(self, config: Config):
        """
        Initialize the session discovery engine.
        
        Args:
            config: Application configuration
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        self._cache: Dict[str, SessionResult] = {}
        
        # Pre-defined session info
        self._session_infos = self._build_session_infos()
        
        # Volatility thresholds (percentage of ATR)
        self._high_volatility_threshold = 0.02
        self._low_volatility_threshold = 0.005
    
    def _build_session_infos(self) -> Dict[SessionType, SessionInfo]:
        """Build predefined session information."""
        return {
            SessionType.ASIAN: SessionInfo(
                session_type=SessionType.ASIAN,
                name="Asian Session",
                timezone="UTC",
                open_time=self.ASIAN_OPEN,
                close_time=self.ASIAN_CLOSE,
                metadata={
                    'major_currency': 'JPY',
                    'volatility': 'moderate',
                    'liquidity': 'medium',
                },
            ),
            SessionType.EUROPEAN: SessionInfo(
                session_type=SessionType.EUROPEAN,
                name="European Session",
                timezone="UTC",
                open_time=self.EUROPEAN_OPEN,
                close_time=self.EUROPEAN_CLOSE,
                metadata={
                    'major_currency': 'EUR',
                    'volatility': 'high',
                    'liquidity': 'high',
                },
            ),
            SessionType.US: SessionInfo(
                session_type=SessionType.US,
                name="US Session",
                timezone="UTC",
                open_time=self.US_OPEN,
                close_time=self.US_CLOSE,
                metadata={
                    'major_currency': 'USD',
                    'volatility': 'high',
                    'liquidity': 'very_high',
                },
            ),
            SessionType.PACIFIC: SessionInfo(
                session_type=SessionType.PACIFIC,
                name="Pacific Session",
                timezone="UTC",
                open_time=self.PACIFIC_OPEN,
                close_time=self.PACIFIC_CLOSE,
                metadata={
                    'major_currency': 'AUD',
                    'volatility': 'moderate',
                    'liquidity': 'low',
                },
            ),
            SessionType.OVERLAP: SessionInfo(
                session_type=SessionType.OVERLAP,
                name="Session Overlap",
                timezone="UTC",
                open_time=time(0, 0),
                close_time=time(0, 0),
                metadata={
                    'description': 'Overlap between sessions',
                },
            ),
        }
    
    # ==========================================================================
    # PUBLIC METHODS
    # ==========================================================================
    
    def discover(
        self,
        candles: List[Dict[str, Any]],
        symbol: str,
        timeframe: str,
    ) -> SessionResult:
        """
        Discover trading sessions from candle data.
        
        Args:
            candles: List of candle dictionaries
            symbol: Symbol name
            timeframe: Timeframe
            
        Returns:
            SessionResult object
        """
        cache_key = f"{symbol}_{timeframe}"
        if cache_key in self._cache:
            self.logger.debug(f"Cache hit: {cache_key}")
            return self._cache[cache_key]
        
        self.logger.debug(f"Discovering sessions for {symbol} {timeframe}")
        
        try:
            # Convert candles to datetime objects
            candle_dates = self._extract_candle_dates(candles)
            
            if len(candle_dates) < 10:
                raise DataValidationError("Not enough candles for session detection")
            
            # Detect sessions
            sessions = self._detect_sessions(candle_dates)
            
            # Calculate session metrics
            for session in sessions:
                self._calculate_session_metrics(session, candles)
            
            # Get current session
            current = self._get_current_session(sessions)
            
            result = SessionResult(
                symbol=symbol,
                timeframe=timeframe,
                timestamp=datetime.now(),
                sessions=sessions,
                current_session=current,
                metadata={
                    'candle_count': len(candles),
                    'date_range': (candle_dates[0], candle_dates[-1]),
                },
            )
            
            self._cache[cache_key] = result
            return result
            
        except Exception as e:
            raise DiscoveryError(f"Failed to discover sessions for {symbol}: {e}")
    
    def get_current_session(self, symbol: str, timeframe: str) -> Optional[DetectedSession]:
        """
        Get current trading session for a symbol.
        
        Args:
            symbol: Symbol name
            timeframe: Timeframe
            
        Returns:
            Current DetectedSession or None
        """
        result = self.discover([], symbol, timeframe)  # Empty candles
        return result.current_session
    
    def get_session_at_time(
        self,
        symbol: str,
        timeframe: str,
        target_time: datetime,
    ) -> Optional[DetectedSession]:
        """
        Get the session at a specific time.
        
        Args:
            symbol: Symbol name
            timeframe: Timeframe
            target_time: Time to check
            
        Returns:
            DetectedSession or None
        """
        result = self.discover([], symbol, timeframe)
        
        for session in result.sessions:
            if session.start_time <= target_time <= session.end_time:
                return session
        
        return None
    
    def is_session_active(
        self,
        symbol: str,
        timeframe: str,
        target_time: Optional[datetime] = None,
    ) -> bool:
        """
        Check if a trading session is active.
        
        Args:
            symbol: Symbol name
            timeframe: Timeframe
            target_time: Time to check (default: now)
            
        Returns:
            True if session is active
        """
        if target_time is None:
            target_time = datetime.now()
        
        session = self.get_session_at_time(symbol, timeframe, target_time)
        return session is not None and session.state == SessionState.ACTIVE
    
    def get_active_sessions(self, symbol: str, timeframe: str) -> List[DetectedSession]:
        """
        Get all active sessions.
        
        Args:
            symbol: Symbol name
            timeframe: Timeframe
            
        Returns:
            List of active DetectedSession objects
        """
        result = self.discover([], symbol, timeframe)
        return result.get_active_sessions()
    
    def get_sessions_by_type(
        self,
        symbol: str,
        timeframe: str,
        session_type: SessionType,
    ) -> List[DetectedSession]:
        """
        Get sessions of a specific type.
        
        Args:
            symbol: Symbol name
            timeframe: Timeframe
            session_type: Session type to filter
            
        Returns:
            List of DetectedSession objects
        """
        result = self.discover([], symbol, timeframe)
        sessions = []
        
        for session in result.sessions:
            if session.session_type == session_type:
                sessions.append(session)
        
        return sessions
    
    def get_cached(self, symbol: str) -> Optional[SessionResult]:
        """Get cached discovery result."""
        for key, result in self._cache.items():
            if symbol in key:
                return result
        return None
    
    def clear_cache(self) -> None:
        """Clear the discovery cache."""
        self._cache.clear()
        self.logger.debug("Session discovery cache cleared")
    
    # ==========================================================================
    # PRIVATE METHODS
    # ==========================================================================
    
    def _extract_candle_dates(self, candles: List[Dict[str, Any]]) -> List[datetime]:
        """Extract datetime objects from candles."""
        dates = []
        for candle in candles:
            if 'timestamp' in candle:
                dt = candle['timestamp']
                if isinstance(dt, (int, float)):
                    dt = datetime.fromtimestamp(dt)
                elif isinstance(dt, str):
                    dt = datetime.fromisoformat(dt)
                dates.append(dt)
        return sorted(dates)
    
    def _detect_sessions(self, candle_dates: List[datetime]) -> List[DetectedSession]:
        """
        Detect trading sessions from candle dates.
        
        Uses time-based detection with session definitions.
        """
        sessions = []
        
        # Group candles by day
        days = self._group_by_day(candle_dates)
        
        for date, day_candles in days.items():
            # Detect sessions for this day
            day_sessions = self._detect_day_sessions(date, day_candles)
            sessions.extend(day_sessions)
        
        # Merge overlapping sessions
        sessions = self._merge_sessions(sessions)
        
        return sessions
    
    def _group_by_day(self, dates: List[datetime]) -> Dict[datetime, List[datetime]]:
        """Group dates by day."""
        days = {}
        for dt in dates:
            day = dt.replace(hour=0, minute=0, second=0, microsecond=0)
            if day not in days:
                days[day] = []
            days[day].append(dt)
        return days
    
    def _detect_day_sessions(
        self,
        date: datetime,
        day_candles: List[datetime],
    ) -> List[DetectedSession]:
        """Detect sessions for a single day."""
        sessions = []
        
        # Check if it's a weekend
        if date.weekday() >= 5:
            sessions.append(self._create_weekend_session(date, day_candles))
            return sessions
        
        # Check predefined sessions
        for session_type, info in self._session_infos.items():
            if session_type == SessionType.OVERLAP:
                continue
            if session_type == SessionType.HOLIDAY:
                continue
            if session_type == SessionType.WEEKEND:
                continue
            
            session = self._create_session_from_info(
                date, info, day_candles
            )
            if session:
                sessions.append(session)
        
        # Detect overlaps
        overlaps = self._detect_overlaps(date, sessions, day_candles)
        sessions.extend(overlaps)
        
        return sessions
    
    def _create_session_from_info(
        self,
        date: datetime,
        info: SessionInfo,
        day_candles: List[datetime],
    ) -> Optional[DetectedSession]:
        """Create a session from session info."""
        # Create session times
        start_time = datetime.combine(date, info.open_time)
        end_time = datetime.combine(date, info.close_time)
        
        # If session crosses midnight (Pacific session)
        if info.open_time > info.close_time:
            end_time = datetime.combine(date + timedelta(days=1), info.close_time)
        
        # Check if there are candles in this session
        session_candles = [
            dt for dt in day_candles
            if start_time <= dt <= end_time
        ]
        
        if not session_candles:
            return None
        
        # Determine session state
        state = self._determine_session_state(start_time, end_time)
        
        return DetectedSession(
            session_type=info.session_type,
            start_time=start_time,
            end_time=end_time,
            state=state,
            volatility=0.0,
            volume=0.0,
            candle_count=len(session_candles),
            price_range=(0.0, 0.0),
            confidence=0.8,
            metadata={
                'session_name': info.name,
                'timezone': info.timezone,
                'candles_in_session': len(session_candles),
                'session_info': {
                    'open_time': info.open_time.isoformat(),
                    'close_time': info.close_time.isoformat(),
                },
            },
        )
    
    def _detect_overlaps(
        self,
        date: datetime,
        sessions: List[DetectedSession],
        day_candles: List[datetime],
    ) -> List[DetectedSession]:
        """Detect overlapping sessions."""
        overlaps = []
        
        # Check Asian-European overlap (07:00 - 09:00 UTC)
        if len(sessions) >= 2:
            asian = next((s for s in sessions if s.session_type == SessionType.ASIAN), None)
            european = next((s for s in sessions if s.session_type == SessionType.EUROPEAN), None)
            
            if asian and european:
                overlap_start = max(asian.start_time, european.start_time)
                overlap_end = min(asian.end_time, european.end_time)
                
                if overlap_start < overlap_end:
                    overlap_candles = [
                        dt for dt in day_candles
                        if overlap_start <= dt <= overlap_end
                    ]
                    
                    if overlap_candles:
                        overlaps.append(DetectedSession(
                            session_type=SessionType.OVERLAP,
                            start_time=overlap_start,
                            end_time=overlap_end,
                            state=SessionState.ACTIVE,
                            volatility=0.0,
                            volume=0.0,
                            candle_count=len(overlap_candles),
                            price_range=(0.0, 0.0),
                            confidence=0.7,
                            metadata={
                                'overlap_type': 'asian_european',
                                'candles_in_session': len(overlap_candles),
                            },
                        ))
        
        # Check European-US overlap (13:00 - 16:00 UTC)
        if len(sessions) >= 2:
            european = next((s for s in sessions if s.session_type == SessionType.EUROPEAN), None)
            us = next((s for s in sessions if s.session_type == SessionType.US), None)
            
            if european and us:
                overlap_start = max(european.start_time, us.start_time)
                overlap_end = min(european.end_time, us.end_time)
                
                if overlap_start < overlap_end:
                    overlap_candles = [
                        dt for dt in day_candles
                        if overlap_start <= dt <= overlap_end
                    ]
                    
                    if overlap_candles:
                        overlaps.append(DetectedSession(
                            session_type=SessionType.OVERLAP,
                            start_time=overlap_start,
                            end_time=overlap_end,
                            state=SessionState.ACTIVE,
                            volatility=0.0,
                            volume=0.0,
                            candle_count=len(overlap_candles),
                            price_range=(0.0, 0.0),
                            confidence=0.7,
                            metadata={
                                'overlap_type': 'european_us',
                                'candles_in_session': len(overlap_candles),
                            },
                        ))
        
        return overlaps
    
    def _create_weekend_session(
        self,
        date: datetime,
        day_candles: List[datetime],
    ) -> DetectedSession:
        """Create a weekend session."""
        start_time = date.replace(hour=0, minute=0, second=0, microsecond=0)
        end_time = date.replace(hour=23, minute=59, second=59, microsecond=999999)
        
        return DetectedSession(
            session_type=SessionType.WEEKEND,
            start_time=start_time,
            end_time=end_time,
            state=SessionState.CLOSED,
            volatility=0.0,
            volume=0.0,
            candle_count=len(day_candles),
            price_range=(0.0, 0.0),
            confidence=0.95,
            metadata={
                'weekday': date.strftime('%A'),
                'candles_in_session': len(day_candles),
            },
        )
    
    def _determine_session_state(self, start_time: datetime, end_time: datetime) -> SessionState:
        """Determine the state of a session."""
        now = datetime.now()
        
        if now < start_time:
            return SessionState.PRE_OPEN
        elif now < start_time + timedelta(minutes=30):
            return SessionState.OPEN
        elif now < end_time - timedelta(minutes=30):
            return SessionState.ACTIVE
        elif now < end_time:
            return SessionState.CLOSING
        else:
            return SessionState.CLOSED
    
    def _calculate_session_metrics(
        self,
        session: DetectedSession,
        candles: List[Dict[str, Any]],
    ) -> None:
        """
        Calculate metrics for a session.
        
        This is a placeholder - full implementation would calculate:
        - Volatility (ATR percentage)
        - Volume (average volume)
        - Price range (high - low)
        """
        # Find candles in this session
        session_candles = []
        for candle in candles:
            dt = candle.get('timestamp')
            if isinstance(dt, (int, float)):
                dt = datetime.fromtimestamp(dt)
            elif isinstance(dt, str):
                dt = datetime.fromisoformat(dt)
            
            if dt and session.start_time <= dt <= session.end_time:
                session_candles.append(candle)
        
        if not session_candles:
            return
        
        # Calculate price range
        highs = [c['high'] for c in session_candles if 'high' in c]
        lows = [c['low'] for c in session_candles if 'low' in c]
        
        if highs and lows:
            session.price_range = (min(lows), max(highs))
        
        # Calculate volume (placeholder - actual implementation would use real volume)
        session.volume = len(session_candles)
        
        # Calculate volatility (placeholder)
        if session.price_range[1] > 0 and session.price_range[0] > 0:
            range_pct = (session.price_range[1] - session.price_range[0]) / session.price_range[0]
            session.volatility = min(range_pct * 100, 100.0)
        
        session.metadata['candles_in_session'] = len(session_candles)
    
    def _merge_sessions(self, sessions: List[DetectedSession]) -> List[DetectedSession]:
        """Merge overlapping sessions."""
        if not sessions:
            return []
        
        # Sort by start time
        sorted_sessions = sorted(sessions, key=lambda s: s.start_time)
        merged = []
        
        for session in sorted_sessions:
            if not merged:
                merged.append(session)
                continue
            
            last = merged[-1]
            
            # Check if sessions overlap
            if session.start_time <= last.end_time:
                # Merge: keep the longer session
                if session.end_time > last.end_time:
                    merged[-1] = DetectedSession(
                        session_type=session.session_type,
                        start_time=last.start_time,
                        end_time=session.end_time,
                        state=session.state,
                        volatility=max(last.volatility, session.volatility),
                        volume=max(last.volume, session.volume),
                        candle_count=last.candle_count + session.candle_count,
                        price_range=(
                            min(last.price_range[0], session.price_range[0]),
                            max(last.price_range[1], session.price_range[1])
                        ),
                        confidence=max(last.confidence, session.confidence),
                        metadata={
                            'merged_from': [
                                last.metadata.get('session_name', 'unknown'),
                                session.metadata.get('session_name', 'unknown'),
                            ],
                            'candles_in_session': last.candle_count + session.candle_count,
                        },
                    )
                else:
                    # Update candle count
                    last.metadata['candles_in_session'] = last.candle_count + session.candle_count
            else:
                merged.append(session)
        
        return merged
    
    def _get_current_session(self, sessions: List[DetectedSession]) -> Optional[DetectedSession]:
        """Get the current session."""
        now = datetime.now()
        
        for session in sessions:
            if session.start_time <= now <= session.end_time:
                return session
        
        return None


# ==============================================================================
# FACTORY FUNCTION
# ==============================================================================

def create_session_discovery(config: Config) -> SessionDiscovery:
    """
    Factory function for SessionDiscovery creation.
    
    Args:
        config: Application configuration
        
    Returns:
        SessionDiscovery instance
    """
    return SessionDiscovery(config)