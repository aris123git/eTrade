"""
ai/features/session.py - Calendar, time-of-day, and trading-session features

RESPONSIBILITY:
Encode timestamp-derived hour/weekday/month cycles, FX session membership
(Asian / European / American), major open windows (Tokyo, London, NY), and
optional intra-session price behaviour for model consumption.

VERSION: 1.1.0
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Dict, Sequence

import numpy as np
from numpy.typing import NDArray

from ai.utils.math_ops import safe_div
from ai.utils.time_ops import extract_calendar_features, session_from_hour


FeatureMap = Dict[str, NDArray[np.floating]]


class TradingSession(str, Enum):
    """Coarse UTC FX sessions."""

    ASIA = "asia"
    LONDON = "london"
    LONDON_NY_OVERLAP = "london_ny_overlap"
    NEW_YORK = "new_york"
    OFF_HOURS = "off_hours"


# Major open windows in UTC (approximate cash FX / futures opens).
TOKYO_OPEN_UTC = (0, 2)       # 00:00–02:00
LONDON_OPEN_UTC = (7, 9)      # 07:00–09:00
NEW_YORK_OPEN_UTC = (12, 15)  # 12:00–15:00 (includes early NY cash)


def compute_session_features(
    timestamps: Sequence[datetime],
    *,
    open_: NDArray[np.floating] | None = None,
    high: NDArray[np.floating] | None = None,
    low: NDArray[np.floating] | None = None,
    close: NDArray[np.floating] | None = None,
) -> FeatureMap:
    """
    Compute calendar, time-of-day, and session-analysis features.

    When OHLC arrays are provided, also emits intra-session return / range
    statistics reset at each session transition.
    """

    calendar = extract_calendar_features(timestamps)
    features: FeatureMap = {f"calendar_{name}": np.asarray(values, dtype=float) for name, values in calendar.items()}
    months = np.array([ts.month for ts in timestamps], dtype=float)
    days = np.array([ts.day for ts in timestamps], dtype=float)
    hours = np.array([ts.hour + ts.minute / 60.0 for ts in timestamps], dtype=float)
    hour_int = np.array([ts.hour for ts in timestamps], dtype=int)
    minutes = np.array([ts.minute for ts in timestamps], dtype=float)
    sessions = [session_from_hour(int(h)) for h in hour_int]

    # Region aliases requested by product naming
    asian = np.array([1.0 if s == "asia" else 0.0 for s in sessions], dtype=float)
    european = np.array([1.0 if s in {"london", "london_ny_overlap"} else 0.0 for s in sessions], dtype=float)
    american = np.array([1.0 if s in {"london_ny_overlap", "new_york"} else 0.0 for s in sessions], dtype=float)

    tokyo_open = ((hours >= TOKYO_OPEN_UTC[0]) & (hours < TOKYO_OPEN_UTC[1])).astype(float)
    london_open = ((hours >= LONDON_OPEN_UTC[0]) & (hours < LONDON_OPEN_UTC[1])).astype(float)
    ny_open = ((hours >= NEW_YORK_OPEN_UTC[0]) & (hours < NEW_YORK_OPEN_UTC[1])).astype(float)

    features.update(
        {
            "calendar_month_sin": np.sin(2.0 * np.pi * (months - 1.0) / 12.0),
            "calendar_month_cos": np.cos(2.0 * np.pi * (months - 1.0) / 12.0),
            "calendar_day_of_month": days,
            "calendar_day_of_month_sin": np.sin(2.0 * np.pi * (days - 1.0) / 31.0),
            "calendar_day_of_month_cos": np.cos(2.0 * np.pi * (days - 1.0) / 31.0),
            "calendar_is_month_start": np.array([1.0 if ts.day <= 3 else 0.0 for ts in timestamps], dtype=float),
            "calendar_is_month_end": np.array([1.0 if ts.day >= 28 else 0.0 for ts in timestamps], dtype=float),
            # Time-of-day / major opens
            "tod_hour_fraction": hours / 24.0,
            "tod_minute": minutes,
            "tod_tokyo_open_window": tokyo_open,
            "tod_london_open_window": london_open,
            "tod_new_york_open_window": ny_open,
            "calendar_london_open_window": london_open,  # backward compatible
            "calendar_new_york_open_window": ny_open,
            "calendar_session_transition": _session_transition(sessions),
            # Session analysis (Asian / European / American)
            "session_asian": asian,
            "session_european": european,
            "session_american": american,
            "session_minutes_into": _minutes_into_session(hour_int, minutes, sessions),
            "session_progress": _session_progress(hour_int, minutes, sessions),
        }
    )

    if close is not None:
        close_arr = np.asarray(close, dtype=float)
        open_arr = np.asarray(open_, dtype=float) if open_ is not None else close_arr
        high_arr = np.asarray(high, dtype=float) if high is not None else close_arr
        low_arr = np.asarray(low, dtype=float) if low is not None else close_arr
        features.update(
            _session_price_features(sessions, open_arr, high_arr, low_arr, close_arr)
        )

    return features


def _session_transition(sessions: Sequence[str]) -> NDArray[np.floating]:
    out = np.zeros(len(sessions), dtype=float)
    for idx in range(1, len(sessions)):
        out[idx] = 1.0 if sessions[idx] != sessions[idx - 1] else 0.0
    return out


def _session_bounds(session: str) -> tuple[float, float]:
    """Return (start_hour, end_hour] style UTC bounds for progress calc."""
    if session == "asia":
        return 0.0, 7.0
    if session == "london":
        return 7.0, 12.0
    if session == "london_ny_overlap":
        return 12.0, 17.0
    if session == "new_york":
        return 17.0, 22.0
    return 22.0, 24.0


def _minutes_into_session(
    hours: NDArray[np.integer],
    minutes: NDArray[np.floating],
    sessions: Sequence[str],
) -> NDArray[np.floating]:
    out = np.zeros(len(sessions), dtype=float)
    for i, session in enumerate(sessions):
        start, _ = _session_bounds(session)
        current = float(hours[i]) + float(minutes[i]) / 60.0
        out[i] = max(0.0, (current - start) * 60.0)
    return out


def _session_progress(
    hours: NDArray[np.integer],
    minutes: NDArray[np.floating],
    sessions: Sequence[str],
) -> NDArray[np.floating]:
    out = np.zeros(len(sessions), dtype=float)
    for i, session in enumerate(sessions):
        start, end = _session_bounds(session)
        width = max(end - start, 1e-6)
        current = float(hours[i]) + float(minutes[i]) / 60.0
        out[i] = float(np.clip((current - start) / width, 0.0, 1.0))
    return out


def _session_price_features(
    sessions: Sequence[str],
    open_: NDArray[np.floating],
    high: NDArray[np.floating],
    low: NDArray[np.floating],
    close: NDArray[np.floating],
) -> FeatureMap:
    """Cumulative return / range since the start of the current session."""

    n = len(close)
    session_open = np.full(n, np.nan, dtype=float)
    session_high = np.full(n, np.nan, dtype=float)
    session_low = np.full(n, np.nan, dtype=float)
    cur_open = cur_high = cur_low = np.nan
    prev_session = None
    for i in range(n):
        if sessions[i] != prev_session or not np.isfinite(cur_open):
            cur_open = float(open_[i])
            cur_high = float(high[i])
            cur_low = float(low[i])
            prev_session = sessions[i]
        else:
            cur_high = max(cur_high, float(high[i]))
            cur_low = min(cur_low, float(low[i]))
        session_open[i] = cur_open
        session_high[i] = cur_high
        session_low[i] = cur_low

    session_return = safe_div(close - session_open, session_open, default=np.nan)
    session_range = safe_div(session_high - session_low, session_open, default=np.nan)
    dist_from_session_high = safe_div(session_high - close, session_open, default=np.nan)
    dist_from_session_low = safe_div(close - session_low, session_open, default=np.nan)

    return {
        "session_return": session_return,
        "session_range_pct": session_range,
        "session_dist_from_high": dist_from_session_high,
        "session_dist_from_low": dist_from_session_low,
        "session_close_location": safe_div(close - session_low, session_high - session_low, default=0.5),
    }
