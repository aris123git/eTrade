"""
ai/utils/time_ops.py - Time and session utilities

VERSION: 1.0.0
"""

from __future__ import annotations

from datetime import datetime
from typing import Dict, Iterable, List, Optional, Sequence
import numpy as np


TIMEFRAME_MINUTES: Dict[str, int] = {
    "M1": 1,
    "M5": 5,
    "M15": 15,
    "M30": 30,
    "H1": 60,
    "H4": 240,
    "D1": 1440,
    "W1": 10080,
    "MN1": 43200,
}


def timeframe_to_minutes(timeframe: str) -> int:
    key = timeframe.upper()
    if key not in TIMEFRAME_MINUTES:
        raise ValueError(f"Unsupported timeframe: {timeframe}")
    return TIMEFRAME_MINUTES[key]


def align_timestamps(
    primary: Sequence[datetime],
    secondary: Sequence[datetime],
) -> List[int]:
    """
    For each primary timestamp, return the index of the latest secondary
    timestamp that is <= primary. Returns -1 when none exist.
    """
    if not secondary:
        return [-1] * len(primary)
    sec = list(secondary)
    result: List[int] = []
    j = -1
    for ts in primary:
        while j + 1 < len(sec) and sec[j + 1] <= ts:
            j += 1
        result.append(j)
    return result


def session_from_hour(hour: int) -> str:
    """Map UTC hour to a coarse FX session label."""
    if 0 <= hour < 7:
        return "asia"
    if 7 <= hour < 12:
        return "london"
    if 12 <= hour < 17:
        return "london_ny_overlap"
    if 17 <= hour < 22:
        return "new_york"
    return "off_hours"


def session_one_hot(hour: int) -> Dict[str, float]:
    session = session_from_hour(hour)
    keys = ["asia", "london", "london_ny_overlap", "new_york", "off_hours"]
    return {f"session_{k}": 1.0 if k == session else 0.0 for k in keys}


def is_weekend(ts: datetime) -> bool:
    return ts.weekday() >= 5


def extract_calendar_features(timestamps: Sequence[datetime]) -> Dict[str, np.ndarray]:
    hours = np.array([ts.hour for ts in timestamps], dtype=float)
    weekdays = np.array([ts.weekday() for ts in timestamps], dtype=float)
    months = np.array([ts.month for ts in timestamps], dtype=float)
    weekends = np.array([1.0 if is_weekend(ts) else 0.0 for ts in timestamps], dtype=float)
    features = {
        "hour": hours,
        "weekday": weekdays,
        "month": months,
        "is_weekend": weekends,
        "hour_sin": np.sin(2 * np.pi * hours / 24.0),
        "hour_cos": np.cos(2 * np.pi * hours / 24.0),
        "weekday_sin": np.sin(2 * np.pi * weekdays / 7.0),
        "weekday_cos": np.cos(2 * np.pi * weekdays / 7.0),
    }
    for key in ("asia", "london", "london_ny_overlap", "new_york", "off_hours"):
        features[f"session_{key}"] = np.array(
            [1.0 if session_from_hour(int(h)) == key else 0.0 for h in hours],
            dtype=float,
        )
    return features
