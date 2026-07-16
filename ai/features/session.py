"""
ai/features/session.py - Calendar and trading-session features

RESPONSIBILITY:
Encode timestamp-derived hour, weekday, month, cyclic calendar, and FX session
features for model consumption.

VERSION: 1.0.0
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Dict, Sequence

import numpy as np
from numpy.typing import NDArray

from ai.utils.time_ops import extract_calendar_features, session_from_hour


# ==============================================================================
# TYPES
# ==============================================================================


FeatureMap = Dict[str, NDArray[np.floating]]


class TradingSession(str, Enum):
    """Coarse UTC FX sessions."""

    ASIA = "asia"
    LONDON = "london"
    LONDON_NY_OVERLAP = "london_ny_overlap"
    NEW_YORK = "new_york"
    OFF_HOURS = "off_hours"


# ==============================================================================
# PUBLIC API
# ==============================================================================


def compute_session_features(timestamps: Sequence[datetime]) -> FeatureMap:
    """Compute calendar and session features from timestamps."""

    calendar = extract_calendar_features(timestamps)
    features: FeatureMap = {f"calendar_{name}": np.asarray(values, dtype=float) for name, values in calendar.items()}
    months = np.array([ts.month for ts in timestamps], dtype=float)
    days = np.array([ts.day for ts in timestamps], dtype=float)
    hours = np.array([ts.hour for ts in timestamps], dtype=float)
    sessions = [session_from_hour(int(hour)) for hour in hours]

    features.update(
        {
            "calendar_month_sin": np.sin(2.0 * np.pi * (months - 1.0) / 12.0),
            "calendar_month_cos": np.cos(2.0 * np.pi * (months - 1.0) / 12.0),
            "calendar_day_of_month": days,
            "calendar_day_of_month_sin": np.sin(2.0 * np.pi * (days - 1.0) / 31.0),
            "calendar_day_of_month_cos": np.cos(2.0 * np.pi * (days - 1.0) / 31.0),
            "calendar_is_month_start": np.array([1.0 if ts.day <= 3 else 0.0 for ts in timestamps], dtype=float),
            "calendar_is_month_end": np.array([1.0 if ts.day >= 28 else 0.0 for ts in timestamps], dtype=float),
            "calendar_london_open_window": ((hours >= 7.0) & (hours < 9.0)).astype(float),
            "calendar_new_york_open_window": ((hours >= 12.0) & (hours < 15.0)).astype(float),
            "calendar_session_transition": _session_transition(sessions),
        }
    )
    return features


# ==============================================================================
# HELPERS
# ==============================================================================


def _session_transition(sessions: Sequence[str]) -> NDArray[np.floating]:
    out = np.zeros(len(sessions), dtype=float)
    for idx in range(1, len(sessions)):
        out[idx] = 1.0 if sessions[idx] != sessions[idx - 1] else 0.0
    return out
