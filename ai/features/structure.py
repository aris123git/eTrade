"""
ai/features/structure.py - Market structure and trend features

RESPONSIBILITY:
Compute support/resistance distances, swing high/low markers, breakout flags,
trend direction, and trend strength from OHLC arrays.

VERSION: 1.0.0
"""

from __future__ import annotations

from enum import Enum
from typing import Dict

import numpy as np
from numpy.typing import NDArray

from ai.utils.math_ops import rolling_max, rolling_min, safe_div


# ==============================================================================
# TYPES
# ==============================================================================


FeatureMap = Dict[str, NDArray[np.floating]]


class TrendDirection(str, Enum):
    """Trend direction labels encoded numerically in features."""

    DOWN = "down"
    RANGE = "range"
    UP = "up"


# ==============================================================================
# PUBLIC API
# ==============================================================================


def compute_structure_features(
    high: NDArray[np.floating],
    low: NDArray[np.floating],
    close: NDArray[np.floating],
    *,
    swing_lookback: int,
    support_resistance_lookback: int,
) -> FeatureMap:
    """Compute market-structure features."""

    high_arr = np.asarray(high, dtype=float)
    low_arr = np.asarray(low, dtype=float)
    close_arr = np.asarray(close, dtype=float)
    support = _shift(rolling_min(low_arr, support_resistance_lookback), 1)
    resistance = _shift(rolling_max(high_arr, support_resistance_lookback), 1)
    swing_high = _swing_high(high_arr, swing_lookback)
    swing_low = _swing_low(low_arr, swing_lookback)
    slope, r2 = _rolling_linear_regression(close_arr, swing_lookback)
    normalized_slope = safe_div(slope * swing_lookback, close_arr, default=np.nan)
    efficiency = _efficiency_ratio(close_arr, swing_lookback)
    direction = np.where(normalized_slope > 0.0, 1.0, np.where(normalized_slope < 0.0, -1.0, 0.0))
    direction[np.isnan(normalized_slope)] = np.nan

    return {
        f"structure_support_{support_resistance_lookback}": support,
        f"structure_resistance_{support_resistance_lookback}": resistance,
        f"structure_support_distance_{support_resistance_lookback}": safe_div(close_arr - support, close_arr, default=np.nan),
        f"structure_resistance_distance_{support_resistance_lookback}": safe_div(resistance - close_arr, close_arr, default=np.nan),
        f"structure_channel_position_{support_resistance_lookback}": safe_div(
            close_arr - support,
            resistance - support,
            default=np.nan,
        ),
        f"structure_break_resistance_{support_resistance_lookback}": (
            (close_arr > resistance) & np.isfinite(resistance)
        ).astype(float),
        f"structure_break_support_{support_resistance_lookback}": (
            (close_arr < support) & np.isfinite(support)
        ).astype(float),
        f"structure_swing_high_{swing_lookback}": swing_high,
        f"structure_swing_low_{swing_lookback}": swing_low,
        f"structure_higher_high_{swing_lookback}": _higher_high(high_arr, swing_lookback),
        f"structure_lower_low_{swing_lookback}": _lower_low(low_arr, swing_lookback),
        f"structure_trend_slope_{swing_lookback}": normalized_slope,
        f"structure_trend_r2_{swing_lookback}": r2,
        f"structure_trend_direction_{swing_lookback}": direction,
        f"structure_trend_strength_{swing_lookback}": np.abs(normalized_slope) * r2,
        f"structure_efficiency_ratio_{swing_lookback}": efficiency,
    }


# ==============================================================================
# NUMERICAL HELPERS
# ==============================================================================


def _shift(values: NDArray[np.floating], periods: int) -> NDArray[np.floating]:
    arr = np.asarray(values, dtype=float)
    out = np.full(len(arr), np.nan, dtype=float)
    if periods > 0 and len(arr) > periods:
        out[periods:] = arr[:-periods]
    return out


def _swing_high(high: NDArray[np.floating], lookback: int) -> NDArray[np.floating]:
    rolling = rolling_max(high, max(int(lookback), 1))
    out = (high >= rolling).astype(float)
    out[np.isnan(rolling)] = np.nan
    return out


def _swing_low(low: NDArray[np.floating], lookback: int) -> NDArray[np.floating]:
    rolling = rolling_min(low, max(int(lookback), 1))
    out = (low <= rolling).astype(float)
    out[np.isnan(rolling)] = np.nan
    return out


def _higher_high(high: NDArray[np.floating], lookback: int) -> NDArray[np.floating]:
    previous = _shift(rolling_max(high, max(int(lookback), 1)), 1)
    out = (high > previous).astype(float)
    out[np.isnan(previous)] = np.nan
    return out


def _lower_low(low: NDArray[np.floating], lookback: int) -> NDArray[np.floating]:
    previous = _shift(rolling_min(low, max(int(lookback), 1)), 1)
    out = (low < previous).astype(float)
    out[np.isnan(previous)] = np.nan
    return out


def _rolling_linear_regression(values: NDArray[np.floating], window: int) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
    arr = np.asarray(values, dtype=float)
    width = max(int(window), 2)
    slope = np.full(len(arr), np.nan, dtype=float)
    r2 = np.full(len(arr), np.nan, dtype=float)
    if len(arr) < width:
        return slope, r2
    x = np.arange(width, dtype=float)
    x_mean = np.mean(x)
    x_centered = x - x_mean
    x_var = np.sum(x_centered ** 2)
    for idx in range(width - 1, len(arr)):
        y = arr[idx - width + 1 : idx + 1]
        y_mean = np.mean(y)
        y_centered = y - y_mean
        cov = np.sum(x_centered * y_centered)
        slope[idx] = cov / x_var
        fitted = y_mean + slope[idx] * x_centered
        ss_res = np.sum((y - fitted) ** 2)
        ss_tot = np.sum(y_centered ** 2)
        r2[idx] = 1.0 - safe_div(ss_res, ss_tot, default=0.0)
    return slope, r2


def _efficiency_ratio(values: NDArray[np.floating], window: int) -> NDArray[np.floating]:
    arr = np.asarray(values, dtype=float)
    width = max(int(window), 1)
    out = np.full(len(arr), np.nan, dtype=float)
    if len(arr) <= width:
        return out
    absolute_change = np.abs(arr[width:] - arr[:-width])
    path = np.full(len(arr) - width, np.nan, dtype=float)
    for idx in range(width, len(arr)):
        path[idx - width] = np.sum(np.abs(np.diff(arr[idx - width : idx + 1])))
    out[width:] = safe_div(absolute_change, path, default=np.nan)
    return out
