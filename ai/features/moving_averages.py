"""
ai/features/moving_averages.py - Moving average, VWAP, and cross features

RESPONSIBILITY:
Compute SMA, EMA, VWAP, moving-average distances, slopes, and fast/slow cross
state features from OHLCV arrays.

VERSION: 1.0.0
"""

from __future__ import annotations

from enum import Enum
from typing import Dict, Sequence

import numpy as np
from numpy.typing import NDArray

from ai.utils.math_ops import ema, rolling_mean, safe_div, sma


# ==============================================================================
# TYPES
# ==============================================================================


FeatureMap = Dict[str, NDArray[np.floating]]


class MovingAverageKind(str, Enum):
    """Supported moving-average families."""

    SMA = "sma"
    EMA = "ema"
    VWAP = "vwap"


# ==============================================================================
# PUBLIC API
# ==============================================================================


def compute_moving_average_features(
    high: NDArray[np.floating],
    low: NDArray[np.floating],
    close: NDArray[np.floating],
    volume: NDArray[np.floating],
    *,
    sma_periods: Sequence[int],
    ema_periods: Sequence[int],
    vwap_windows: Sequence[int],
) -> FeatureMap:
    """Compute moving-average features with close-relative distances."""

    high_arr = np.asarray(high, dtype=float)
    low_arr = np.asarray(low, dtype=float)
    close_arr = np.asarray(close, dtype=float)
    volume_arr = np.asarray(volume, dtype=float)
    typical = (high_arr + low_arr + close_arr) / 3.0
    features: FeatureMap = {}

    sma_values: Dict[int, NDArray[np.floating]] = {}
    for period in _valid_periods(sma_periods):
        avg = sma(close_arr, period)
        sma_values[period] = avg
        features[f"sma_{period}"] = avg
        features[f"sma_distance_{period}"] = safe_div(close_arr - avg, avg, default=np.nan)
        features[f"sma_slope_{period}"] = _slope(avg, periods=1)

    ema_values: Dict[int, NDArray[np.floating]] = {}
    for period in _valid_periods(ema_periods):
        avg = ema(close_arr, period)
        ema_values[period] = avg
        features[f"ema_{period}"] = avg
        features[f"ema_distance_{period}"] = safe_div(close_arr - avg, avg, default=np.nan)
        features[f"ema_slope_{period}"] = _slope(avg, periods=1)

    cumulative_vwap = _cumulative_vwap(typical, volume_arr)
    features["vwap_cumulative"] = cumulative_vwap
    features["vwap_cumulative_distance"] = safe_div(close_arr - cumulative_vwap, cumulative_vwap, default=np.nan)
    for window in _valid_periods(vwap_windows):
        rolling_vwap = _rolling_vwap(typical, volume_arr, window)
        features[f"vwap_{window}"] = rolling_vwap
        features[f"vwap_distance_{window}"] = safe_div(close_arr - rolling_vwap, rolling_vwap, default=np.nan)
        features[f"volume_weighted_return_{window}"] = safe_div(
            rolling_mean(close_arr * volume_arr, window),
            rolling_mean(volume_arr, window),
            default=np.nan,
        )

    _add_cross_features(features, close_arr, sma_values, prefix="sma")
    _add_cross_features(features, close_arr, ema_values, prefix="ema")
    return features


# ==============================================================================
# NUMERICAL HELPERS
# ==============================================================================


def _valid_periods(periods: Sequence[int]) -> list[int]:
    return sorted({int(period) for period in periods if int(period) > 0})


def _slope(values: NDArray[np.floating], periods: int) -> NDArray[np.floating]:
    arr = np.asarray(values, dtype=float)
    out = np.full(len(arr), np.nan, dtype=float)
    if periods > 0 and len(arr) > periods:
        out[periods:] = arr[periods:] - arr[:-periods]
    return out


def _cumulative_vwap(price: NDArray[np.floating], volume: NDArray[np.floating]) -> NDArray[np.floating]:
    pv = np.cumsum(price * volume)
    vv = np.cumsum(volume)
    return safe_div(pv, vv, default=np.nan)


def _rolling_vwap(price: NDArray[np.floating], volume: NDArray[np.floating], window: int) -> NDArray[np.floating]:
    out = np.full(len(price), np.nan, dtype=float)
    if window <= 0 or len(price) < window:
        return out
    pv = price * volume
    pv_sum = np.cumsum(np.insert(pv, 0, 0.0))
    volume_sum = np.cumsum(np.insert(volume, 0, 0.0))
    numerator = pv_sum[window:] - pv_sum[:-window]
    denominator = volume_sum[window:] - volume_sum[:-window]
    out[window - 1 :] = safe_div(numerator, denominator, default=np.nan)
    return out


def _add_cross_features(
    features: FeatureMap,
    close: NDArray[np.floating],
    averages: Dict[int, NDArray[np.floating]],
    *,
    prefix: str,
) -> None:
    periods = sorted(averages)
    for period in periods:
        avg = averages[period]
        above = (close > avg).astype(float)
        above[np.isnan(avg)] = np.nan
        features[f"{prefix}_close_above_{period}"] = above

    for fast, slow in zip(periods, periods[1:]):
        fast_arr = averages[fast]
        slow_arr = averages[slow]
        spread = fast_arr - slow_arr
        state = np.where(spread > 0.0, 1.0, np.where(spread < 0.0, -1.0, 0.0))
        state[np.isnan(spread)] = np.nan
        cross_up = np.zeros(len(close), dtype=float)
        cross_down = np.zeros(len(close), dtype=float)
        prev = np.roll(spread, 1)
        prev[0] = np.nan
        cross_up[(prev <= 0.0) & (spread > 0.0)] = 1.0
        cross_down[(prev >= 0.0) & (spread < 0.0)] = 1.0
        cross_up[np.isnan(prev) | np.isnan(spread)] = np.nan
        cross_down[np.isnan(prev) | np.isnan(spread)] = np.nan
        features[f"{prefix}_cross_state_{fast}_{slow}"] = state
        features[f"{prefix}_cross_up_{fast}_{slow}"] = cross_up
        features[f"{prefix}_cross_down_{fast}_{slow}"] = cross_down
        features[f"{prefix}_spread_{fast}_{slow}"] = safe_div(spread, slow_arr, default=np.nan)
