"""
ai/features/regime.py - Market regime classification features

RESPONSIBILITY:
Compute trend, range, and volatility regime features from rolling return,
efficiency, range, and volatility statistics.

VERSION: 1.0.0
"""

from __future__ import annotations

from enum import Enum
from typing import Dict, Sequence

import numpy as np
from numpy.typing import NDArray

from ai.utils.math_ops import rolling_max, rolling_min, safe_div


# ==============================================================================
# TYPES
# ==============================================================================


FeatureMap = Dict[str, NDArray[np.floating]]


class MarketRegime(str, Enum):
    """Regime labels represented by numeric feature flags."""

    TREND_UP = "trend_up"
    TREND_DOWN = "trend_down"
    RANGE = "range"
    HIGH_VOL = "high_vol"
    LOW_VOL = "low_vol"


# ==============================================================================
# PUBLIC API
# ==============================================================================


def compute_regime_features(
    high: NDArray[np.floating],
    low: NDArray[np.floating],
    close: NDArray[np.floating],
    *,
    rolling_windows: Sequence[int],
) -> FeatureMap:
    """Compute rolling market regime features."""

    high_arr = np.asarray(high, dtype=float)
    low_arr = np.asarray(low, dtype=float)
    close_arr = np.asarray(close, dtype=float)
    log_ret = _log_returns(close_arr, 1)
    features: FeatureMap = {}

    for window in _valid_windows(rolling_windows):
        vol = _rolling_nan_std(log_ret, window)
        ret_mean = _rolling_nan_mean(log_ret, window)
        efficiency = _efficiency_ratio(close_arr, window)
        direction = _window_direction(close_arr, window)
        trend_score = efficiency * direction
        range_width = safe_div(rolling_max(high_arr, window) - rolling_min(low_arr, window), close_arr, default=np.nan)
        vol_percentile = _rolling_percent_rank(vol, window)

        features[f"regime_return_mean_{window}"] = ret_mean
        features[f"regime_volatility_{window}"] = vol
        features[f"regime_vol_percentile_{window}"] = vol_percentile
        features[f"regime_efficiency_ratio_{window}"] = efficiency
        features[f"regime_trend_score_{window}"] = trend_score
        features[f"regime_range_width_{window}"] = range_width
        features[f"regime_trend_up_{window}"] = ((trend_score > 0.35) & np.isfinite(trend_score)).astype(float)
        features[f"regime_trend_down_{window}"] = ((trend_score < -0.35) & np.isfinite(trend_score)).astype(float)
        features[f"regime_range_{window}"] = ((efficiency < 0.25) & np.isfinite(efficiency)).astype(float)
        features[f"regime_high_vol_{window}"] = ((vol_percentile >= 0.75) & np.isfinite(vol_percentile)).astype(float)
        features[f"regime_low_vol_{window}"] = ((vol_percentile <= 0.25) & np.isfinite(vol_percentile)).astype(float)

    return features


# ==============================================================================
# NUMERICAL HELPERS
# ==============================================================================


def _valid_windows(windows: Sequence[int]) -> list[int]:
    return sorted({int(window) for window in windows if int(window) > 0})


def _log_returns(values: NDArray[np.floating], periods: int) -> NDArray[np.floating]:
    arr = np.asarray(values, dtype=float)
    out = np.full(len(arr), np.nan, dtype=float)
    if periods <= 0 or len(arr) <= periods:
        return out
    previous = arr[:-periods]
    current = arr[periods:]
    valid = (previous > 0.0) & (current > 0.0)
    out_slice = out[periods:]
    out_slice[valid] = np.log(current[valid] / previous[valid])
    out[periods:] = out_slice
    return out


def _rolling_nan_mean(values: NDArray[np.floating], window: int) -> NDArray[np.floating]:
    arr = np.asarray(values, dtype=float)
    out = np.full(len(arr), np.nan, dtype=float)
    if window <= 0 or len(arr) < window:
        return out
    for idx in range(window - 1, len(arr)):
        sample = arr[idx - window + 1 : idx + 1]
        finite = sample[np.isfinite(sample)]
        if len(finite) == window:
            out[idx] = float(np.mean(finite))
    return out


def _rolling_nan_std(values: NDArray[np.floating], window: int) -> NDArray[np.floating]:
    arr = np.asarray(values, dtype=float)
    out = np.full(len(arr), np.nan, dtype=float)
    if window <= 1 or len(arr) < window:
        return out
    for idx in range(window - 1, len(arr)):
        sample = arr[idx - window + 1 : idx + 1]
        finite = sample[np.isfinite(sample)]
        if len(finite) == window:
            out[idx] = float(np.std(finite, ddof=0))
    return out


def _efficiency_ratio(values: NDArray[np.floating], window: int) -> NDArray[np.floating]:
    arr = np.asarray(values, dtype=float)
    width = max(int(window), 1)
    out = np.full(len(arr), np.nan, dtype=float)
    if len(arr) <= width:
        return out
    for idx in range(width, len(arr)):
        direct = abs(arr[idx] - arr[idx - width])
        path = np.sum(np.abs(np.diff(arr[idx - width : idx + 1])))
        out[idx] = safe_div(direct, path, default=np.nan)
    return out


def _window_direction(values: NDArray[np.floating], window: int) -> NDArray[np.floating]:
    arr = np.asarray(values, dtype=float)
    out = np.full(len(arr), np.nan, dtype=float)
    if window > 0 and len(arr) > window:
        out[window:] = np.sign(arr[window:] - arr[:-window])
    return out


def _rolling_percent_rank(values: NDArray[np.floating], window: int) -> NDArray[np.floating]:
    arr = np.asarray(values, dtype=float)
    out = np.full(len(arr), np.nan, dtype=float)
    width = max(int(window), 1)
    if len(arr) < width:
        return out
    for idx in range(width - 1, len(arr)):
        sample = arr[idx - width + 1 : idx + 1]
        finite = sample[np.isfinite(sample)]
        current = arr[idx]
        if len(finite) == width and np.isfinite(current):
            out[idx] = np.count_nonzero(finite <= current) / float(width)
    return out
