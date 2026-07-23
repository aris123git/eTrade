"""
ai/features/volatility.py - Volatility, channel, SuperTrend, and Ichimoku features

RESPONSIBILITY:
Compute ATR, Bollinger Bands, Donchian Channels, Keltner Channels, SuperTrend,
rolling standard deviations, and Ichimoku components using pure NumPy.

VERSION: 1.0.0
"""

from __future__ import annotations

from enum import Enum
from typing import Dict, Sequence

import numpy as np
from numpy.typing import NDArray

from ai.utils.math_ops import atr, ema, rolling_max, rolling_mean, rolling_min, rolling_std, safe_div


# ==============================================================================
# TYPES
# ==============================================================================


FeatureMap = Dict[str, NDArray[np.floating]]


class VolatilityIndicator(str, Enum):
    """Supported volatility indicator families."""

    ATR = "atr"
    BOLLINGER = "bollinger"
    DONCHIAN = "donchian"
    KELTNER = "keltner"
    SUPERTREND = "supertrend"
    ICHIMOKU = "ichimoku"


# ==============================================================================
# PUBLIC API
# ==============================================================================


def compute_volatility_features(
    high: NDArray[np.floating],
    low: NDArray[np.floating],
    close: NDArray[np.floating],
    *,
    rolling_windows: Sequence[int],
    atr_period: int,
    bollinger_period: int,
    bollinger_std: float,
    donchian_period: int,
    keltner_period: int,
    keltner_atr_mult: float,
    supertrend_period: int,
    supertrend_mult: float,
    include_volatility: bool = True,
    include_channels: bool = True,
) -> FeatureMap:
    """Compute volatility and channel features."""

    high_arr = np.asarray(high, dtype=float)
    low_arr = np.asarray(low, dtype=float)
    close_arr = np.asarray(close, dtype=float)
    features: FeatureMap = {}

    atr_values = atr(high_arr, low_arr, close_arr, atr_period)
    if include_volatility:
        features[f"atr_{atr_period}"] = atr_values
        features[f"atr_pct_{atr_period}"] = safe_div(atr_values, close_arr, default=np.nan)
        one_log_return = _log_returns(close_arr, 1)
        for window in _valid_windows(rolling_windows):
            features[f"rolling_std_close_{window}"] = rolling_std(close_arr, window)
            features[f"rolling_std_log_return_{window}"] = _rolling_nan_std(one_log_return, window)
            features[f"rolling_range_pct_{window}"] = safe_div(
                rolling_max(high_arr, window) - rolling_min(low_arr, window),
                close_arr,
                default=np.nan,
            )

    if include_channels:
        features.update(_bollinger_features(close_arr, bollinger_period, bollinger_std))
        features.update(_donchian_features(high_arr, low_arr, close_arr, donchian_period))
        features.update(_keltner_features(high_arr, low_arr, close_arr, keltner_period, keltner_atr_mult))
        supertrend_line, supertrend_direction, supertrend_upper, supertrend_lower = supertrend(
            high_arr,
            low_arr,
            close_arr,
            period=supertrend_period,
            multiplier=supertrend_mult,
        )
        features[f"supertrend_{supertrend_period}"] = supertrend_line
        features[f"supertrend_direction_{supertrend_period}"] = supertrend_direction
        features[f"supertrend_upper_{supertrend_period}"] = supertrend_upper
        features[f"supertrend_lower_{supertrend_period}"] = supertrend_lower
        features.update(ichimoku(high_arr, low_arr, close_arr))

    return features


def supertrend(
    high: NDArray[np.floating],
    low: NDArray[np.floating],
    close: NDArray[np.floating],
    *,
    period: int = 10,
    multiplier: float = 3.0,
) -> tuple[NDArray[np.floating], NDArray[np.floating], NDArray[np.floating], NDArray[np.floating]]:
    """SuperTrend line, direction, final upper band, and final lower band."""

    high_arr = np.asarray(high, dtype=float)
    low_arr = np.asarray(low, dtype=float)
    close_arr = np.asarray(close, dtype=float)
    n = len(close_arr)
    atr_values = atr(high_arr, low_arr, close_arr, period)
    hl2 = (high_arr + low_arr) / 2.0
    basic_upper = hl2 + multiplier * atr_values
    basic_lower = hl2 - multiplier * atr_values
    final_upper = np.full(n, np.nan, dtype=float)
    final_lower = np.full(n, np.nan, dtype=float)
    trend = np.full(n, np.nan, dtype=float)
    direction = np.full(n, np.nan, dtype=float)
    valid = np.flatnonzero(np.isfinite(basic_upper) & np.isfinite(basic_lower))
    if len(valid) == 0:
        return trend, direction, final_upper, final_lower

    start = int(valid[0])
    final_upper[start] = basic_upper[start]
    final_lower[start] = basic_lower[start]
    direction[start] = 1.0
    trend[start] = final_lower[start]
    for idx in range(start + 1, n):
        prev = idx - 1
        if basic_upper[idx] < final_upper[prev] or close_arr[prev] > final_upper[prev]:
            final_upper[idx] = basic_upper[idx]
        else:
            final_upper[idx] = final_upper[prev]

        if basic_lower[idx] > final_lower[prev] or close_arr[prev] < final_lower[prev]:
            final_lower[idx] = basic_lower[idx]
        else:
            final_lower[idx] = final_lower[prev]

        if trend[prev] == final_upper[prev]:
            trend[idx] = final_upper[idx] if close_arr[idx] <= final_upper[idx] else final_lower[idx]
        else:
            trend[idx] = final_lower[idx] if close_arr[idx] >= final_lower[idx] else final_upper[idx]
        direction[idx] = 1.0 if trend[idx] == final_lower[idx] else -1.0
    return trend, direction, final_upper, final_lower


def ichimoku(
    high: NDArray[np.floating],
    low: NDArray[np.floating],
    close: NDArray[np.floating],
    *,
    conversion_period: int = 9,
    base_period: int = 26,
    span_b_period: int = 52,
    displacement: int = 26,
) -> FeatureMap:
    """Ichimoku components aligned causally to the current row."""

    high_arr = np.asarray(high, dtype=float)
    low_arr = np.asarray(low, dtype=float)
    close_arr = np.asarray(close, dtype=float)
    tenkan = (rolling_max(high_arr, conversion_period) + rolling_min(low_arr, conversion_period)) / 2.0
    kijun = (rolling_max(high_arr, base_period) + rolling_min(low_arr, base_period)) / 2.0
    span_a_raw = (tenkan + kijun) / 2.0
    span_b_raw = (rolling_max(high_arr, span_b_period) + rolling_min(low_arr, span_b_period)) / 2.0
    span_a = _lag(span_a_raw, displacement)
    span_b = _lag(span_b_raw, displacement)
    chikou_lag = _lag(close_arr, displacement)
    cloud_top = np.maximum(span_a, span_b)
    cloud_bottom = np.minimum(span_a, span_b)
    return {
        f"ichimoku_tenkan_{conversion_period}": tenkan,
        f"ichimoku_kijun_{base_period}": kijun,
        f"ichimoku_senkou_a_lagged_{displacement}": span_a,
        f"ichimoku_senkou_b_lagged_{displacement}": span_b,
        f"ichimoku_chikou_lag_{displacement}": chikou_lag,
        "ichimoku_cloud_width": safe_div(cloud_top - cloud_bottom, close_arr, default=np.nan),
        "ichimoku_close_above_cloud": _above_cloud(close_arr, cloud_top),
        "ichimoku_close_below_cloud": _below_cloud(close_arr, cloud_bottom),
        "ichimoku_tenkan_kijun_spread": safe_div(tenkan - kijun, kijun, default=np.nan),
    }


# ==============================================================================
# CHANNEL HELPERS
# ==============================================================================


def _bollinger_features(close: NDArray[np.floating], period: int, std_mult: float) -> FeatureMap:
    middle = rolling_mean(close, period)
    std = rolling_std(close, period)
    upper = middle + std_mult * std
    lower = middle - std_mult * std
    return {
        f"bollinger_middle_{period}": middle,
        f"bollinger_upper_{period}": upper,
        f"bollinger_lower_{period}": lower,
        f"bollinger_width_{period}": safe_div(upper - lower, middle, default=np.nan),
        f"bollinger_percent_b_{period}": safe_div(close - lower, upper - lower, default=np.nan),
    }


def _donchian_features(
    high: NDArray[np.floating],
    low: NDArray[np.floating],
    close: NDArray[np.floating],
    period: int,
) -> FeatureMap:
    upper = rolling_max(high, period)
    lower = rolling_min(low, period)
    middle = (upper + lower) / 2.0
    return {
        f"donchian_upper_{period}": upper,
        f"donchian_lower_{period}": lower,
        f"donchian_middle_{period}": middle,
        f"donchian_width_{period}": safe_div(upper - lower, middle, default=np.nan),
        f"donchian_position_{period}": safe_div(close - lower, upper - lower, default=np.nan),
    }


def _keltner_features(
    high: NDArray[np.floating],
    low: NDArray[np.floating],
    close: NDArray[np.floating],
    period: int,
    atr_mult: float,
) -> FeatureMap:
    middle = ema(close, period)
    atr_values = atr(high, low, close, period)
    upper = middle + atr_mult * atr_values
    lower = middle - atr_mult * atr_values
    return {
        f"keltner_middle_{period}": middle,
        f"keltner_upper_{period}": upper,
        f"keltner_lower_{period}": lower,
        f"keltner_width_{period}": safe_div(upper - lower, middle, default=np.nan),
        f"keltner_position_{period}": safe_div(close - lower, upper - lower, default=np.nan),
    }


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


def _lag(values: NDArray[np.floating], periods: int) -> NDArray[np.floating]:
    arr = np.asarray(values, dtype=float)
    out = np.full(len(arr), np.nan, dtype=float)
    if periods > 0 and len(arr) > periods:
        out[periods:] = arr[:-periods]
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


def _above_cloud(close: NDArray[np.floating], cloud_top: NDArray[np.floating]) -> NDArray[np.floating]:
    out = (close > cloud_top).astype(float)
    out[np.isnan(cloud_top)] = np.nan
    return out


def _below_cloud(close: NDArray[np.floating], cloud_bottom: NDArray[np.floating]) -> NDArray[np.floating]:
    out = (close < cloud_bottom).astype(float)
    out[np.isnan(cloud_bottom)] = np.nan
    return out
