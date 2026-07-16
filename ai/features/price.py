"""
ai/features/price.py - Price and return-derived features

RESPONSIBILITY:
Create OHLC-derived price transforms, percentage returns, log returns, and
realized volatility features from candle arrays.

VERSION: 1.0.0
"""

from __future__ import annotations

from enum import Enum
from typing import Dict, Sequence

import numpy as np
from numpy.typing import NDArray

from ai.utils.math_ops import returns as pct_returns
from ai.utils.math_ops import rolling_mean, safe_div


# ==============================================================================
# TYPES
# ==============================================================================


FeatureMap = Dict[str, NDArray[np.floating]]


class PriceBasis(str, Enum):
    """Common OHLC price bases."""

    HL2 = "hl2"
    HLC3 = "hlc3"
    OHLC4 = "ohlc4"


# ==============================================================================
# PUBLIC API
# ==============================================================================


def compute_price_features(
    open_: NDArray[np.floating],
    high: NDArray[np.floating],
    low: NDArray[np.floating],
    close: NDArray[np.floating],
    volume: NDArray[np.floating],
    *,
    rolling_windows: Sequence[int],
    include_price: bool = True,
    include_returns: bool = True,
) -> FeatureMap:
    """Compute price-basis and return features."""

    open_arr = np.asarray(open_, dtype=float)
    high_arr = np.asarray(high, dtype=float)
    low_arr = np.asarray(low, dtype=float)
    close_arr = np.asarray(close, dtype=float)
    volume_arr = np.asarray(volume, dtype=float)
    features: FeatureMap = {}

    if include_price:
        range_arr = high_arr - low_arr
        body = close_arr - open_arr
        hl2 = (high_arr + low_arr) / 2.0
        hlc3 = (high_arr + low_arr + close_arr) / 3.0
        ohlc4 = (open_arr + high_arr + low_arr + close_arr) / 4.0
        prev_close = _lag(close_arr, 1)
        features.update(
            {
                "price_open": open_arr,
                "price_high": high_arr,
                "price_low": low_arr,
                "price_close": close_arr,
                "price_hl2": hl2,
                "price_hlc3": hlc3,
                "price_ohlc4": ohlc4,
                "price_range": range_arr,
                "price_range_pct": safe_div(range_arr, close_arr, default=np.nan),
                "price_body": body,
                "price_body_pct": safe_div(body, open_arr, default=np.nan),
                "price_close_position": safe_div(close_arr - low_arr, range_arr, default=0.5),
                "price_gap": open_arr - prev_close,
                "price_gap_pct": safe_div(open_arr - prev_close, prev_close, default=np.nan),
                "price_typical_volume": hlc3 * volume_arr,
            }
        )

    if include_returns:
        one_return = pct_returns(close_arr, periods=1)
        one_log_return = _log_returns(close_arr, periods=1)
        features["return_1"] = one_return
        features["log_return_1"] = one_log_return
        features["return_open_close"] = safe_div(close_arr - open_arr, open_arr, default=np.nan)
        features["return_high_low"] = safe_div(high_arr - low_arr, low_arr, default=np.nan)

        for window in _valid_windows(rolling_windows):
            ret = pct_returns(close_arr, periods=window)
            log_ret = _log_returns(close_arr, periods=window)
            features[f"return_{window}"] = ret
            features[f"log_return_{window}"] = log_ret
            features[f"realized_volatility_{window}"] = _rolling_nan_std(one_log_return, window) * np.sqrt(float(window))
            features[f"return_mean_{window}"] = _rolling_nan_mean(one_return, window)
            features[f"parkinson_volatility_{window}"] = _parkinson_volatility(high_arr, low_arr, window)

    return features


# ==============================================================================
# NUMERICAL HELPERS
# ==============================================================================


def _valid_windows(windows: Sequence[int]) -> list[int]:
    return sorted({int(window) for window in windows if int(window) > 0})


def _lag(values: NDArray[np.floating], periods: int) -> NDArray[np.floating]:
    arr = np.asarray(values, dtype=float)
    out = np.full(len(arr), np.nan, dtype=float)
    if periods > 0 and len(arr) > periods:
        out[periods:] = arr[:-periods]
    return out


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


def _parkinson_volatility(
    high: NDArray[np.floating],
    low: NDArray[np.floating],
    window: int,
) -> NDArray[np.floating]:
    high_arr = np.asarray(high, dtype=float)
    low_arr = np.asarray(low, dtype=float)
    out = np.full(len(high_arr), np.nan, dtype=float)
    valid = (high_arr > 0.0) & (low_arr > 0.0)
    log_range_sq = np.full(len(high_arr), np.nan, dtype=float)
    log_range_sq[valid] = np.log(high_arr[valid] / low_arr[valid]) ** 2
    mean = rolling_mean(log_range_sq, window)
    out[:] = np.sqrt(mean / (4.0 * np.log(2.0)))
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
