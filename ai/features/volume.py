"""
ai/features/volume.py - Volume flow and participation features

RESPONSIBILITY:
Compute OBV, MFI, CMF, signed volume delta, volume ratios, and rolling volume
participation features from OHLCV arrays.

VERSION: 1.0.0
"""

from __future__ import annotations

from enum import Enum
from typing import Dict, Sequence

import numpy as np
from numpy.typing import NDArray

from ai.utils.math_ops import rolling_mean, rolling_std, safe_div


# ==============================================================================
# TYPES
# ==============================================================================


FeatureMap = Dict[str, NDArray[np.floating]]


class VolumeIndicator(str, Enum):
    """Supported volume indicator families."""

    OBV = "obv"
    MFI = "mfi"
    CMF = "cmf"
    VOLUME_DELTA = "volume_delta"


# ==============================================================================
# PUBLIC API
# ==============================================================================


def compute_volume_features(
    open_: NDArray[np.floating],
    high: NDArray[np.floating],
    low: NDArray[np.floating],
    close: NDArray[np.floating],
    volume: NDArray[np.floating],
    *,
    rolling_windows: Sequence[int],
    mfi_period: int,
    cmf_period: int,
) -> FeatureMap:
    """Compute volume-derived features."""

    open_arr = np.asarray(open_, dtype=float)
    high_arr = np.asarray(high, dtype=float)
    low_arr = np.asarray(low, dtype=float)
    close_arr = np.asarray(close, dtype=float)
    volume_arr = np.asarray(volume, dtype=float)
    volume_delta = signed_volume_delta(open_arr, close_arr, volume_arr)
    cumulative_delta = np.cumsum(np.nan_to_num(volume_delta, nan=0.0))

    features: FeatureMap = {
        "volume": volume_arr,
        "dollar_volume": close_arr * volume_arr,
        "obv": obv(close_arr, volume_arr),
        f"mfi_{mfi_period}": mfi(high_arr, low_arr, close_arr, volume_arr, mfi_period),
        f"cmf_{cmf_period}": cmf(high_arr, low_arr, close_arr, volume_arr, cmf_period),
        "volume_delta": volume_delta,
        "volume_delta_cumulative": cumulative_delta,
        "volume_buy_ratio": _buy_ratio(open_arr, close_arr),
    }

    for window in _valid_windows(rolling_windows):
        volume_mean = rolling_mean(volume_arr, window)
        volume_std = rolling_std(volume_arr, window)
        delta_sum = _rolling_sum(volume_delta, window)
        features[f"volume_sma_{window}"] = volume_mean
        features[f"volume_ratio_{window}"] = safe_div(volume_arr, volume_mean, default=np.nan)
        features[f"volume_zscore_{window}"] = safe_div(volume_arr - volume_mean, volume_std, default=np.nan)
        features[f"volume_delta_sum_{window}"] = delta_sum
        features[f"volume_delta_ratio_{window}"] = safe_div(delta_sum, _rolling_sum(volume_arr, window), default=np.nan)
        features[f"dollar_volume_sma_{window}"] = rolling_mean(close_arr * volume_arr, window)
    return features


def obv(close: NDArray[np.floating], volume: NDArray[np.floating]) -> NDArray[np.floating]:
    """On-Balance Volume."""

    close_arr = np.asarray(close, dtype=float)
    volume_arr = np.asarray(volume, dtype=float)
    direction = np.zeros(len(close_arr), dtype=float)
    if len(close_arr) > 1:
        direction[1:] = np.sign(close_arr[1:] - close_arr[:-1])
    return np.cumsum(direction * volume_arr)


def mfi(
    high: NDArray[np.floating],
    low: NDArray[np.floating],
    close: NDArray[np.floating],
    volume: NDArray[np.floating],
    period: int = 14,
) -> NDArray[np.floating]:
    """Money Flow Index."""

    typical = (np.asarray(high, dtype=float) + np.asarray(low, dtype=float) + np.asarray(close, dtype=float)) / 3.0
    money_flow = typical * np.asarray(volume, dtype=float)
    positive = np.zeros(len(typical), dtype=float)
    negative = np.zeros(len(typical), dtype=float)
    if len(typical) > 1:
        up = typical[1:] > typical[:-1]
        down = typical[1:] < typical[:-1]
        positive[1:] = np.where(up, money_flow[1:], 0.0)
        negative[1:] = np.where(down, money_flow[1:], 0.0)
    pos_sum = _rolling_sum(positive, period)
    neg_sum = _rolling_sum(negative, period)
    ratio = safe_div(pos_sum, neg_sum, default=np.nan)
    out = 100.0 - (100.0 / (1.0 + ratio))
    out[(neg_sum == 0.0) & (pos_sum > 0.0)] = 100.0
    out[(neg_sum == 0.0) & (pos_sum == 0.0)] = 50.0
    return out


def cmf(
    high: NDArray[np.floating],
    low: NDArray[np.floating],
    close: NDArray[np.floating],
    volume: NDArray[np.floating],
    period: int = 20,
) -> NDArray[np.floating]:
    """Chaikin Money Flow."""

    high_arr = np.asarray(high, dtype=float)
    low_arr = np.asarray(low, dtype=float)
    close_arr = np.asarray(close, dtype=float)
    volume_arr = np.asarray(volume, dtype=float)
    multiplier = safe_div((close_arr - low_arr) - (high_arr - close_arr), high_arr - low_arr, default=0.0)
    money_flow_volume = multiplier * volume_arr
    return safe_div(_rolling_sum(money_flow_volume, period), _rolling_sum(volume_arr, period), default=np.nan)


def signed_volume_delta(
    open_: NDArray[np.floating],
    close: NDArray[np.floating],
    volume: NDArray[np.floating],
) -> NDArray[np.floating]:
    """Signed candle volume using open-close direction."""

    direction = np.sign(np.asarray(close, dtype=float) - np.asarray(open_, dtype=float))
    return direction * np.asarray(volume, dtype=float)


# ==============================================================================
# NUMERICAL HELPERS
# ==============================================================================


def _valid_windows(windows: Sequence[int]) -> list[int]:
    return sorted({int(window) for window in windows if int(window) > 0})


def _rolling_sum(values: NDArray[np.floating], window: int) -> NDArray[np.floating]:
    arr = np.asarray(values, dtype=float)
    out = np.full(len(arr), np.nan, dtype=float)
    if window <= 0 or len(arr) < window:
        return out
    cumsum = np.cumsum(np.insert(arr, 0, 0.0))
    out[window - 1 :] = cumsum[window:] - cumsum[:-window]
    return out


def _buy_ratio(open_: NDArray[np.floating], close: NDArray[np.floating]) -> NDArray[np.floating]:
    out = np.full(len(close), 0.5, dtype=float)
    out[close > open_] = 1.0
    out[close < open_] = 0.0
    return out
