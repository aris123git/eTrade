"""
ai/features/regime.py - Market regime classification features

RESPONSIBILITY:
Compute trend vs mean-reversion regime flags, range/efficiency statistics, and
an FX VIX-like volatility-regime measure (realized vol level, percentile, and
vol-of-vol).

VERSION: 1.1.0
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
    TRENDING = "trending"
    MEAN_REVERTING = "mean_reverting"
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
    fx_vix_window: int = 20,
    fx_vix_lookback: int = 252,
) -> FeatureMap:
    """Compute rolling market regime and FX VIX-like features."""

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
        vol_percentile = _rolling_percent_rank(vol, max(window, 20))
        variance_ratio = _variance_ratio(log_ret, window)
        # Hurst-like proxy from variance ratio: VR≈1 random, >1 persistent/trend, <1 mean-revert
        trending = ((efficiency >= 0.35) & (np.abs(trend_score) >= 0.25) & np.isfinite(efficiency)).astype(float)
        mean_reverting = (
            ((efficiency < 0.25) | (variance_ratio < 0.85)) & np.isfinite(efficiency)
        ).astype(float)

        features[f"regime_return_mean_{window}"] = ret_mean
        features[f"regime_volatility_{window}"] = vol
        features[f"regime_vol_percentile_{window}"] = vol_percentile
        features[f"regime_efficiency_ratio_{window}"] = efficiency
        features[f"regime_variance_ratio_{window}"] = variance_ratio
        features[f"regime_trend_score_{window}"] = trend_score
        features[f"regime_range_width_{window}"] = range_width
        features[f"regime_trend_up_{window}"] = ((trend_score > 0.35) & np.isfinite(trend_score)).astype(float)
        features[f"regime_trend_down_{window}"] = ((trend_score < -0.35) & np.isfinite(trend_score)).astype(float)
        features[f"regime_trending_{window}"] = trending
        features[f"regime_mean_reverting_{window}"] = mean_reverting
        features[f"regime_range_{window}"] = ((efficiency < 0.25) & np.isfinite(efficiency)).astype(float)
        features[f"regime_high_vol_{window}"] = ((vol_percentile >= 0.75) & np.isfinite(vol_percentile)).astype(float)
        features[f"regime_low_vol_{window}"] = ((vol_percentile <= 0.25) & np.isfinite(vol_percentile)).astype(float)

    # FX VIX-like: short realized vol, its long-history percentile, and vol-of-vol.
    features.update(
        _fx_vix_features(
            log_ret,
            high_arr,
            low_arr,
            close_arr,
            short_window=int(fx_vix_window),
            lookback=int(fx_vix_lookback),
        )
    )
    return features


def _fx_vix_features(
    log_ret: NDArray[np.floating],
    high: NDArray[np.floating],
    low: NDArray[np.floating],
    close: NDArray[np.floating],
    *,
    short_window: int,
    lookback: int,
) -> FeatureMap:
    """
    VIX-like measure for FX without an options surface:

    - realized volatility (std of log returns)
    - Parkinson high-low volatility estimator
    - percentile of realized vol vs long lookback (fear gauge)
    - vol-of-vol (std of realized vol)
    """

    w = max(int(short_window), 2)
    lb = max(int(lookback), w + 2)
    realized = _rolling_nan_std(log_ret, w) * np.sqrt(252.0)  # annualized scale
    # Parkinson estimator: sqrt(1/(4 ln 2) * mean(ln(H/L)^2))
    hl = safe_div(high, low, default=np.nan)
    log_hl2 = np.square(np.log(np.where(hl > 0, hl, np.nan)))
    parkinson = np.sqrt(np.maximum(_rolling_nan_mean(log_hl2, w) / (4.0 * np.log(2.0)), 0.0)) * np.sqrt(252.0)
    vol_pct = _rolling_percent_rank(realized, lb)
    vol_of_vol = _rolling_nan_std(realized, w)
    # Composite FX VIX proxy in ~0-100 units (percentile × 100, blended with level)
    fx_vix = 100.0 * np.clip(vol_pct, 0.0, 1.0)
    stress = ((vol_pct >= 0.80) & np.isfinite(vol_pct)).astype(float)
    calm = ((vol_pct <= 0.20) & np.isfinite(vol_pct)).astype(float)

    return {
        "fx_vix": fx_vix,
        "fx_vix_realized_vol": realized,
        "fx_vix_parkinson_vol": parkinson,
        "fx_vix_vol_of_vol": vol_of_vol,
        "fx_vix_percentile": vol_pct,
        "fx_vix_stress": stress,
        "fx_vix_calm": calm,
        "fx_vix_term_structure": safe_div(realized, _rolling_nan_std(log_ret, max(w * 3, 6)) * np.sqrt(252.0), default=np.nan),
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
        sample = arr[idx - window + 1 : idx + 1]
        finite = sample[np.isfinite(sample)]
        current = arr[idx]
        if len(finite) == width and np.isfinite(current):
            out[idx] = np.count_nonzero(finite <= current) / float(width)
    return out


def _variance_ratio(log_returns: NDArray[np.floating], window: int) -> NDArray[np.floating]:
    """
    Lo-MacKinlay style variance ratio using lag-q=2 inside a rolling window.

    VR < 1 ⇒ mean-reverting tendencies; VR > 1 ⇒ trending / persistent.
    """

    arr = np.asarray(log_returns, dtype=float)
    out = np.full(len(arr), np.nan, dtype=float)
    q = 2
    width = max(int(window), q * 2)
    if len(arr) < width:
        return out
    for idx in range(width - 1, len(arr)):
        sample = arr[idx - width + 1 : idx + 1]
        finite = sample[np.isfinite(sample)]
        if len(finite) < width:
            continue
        var1 = float(np.var(finite, ddof=1))
        if var1 <= 1e-16:
            out[idx] = 1.0
            continue
        summed = np.convolve(finite, np.ones(q), mode="valid")
        varq = float(np.var(summed, ddof=1))
        out[idx] = safe_div(varq, q * var1, default=np.nan)
    return out
