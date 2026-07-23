"""
ai/features/momentum.py - Oscillators and trend-momentum indicators

RESPONSIBILITY:
Compute RSI, MACD, ADX, CCI, ROC, Momentum, Williams %R, and Stochastic
features using pure NumPy implementations.

VERSION: 1.0.0
"""

from __future__ import annotations

from enum import Enum
from typing import Dict

import numpy as np
from numpy.typing import NDArray

from ai.utils.math_ops import ema, rolling_max, rolling_mean, rolling_min, safe_div, true_range


# ==============================================================================
# TYPES
# ==============================================================================


FeatureMap = Dict[str, NDArray[np.floating]]


class MomentumIndicator(str, Enum):
    """Supported momentum indicator families."""

    RSI = "rsi"
    MACD = "macd"
    ADX = "adx"
    CCI = "cci"
    ROC = "roc"
    STOCHASTIC = "stochastic"


# ==============================================================================
# PUBLIC API
# ==============================================================================


def compute_momentum_features(
    high: NDArray[np.floating],
    low: NDArray[np.floating],
    close: NDArray[np.floating],
    *,
    rsi_period: int,
    adx_period: int,
    macd_fast: int,
    macd_slow: int,
    macd_signal: int,
    stochastic_k: int,
    stochastic_d: int,
    williams_period: int,
    cci_period: int,
    roc_period: int,
    momentum_period: int,
) -> FeatureMap:
    """Compute all configured momentum indicators."""

    high_arr = np.asarray(high, dtype=float)
    low_arr = np.asarray(low, dtype=float)
    close_arr = np.asarray(close, dtype=float)

    macd_line, signal_line, histogram = macd(close_arr, macd_fast, macd_slow, macd_signal)
    plus_di, minus_di, adx_values = adx(high_arr, low_arr, close_arr, adx_period)
    stochastic_k_values, stochastic_d_values = stochastic(high_arr, low_arr, close_arr, stochastic_k, stochastic_d)
    momentum_values = momentum(close_arr, momentum_period)

    return {
        f"rsi_{rsi_period}": rsi(close_arr, rsi_period),
        "macd_line": macd_line,
        "macd_signal": signal_line,
        "macd_histogram": histogram,
        f"adx_{adx_period}": adx_values,
        f"plus_di_{adx_period}": plus_di,
        f"minus_di_{adx_period}": minus_di,
        f"cci_{cci_period}": cci(high_arr, low_arr, close_arr, cci_period),
        f"roc_{roc_period}": roc(close_arr, roc_period),
        f"momentum_{momentum_period}": momentum_values,
        f"momentum_pct_{momentum_period}": safe_div(momentum_values, _lag(close_arr, momentum_period), default=np.nan),
        f"williams_r_{williams_period}": williams_r(high_arr, low_arr, close_arr, williams_period),
        f"stochastic_k_{stochastic_k}": stochastic_k_values,
        f"stochastic_d_{stochastic_d}": stochastic_d_values,
        f"stochastic_spread_{stochastic_k}_{stochastic_d}": stochastic_k_values - stochastic_d_values,
    }


def rsi(close: NDArray[np.floating], period: int = 14) -> NDArray[np.floating]:
    """Relative Strength Index with Wilder smoothing."""

    values = np.asarray(close, dtype=float)
    out = np.full(len(values), np.nan, dtype=float)
    if period <= 0 or len(values) <= period:
        return out
    delta = np.diff(values, prepend=values[0])
    gains = np.maximum(delta, 0.0)
    losses = np.maximum(-delta, 0.0)
    avg_gain = np.mean(gains[1 : period + 1])
    avg_loss = np.mean(losses[1 : period + 1])
    out[period] = _rsi_value(avg_gain, avg_loss)
    for idx in range(period + 1, len(values)):
        avg_gain = ((avg_gain * (period - 1)) + gains[idx]) / period
        avg_loss = ((avg_loss * (period - 1)) + losses[idx]) / period
        out[idx] = _rsi_value(avg_gain, avg_loss)
    return out


def macd(
    close: NDArray[np.floating],
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
) -> tuple[NDArray[np.floating], NDArray[np.floating], NDArray[np.floating]]:
    """Moving Average Convergence Divergence."""

    values = np.asarray(close, dtype=float)
    fast_ema = ema(values, fast)
    slow_ema = ema(values, slow)
    line = fast_ema - slow_ema
    signal_line = _ema_from_first_valid(line, signal)
    histogram = line - signal_line
    return line, signal_line, histogram


def adx(
    high: NDArray[np.floating],
    low: NDArray[np.floating],
    close: NDArray[np.floating],
    period: int = 14,
) -> tuple[NDArray[np.floating], NDArray[np.floating], NDArray[np.floating]]:
    """Average Directional Index with +DI and -DI."""

    high_arr = np.asarray(high, dtype=float)
    low_arr = np.asarray(low, dtype=float)
    close_arr = np.asarray(close, dtype=float)
    plus_dm = np.zeros(len(high_arr), dtype=float)
    minus_dm = np.zeros(len(high_arr), dtype=float)
    up_move = high_arr[1:] - high_arr[:-1]
    down_move = low_arr[:-1] - low_arr[1:]
    plus_dm[1:] = np.where((up_move > down_move) & (up_move > 0.0), up_move, 0.0)
    minus_dm[1:] = np.where((down_move > up_move) & (down_move > 0.0), down_move, 0.0)
    tr_avg = _wilder_average(true_range(high_arr, low_arr, close_arr), period)
    plus_avg = _wilder_average(plus_dm, period)
    minus_avg = _wilder_average(minus_dm, period)
    plus_di = 100.0 * safe_div(plus_avg, tr_avg, default=np.nan)
    minus_di = 100.0 * safe_div(minus_avg, tr_avg, default=np.nan)
    dx = 100.0 * safe_div(np.abs(plus_di - minus_di), plus_di + minus_di, default=np.nan)
    adx_values = _wilder_average(dx, period)
    return plus_di, minus_di, adx_values


def cci(
    high: NDArray[np.floating],
    low: NDArray[np.floating],
    close: NDArray[np.floating],
    period: int = 20,
) -> NDArray[np.floating]:
    """Commodity Channel Index."""

    typical = (np.asarray(high, dtype=float) + np.asarray(low, dtype=float) + np.asarray(close, dtype=float)) / 3.0
    mean = rolling_mean(typical, period)
    mean_deviation = np.full(len(typical), np.nan, dtype=float)
    if period > 0:
        for idx in range(period - 1, len(typical)):
            window_values = typical[idx - period + 1 : idx + 1]
            mean_deviation[idx] = np.mean(np.abs(window_values - mean[idx]))
    return safe_div(typical - mean, 0.015 * mean_deviation, default=np.nan)


def roc(close: NDArray[np.floating], period: int = 12) -> NDArray[np.floating]:
    """Rate of Change in percent."""

    values = np.asarray(close, dtype=float)
    out = np.full(len(values), np.nan, dtype=float)
    if period > 0 and len(values) > period:
        out[period:] = 100.0 * safe_div(values[period:] - values[:-period], values[:-period], default=np.nan)
    return out


def momentum(close: NDArray[np.floating], period: int = 10) -> NDArray[np.floating]:
    """Absolute momentum over a fixed period."""

    values = np.asarray(close, dtype=float)
    out = np.full(len(values), np.nan, dtype=float)
    if period > 0 and len(values) > period:
        out[period:] = values[period:] - values[:-period]
    return out


def williams_r(
    high: NDArray[np.floating],
    low: NDArray[np.floating],
    close: NDArray[np.floating],
    period: int = 14,
) -> NDArray[np.floating]:
    """Williams %R oscillator."""

    highest = rolling_max(np.asarray(high, dtype=float), period)
    lowest = rolling_min(np.asarray(low, dtype=float), period)
    return -100.0 * safe_div(highest - np.asarray(close, dtype=float), highest - lowest, default=np.nan)


def stochastic(
    high: NDArray[np.floating],
    low: NDArray[np.floating],
    close: NDArray[np.floating],
    k_period: int = 14,
    d_period: int = 3,
) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
    """Fast stochastic %K and smoothed %D."""

    highest = rolling_max(np.asarray(high, dtype=float), k_period)
    lowest = rolling_min(np.asarray(low, dtype=float), k_period)
    k_values = 100.0 * safe_div(np.asarray(close, dtype=float) - lowest, highest - lowest, default=np.nan)
    d_values = _rolling_nan_mean(k_values, d_period)
    return k_values, d_values


# ==============================================================================
# NUMERICAL HELPERS
# ==============================================================================


def _rsi_value(avg_gain: float, avg_loss: float) -> float:
    if avg_gain == 0.0 and avg_loss == 0.0:
        return 50.0
    if avg_loss == 0.0:
        return 100.0
    rs = avg_gain / avg_loss
    return 100.0 - (100.0 / (1.0 + rs))


def _lag(values: NDArray[np.floating], periods: int) -> NDArray[np.floating]:
    arr = np.asarray(values, dtype=float)
    out = np.full(len(arr), np.nan, dtype=float)
    if periods > 0 and len(arr) > periods:
        out[periods:] = arr[:-periods]
    return out


def _ema_from_first_valid(values: NDArray[np.floating], period: int) -> NDArray[np.floating]:
    arr = np.asarray(values, dtype=float)
    out = np.full(len(arr), np.nan, dtype=float)
    valid_idx = np.flatnonzero(np.isfinite(arr))
    if period <= 0 or len(valid_idx) < period:
        return out
    start_pos = valid_idx[period - 1]
    seed_values = arr[valid_idx[:period]]
    out[start_pos] = np.mean(seed_values)
    alpha = 2.0 / (period + 1.0)
    for idx in range(start_pos + 1, len(arr)):
        if np.isfinite(arr[idx]):
            out[idx] = alpha * arr[idx] + (1.0 - alpha) * out[idx - 1]
        else:
            out[idx] = out[idx - 1]
    return out


def _wilder_average(values: NDArray[np.floating], period: int) -> NDArray[np.floating]:
    arr = np.asarray(values, dtype=float)
    out = np.full(len(arr), np.nan, dtype=float)
    finite = np.isfinite(arr)
    if period <= 0:
        return out
    for idx in range(len(arr)):
        start = idx - period + 1
        if start < 0:
            continue
        window = arr[start : idx + 1]
        if np.all(finite[start : idx + 1]):
            out[idx] = np.mean(window)
            seed_idx = idx
            break
    else:
        return out
    for idx in range(seed_idx + 1, len(arr)):
        value = arr[idx] if np.isfinite(arr[idx]) else out[idx - 1]
        out[idx] = ((out[idx - 1] * (period - 1)) + value) / period
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
