"""
ai/utils/math_ops.py - Vectorized mathematical primitives

VERSION: 1.0.0
"""

from __future__ import annotations

from typing import Optional, Tuple
import numpy as np
from numpy.typing import NDArray


def safe_div(
    numerator: NDArray[np.floating] | float,
    denominator: NDArray[np.floating] | float,
    default: float = 0.0,
) -> NDArray[np.floating]:
    """Divide with zero-safe fallback."""
    num = np.asarray(numerator, dtype=float)
    den = np.asarray(denominator, dtype=float)
    out = np.full(np.broadcast(num, den).shape, default, dtype=float)
    mask = den != 0
    np.divide(num, den, out=out, where=mask)
    return out


def rolling_window(values: NDArray[np.floating], window: int) -> NDArray[np.floating]:
    """Create a rolling window view (copy-based for safety)."""
    values = np.asarray(values, dtype=float)
    if window <= 0:
        raise ValueError("window must be > 0")
    if len(values) < window:
        return np.empty((0, window), dtype=float)
    shape = (len(values) - window + 1, window)
    strides = (values.strides[0], values.strides[0])
    return np.lib.stride_tricks.as_strided(values, shape=shape, strides=strides).copy()


def rolling_mean(values: NDArray[np.floating], window: int) -> NDArray[np.floating]:
    values = np.asarray(values, dtype=float)
    out = np.full(len(values), np.nan, dtype=float)
    if window <= 0 or len(values) < window:
        return out
    cumsum = np.cumsum(np.insert(values, 0, 0.0))
    out[window - 1 :] = (cumsum[window:] - cumsum[:-window]) / window
    return out


def rolling_std(values: NDArray[np.floating], window: int, ddof: int = 0) -> NDArray[np.floating]:
    values = np.asarray(values, dtype=float)
    out = np.full(len(values), np.nan, dtype=float)
    if window <= 1 or len(values) < window:
        return out
    means = rolling_mean(values, window)
    # Efficient rolling variance via cumulative sums
    cumsum = np.cumsum(np.insert(values, 0, 0.0))
    cumsum2 = np.cumsum(np.insert(values ** 2, 0, 0.0))
    total = cumsum[window:] - cumsum[:-window]
    total2 = cumsum2[window:] - cumsum2[:-window]
    mean = total / window
    var = (total2 / window) - (mean ** 2)
    if ddof:
        var = var * window / max(window - ddof, 1)
    out[window - 1 :] = np.sqrt(np.maximum(var, 0.0))
    # Align with means length usage
    _ = means
    return out


def rolling_min(values: NDArray[np.floating], window: int) -> NDArray[np.floating]:
    values = np.asarray(values, dtype=float)
    out = np.full(len(values), np.nan, dtype=float)
    if window <= 0 or len(values) < window:
        return out
    windows = rolling_window(values, window)
    out[window - 1 :] = windows.min(axis=1)
    return out


def rolling_max(values: NDArray[np.floating], window: int) -> NDArray[np.floating]:
    values = np.asarray(values, dtype=float)
    out = np.full(len(values), np.nan, dtype=float)
    if window <= 0 or len(values) < window:
        return out
    windows = rolling_window(values, window)
    out[window - 1 :] = windows.max(axis=1)
    return out


def sma(values: NDArray[np.floating], period: int) -> NDArray[np.floating]:
    return rolling_mean(values, period)


def ema(values: NDArray[np.floating], period: int) -> NDArray[np.floating]:
    values = np.asarray(values, dtype=float)
    out = np.full(len(values), np.nan, dtype=float)
    if period <= 0 or len(values) == 0:
        return out
    alpha = 2.0 / (period + 1.0)
    start = min(period - 1, len(values) - 1)
    out[start] = np.mean(values[: start + 1])
    for i in range(start + 1, len(values)):
        out[i] = alpha * values[i] + (1.0 - alpha) * out[i - 1]
    return out


def true_range(
    high: NDArray[np.floating],
    low: NDArray[np.floating],
    close: NDArray[np.floating],
) -> NDArray[np.floating]:
    high = np.asarray(high, dtype=float)
    low = np.asarray(low, dtype=float)
    close = np.asarray(close, dtype=float)
    prev_close = np.roll(close, 1)
    prev_close[0] = close[0]
    ranges = np.vstack(
        [
            high - low,
            np.abs(high - prev_close),
            np.abs(low - prev_close),
        ]
    )
    return ranges.max(axis=0)


def atr(
    high: NDArray[np.floating],
    low: NDArray[np.floating],
    close: NDArray[np.floating],
    period: int = 14,
) -> NDArray[np.floating]:
    return ema(true_range(high, low, close), period)


def returns(values: NDArray[np.floating], periods: int = 1) -> NDArray[np.floating]:
    values = np.asarray(values, dtype=float)
    out = np.full(len(values), np.nan, dtype=float)
    if periods <= 0 or len(values) <= periods:
        return out
    out[periods:] = safe_div(values[periods:] - values[:-periods], values[:-periods])
    return out


def log_returns(values: NDArray[np.floating], periods: int = 1) -> NDArray[np.floating]:
    values = np.asarray(values, dtype=float)
    out = np.full(len(values), np.nan, dtype=float)
    if periods <= 0 or len(values) <= periods:
        return out
    ratio = safe_div(values[periods:], values[:-periods], default=np.nan)
    valid = ratio > 0
    out[periods:][valid] = np.log(ratio[valid])
    return out


def clip_array(
    values: NDArray[np.floating],
    lower: Optional[float] = None,
    upper: Optional[float] = None,
) -> NDArray[np.floating]:
    return np.clip(np.asarray(values, dtype=float), lower, upper)


def zscore(values: NDArray[np.floating], ddof: int = 0) -> NDArray[np.floating]:
    values = np.asarray(values, dtype=float)
    mean = np.nanmean(values)
    std = np.nanstd(values, ddof=ddof)
    if std == 0 or np.isnan(std):
        return np.zeros_like(values)
    return (values - mean) / std


def softmax(logits: NDArray[np.floating]) -> NDArray[np.floating]:
    logits = np.asarray(logits, dtype=float)
    shifted = logits - np.max(logits, axis=-1, keepdims=True)
    exp = np.exp(shifted)
    return safe_div(exp, np.sum(exp, axis=-1, keepdims=True), default=0.0)


def sharpe_ratio(returns_arr: NDArray[np.floating], risk_free: float = 0.0, periods: int = 252) -> float:
    arr = np.asarray(returns_arr, dtype=float)
    arr = arr[~np.isnan(arr)]
    if len(arr) < 2:
        return 0.0
    excess = arr - risk_free / periods
    std = np.std(excess, ddof=1)
    if std == 0:
        return 0.0
    return float(np.mean(excess) / std * np.sqrt(periods))


def sortino_ratio(returns_arr: NDArray[np.floating], risk_free: float = 0.0, periods: int = 252) -> float:
    arr = np.asarray(returns_arr, dtype=float)
    arr = arr[~np.isnan(arr)]
    if len(arr) < 2:
        return 0.0
    excess = arr - risk_free / periods
    downside = excess[excess < 0]
    if len(downside) == 0:
        return float("inf") if np.mean(excess) > 0 else 0.0
    downside_std = np.std(downside, ddof=1)
    if downside_std == 0:
        return 0.0
    return float(np.mean(excess) / downside_std * np.sqrt(periods))


def max_drawdown(equity: NDArray[np.floating]) -> Tuple[float, int, int]:
    equity = np.asarray(equity, dtype=float)
    if len(equity) == 0:
        return 0.0, 0, 0
    peaks = np.maximum.accumulate(equity)
    drawdowns = safe_div(equity - peaks, peaks, default=0.0)
    trough = int(np.argmin(drawdowns))
    peak = int(np.argmax(equity[: trough + 1])) if trough > 0 else 0
    return float(drawdowns[trough]), peak, trough
