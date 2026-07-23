"""
ai/features/patterns.py - Candle pattern detectors

RESPONSIBILITY:
Detect engulfing candles, pin bars, inside/outside bars, and confirmed fractal
high/low patterns from OHLC arrays.

VERSION: 1.0.0
"""

from __future__ import annotations

from enum import Enum
from typing import Dict

import numpy as np
from numpy.typing import NDArray

from ai.utils.math_ops import safe_div


# ==============================================================================
# TYPES
# ==============================================================================


FeatureMap = Dict[str, NDArray[np.floating]]


class PatternKind(str, Enum):
    """Supported price-action pattern families."""

    ENGULFING = "engulfing"
    PIN_BAR = "pin_bar"
    INSIDE_BAR = "inside_bar"
    OUTSIDE_BAR = "outside_bar"
    FRACTAL = "fractal"


# ==============================================================================
# PUBLIC API
# ==============================================================================


def compute_pattern_features(
    open_: NDArray[np.floating],
    high: NDArray[np.floating],
    low: NDArray[np.floating],
    close: NDArray[np.floating],
    *,
    fractal_window: int,
) -> FeatureMap:
    """Compute binary candle pattern features."""

    open_arr = np.asarray(open_, dtype=float)
    high_arr = np.asarray(high, dtype=float)
    low_arr = np.asarray(low, dtype=float)
    close_arr = np.asarray(close, dtype=float)
    bullish_engulfing, bearish_engulfing = engulfing(open_arr, close_arr)
    bullish_pin, bearish_pin = pin_bar(open_arr, high_arr, low_arr, close_arr)
    fractal_high, fractal_low = confirmed_fractals(high_arr, low_arr, fractal_window)
    return {
        "pattern_bullish_engulfing": bullish_engulfing,
        "pattern_bearish_engulfing": bearish_engulfing,
        "pattern_bullish_pin_bar": bullish_pin,
        "pattern_bearish_pin_bar": bearish_pin,
        "pattern_inside_bar": inside_bar(high_arr, low_arr),
        "pattern_outside_bar": outside_bar(high_arr, low_arr),
        f"pattern_confirmed_fractal_high_{fractal_window}": fractal_high,
        f"pattern_confirmed_fractal_low_{fractal_window}": fractal_low,
        "pattern_directional_score": bullish_engulfing - bearish_engulfing + bullish_pin - bearish_pin,
    }


def engulfing(
    open_: NDArray[np.floating],
    close: NDArray[np.floating],
) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
    """Bullish and bearish real-body engulfing patterns."""

    open_arr = np.asarray(open_, dtype=float)
    close_arr = np.asarray(close, dtype=float)
    bullish = np.zeros(len(close_arr), dtype=float)
    bearish = np.zeros(len(close_arr), dtype=float)
    if len(close_arr) < 2:
        return bullish, bearish
    prev_open = open_arr[:-1]
    prev_close = close_arr[:-1]
    cur_open = open_arr[1:]
    cur_close = close_arr[1:]
    prev_bearish = prev_close < prev_open
    prev_bullish = prev_close > prev_open
    cur_bullish = cur_close > cur_open
    cur_bearish = cur_close < cur_open
    bullish[1:] = (
        prev_bearish
        & cur_bullish
        & (cur_open <= prev_close)
        & (cur_close >= prev_open)
    ).astype(float)
    bearish[1:] = (
        prev_bullish
        & cur_bearish
        & (cur_open >= prev_close)
        & (cur_close <= prev_open)
    ).astype(float)
    return bullish, bearish


def pin_bar(
    open_: NDArray[np.floating],
    high: NDArray[np.floating],
    low: NDArray[np.floating],
    close: NDArray[np.floating],
) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
    """Bullish and bearish pin bars based on wick dominance."""

    open_arr = np.asarray(open_, dtype=float)
    high_arr = np.asarray(high, dtype=float)
    low_arr = np.asarray(low, dtype=float)
    close_arr = np.asarray(close, dtype=float)
    total_range = high_arr - low_arr
    body = np.abs(close_arr - open_arr)
    upper_wick = high_arr - np.maximum(open_arr, close_arr)
    lower_wick = np.minimum(open_arr, close_arr) - low_arr
    body_ratio = safe_div(body, total_range, default=np.nan)
    close_location = safe_div(close_arr - low_arr, total_range, default=0.5)
    bullish = (
        (body_ratio <= 0.35)
        & (lower_wick >= 2.0 * np.maximum(body, 1e-12))
        & (lower_wick > upper_wick)
        & (close_location >= 0.55)
    ).astype(float)
    bearish = (
        (body_ratio <= 0.35)
        & (upper_wick >= 2.0 * np.maximum(body, 1e-12))
        & (upper_wick > lower_wick)
        & (close_location <= 0.45)
    ).astype(float)
    bullish[~np.isfinite(body_ratio)] = np.nan
    bearish[~np.isfinite(body_ratio)] = np.nan
    return bullish, bearish


def inside_bar(high: NDArray[np.floating], low: NDArray[np.floating]) -> NDArray[np.floating]:
    """Inside bar where current range is contained by the prior range."""

    high_arr = np.asarray(high, dtype=float)
    low_arr = np.asarray(low, dtype=float)
    out = np.zeros(len(high_arr), dtype=float)
    if len(high_arr) > 1:
        out[1:] = ((high_arr[1:] <= high_arr[:-1]) & (low_arr[1:] >= low_arr[:-1])).astype(float)
    return out


def outside_bar(high: NDArray[np.floating], low: NDArray[np.floating]) -> NDArray[np.floating]:
    """Outside bar where current range exceeds the prior range."""

    high_arr = np.asarray(high, dtype=float)
    low_arr = np.asarray(low, dtype=float)
    out = np.zeros(len(high_arr), dtype=float)
    if len(high_arr) > 1:
        out[1:] = ((high_arr[1:] >= high_arr[:-1]) & (low_arr[1:] <= low_arr[:-1])).astype(float)
    return out


def confirmed_fractals(
    high: NDArray[np.floating],
    low: NDArray[np.floating],
    window: int = 2,
) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
    """Confirmed fractal high/low flags emitted after the right-side bars close."""

    high_arr = np.asarray(high, dtype=float)
    low_arr = np.asarray(low, dtype=float)
    width = max(int(window), 1)
    high_out = np.zeros(len(high_arr), dtype=float)
    low_out = np.zeros(len(low_arr), dtype=float)
    if len(high_arr) < (2 * width + 1):
        return high_out, low_out

    for center in range(width, len(high_arr) - width):
        left = center - width
        right = center + width + 1
        confirm_at = center + width
        high_window = high_arr[left:right]
        low_window = low_arr[left:right]
        if high_arr[center] == np.max(high_window) and np.count_nonzero(high_window == high_arr[center]) == 1:
            high_out[confirm_at] = 1.0
        if low_arr[center] == np.min(low_window) and np.count_nonzero(low_window == low_arr[center]) == 1:
            low_out[confirm_at] = 1.0
    return high_out, low_out
