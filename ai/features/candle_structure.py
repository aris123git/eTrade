"""
ai/features/candle_structure.py - Candle anatomy and gap features

RESPONSIBILITY:
Compute candle body, wick, ratio, direction, spread, and gap features from
OHLC arrays.

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


class CandleDirection(str, Enum):
    """Discrete candle direction labels."""

    BULLISH = "bullish"
    BEARISH = "bearish"
    NEUTRAL = "neutral"


# ==============================================================================
# PUBLIC API
# ==============================================================================


def compute_candle_structure_features(
    open_: NDArray[np.floating],
    high: NDArray[np.floating],
    low: NDArray[np.floating],
    close: NDArray[np.floating],
    spread: NDArray[np.floating] | None = None,
) -> FeatureMap:
    """Compute candle structure and gap features."""

    open_arr = np.asarray(open_, dtype=float)
    high_arr = np.asarray(high, dtype=float)
    low_arr = np.asarray(low, dtype=float)
    close_arr = np.asarray(close, dtype=float)
    spread_arr = np.zeros(len(close_arr), dtype=float) if spread is None else np.asarray(spread, dtype=float)

    total_range = high_arr - low_arr
    signed_body = close_arr - open_arr
    body = np.abs(signed_body)
    upper_wick = high_arr - np.maximum(open_arr, close_arr)
    lower_wick = np.minimum(open_arr, close_arr) - low_arr
    prev_close = _lag(close_arr, 1)
    gap = open_arr - prev_close

    bullish = (close_arr > open_arr).astype(float)
    bearish = (close_arr < open_arr).astype(float)
    neutral = (close_arr == open_arr).astype(float)
    doji = (safe_div(body, total_range, default=0.0) <= 0.1).astype(float)

    return {
        "candle_body": body,
        "candle_signed_body": signed_body,
        "candle_body_pct": safe_div(signed_body, open_arr, default=np.nan),
        "candle_range": total_range,
        "candle_range_pct": safe_div(total_range, close_arr, default=np.nan),
        "candle_upper_wick": upper_wick,
        "candle_lower_wick": lower_wick,
        "candle_upper_wick_ratio": safe_div(upper_wick, total_range, default=np.nan),
        "candle_lower_wick_ratio": safe_div(lower_wick, total_range, default=np.nan),
        "candle_body_ratio": safe_div(body, total_range, default=np.nan),
        "candle_close_location": safe_div(close_arr - low_arr, total_range, default=0.5),
        "candle_open_location": safe_div(open_arr - low_arr, total_range, default=0.5),
        "candle_is_bullish": bullish,
        "candle_is_bearish": bearish,
        "candle_is_neutral": neutral,
        "candle_is_doji": doji,
        "candle_gap": gap,
        "candle_gap_pct": safe_div(gap, prev_close, default=np.nan),
        "candle_gap_up": ((open_arr > prev_close) & np.isfinite(prev_close)).astype(float),
        "candle_gap_down": ((open_arr < prev_close) & np.isfinite(prev_close)).astype(float),
        "candle_true_gap_up": ((low_arr > _lag(high_arr, 1)) & np.isfinite(prev_close)).astype(float),
        "candle_true_gap_down": ((high_arr < _lag(low_arr, 1)) & np.isfinite(prev_close)).astype(float),
        "candle_spread": spread_arr,
        "candle_spread_pct": safe_div(spread_arr, close_arr, default=np.nan),
    }


# ==============================================================================
# NUMERICAL HELPERS
# ==============================================================================


def _lag(values: NDArray[np.floating], periods: int) -> NDArray[np.floating]:
    arr = np.asarray(values, dtype=float)
    out = np.full(len(arr), np.nan, dtype=float)
    if periods > 0 and len(arr) > periods:
        out[periods:] = arr[:-periods]
    return out
