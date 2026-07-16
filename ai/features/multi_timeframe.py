"""
ai/features/multi_timeframe.py - Higher-timeframe feature alignment

RESPONSIBILITY:
Compute compact higher-timeframe features and align them onto base timeframe rows
using timestamp-as-of alignment.

VERSION: 1.0.0
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Dict, Mapping, Sequence

import numpy as np
from numpy.typing import NDArray

from ai.features.engine import candles_to_arrays
from ai.utils.math_ops import atr, ema, safe_div
from ai.utils.time_ops import align_timestamps
from ai.utils.types import CandleDict


# ==============================================================================
# TYPES
# ==============================================================================


FeatureMap = Dict[str, NDArray[np.floating]]


class AlignmentMode(str, Enum):
    """Supported timestamp alignment mode."""

    ASOF = "asof"


@dataclass(frozen=True)
class TimeframeFeatureSpec:
    """Configuration for compact higher-timeframe feature extraction."""

    ema_fast: int
    ema_slow: int
    atr_period: int


# ==============================================================================
# PUBLIC API
# ==============================================================================


def compute_multi_timeframe_features(
    base_timestamps: Sequence[datetime],
    higher_timeframes: Mapping[str, Sequence[CandleDict]],
    *,
    ema_periods: Sequence[int],
    atr_period: int,
) -> FeatureMap:
    """Compute higher-timeframe features aligned to base timestamps."""

    features: FeatureMap = {}
    periods = sorted({int(period) for period in ema_periods if int(period) > 0})
    fast = periods[0] if periods else 12
    slow = periods[-1] if periods else 26
    spec = TimeframeFeatureSpec(ema_fast=fast, ema_slow=slow, atr_period=atr_period)

    for timeframe, candles in sorted(higher_timeframes.items()):
        if not candles:
            continue
        arrays = candles_to_arrays(candles)
        raw = _higher_timeframe_feature_map(arrays.close, arrays.high, arrays.low, spec)
        index_map = align_timestamps(base_timestamps, arrays.timestamps)
        prefix = f"mtf_{timeframe.lower()}"
        for name, values in raw.items():
            features[f"{prefix}_{name}"] = _align_array(values, index_map)
    return features


# ==============================================================================
# FEATURE HELPERS
# ==============================================================================


def _higher_timeframe_feature_map(
    close: NDArray[np.floating],
    high: NDArray[np.floating],
    low: NDArray[np.floating],
    spec: TimeframeFeatureSpec,
) -> FeatureMap:
    fast_ema = ema(close, spec.ema_fast)
    slow_ema = ema(close, spec.ema_slow)
    atr_values = atr(high, low, close, spec.atr_period)
    return {
        "close": np.asarray(close, dtype=float),
        "return_1": _returns(close, 1),
        f"ema_distance_{spec.ema_fast}": safe_div(close - fast_ema, fast_ema, default=np.nan),
        f"ema_distance_{spec.ema_slow}": safe_div(close - slow_ema, slow_ema, default=np.nan),
        f"ema_spread_{spec.ema_fast}_{spec.ema_slow}": safe_div(fast_ema - slow_ema, slow_ema, default=np.nan),
        f"atr_pct_{spec.atr_period}": safe_div(atr_values, close, default=np.nan),
    }


def _align_array(values: NDArray[np.floating], index_map: Sequence[int]) -> NDArray[np.floating]:
    arr = np.asarray(values, dtype=float)
    out = np.full(len(index_map), np.nan, dtype=float)
    for row, idx in enumerate(index_map):
        if idx >= 0:
            out[row] = arr[idx]
    return out


def _returns(values: NDArray[np.floating], periods: int) -> NDArray[np.floating]:
    arr = np.asarray(values, dtype=float)
    out = np.full(len(arr), np.nan, dtype=float)
    if periods > 0 and len(arr) > periods:
        out[periods:] = safe_div(arr[periods:] - arr[:-periods], arr[:-periods], default=np.nan)
    return out
