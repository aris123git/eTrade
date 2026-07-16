"""
ai/features/correlation.py - Cross-asset correlation features

RESPONSIBILITY:
Align peer-symbol candles to base rows and compute rolling correlation, beta,
relative return, and spread features.

VERSION: 1.0.0
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Dict, Mapping, Sequence

import numpy as np
from numpy.typing import NDArray

from ai.features.engine import candles_to_arrays
from ai.utils.math_ops import safe_div
from ai.utils.time_ops import align_timestamps
from ai.utils.types import CandleDict


# ==============================================================================
# TYPES
# ==============================================================================


FeatureMap = Dict[str, NDArray[np.floating]]


class CorrelationMeasure(str, Enum):
    """Supported cross-asset measures."""

    CORRELATION = "correlation"
    BETA = "beta"
    RELATIVE_RETURN = "relative_return"


@dataclass(frozen=True)
class CorrelationSpec:
    """Rolling correlation settings."""

    window: int = 50


# ==============================================================================
# PUBLIC API
# ==============================================================================


def compute_correlation_features(
    base_candles: Sequence[CandleDict],
    peer_symbols: Mapping[str, Sequence[CandleDict]],
    *,
    window: int,
) -> FeatureMap:
    """Compute cross-asset features aligned to base candles."""

    base = candles_to_arrays(base_candles)
    base_returns = _returns(base.close, 1)
    features: FeatureMap = {}
    spec = CorrelationSpec(window=max(int(window), 2))

    for symbol, candles in sorted(peer_symbols.items()):
        if not candles:
            continue
        peer = candles_to_arrays(candles)
        index_map = align_timestamps(base.timestamps, peer.timestamps)
        peer_close = _align_array(peer.close, index_map)
        peer_returns = _align_array(_returns(peer.close, 1), index_map)
        prefix = f"corr_{_sanitize_symbol(symbol)}"
        correlation = _rolling_correlation(base_returns, peer_returns, spec.window)
        beta = _rolling_beta(base_returns, peer_returns, spec.window)
        relative_return = base_returns - peer_returns
        spread = _normalized_spread(base.close, peer_close, spec.window)
        features[f"{prefix}_{spec.window}"] = correlation
        features[f"{prefix}_beta_{spec.window}"] = beta
        features[f"{prefix}_relative_return"] = relative_return
        features[f"{prefix}_spread_zscore_{spec.window}"] = spread
    return features


# ==============================================================================
# NUMERICAL HELPERS
# ==============================================================================


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


def _rolling_correlation(
    left: NDArray[np.floating],
    right: NDArray[np.floating],
    window: int,
) -> NDArray[np.floating]:
    left_arr = np.asarray(left, dtype=float)
    right_arr = np.asarray(right, dtype=float)
    out = np.full(len(left_arr), np.nan, dtype=float)
    if len(left_arr) < window:
        return out
    for idx in range(window - 1, len(left_arr)):
        x = left_arr[idx - window + 1 : idx + 1]
        y = right_arr[idx - window + 1 : idx + 1]
        mask = np.isfinite(x) & np.isfinite(y)
        if np.count_nonzero(mask) == window:
            x_centered = x[mask] - np.mean(x[mask])
            y_centered = y[mask] - np.mean(y[mask])
            denom = np.sqrt(np.sum(x_centered ** 2) * np.sum(y_centered ** 2))
            out[idx] = safe_div(np.sum(x_centered * y_centered), denom, default=np.nan)
    return out


def _rolling_beta(
    target: NDArray[np.floating],
    benchmark: NDArray[np.floating],
    window: int,
) -> NDArray[np.floating]:
    target_arr = np.asarray(target, dtype=float)
    bench_arr = np.asarray(benchmark, dtype=float)
    out = np.full(len(target_arr), np.nan, dtype=float)
    if len(target_arr) < window:
        return out
    for idx in range(window - 1, len(target_arr)):
        y = target_arr[idx - window + 1 : idx + 1]
        x = bench_arr[idx - window + 1 : idx + 1]
        mask = np.isfinite(x) & np.isfinite(y)
        if np.count_nonzero(mask) == window:
            x_centered = x[mask] - np.mean(x[mask])
            y_centered = y[mask] - np.mean(y[mask])
            out[idx] = safe_div(np.sum(x_centered * y_centered), np.sum(x_centered ** 2), default=np.nan)
    return out


def _normalized_spread(
    base_close: NDArray[np.floating],
    peer_close: NDArray[np.floating],
    window: int,
) -> NDArray[np.floating]:
    ratio = safe_div(base_close, peer_close, default=np.nan)
    mean = _rolling_nan_mean(ratio, window)
    std = _rolling_nan_std(ratio, window)
    return safe_div(ratio - mean, std, default=np.nan)


def _rolling_nan_mean(values: NDArray[np.floating], window: int) -> NDArray[np.floating]:
    arr = np.asarray(values, dtype=float)
    out = np.full(len(arr), np.nan, dtype=float)
    if len(arr) < window:
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
    if len(arr) < window:
        return out
    for idx in range(window - 1, len(arr)):
        sample = arr[idx - window + 1 : idx + 1]
        finite = sample[np.isfinite(sample)]
        if len(finite) == window:
            out[idx] = float(np.std(finite, ddof=0))
    return out


def _sanitize_symbol(symbol: str) -> str:
    cleaned = []
    for char in symbol.lower():
        cleaned.append(char if char.isalnum() else "_")
    return "".join(cleaned).strip("_") or "peer"
