"""
ai/evaluation/regression_metrics.py - Regression metric primitives.

VERSION: 1.0.0
"""

from __future__ import annotations

from typing import Dict, Sequence

import numpy as np
from numpy.typing import NDArray


def rmse(y_true: Sequence[float], y_pred: Sequence[float]) -> float:
    """Root mean squared error."""
    true, pred = _aligned_float_arrays(y_true, y_pred)
    if len(true) == 0:
        return 0.0
    return float(np.sqrt(np.mean((true - pred) ** 2)))


def mae(y_true: Sequence[float], y_pred: Sequence[float]) -> float:
    """Mean absolute error."""
    true, pred = _aligned_float_arrays(y_true, y_pred)
    if len(true) == 0:
        return 0.0
    return float(np.mean(np.abs(true - pred)))


def mape(y_true: Sequence[float], y_pred: Sequence[float], epsilon: float = 1e-12) -> float:
    """Mean absolute percentage error as a percentage."""
    true, pred = _aligned_float_arrays(y_true, y_pred)
    if len(true) == 0:
        return 0.0
    denominator = np.where(np.abs(true) > epsilon, np.abs(true), np.nan)
    values = np.abs((true - pred) / denominator)
    if np.all(np.isnan(values)):
        return 0.0
    return float(np.nanmean(values) * 100.0)


def r2(y_true: Sequence[float], y_pred: Sequence[float]) -> float:
    """Coefficient of determination."""
    true, pred = _aligned_float_arrays(y_true, y_pred)
    if len(true) == 0:
        return 0.0
    residual_sum = float(np.sum((true - pred) ** 2))
    total_sum = float(np.sum((true - np.mean(true)) ** 2))
    if total_sum == 0.0:
        return 1.0 if residual_sum == 0.0 else 0.0
    return float(1.0 - residual_sum / total_sum)


def regression_metrics(y_true: Sequence[float], y_pred: Sequence[float]) -> Dict[str, float]:
    """Aggregate common regression metrics into a serializable dictionary."""
    return {
        "rmse": rmse(y_true, y_pred),
        "mae": mae(y_true, y_pred),
        "mape": mape(y_true, y_pred),
        "r2": r2(y_true, y_pred),
    }


def _aligned_float_arrays(
    y_true: Sequence[float],
    y_pred: Sequence[float],
) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
    true = np.asarray(y_true, dtype=float).reshape(-1)
    pred = np.asarray(y_pred, dtype=float).reshape(-1)
    if len(true) != len(pred):
        raise ValueError("y_true and y_pred must have the same length")
    mask = np.isfinite(true) & np.isfinite(pred)
    return true[mask], pred[mask]
