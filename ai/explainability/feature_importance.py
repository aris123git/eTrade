"""
ai/explainability/feature_importance.py - Feature importance utilities.

VERSION: 1.0.0
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Sequence

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class FeatureImportance:
    """Named feature importance vector."""

    scores: Dict[str, float]
    source: str

    def top(self, k: int = 10) -> Dict[str, float]:
        """Return top-k features by absolute importance."""

        ordered = sorted(self.scores.items(), key=lambda item: abs(item[1]), reverse=True)
        return dict(ordered[: max(0, int(k))])


def normalize_importances(
    values: Sequence[float] | NDArray[np.floating],
    feature_names: Sequence[str] | None = None,
    absolute: bool = True,
) -> Dict[str, float]:
    """Normalize raw importances so absolute values sum to one."""

    arr = np.asarray(values, dtype=float).reshape(-1)
    if absolute:
        arr = np.abs(arr)
    names = list(feature_names or [f"feature_{idx}" for idx in range(arr.size)])
    if len(names) != arr.size:
        raise ValueError("feature_names length must match values")
    total = float(np.sum(np.abs(arr)))
    if total > 0.0:
        arr = arr / total
    return {name: float(value) for name, value in zip(names, arr)}


def model_feature_importance(
    model: object,
    feature_names: Sequence[str] | None = None,
) -> FeatureImportance:
    """Read feature_importances_ or coef_ from a fitted model."""

    raw = getattr(model, "feature_importances_", None)
    if raw is None:
        raw = getattr(model, "coef_", None)
    if raw is None and hasattr(model, "estimator"):
        raw = getattr(getattr(model, "estimator"), "feature_importances_", None)
    if raw is None:
        raise ValueError("model does not expose feature_importances_ or coef_")
    arr = np.asarray(raw, dtype=float)
    if arr.ndim > 1:
        arr = np.mean(np.abs(arr), axis=0)
    return FeatureImportance(scores=normalize_importances(arr, feature_names), source="model")


def aggregate_importances(
    importances: Sequence[FeatureImportance | Dict[str, float]],
    weights: Sequence[float] | None = None,
) -> FeatureImportance:
    """Average multiple importance maps."""

    if not importances:
        return FeatureImportance(scores={}, source="aggregate")
    maps = [item.scores if isinstance(item, FeatureImportance) else item for item in importances]
    all_names = sorted({name for mapping in maps for name in mapping})
    raw_weights = np.asarray(weights if weights is not None else np.ones(len(maps)), dtype=float)
    raw_weights = raw_weights / (np.sum(raw_weights) or 1.0)
    scores: Dict[str, float] = {}
    for name in all_names:
        scores[name] = float(sum(weight * mapping.get(name, 0.0) for weight, mapping in zip(raw_weights, maps)))
    return FeatureImportance(scores=normalize_importances(list(scores.values()), list(scores)), source="aggregate")
