"""
ai/evaluation/importance.py - Feature importance utilities.

VERSION: 1.0.0
"""

from __future__ import annotations

from typing import Any, Callable, Dict, Sequence

import numpy as np
from numpy.typing import NDArray

from ai.evaluation.classifier_metrics import accuracy
from ai.evaluation.regression_metrics import rmse


MetricFn = Callable[[Sequence[Any], Sequence[Any]], float]


def feature_importance(
    model: Any,
    feature_names: Sequence[str] | None = None,
    normalize: bool = True,
) -> Dict[str, float]:
    """
    Extract feature importance from common estimator attributes.

    Supports tree-style ``feature_importances_`` and linear ``coef_`` attributes.
    """
    if model is None:
        return {}
    raw = _raw_importance(model)
    if raw is None:
        return {}
    values = np.asarray(raw, dtype=float).reshape(-1)
    if normalize:
        total = float(np.sum(np.abs(values)))
        if total > 0.0:
            values = values / total
    names = _feature_names(feature_names, len(values))
    return {name: float(value) for name, value in zip(names, values)}


def permutation_importance(
    model: Any,
    X: Sequence[Sequence[float]],
    y: Sequence[Any],
    feature_names: Sequence[str] | None = None,
    metric: MetricFn | None = None,
    n_repeats: int = 5,
    random_seed: int | None = None,
    greater_is_better: bool = True,
) -> Dict[str, Dict[str, float]]:
    """
    Compute permutation importance as score degradation after shuffling a column.

    The default metric is accuracy for discrete targets and negative RMSE for
    continuous targets, making larger scores better in both cases.
    """
    if model is None or not hasattr(model, "predict"):
        raise ValueError("model must expose a predict method")
    if n_repeats <= 0:
        raise ValueError("n_repeats must be > 0")

    x = np.asarray(X, dtype=float)
    if x.ndim != 2:
        raise ValueError("X must be a 2D array")
    target = np.asarray(y).reshape(-1)
    if len(target) != x.shape[0]:
        raise ValueError("X and y must contain the same number of rows")

    scorer = metric or _default_metric(target)
    baseline = _score(model, x, target, scorer, greater_is_better)
    rng = np.random.default_rng(random_seed)
    names = _feature_names(feature_names, x.shape[1])
    result: Dict[str, Dict[str, float]] = {}
    for column, name in enumerate(names):
        drops = np.empty(n_repeats, dtype=float)
        for repeat in range(n_repeats):
            shuffled = x.copy()
            shuffled[:, column] = rng.permutation(shuffled[:, column])
            shuffled_score = _score(model, shuffled, target, scorer, greater_is_better)
            drops[repeat] = baseline - shuffled_score
        result[name] = {
            "importance": float(np.mean(drops)),
            "std": float(np.std(drops, ddof=1)) if n_repeats > 1 else 0.0,
            "baseline": float(baseline),
        }
    return result


def importance_summary(
    model: Any,
    feature_names: Sequence[str] | None = None,
) -> Dict[str, float]:
    """Alias for model-derived feature importance."""
    return feature_importance(model, feature_names=feature_names)


def _raw_importance(model: Any) -> NDArray[np.floating] | None:
    if hasattr(model, "feature_importances_"):
        return np.asarray(getattr(model, "feature_importances_"), dtype=float)
    if hasattr(model, "coef_"):
        coef = np.asarray(getattr(model, "coef_"), dtype=float)
        if coef.ndim == 1:
            return np.abs(coef)
        return np.mean(np.abs(coef), axis=0)
    return None


def _feature_names(feature_names: Sequence[str] | None, count: int) -> list[str]:
    names = list(feature_names or [])
    if len(names) < count:
        names.extend(f"feature_{idx}" for idx in range(len(names), count))
    return names[:count]


def _default_metric(y: NDArray[Any]) -> MetricFn:
    finite_y = y
    if np.issubdtype(finite_y.dtype, np.number):
        numeric = finite_y.astype(float)
        unique = np.unique(numeric[np.isfinite(numeric)])
        if len(unique) > 20 or np.any(unique != np.round(unique)):
            return lambda true, pred: -rmse(true, pred)
    return accuracy


def _score(
    model: Any,
    X: NDArray[np.floating],
    y: NDArray[Any],
    metric: MetricFn,
    greater_is_better: bool,
) -> float:
    predictions = np.asarray(model.predict(X)).reshape(-1)
    value = float(metric(y, predictions))
    return value if greater_is_better else -value
