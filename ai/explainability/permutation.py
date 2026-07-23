"""
ai/explainability/permutation.py - Permutation importance.

VERSION: 1.0.0
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np
from numpy.typing import NDArray


ScoreFn = Callable[[NDArray[np.floating], NDArray[np.floating]], float]


@dataclass(frozen=True)
class PermutationImportanceResult:
    """Permutation importance output."""

    feature_names: list[str]
    importances: NDArray[np.floating]
    importances_std: NDArray[np.floating]
    baseline_score: float

    def as_dict(self) -> dict[str, float | dict[str, float]]:
        return {
            "baseline_score": float(self.baseline_score),
            "importances": {
                name: float(value) for name, value in zip(self.feature_names, self.importances)
            },
            "importances_std": {
                name: float(value) for name, value in zip(self.feature_names, self.importances_std)
            },
        }


def permutation_importance(
    model: object,
    X: NDArray[np.floating] | Sequence[Sequence[float]],
    y: NDArray[np.floating] | Sequence[float],
    feature_names: Sequence[str] | None = None,
    scoring: ScoreFn | None = None,
    n_repeats: int = 5,
    random_seed: int | None = 42,
) -> PermutationImportanceResult:
    """Compute feature importance as score decrease after column shuffling."""

    matrix = np.asarray(X, dtype=float)
    if matrix.ndim == 1:
        matrix = matrix.reshape(-1, 1)
    target = np.asarray(y, dtype=float).reshape(-1)
    if len(matrix) != len(target):
        raise ValueError("X and y must have matching rows")
    names = list(feature_names) if feature_names is not None else [f"feature_{idx}" for idx in range(matrix.shape[1])]
    if len(names) != matrix.shape[1]:
        raise ValueError("feature_names length must match X columns")
    score_fn = scoring or _default_score
    baseline_pred = _predict(model, matrix)
    baseline = float(score_fn(target, baseline_pred))
    rng = np.random.default_rng(random_seed)
    repeats = max(1, int(n_repeats))
    values = np.zeros((matrix.shape[1], repeats), dtype=float)
    for feature_idx in range(matrix.shape[1]):
        for repeat in range(repeats):
            shuffled = matrix.copy()
            shuffled[:, feature_idx] = rng.permutation(shuffled[:, feature_idx])
            values[feature_idx, repeat] = baseline - float(score_fn(target, _predict(model, shuffled)))
    return PermutationImportanceResult(
        feature_names=names,
        importances=np.mean(values, axis=1),
        importances_std=np.std(values, axis=1),
        baseline_score=baseline,
    )


def _predict(model: object, X: NDArray[np.floating]) -> NDArray[np.floating]:
    if not hasattr(model, "predict"):
        raise TypeError("model must expose predict(X)")
    return np.asarray(getattr(model, "predict")(X), dtype=float).reshape(-1)


def _default_score(y_true: NDArray[np.floating], y_pred: NDArray[np.floating]) -> float:
    true = y_true.reshape(-1)
    pred = y_pred.reshape(-1)
    if true.size == 0:
        return 0.0
    unique = np.unique(true)
    if unique.size <= 10 and np.allclose(unique, np.round(unique)):
        return float(np.mean(true == np.round(pred)))
    return float(-np.mean((true - pred) ** 2))
