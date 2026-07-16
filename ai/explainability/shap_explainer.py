"""
ai/explainability/shap_explainer.py - SHAP and fallback explainers.

VERSION: 1.0.0
"""

from __future__ import annotations

from dataclasses import dataclass
from importlib import import_module
from typing import Sequence

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class ShapExplanation:
    """SHAP-style contribution matrix."""

    values: NDArray[np.floating]
    base_values: NDArray[np.floating]
    feature_names: list[str]
    method: str

    def mean_abs(self) -> dict[str, float]:
        scores = np.mean(np.abs(self.values), axis=0) if self.values.size else np.zeros(len(self.feature_names))
        return {name: float(value) for name, value in zip(self.feature_names, scores)}


@dataclass
class ShapExplainer:
    """Use SHAP when installed, otherwise approximate feature attributions."""

    model: object
    background: NDArray[np.floating] | Sequence[Sequence[float]]
    feature_names: Sequence[str] | None = None
    nsamples: int = 100

    def __post_init__(self) -> None:
        self.background = _matrix(self.background)
        self.feature_names = list(self.feature_names or [f"feature_{idx}" for idx in range(self.background.shape[1])])
        self._native = None
        try:
            shap = import_module("shap")
            self._native = shap.Explainer(self._predict, self.background)
        except Exception:
            self._native = None

    def explain(self, X: NDArray[np.floating] | Sequence[Sequence[float]]) -> ShapExplanation:
        """Return native SHAP values or a permutation-style local fallback."""

        matrix = _matrix(X)
        if len(self.feature_names) != matrix.shape[1]:
            raise ValueError("feature_names length must match X columns")
        if self._native is not None:
            explanation = self._native(matrix)
            values = np.asarray(explanation.values, dtype=float)
            if values.ndim == 3:
                values = values[:, :, -1]
            base = np.asarray(explanation.base_values, dtype=float).reshape(len(matrix), -1)[:, 0]
            return ShapExplanation(values=values, base_values=base, feature_names=list(self.feature_names), method="shap")
        return self._fallback(matrix)

    def _fallback(self, matrix: NDArray[np.floating]) -> ShapExplanation:
        background_mean = np.mean(self.background, axis=0)
        baseline_matrix = np.tile(background_mean, (len(matrix), 1))
        baseline = self._predict(baseline_matrix)
        full = self._predict(matrix)
        values = np.zeros((len(matrix), matrix.shape[1]), dtype=float)
        for feature_idx in range(matrix.shape[1]):
            perturbed = baseline_matrix.copy()
            perturbed[:, feature_idx] = matrix[:, feature_idx]
            values[:, feature_idx] = self._predict(perturbed) - baseline
        residual = full - baseline - np.sum(values, axis=1)
        if values.shape[1] > 0:
            values += residual.reshape(-1, 1) / values.shape[1]
        return ShapExplanation(
            values=values,
            base_values=baseline,
            feature_names=list(self.feature_names),
            method="kernel_fallback",
        )

    def _predict(self, X: NDArray[np.floating]) -> NDArray[np.floating]:
        if hasattr(self.model, "predict_proba"):
            proba = getattr(self.model, "predict_proba")(X)
            if proba is not None:
                arr = np.asarray(proba, dtype=float)
                if arr.ndim == 2 and arr.shape[1] > 1:
                    return arr[:, -1]
        if not hasattr(self.model, "predict"):
            raise TypeError("model must expose predict(X)")
        return np.asarray(getattr(self.model, "predict")(X), dtype=float).reshape(-1)


def explain_shap(
    model: object,
    X: NDArray[np.floating] | Sequence[Sequence[float]],
    background: NDArray[np.floating] | Sequence[Sequence[float]] | None = None,
    feature_names: Sequence[str] | None = None,
) -> ShapExplanation:
    """Convenience function for SHAP-style explanations."""

    matrix = _matrix(X)
    base = matrix if background is None else _matrix(background)
    return ShapExplainer(model=model, background=base, feature_names=feature_names).explain(matrix)


def _matrix(values: NDArray[np.floating] | Sequence[Sequence[float]]) -> NDArray[np.floating]:
    arr = np.asarray(values, dtype=float)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    if arr.ndim != 2:
        raise ValueError("Expected a 2D matrix")
    return np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
