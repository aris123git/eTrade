"""
ai/explainability/decision.py - Decision explanation services.

VERSION: 1.0.0
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, Sequence

import numpy as np
from numpy.typing import NDArray

from ai.explainability.feature_importance import FeatureImportance, normalize_importances
from ai.explainability.shap_explainer import ShapExplainer


@dataclass(frozen=True)
class PredictionBreakdown:
    """Prediction components and feature contributions."""

    prediction: float | int | str
    base_value: float
    contributions: Dict[str, float]
    top_features: Dict[str, float]


@dataclass(frozen=True)
class ConfidenceExplanation:
    """Confidence score with interpretable drivers."""

    confidence: float
    method: str
    drivers: Dict[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class DecisionExplanation:
    """Complete explanation for a model decision."""

    decision: str
    prediction: PredictionBreakdown
    confidence: ConfidenceExplanation
    feature_importance: FeatureImportance
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "decision": self.decision,
            "prediction": {
                "prediction": self.prediction.prediction,
                "base_value": self.prediction.base_value,
                "contributions": self.prediction.contributions,
                "top_features": self.prediction.top_features,
            },
            "confidence": {
                "confidence": self.confidence.confidence,
                "method": self.confidence.method,
                "drivers": self.confidence.drivers,
            },
            "feature_importance": self.feature_importance.scores,
            "feature_importance_source": self.feature_importance.source,
            "timestamp": self.timestamp.isoformat(),
            "metadata": self.metadata,
        }


@dataclass
class Explainer:
    """High-level service for model decision explanations."""

    model: object
    background: NDArray[np.floating] | Sequence[Sequence[float]]
    feature_names: Sequence[str] | None = None

    def __post_init__(self) -> None:
        self.background = _matrix(self.background)
        self.feature_names = list(self.feature_names or [f"feature_{idx}" for idx in range(self.background.shape[1])])
        self.shap = ShapExplainer(self.model, self.background, self.feature_names)

    def explain(
        self,
        X: NDArray[np.floating] | Sequence[Sequence[float]],
        decision: str | None = None,
        metadata: Dict[str, Any] | None = None,
    ) -> DecisionExplanation:
        """Explain the first row of X as a trading decision."""

        matrix = _matrix(X)
        row = matrix[:1]
        prediction = self._prediction(row)
        shap_values = self.shap.explain(row)
        contributions = {
            name: float(value) for name, value in zip(self.feature_names, shap_values.values[0])
        }
        feature_importance = FeatureImportance(
            scores=normalize_importances(list(contributions.values()), list(contributions), absolute=True),
            source=shap_values.method,
        )
        top_features = dict(sorted(contributions.items(), key=lambda item: abs(item[1]), reverse=True)[:10])
        confidence = self._confidence(row, prediction)
        return DecisionExplanation(
            decision=decision or self._decision_from_prediction(prediction),
            prediction=PredictionBreakdown(
                prediction=prediction,
                base_value=float(shap_values.base_values[0]) if len(shap_values.base_values) else 0.0,
                contributions=contributions,
                top_features=top_features,
            ),
            confidence=confidence,
            feature_importance=feature_importance,
            metadata=metadata or {},
        )

    def _prediction(self, X: NDArray[np.floating]) -> float | int:
        pred = np.asarray(getattr(self.model, "predict")(X)).reshape(-1)
        value = pred[0]
        return int(value) if np.issubdtype(pred.dtype, np.integer) else float(value)

    def _confidence(self, X: NDArray[np.floating], prediction: float | int) -> ConfidenceExplanation:
        if hasattr(self.model, "predict_proba"):
            proba = getattr(self.model, "predict_proba")(X)
            if proba is not None:
                arr = np.asarray(proba, dtype=float).reshape(1, -1)
                confidence = float(np.max(arr[0]))
                return ConfidenceExplanation(
                    confidence=confidence,
                    method="predict_proba",
                    drivers={f"class_{idx}": float(value) for idx, value in enumerate(arr[0])},
                )
        magnitude = min(1.0, abs(float(prediction)))
        return ConfidenceExplanation(confidence=magnitude, method="prediction_magnitude")

    @staticmethod
    def _decision_from_prediction(prediction: float | int) -> str:
        value = float(prediction)
        if value > 0.0:
            return "buy"
        if value < 0.0:
            return "sell"
        return "hold"


def _matrix(values: NDArray[np.floating] | Sequence[Sequence[float]]) -> NDArray[np.floating]:
    arr = np.asarray(values, dtype=float)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    if arr.ndim != 2:
        raise ValueError("Expected a 2D feature matrix")
    return np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
