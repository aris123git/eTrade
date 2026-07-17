"""
ai/explainability/explainer.py - Unified explainability facade.

Provides feature importance, SHAP values, decision-tree style explanations,
and confidence intervals for model predictions.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
from numpy.typing import NDArray

from ai.explainability.decision import DecisionExplanation, Explainer as DecisionExplainer
from ai.explainability.feature_importance import FeatureImportance, model_feature_importance, normalize_importances
from ai.explainability.shap_explainer import ShapExplanation, ShapExplainer, explain_shap

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ConfidenceInterval:
    """Prediction confidence interval."""

    low: float
    high: float
    mean: float
    confidence_level: float = 0.95
    method: str = "bootstrap"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "low": self.low,
            "high": self.high,
            "mean": self.mean,
            "confidence_level": self.confidence_level,
            "method": self.method,
        }


@dataclass(frozen=True)
class TreeExplanation:
    """Human-readable decision path approximation."""

    rules: List[str]
    prediction: float | int | str
    depth: int

    def to_dict(self) -> Dict[str, Any]:
        return {"rules": list(self.rules), "prediction": self.prediction, "depth": self.depth}


@dataclass(frozen=True)
class ExplanationReport:
    """Full explainability payload for one prediction."""

    decision: str
    feature_importance: Dict[str, float]
    shap: Dict[str, Any]
    tree: Dict[str, Any]
    confidence_interval: Dict[str, Any]
    decision_explanation: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "decision": self.decision,
            "feature_importance": self.feature_importance,
            "shap": self.shap,
            "tree": self.tree,
            "confidence_interval": self.confidence_interval,
            "decision_explanation": self.decision_explanation,
        }


@dataclass
class ModelExplainer:
    """
    Production explainability service.

    Wraps existing SHAP / feature-importance / decision modules behind a
    single API used by the autonomous trading system.
    """

    model: object
    background: NDArray[np.floating] | Sequence[Sequence[float]]
    feature_names: Sequence[str] | None = None

    def __post_init__(self) -> None:
        self.background = _matrix(self.background)
        n_features = int(self.background.shape[1])
        self.feature_names = list(
            self.feature_names or [f"feature_{i}" for i in range(n_features)]
        )
        self._decision = DecisionExplainer(
            model=self.model,
            background=self.background,
            feature_names=self.feature_names,
        )
        self._shap = ShapExplainer(self.model, self.background, self.feature_names)
        logger.info("ModelExplainer ready features=%s", len(self.feature_names))

    def feature_importance(self, top_k: int = 15) -> FeatureImportance:
        """Global feature importance from the fitted model."""

        try:
            importance = model_feature_importance(self.model, self.feature_names)
        except Exception:
            logger.debug("model feature_importances_ unavailable; using SHAP mean |value|")
            shap_values = self._shap.explain(self.background[: min(50, len(self.background))])
            mean_abs = np.mean(np.abs(shap_values.values), axis=0)
            importance = FeatureImportance(
                scores=normalize_importances(mean_abs, self.feature_names, absolute=True),
                source="shap_mean_abs",
            )
        logger.info("top features: %s", list(importance.top(top_k).items())[:5])
        return importance

    def shap_values(
        self,
        X: NDArray[np.floating] | Sequence[Sequence[float]],
    ) -> ShapExplanation:
        """Local SHAP explanation for rows in X."""

        return explain_shap(self.model, X, self.background, feature_names=self.feature_names)

    def decision_tree_explanation(
        self,
        X: NDArray[np.floating] | Sequence[Sequence[float]],
        *,
        max_rules: int = 8,
    ) -> TreeExplanation:
        """
        Approximate a decision-tree style explanation using top SHAP drivers.

        Works for any model (not only sklearn trees).
        """

        matrix = _matrix(X)
        row = matrix[:1]
        prediction = _predict_scalar(self.model, row)
        shap_values = self._shap.explain(row)
        contribs = sorted(
            zip(self.feature_names, shap_values.values[0]),
            key=lambda item: abs(float(item[1])),
            reverse=True,
        )[:max_rules]
        rules: List[str] = []
        for name, value in contribs:
            direction = "increased" if float(value) > 0 else "decreased"
            rules.append(f"{name} {direction} the score by {abs(float(value)):.4f}")
        if not rules:
            rules.append("No dominant feature contributions detected")
        decision = "buy" if float(prediction) > 0 else ("sell" if float(prediction) < 0 else "hold")
        rules.insert(0, f"Decision ≈ {decision} (raw prediction={prediction})")
        return TreeExplanation(rules=rules, prediction=prediction, depth=len(rules))

    def confidence_interval(
        self,
        X: NDArray[np.floating] | Sequence[Sequence[float]],
        *,
        confidence_level: float = 0.95,
        n_bootstrap: int = 40,
    ) -> ConfidenceInterval:
        """
        Bootstrap confidence interval around the prediction.

        Perturbs the feature row with background noise and re-predicts.
        """

        matrix = _matrix(X)
        row = matrix[0].astype(float)
        base = float(_predict_scalar(self.model, row.reshape(1, -1)))
        rng = np.random.default_rng(42)
        samples: List[float] = []
        noise_scale = np.std(self.background, axis=0)
        noise_scale = np.where(noise_scale > 0, noise_scale, 1e-6)
        for _ in range(max(10, int(n_bootstrap))):
            perturbed = row + rng.normal(0.0, 0.15, size=row.shape) * noise_scale
            samples.append(float(_predict_scalar(self.model, perturbed.reshape(1, -1))))
        arr = np.asarray(samples, dtype=float)
        alpha = (1.0 - float(confidence_level)) / 2.0
        low, high = np.quantile(arr, [alpha, 1.0 - alpha])
        return ConfidenceInterval(
            low=float(low),
            high=float(high),
            mean=float(np.mean(arr)) if arr.size else base,
            confidence_level=float(confidence_level),
            method="feature_bootstrap",
        )

    def explain(
        self,
        X: NDArray[np.floating] | Sequence[Sequence[float]],
        *,
        decision: str | None = None,
    ) -> ExplanationReport:
        """Full explainability report for the first row of X."""

        matrix = _matrix(X)
        decision_expl: DecisionExplanation = self._decision.explain(matrix, decision=decision)
        importance = self.feature_importance()
        shap_local = self.shap_values(matrix[:1])
        tree = self.decision_tree_explanation(matrix[:1])
        interval = self.confidence_interval(matrix[:1])
        shap_payload = {
            "method": shap_local.method,
            "base_value": float(shap_local.base_values[0]) if len(shap_local.base_values) else 0.0,
            "values": {
                name: float(value)
                for name, value in zip(self.feature_names, shap_local.values[0])
            },
        }
        report = ExplanationReport(
            decision=decision_expl.decision,
            feature_importance=importance.top(15),
            shap=shap_payload,
            tree=tree.to_dict(),
            confidence_interval=interval.to_dict(),
            decision_explanation=decision_expl.to_dict(),
        )
        logger.info(
            "explained decision=%s top=%s ci=[%.4f, %.4f]",
            report.decision,
            list(report.feature_importance.items())[:3],
            interval.low,
            interval.high,
        )
        return report


def create_model_explainer(
    model: object,
    background: NDArray[np.floating] | Sequence[Sequence[float]],
    feature_names: Sequence[str] | None = None,
) -> ModelExplainer:
    return ModelExplainer(model=model, background=background, feature_names=feature_names)


def _matrix(values: NDArray[np.floating] | Sequence[Sequence[float]]) -> NDArray[np.floating]:
    arr = np.asarray(values, dtype=float)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    if arr.ndim != 2:
        raise ValueError("Expected a 2D feature matrix")
    return np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)


def _predict_scalar(model: object, X: NDArray[np.floating]) -> float | int:
    pred = np.asarray(getattr(model, "predict")(X)).reshape(-1)
    value = pred[0]
    return int(value) if np.issubdtype(pred.dtype, np.integer) else float(value)
