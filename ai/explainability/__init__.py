"""Explainability tools for predictions and trading decisions."""

from ai.explainability.decision import (
    ConfidenceExplanation,
    DecisionExplanation,
    Explainer,
    PredictionBreakdown,
)
from ai.explainability.feature_importance import (
    FeatureImportance,
    aggregate_importances,
    model_feature_importance,
    normalize_importances,
)
from ai.explainability.permutation import PermutationImportanceResult, permutation_importance
from ai.explainability.shap_explainer import ShapExplainer, ShapExplanation, explain_shap

__all__ = [
    "ConfidenceExplanation",
    "DecisionExplanation",
    "Explainer",
    "PredictionBreakdown",
    "FeatureImportance",
    "aggregate_importances",
    "model_feature_importance",
    "normalize_importances",
    "PermutationImportanceResult",
    "permutation_importance",
    "ShapExplainer",
    "ShapExplanation",
    "explain_shap",
]
