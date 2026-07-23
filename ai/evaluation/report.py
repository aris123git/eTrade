"""
ai/evaluation/report.py - Evaluation report aggregation.

VERSION: 1.0.0
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, Sequence

import numpy as np

from ai.config.settings import AIConfig
from ai.evaluation.classifier_metrics import classification_metrics
from ai.evaluation.importance import feature_importance
from ai.evaluation.regression_metrics import regression_metrics
from ai.evaluation.trading_metrics import trading_metrics


@dataclass(frozen=True)
class EvaluationReport:
    """Serializable aggregate of model and trading evaluation metrics."""

    task: str
    classification: Dict[str, Any] = field(default_factory=dict)
    regression: Dict[str, float] = field(default_factory=dict)
    trading: Dict[str, Any] = field(default_factory=dict)
    importance: Dict[str, float] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> Dict[str, Any]:
        """Serialize the report to JSON-compatible primitives."""
        return {
            "task": self.task,
            "classification": self.classification,
            "regression": self.regression,
            "trading": self.trading,
            "importance": self.importance,
            "metadata": self.metadata,
            "created_at": self.created_at,
        }


@dataclass
class Evaluator:
    """High-level evaluation facade driven by AIConfig defaults."""

    config: AIConfig = field(default_factory=AIConfig)
    task: str | None = None

    def evaluate(
        self,
        y_true: Sequence[Any],
        y_pred: Sequence[Any],
        y_proba: Sequence[float] | Sequence[Sequence[float]] | None = None,
        returns: Sequence[float] | None = None,
        feature_names: Sequence[str] | None = None,
        model: Any = None,
    ) -> EvaluationReport:
        """Evaluate predictions, optional trading returns, and optional model importance."""
        inferred_task = self.task or _infer_task(y_true, y_pred, y_proba)
        classification: Dict[str, Any] = {}
        regression: Dict[str, float] = {}
        if inferred_task == "classification":
            classification = classification_metrics(y_true, y_pred, y_proba=y_proba)
        else:
            regression = regression_metrics(y_true, y_pred)

        trade_metrics = (
            trading_metrics(returns, initial_equity=1.0, periods=252)
            if returns is not None
            else {}
        )
        importances = feature_importance(model, feature_names=feature_names) if model is not None else {}
        metadata = {
            "project_name": self.config.project_name,
            "config_version": self.config.version,
            "n_samples": int(len(np.asarray(y_true).reshape(-1))),
        }
        if feature_names is not None:
            metadata["feature_count"] = len(feature_names)
        return EvaluationReport(
            task=inferred_task,
            classification=classification,
            regression=regression,
            trading=trade_metrics,
            importance=importances,
            metadata=metadata,
        )


def create_evaluator(config: AIConfig | None = None, task: str | None = None) -> Evaluator:
    """Factory for Evaluator."""
    return Evaluator(config=config or AIConfig(), task=task)


def _infer_task(
    y_true: Sequence[Any],
    y_pred: Sequence[Any],
    y_proba: Sequence[float] | Sequence[Sequence[float]] | None,
) -> str:
    if y_proba is not None:
        return "classification"
    true = np.asarray(y_true)
    pred = np.asarray(y_pred)
    if not np.issubdtype(true.dtype, np.number) or not np.issubdtype(pred.dtype, np.number):
        return "classification"
    true_flat = true.astype(float).reshape(-1)
    pred_flat = pred.astype(float).reshape(-1)
    unique_true = np.unique(true_flat[np.isfinite(true_flat)])
    integer_like = np.all(unique_true == np.round(unique_true)) and np.all(pred_flat == np.round(pred_flat))
    if integer_like and len(unique_true) <= max(20, int(np.sqrt(max(len(true_flat), 1))) + 1):
        return "classification"
    return "regression"
