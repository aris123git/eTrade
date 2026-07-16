"""
ai/research/gate.py - Champion / challenger promotion gate.

Keep only models that improve the primary metric. Discard regressions.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional


@dataclass(frozen=True)
class GateDecision:
    """Outcome of comparing a challenger against the champion."""

    accepted: bool
    reason: str
    metric_name: str
    challenger_score: Optional[float]
    champion_score: Optional[float]
    delta: Optional[float]
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "accepted": self.accepted,
            "reason": self.reason,
            "metric_name": self.metric_name,
            "challenger_score": self.challenger_score,
            "champion_score": self.champion_score,
            "delta": self.delta,
            "metadata": self.metadata,
        }


def extract_metric(metrics: Dict[str, Any], name: str) -> Optional[float]:
    """Pull a scalar metric, trying common prefixes when exact key is missing."""

    if not metrics:
        return None
    if name in metrics and _is_number(metrics[name]):
        return float(metrics[name])
    for prefix in ("test_", "val_", ""):
        key = f"{prefix}{name}" if prefix and not name.startswith(prefix) else name
        if key in metrics and _is_number(metrics[key]):
            return float(metrics[key])
    # Fallback: f1 / accuracy / sharpe family
    for candidate in (
        name,
        f"test_{name}",
        f"val_{name}",
        "test_f1",
        "val_f1",
        "f1",
        "test_accuracy",
        "val_accuracy",
        "accuracy",
        "test_sharpe",
        "sharpe",
    ):
        if candidate in metrics and _is_number(metrics[candidate]):
            return float(metrics[candidate])
    return None


def decide_promotion(
    *,
    challenger_metrics: Dict[str, Any],
    champion_metrics: Dict[str, Any] | None,
    metric_name: str = "test_f1",
    minimize: bool = False,
    min_improvement: float = 0.005,
) -> GateDecision:
    """
    Accept challenger only if it beats the champion by ``min_improvement``.

    First model (no champion) is accepted as baseline champion.
    """

    challenger = extract_metric(challenger_metrics, metric_name)
    if challenger is None:
        return GateDecision(
            accepted=False,
            reason="challenger_missing_metric",
            metric_name=metric_name,
            challenger_score=None,
            champion_score=None,
            delta=None,
        )

    if not champion_metrics:
        return GateDecision(
            accepted=True,
            reason="baseline_champion",
            metric_name=metric_name,
            challenger_score=challenger,
            champion_score=None,
            delta=None,
        )

    champion = extract_metric(champion_metrics, metric_name)
    if champion is None:
        return GateDecision(
            accepted=True,
            reason="champion_missing_metric",
            metric_name=metric_name,
            challenger_score=challenger,
            champion_score=None,
            delta=None,
        )

    delta = (champion - challenger) if minimize else (challenger - champion)
    if delta >= float(min_improvement):
        return GateDecision(
            accepted=True,
            reason="improved",
            metric_name=metric_name,
            challenger_score=challenger,
            champion_score=champion,
            delta=float(delta),
        )
    return GateDecision(
        accepted=False,
        reason="no_improvement",
        metric_name=metric_name,
        challenger_score=challenger,
        champion_score=champion,
        delta=float(delta),
    )


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)
