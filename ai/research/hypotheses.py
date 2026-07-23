"""
ai/research/hypotheses.py - Weakness detection and research hypotheses.

Example:
  "I perform poorly on XAUUSD" → download more gold history, gold features, retrain.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence


@dataclass(frozen=True)
class Hypothesis:
    """Actionable research idea produced from measured weaknesses."""

    id: str
    symbol: str
    kind: str
    priority: float
    rationale: str
    actions: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "symbol": self.symbol,
            "kind": self.kind,
            "priority": self.priority,
            "rationale": self.rationale,
            "actions": list(self.actions),
            "metadata": self.metadata,
        }


def generate_hypotheses(
    *,
    per_symbol_metrics: Dict[str, Dict[str, Any]],
    validation_failures: Sequence[str] | None = None,
    drift_by_symbol: Dict[str, bool] | None = None,
    primary_metric: str = "test_f1",
    weak_threshold: float = 0.55,
) -> List[Hypothesis]:
    """
    Turn measured weaknesses into the next research actions.

    Does not invent data — actions request more history or feature work from
    real sources.
    """

    hypotheses: List[Hypothesis] = []
    validation_failures = list(validation_failures or [])
    drift_by_symbol = dict(drift_by_symbol or {})

    for idx, series_key in enumerate(validation_failures):
        symbol = series_key.split(":")[0] if ":" in series_key else series_key
        hypotheses.append(
            Hypothesis(
                id=f"data_gap_{idx}_{symbol}",
                symbol=symbol,
                kind="data_coverage",
                priority=1.0,
                rationale=f"Validation failed for {series_key}; history is incomplete or inconsistent.",
                actions=[
                    "download_missing_history",
                    "repair_gaps",
                    "revalidate",
                ],
                metadata={"series": series_key},
            )
        )

    scores: List[tuple[str, float]] = []
    for symbol, metrics in per_symbol_metrics.items():
        score = _metric(metrics, primary_metric)
        if score is None:
            continue
        scores.append((symbol, score))
        if score < weak_threshold:
            hypotheses.append(
                Hypothesis(
                    id=f"weak_edge_{symbol}",
                    symbol=symbol,
                    kind="weak_performance",
                    priority=float(max(0.0, weak_threshold - score)),
                    rationale=f"Poor {primary_metric}={score:.4f} on {symbol}.",
                    actions=[
                        "download_more_history",
                        "create_symbol_specific_features",
                        "retrain",
                        "compare_and_keep_improvements_only",
                    ],
                    metadata={"metric": primary_metric, "score": score},
                )
            )
        if drift_by_symbol.get(symbol):
            hypotheses.append(
                Hypothesis(
                    id=f"drift_{symbol}",
                    symbol=symbol,
                    kind="drift",
                    priority=0.8,
                    rationale=f"Feature/prediction drift detected on {symbol}.",
                    actions=["retrain", "refresh_features", "monitor"],
                    metadata={"drift": True},
                )
            )

    if len(scores) >= 2:
        scores.sort(key=lambda item: item[1])
        weakest, weak_score = scores[0]
        strongest, strong_score = scores[-1]
        if strong_score - weak_score >= 0.05:
            hypotheses.append(
                Hypothesis(
                    id=f"relative_weak_{weakest}",
                    symbol=weakest,
                    kind="relative_underperformance",
                    priority=float(strong_score - weak_score),
                    rationale=(
                        f"{weakest} underperforms {strongest} "
                        f"({weak_score:.4f} vs {strong_score:.4f} on {primary_metric})."
                    ),
                    actions=[
                        "expand_history_for_symbol",
                        "engineer_regime_features",
                        "retrain_symbol_specialist",
                        "discard_if_no_improvement",
                    ],
                    metadata={
                        "weak_score": weak_score,
                        "strong_symbol": strongest,
                        "strong_score": strong_score,
                    },
                )
            )

    hypotheses.sort(key=lambda h: h.priority, reverse=True)
    return hypotheses


def _metric(metrics: Dict[str, Any], name: str) -> Optional[float]:
    from ai.research.gate import extract_metric

    return extract_metric(metrics, name)
