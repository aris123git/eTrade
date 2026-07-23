"""
ai/monitoring/drift.py - Data, prediction, feature, and model drift detectors.

VERSION: 1.0.0
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Sequence

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class DriftResult:
    """Drift detector output."""

    drifted: bool
    score: float
    threshold: float
    method: str
    details: Dict[str, Any] = field(default_factory=dict)


def population_stability_index(
    reference: Sequence[float] | NDArray[np.floating],
    current: Sequence[float] | NDArray[np.floating],
    bins: int = 10,
    epsilon: float = 1e-8,
) -> float:
    """Compute PSI between two numeric samples."""

    ref = _finite(reference)
    cur = _finite(current)
    if ref.size == 0 or cur.size == 0:
        return 0.0
    edges = np.unique(np.percentile(ref, np.linspace(0.0, 100.0, max(2, int(bins)) + 1)))
    if edges.size < 2:
        return 0.0
    ref_counts, _ = np.histogram(ref, bins=edges)
    cur_counts, _ = np.histogram(cur, bins=edges)
    ref_pct = ref_counts / max(float(np.sum(ref_counts)), 1.0)
    cur_pct = cur_counts / max(float(np.sum(cur_counts)), 1.0)
    ref_pct = np.maximum(ref_pct, epsilon)
    cur_pct = np.maximum(cur_pct, epsilon)
    return float(np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct)))


def mean_shift_score(
    reference: Sequence[float] | NDArray[np.floating],
    current: Sequence[float] | NDArray[np.floating],
    epsilon: float = 1e-8,
) -> float:
    """Return absolute mean shift measured in reference standard deviations."""

    ref = _finite(reference)
    cur = _finite(current)
    if ref.size == 0 or cur.size == 0:
        return 0.0
    return float(abs(np.mean(cur) - np.mean(ref)) / (np.std(ref, ddof=1) + epsilon))


@dataclass
class FeatureDriftDetector:
    """Detect per-feature drift with PSI and mean shift."""

    threshold: float = 0.2
    mean_shift_threshold: float = 3.0
    bins: int = 10

    def detect(
        self,
        reference: NDArray[np.floating],
        current: NDArray[np.floating],
        feature_names: Sequence[str] | None = None,
    ) -> DriftResult:
        ref = _matrix(reference)
        cur = _matrix(current)
        if ref.shape[1] != cur.shape[1]:
            raise ValueError("reference and current feature counts must match")
        names = list(feature_names or [f"feature_{idx}" for idx in range(ref.shape[1])])
        scores: Dict[str, float] = {}
        mean_scores: Dict[str, float] = {}
        for idx, name in enumerate(names):
            scores[name] = population_stability_index(ref[:, idx], cur[:, idx], bins=self.bins)
            mean_scores[name] = mean_shift_score(ref[:, idx], cur[:, idx])
        max_psi = max(scores.values()) if scores else 0.0
        max_shift = max(mean_scores.values()) if mean_scores else 0.0
        drifted = max_psi > self.threshold or max_shift > self.mean_shift_threshold
        details = {f"psi_{key}": value for key, value in scores.items()}
        details.update({f"mean_shift_{key}": value for key, value in mean_scores.items()})
        return DriftResult(
            drifted=drifted,
            score=float(max(max_psi, max_shift)),
            threshold=float(self.threshold),
            method="feature_psi_mean_shift",
            details=details,
        )


@dataclass
class DataDriftDetector(FeatureDriftDetector):
    """Alias for full input data drift detection."""


@dataclass
class PredictionDriftDetector:
    """Detect drift in model predictions."""

    threshold: float = 0.2
    bins: int = 10

    def detect(self, reference: Sequence[float], current: Sequence[float]) -> DriftResult:
        psi = population_stability_index(reference, current, bins=self.bins)
        shift = mean_shift_score(reference, current)
        return DriftResult(
            drifted=psi > self.threshold,
            score=float(psi),
            threshold=float(self.threshold),
            method="prediction_psi",
            details={"psi": float(psi), "mean_shift": float(shift)},
        )


@dataclass
class ModelDriftDetector:
    """Detect model quality degradation."""

    threshold: float = 0.05
    metric_name: str = "score"

    def detect(self, reference_metric: float, current_metric: float) -> DriftResult:
        degradation = float(reference_metric) - float(current_metric)
        return DriftResult(
            drifted=degradation > float(self.threshold),
            score=degradation,
            threshold=float(self.threshold),
            method="metric_degradation",
            details={
                "reference": float(reference_metric),
                "current": float(current_metric),
                "metric": self.metric_name,
            },
        )


def _finite(values: Sequence[float] | NDArray[np.floating]) -> NDArray[np.floating]:
    arr = np.asarray(values, dtype=float).reshape(-1)
    return arr[np.isfinite(arr)]


def _matrix(values: NDArray[np.floating]) -> NDArray[np.floating]:
    arr = np.asarray(values, dtype=float)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    if arr.ndim != 2:
        raise ValueError("Expected 2D matrix")
    return arr
