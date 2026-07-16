"""
ai/monitoring/tracker.py - Performance tracking service.

VERSION: 1.0.0
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Sequence

import numpy as np

from ai.config.settings import AIConfig
from ai.monitoring.alerts import Alert, AlertManager
from ai.monitoring.drift import DriftResult, PredictionDriftDetector
from ai.monitoring.metrics import MetricRecorder, collect_resource_metrics


@dataclass(frozen=True)
class PerformanceRecord:
    """One prediction/training performance event."""

    timestamp: datetime
    metrics: Dict[str, float]
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PerformanceTracker:
    """Track metrics, resources, drift, and alerts for AI services."""

    config: AIConfig = field(default_factory=AIConfig)
    recorder: MetricRecorder = field(default_factory=MetricRecorder)
    alert_manager: AlertManager = field(default_factory=AlertManager)
    records: List[PerformanceRecord] = field(default_factory=list)
    reference_predictions: List[float] = field(default_factory=list)
    prediction_drift: PredictionDriftDetector | None = None

    def __post_init__(self) -> None:
        if self.prediction_drift is None:
            self.prediction_drift = PredictionDriftDetector(threshold=self.config.monitoring.drift_threshold)
        self.alert_manager.add_threshold(
            "prediction_latency_warn",
            "prediction_ms_p95",
            self.config.monitoring.prediction_latency_warn_ms,
            ">",
            "warning",
        )
        self.alert_manager.add_threshold(
            "training_memory_warn",
            "memory_mb",
            self.config.monitoring.training_memory_warn_mb,
            ">",
            "warning",
        )

    def record_prediction(
        self,
        prediction: float,
        latency_ms: float | None = None,
        metadata: Dict[str, Any] | None = None,
    ) -> List[Alert]:
        """Record one prediction event."""

        if latency_ms is not None:
            self.recorder.record_prediction_time(latency_ms)
        metrics = self.snapshot_metrics()
        metrics["prediction"] = float(prediction)
        self.records.append(PerformanceRecord(datetime.now(timezone.utc), metrics, metadata or {}))
        return self.alert_manager.evaluate(metrics)

    def record_training(
        self,
        duration_ms: float,
        metrics: Dict[str, float] | None = None,
        metadata: Dict[str, Any] | None = None,
    ) -> List[Alert]:
        """Record one training event."""

        self.recorder.record_training_time(duration_ms)
        payload = self.snapshot_metrics()
        payload.update(metrics or {})
        self.records.append(PerformanceRecord(datetime.now(timezone.utc), payload, metadata or {"event": "training"}))
        return self.alert_manager.evaluate(payload)

    def set_reference_predictions(self, predictions: Sequence[float]) -> None:
        self.reference_predictions = [float(value) for value in predictions]

    def detect_prediction_drift(self, current_predictions: Sequence[float]) -> DriftResult:
        if not self.reference_predictions:
            self.set_reference_predictions(current_predictions)
            return DriftResult(False, 0.0, self.config.monitoring.drift_threshold, "prediction_psi")
        assert self.prediction_drift is not None
        return self.prediction_drift.detect(self.reference_predictions, current_predictions)

    def snapshot_metrics(self) -> Dict[str, float]:
        """Collect timing and resource metrics."""

        metrics = self.recorder.summary()
        resources = collect_resource_metrics()
        metrics.update(
            {
                "memory_mb": resources.memory_mb,
                "cpu_percent": resources.cpu_percent,
                "gpu_memory_mb": resources.gpu_memory_mb,
                "gpu_utilization_percent": resources.gpu_utilization_percent,
            }
        )
        return metrics

    def summary(self, limit: int | None = None) -> Dict[str, float]:
        """Summarize tracked metric records."""

        records = self.records[-int(limit) :] if limit is not None else self.records
        values: Dict[str, List[float]] = {}
        for record in records:
            for key, value in record.metrics.items():
                if isinstance(value, (int, float)):
                    values.setdefault(key, []).append(float(value))
        return {f"{key}_mean": float(np.mean(items)) for key, items in values.items() if items}
