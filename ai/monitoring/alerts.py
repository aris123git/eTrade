"""
ai/monitoring/alerts.py - Alert management.

VERSION: 1.0.0
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Callable, Dict, List


@dataclass(frozen=True)
class Alert:
    """Monitoring alert event."""

    name: str
    severity: str
    message: str
    value: float
    threshold: float
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: Dict[str, str | float | bool] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, object]:
        return {
            "name": self.name,
            "severity": self.severity,
            "message": self.message,
            "value": self.value,
            "threshold": self.threshold,
            "timestamp": self.timestamp.isoformat(),
            "metadata": self.metadata,
        }


@dataclass(frozen=True)
class AlertRule:
    """Threshold alert rule."""

    name: str
    metric: str
    threshold: float
    comparison: str = ">"
    severity: str = "warning"
    message: str | None = None

    def evaluate(self, metrics: Dict[str, float]) -> Alert | None:
        if self.metric not in metrics:
            return None
        value = float(metrics[self.metric])
        triggered = _compare(value, float(self.threshold), self.comparison)
        if not triggered:
            return None
        message = self.message or f"{self.metric} {self.comparison} {self.threshold}"
        return Alert(
            name=self.name,
            severity=self.severity,
            message=message,
            value=value,
            threshold=float(self.threshold),
            metadata={"metric": self.metric},
        )


@dataclass
class AlertManager:
    """Evaluate alert rules and dispatch alert events."""

    rules: List[AlertRule] = field(default_factory=list)
    handlers: List[Callable[[Alert], None]] = field(default_factory=list)
    history: List[Alert] = field(default_factory=list)

    def add_rule(self, rule: AlertRule) -> None:
        self.rules.append(rule)

    def add_threshold(
        self,
        name: str,
        metric: str,
        threshold: float,
        comparison: str = ">",
        severity: str = "warning",
    ) -> None:
        self.add_rule(AlertRule(name=name, metric=metric, threshold=threshold, comparison=comparison, severity=severity))

    def add_handler(self, handler: Callable[[Alert], None]) -> None:
        self.handlers.append(handler)

    def evaluate(self, metrics: Dict[str, float]) -> List[Alert]:
        alerts: List[Alert] = []
        for rule in self.rules:
            alert = rule.evaluate(metrics)
            if alert is None:
                continue
            self.history.append(alert)
            alerts.append(alert)
            for handler in self.handlers:
                handler(alert)
        return alerts

    def latest(self, limit: int = 50) -> List[Alert]:
        return self.history[-max(0, int(limit)) :]


def _compare(value: float, threshold: float, comparison: str) -> bool:
    if comparison == ">":
        return value > threshold
    if comparison == ">=":
        return value >= threshold
    if comparison == "<":
        return value < threshold
    if comparison == "<=":
        return value <= threshold
    if comparison == "==":
        return value == threshold
    raise ValueError(f"Unsupported comparison: {comparison!r}")
