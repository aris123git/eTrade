"""
ai/monitoring/live.py - Live trading monitor for PnL, win rate, Sharpe, drift.

Tracks paper and live sessions without requiring a broker connection.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Mapping, Optional, Sequence

import numpy as np

from ai.config.settings import AIConfig
from ai.evaluation.trading_metrics import sharpe as sharpe_ratio
from ai.monitoring.drift import PredictionDriftDetector
from ai.portfolio.manager import PortfolioManager

logger = logging.getLogger(__name__)


@dataclass
class LiveSnapshot:
    """Point-in-time live monitoring snapshot."""

    timestamp: datetime
    equity: float
    realized_pnl: float
    unrealized_pnl: float
    win_rate: float
    loss_rate: float
    sharpe: float
    drawdown: float
    open_positions: int
    closed_trades: int
    accuracy: float | None
    accuracy_ok: bool
    drift_score: float
    drifted: bool
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp.isoformat(),
            "equity": self.equity,
            "realized_pnl": self.realized_pnl,
            "unrealized_pnl": self.unrealized_pnl,
            "win_rate": self.win_rate,
            "loss_rate": self.loss_rate,
            "sharpe": self.sharpe,
            "drawdown": self.drawdown,
            "open_positions": self.open_positions,
            "closed_trades": self.closed_trades,
            "accuracy": self.accuracy,
            "accuracy_ok": self.accuracy_ok,
            "drift_score": self.drift_score,
            "drifted": self.drifted,
            "metadata": self.metadata,
        }


@dataclass
class LiveMonitor:
    """
    Real-time (or replay) monitoring for autonomous trading.

    Accuracy drift: flags when rolling prediction accuracy falls below
    ``min_accuracy`` (default 55%).
    """

    config: AIConfig = field(default_factory=AIConfig)
    portfolio: PortfolioManager | None = None
    min_accuracy: float = 0.55
    prediction_outcomes: List[int] = field(default_factory=list)  # 1=correct, 0=wrong
    reference_predictions: List[float] = field(default_factory=list)
    current_predictions: List[float] = field(default_factory=list)
    snapshots: List[LiveSnapshot] = field(default_factory=list)
    _shutdown: bool = False

    def __post_init__(self) -> None:
        self._drift = PredictionDriftDetector(
            threshold=float(self.config.monitoring.drift_threshold),
        )
        logger.info(
            "LiveMonitor ready min_accuracy=%.0f%% drift_threshold=%.3f",
            self.min_accuracy * 100.0,
            self.config.monitoring.drift_threshold,
        )

    def request_shutdown(self) -> None:
        self._shutdown = True
        logger.warning("LiveMonitor shutdown requested")

    def record_prediction(
        self,
        *,
        prediction: float,
        realized_direction: float | None = None,
        correct: bool | None = None,
    ) -> None:
        """Record a prediction for accuracy / drift tracking."""

        if self._shutdown:
            return
        self.current_predictions.append(float(prediction))
        if correct is not None:
            self.prediction_outcomes.append(1 if correct else 0)
        elif realized_direction is not None:
            pred_dir = 1.0 if prediction > 0 else (-1.0 if prediction < 0 else 0.0)
            real_dir = 1.0 if realized_direction > 0 else (-1.0 if realized_direction < 0 else 0.0)
            self.prediction_outcomes.append(1 if pred_dir != 0 and pred_dir == real_dir else 0)

    def set_reference_predictions(self, values: Sequence[float]) -> None:
        self.reference_predictions = [float(v) for v in values]

    def accuracy(self) -> float | None:
        if not self.prediction_outcomes:
            return None
        return float(np.mean(self.prediction_outcomes))

    def snapshot(self, *, timestamp: datetime | None = None, metadata: Mapping[str, Any] | None = None) -> LiveSnapshot:
        """Compute current monitoring metrics."""

        ts = timestamp or datetime.now(timezone.utc)
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)

        portfolio = self.portfolio
        metrics = portfolio.performance_metrics() if portfolio is not None else {}
        pnls = [t.pnl for t in (portfolio.closed_trades if portfolio else [])]
        returns = list(portfolio.returns_history) if portfolio is not None else []
        if not returns and pnls:
            base = max(float(metrics.get("equity", 100_000.0)), 1e-12)
            returns = [p / base for p in pnls]

        accuracy = self.accuracy()
        accuracy_ok = True if accuracy is None else accuracy >= self.min_accuracy

        drift_score = 0.0
        drifted = False
        if self.reference_predictions and self.current_predictions:
            n = min(len(self.reference_predictions), len(self.current_predictions))
            if n >= 10:
                result = self._drift.detect(
                    np.asarray(self.reference_predictions[-n:], dtype=float),
                    np.asarray(self.current_predictions[-n:], dtype=float),
                )
                drift_score = float(result.score)
                drifted = bool(result.drifted)

        snap = LiveSnapshot(
            timestamp=ts,
            equity=float(metrics.get("equity", 0.0)),
            realized_pnl=float(metrics.get("realized_pnl", 0.0)),
            unrealized_pnl=float(metrics.get("unrealized_pnl", 0.0)),
            win_rate=float(metrics.get("win_rate", 0.0)),
            loss_rate=float(metrics.get("loss_rate", 1.0 - metrics.get("win_rate", 0.0))),
            sharpe=float(sharpe_ratio(returns)) if len(returns) >= 2 else 0.0,
            drawdown=float(metrics.get("drawdown", 0.0)),
            open_positions=int(metrics.get("open_positions", 0)),
            closed_trades=int(metrics.get("closed_trades", 0)),
            accuracy=accuracy,
            accuracy_ok=accuracy_ok,
            drift_score=drift_score,
            drifted=drifted,
            metadata=dict(metadata or {}),
        )
        self.snapshots.append(snap)
        if not accuracy_ok:
            logger.warning(
                "accuracy drifted below %.0f%%: current=%.1f%%",
                self.min_accuracy * 100.0,
                (accuracy or 0.0) * 100.0,
            )
        logger.info(
            "live snap equity=%.2f pnl=%.2f win=%.1f%% sharpe=%.2f acc=%s",
            snap.equity,
            snap.realized_pnl,
            snap.win_rate * 100.0,
            snap.sharpe,
            f"{accuracy * 100.0:.1f}%" if accuracy is not None else "n/a",
        )
        return snap

    def summary(self) -> Dict[str, Any]:
        latest = self.snapshots[-1].to_dict() if self.snapshots else {}
        return {
            "snapshots": len(self.snapshots),
            "latest": latest,
            "accuracy": self.accuracy(),
            "min_accuracy": self.min_accuracy,
            "prediction_count": len(self.prediction_outcomes),
        }


def create_live_monitor(
    config: AIConfig | None = None,
    portfolio: PortfolioManager | None = None,
    *,
    min_accuracy: float = 0.55,
) -> LiveMonitor:
    return LiveMonitor(
        config=config or AIConfig(),
        portfolio=portfolio,
        min_accuracy=min_accuracy,
    )
