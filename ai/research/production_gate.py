"""
ai/research/production_gate.py - Live trading readiness.

Live trading is enabled only after prolonged paper-trading performance
meets configurable risk thresholds. No shortcuts.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, Optional

from database.repositories.research_repository import ResearchRepository


@dataclass
class ProductionThresholds:
    min_paper_trades: int = 50
    min_paper_days: float = 14.0
    min_sharpe: float = 0.5
    max_drawdown: float = 0.20
    min_profit_factor: float = 1.2
    min_accuracy: float = 0.52
    min_resolved_predictions: int = 50


@dataclass
class ProductionGateResult:
    passed: bool
    live_enabled: bool
    symbol: str
    timeframe: str
    metrics: Dict[str, Any] = field(default_factory=dict)
    failures: list[str] = field(default_factory=list)
    thresholds: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "passed": self.passed,
            "live_enabled": self.live_enabled,
            "symbol": self.symbol,
            "timeframe": self.timeframe,
            "metrics": self.metrics,
            "failures": self.failures,
            "thresholds": self.thresholds,
        }


class ProductionReadinessGate:
    """Evaluate whether a symbol/timeframe may leave paper trading."""

    def __init__(
        self,
        research_repo: ResearchRepository,
        thresholds: ProductionThresholds | None = None,
    ):
        self.repo = research_repo
        self.thresholds = thresholds or ProductionThresholds()

    def evaluate(
        self,
        symbol: str,
        timeframe: str,
        *,
        model_id: int | None = None,
        enable_live_if_passed: bool = False,
        paper_started_at: str | None = None,
    ) -> ProductionGateResult:
        stats = self.repo.paper_trade_stats(symbol, timeframe)
        trades = stats.get("trades") or {}
        preds = stats.get("predictions") or {}

        n_trades = int(trades.get("n_trades") or 0)
        n_preds = int(preds.get("n_preds") or 0)
        accuracy = float(preds.get("accuracy") or 0.0)
        total_pnl = float(trades.get("total_pnl") or preds.get("pred_pnl") or 0.0)
        worst = float(trades.get("worst_pnl") or preds.get("min_drawdown") or 0.0)

        # Approximate sharpe / PF from paper prediction pnl stream
        sharpe, profit_factor, max_dd, paper_days = self._compute_series_metrics(
            symbol, timeframe, paper_started_at=paper_started_at
        )

        metrics = {
            "paper_trades": n_trades,
            "resolved_predictions": n_preds,
            "accuracy": accuracy,
            "total_pnl": total_pnl,
            "sharpe": sharpe,
            "profit_factor": profit_factor,
            "max_drawdown": max_dd,
            "paper_days": paper_days,
            "worst_pnl": worst,
        }
        thr = self.thresholds
        failures: list[str] = []
        if n_trades < thr.min_paper_trades and n_preds < thr.min_resolved_predictions:
            failures.append(
                f"insufficient_paper_sample(trades={n_trades}, preds={n_preds})"
            )
        if paper_days < thr.min_paper_days:
            failures.append(f"paper_days={paper_days:.2f}<{thr.min_paper_days}")
        if sharpe < thr.min_sharpe:
            failures.append(f"sharpe={sharpe:.3f}<{thr.min_sharpe}")
        if abs(max_dd) > thr.max_drawdown:
            failures.append(f"max_drawdown={abs(max_dd):.3f}>{thr.max_drawdown}")
        if profit_factor < thr.min_profit_factor:
            failures.append(f"profit_factor={profit_factor:.3f}<{thr.min_profit_factor}")
        if accuracy < thr.min_accuracy and n_preds >= thr.min_resolved_predictions:
            failures.append(f"accuracy={accuracy:.3f}<{thr.min_accuracy}")

        passed = len(failures) == 0
        live_enabled = False
        if passed and enable_live_if_passed and model_id is not None:
            self.repo.record_deployment(
                model_id=model_id,
                symbol=symbol,
                timeframe=timeframe,
                environment="live",
                status="active",
                reason="production_gate_passed",
                metrics=metrics,
            )
            live_enabled = True

        self.repo.upsert_production_gate(
            symbol=symbol,
            timeframe=timeframe,
            model_id=model_id,
            paper_trades=n_trades,
            paper_days=paper_days,
            sharpe=sharpe,
            max_drawdown=max_dd,
            profit_factor=profit_factor,
            passed=passed,
            live_enabled=live_enabled,
            thresholds={
                "min_paper_trades": thr.min_paper_trades,
                "min_paper_days": thr.min_paper_days,
                "min_sharpe": thr.min_sharpe,
                "max_drawdown": thr.max_drawdown,
                "min_profit_factor": thr.min_profit_factor,
                "min_accuracy": thr.min_accuracy,
                "min_resolved_predictions": thr.min_resolved_predictions,
            },
            details={"failures": failures, "metrics": metrics},
        )
        return ProductionGateResult(
            passed=passed,
            live_enabled=live_enabled,
            symbol=symbol.upper(),
            timeframe=timeframe.upper(),
            metrics=metrics,
            failures=failures,
            thresholds={
                "min_paper_trades": thr.min_paper_trades,
                "min_paper_days": thr.min_paper_days,
                "min_sharpe": thr.min_sharpe,
                "max_drawdown": thr.max_drawdown,
                "min_profit_factor": thr.min_profit_factor,
                "min_accuracy": thr.min_accuracy,
            },
        )

    def _compute_series_metrics(
        self,
        symbol: str,
        timeframe: str,
        *,
        paper_started_at: str | None,
    ) -> tuple[float, float, float, float]:
        rows = self.repo._fetch_all(
            """
            SELECT pnl, created_at, timestamp
            FROM research_predictions
            WHERE symbol=? AND timeframe=? AND resolved=1 AND pnl IS NOT NULL
            ORDER BY timestamp ASC
            """,
            (symbol.upper(), timeframe.upper()),
        )
        pnls = [float(r["pnl"]) for r in rows if r.get("pnl") is not None]
        if not pnls:
            return 0.0, 0.0, 0.0, 0.0
        arr = __import__("numpy").asarray(pnls, dtype=float)
        mean = float(arr.mean())
        std = float(arr.std(ddof=1)) if len(arr) > 1 else 0.0
        sharpe = (mean / std) if std > 1e-12 else 0.0
        gains = float(arr[arr > 0].sum())
        losses = float(-arr[arr < 0].sum())
        pf = (gains / losses) if losses > 1e-12 else (999.0 if gains > 0 else 0.0)
        equity = arr.cumsum()
        peak = __import__("numpy").maximum.accumulate(equity)
        dd = float(((equity - peak) / __import__("numpy").maximum(peak, 1e-9)).min()) if len(equity) else 0.0

        first = rows[0].get("timestamp") or rows[0].get("created_at") or paper_started_at
        last = rows[-1].get("timestamp") or rows[-1].get("created_at")
        paper_days = 0.0
        if first and last:
            try:
                t0 = datetime.fromisoformat(str(first).replace("Z", ""))
                t1 = datetime.fromisoformat(str(last).replace("Z", ""))
                paper_days = max(0.0, (t1 - t0).total_seconds() / 86400.0)
            except Exception:
                paper_days = float(len(rows)) / 24.0
        return sharpe, pf, abs(dd), paper_days
