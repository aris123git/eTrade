"""
ai/research/validation_gate.py - Strict validation before paper trading.

Every discovered edge must pass:
  train → validation → walk-forward → out-of-sample → Monte-Carlo →
  realistic spread / commission / slippage.

Failures never reach paper trading.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from ai.config.settings import AIConfig
from ai.datasets.schema import DatasetBundle
from ai.evaluation.backtest import BacktestEngine, BacktestSignal, Candle
from ai.evaluation.monte_carlo import monte_carlo_reshuffle
from ai.models.base import BaseModel, flatten_target
from ai.training.validation import default_metrics, walk_forward_validation

logger = logging.getLogger(__name__)


@dataclass
class ValidationThresholds:
    min_train_score: float = 0.52
    min_val_score: float = 0.52
    min_oos_score: float = 0.50
    min_walk_forward_score: float = 0.50
    min_wf_folds_passed: int = 3
    max_mc_ruin_prob: float = 0.25  # P(final equity < initial)
    min_mc_median_return: float = -0.05
    min_backtest_trades: int = 5
    max_drawdown: float = 0.35
    min_profit_factor: float = 1.0


@dataclass
class StageOutcome:
    stage: str
    passed: bool
    metrics: Dict[str, Any] = field(default_factory=dict)
    details: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "stage": self.stage,
            "passed": self.passed,
            "metrics": self.metrics,
            "details": self.details,
        }


@dataclass
class StrictValidationReport:
    passed: bool
    stages: List[StageOutcome] = field(default_factory=list)
    reason: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "passed": self.passed,
            "reason": self.reason,
            "stages": [s.to_dict() for s in self.stages],
        }


class StrictValidationGate:
    """Hard gate between research champions and paper trading."""

    def __init__(
        self,
        config: AIConfig,
        thresholds: ValidationThresholds | None = None,
    ):
        self.config = config
        self.thresholds = thresholds or ValidationThresholds()

    def validate(
        self,
        model: BaseModel,
        bundle: DatasetBundle,
        *,
        candles: Sequence[Dict[str, Any]] | None = None,
        symbol: str = "",
        timeframe: str = "",
    ) -> StrictValidationReport:
        stages: List[StageOutcome] = []

        train_stage = self._score_split(model, bundle.X_train, bundle.y_train, "train")
        stages.append(train_stage)
        if not train_stage.passed:
            return StrictValidationReport(False, stages, reason="train_failed")

        val_stage = self._score_split(model, bundle.X_val, bundle.y_val, "validation")
        stages.append(val_stage)
        if not val_stage.passed:
            return StrictValidationReport(False, stages, reason="validation_failed")

        oos_stage = self._score_split(model, bundle.X_test, bundle.y_test, "out_of_sample")
        stages.append(oos_stage)
        if not oos_stage.passed:
            return StrictValidationReport(False, stages, reason="oos_failed")

        wf_stage = self._walk_forward(model, bundle)
        stages.append(wf_stage)
        if not wf_stage.passed:
            return StrictValidationReport(False, stages, reason="walk_forward_failed")

        bt_stage = self._cost_backtest(model, bundle, candles=candles, symbol=symbol, timeframe=timeframe)
        stages.append(bt_stage)
        if not bt_stage.passed:
            return StrictValidationReport(False, stages, reason="cost_backtest_failed")

        mc_stage = self._monte_carlo(bt_stage.metrics.get("trade_returns") or [])
        stages.append(mc_stage)
        if not mc_stage.passed:
            return StrictValidationReport(False, stages, reason="monte_carlo_failed")

        return StrictValidationReport(True, stages, reason="all_stages_passed")

    def _score_split(
        self,
        model: BaseModel,
        X: Any,
        y: Any,
        stage: str,
    ) -> StageOutcome:
        if X is None or len(X) == 0:
            return StageOutcome(stage, False, details={"error": "empty_split"})
        y_true = flatten_target(y)
        y_pred = np.asarray(model.predict(X)).reshape(-1)
        metrics = default_metrics(y_true, y_pred, task=model.task)
        score = float(metrics.get("f1") or metrics.get("accuracy") or metrics.get("r2") or 0.0)
        metrics["score"] = score
        minimum = {
            "train": self.thresholds.min_train_score,
            "validation": self.thresholds.min_val_score,
            "out_of_sample": self.thresholds.min_oos_score,
        }.get(stage, self.thresholds.min_val_score)
        return StageOutcome(stage, score >= minimum, metrics=metrics, details={"min_required": minimum})

    def _walk_forward(self, model: BaseModel, bundle: DatasetBundle) -> StageOutcome:
        X = np.vstack([bundle.X_train, bundle.X_val, bundle.X_test])
        y = np.concatenate(
            [
                flatten_target(bundle.y_train),
                flatten_target(bundle.y_val),
                flatten_target(bundle.y_test),
            ]
        )
        if len(X) < 100:
            return StageOutcome("walk_forward", False, details={"error": "insufficient_rows"})

        try:
            fold_metrics = walk_forward_validation(
                model,
                X,
                y,
                folds=max(3, int(self.config.datasets.walk_forward_folds)),
                embargo=int(self.config.datasets.walk_forward_embargo),
            )
        except Exception as exc:
            return StageOutcome(
                "walk_forward",
                False,
                details={"error": f"{exc.__class__.__name__}: {exc}"},
            )

        from ai.training.validation import summarize_scores

        summary = summarize_scores(fold_metrics)
        scores = [
            float(m.get("f1") or m.get("accuracy") or m.get("r2") or 0.0)
            for m in fold_metrics
        ]
        mean_score = float(
            summary.get("f1")
            or summary.get("accuracy")
            or summary.get("r2")
            or (np.mean(scores) if scores else 0.0)
        )
        folds_passed = sum(1 for s in scores if float(s) >= self.thresholds.min_walk_forward_score)
        passed = (
            mean_score >= self.thresholds.min_walk_forward_score
            and folds_passed >= min(self.thresholds.min_wf_folds_passed, max(1, len(scores)))
        )
        return StageOutcome(
            "walk_forward",
            passed,
            metrics={
                "mean_score": mean_score,
                "folds_passed": folds_passed,
                "n_folds": len(scores),
                **{f"mean_{k}": v for k, v in summary.items()},
            },
            details={"fold_metrics": fold_metrics},
        )

    def _cost_backtest(
        self,
        model: BaseModel,
        bundle: DatasetBundle,
        *,
        candles: Sequence[Dict[str, Any]] | None,
        symbol: str,
        timeframe: str,
    ) -> StageOutcome:
        if not candles or len(candles) < 40:
            return StageOutcome("cost_backtest", False, details={"error": "insufficient_candles"})

        # Generate signals on test portion using model predictions
        X_test = bundle.X_test
        if len(X_test) == 0:
            return StageOutcome("cost_backtest", False, details={"error": "empty_test"})
        preds = np.asarray(model.predict(X_test)).reshape(-1)
        n = min(len(preds), len(candles))
        candle_slice = list(candles)[-n:]
        signals: List[BacktestSignal] = []
        for i, (pred, candle) in enumerate(zip(preds, candle_slice)):
            if i % max(1, self.config.labels.horizon) != 0:
                continue
            side = "buy" if float(pred) > 0 else "sell" if float(pred) < 0 else "close"
            if side == "close":
                continue
            signals.append(
                BacktestSignal(
                    symbol=symbol or str(candle.get("symbol") or ""),
                    timestamp=candle["timestamp"],
                    side=side,
                    quantity=0.1,
                    timeframe=timeframe or self.config.primary_timeframe,
                )
            )
        if signals:
            signals.append(
                BacktestSignal(
                    symbol=signals[0].symbol,
                    timestamp=candle_slice[-1]["timestamp"],
                    side="close",
                    quantity=0.1,
                    timeframe=timeframe or self.config.primary_timeframe,
                )
            )

        bt_candles = [
            Candle(
                symbol=symbol or str(c.get("symbol") or ""),
                timeframe=timeframe or self.config.primary_timeframe,
                timestamp=c["timestamp"],
                open=float(c["open"]),
                high=float(c["high"]),
                low=float(c["low"]),
                close=float(c["close"]),
                volume=float(c.get("volume") or 0.0),
            )
            for c in candle_slice
        ]
        engine = BacktestEngine(
            config=self.config,
            spread_points=float(getattr(self.config.execution, "slippage_points", 0.5) or 0.5),
            commission_per_lot=float(self.config.execution.commission_per_lot),
            slippage_points=float(self.config.execution.slippage_points),
        )
        result = engine.run(signals=signals, candles=bt_candles)
        metrics = dict(result.metrics or {})
        trade_returns = [float(t.return_pct) for t in result.trades]
        metrics["trade_returns"] = trade_returns
        metrics["n_trades"] = len(result.trades)
        max_dd = abs(float(metrics.get("max_drawdown") or 0.0))
        pf = float(metrics.get("profit_factor") or 0.0)
        passed = (
            len(result.trades) >= self.thresholds.min_backtest_trades
            and max_dd <= self.thresholds.max_drawdown
            and (pf >= self.thresholds.min_profit_factor or len(result.trades) == 0)
        )
        # If we have trades, require PF; if somehow zero trades, fail via min_trades
        if len(result.trades) > 0 and pf < self.thresholds.min_profit_factor:
            passed = False
        return StageOutcome(
            "cost_backtest",
            passed,
            metrics=metrics,
            details={
                "spread_points": engine.spread_points,
                "commission_per_lot": engine.commission_per_lot,
                "slippage_points": engine.slippage_points,
            },
        )

    def _monte_carlo(self, trade_returns: Sequence[float]) -> StageOutcome:
        if len(trade_returns) < self.thresholds.min_backtest_trades:
            return StageOutcome(
                "monte_carlo",
                False,
                details={"error": "insufficient_trades", "n": len(trade_returns)},
            )
        mc = monte_carlo_reshuffle(
            trade_returns,
            n_simulations=500,
            confidence=0.95,
            initial_equity=1.0,
            replacement=True,
            random_seed=self.config.random_seed,
        )
        finals = np.asarray(mc.final_equity, dtype=float)
        ruin_prob = float(np.mean(finals < 1.0))
        median_return = float(np.median(finals) - 1.0)
        passed = (
            ruin_prob <= self.thresholds.max_mc_ruin_prob
            and median_return >= self.thresholds.min_mc_median_return
        )
        return StageOutcome(
            "monte_carlo",
            passed,
            metrics={
                "ruin_prob": ruin_prob,
                "median_return": median_return,
                "mean_final_equity": float(np.mean(finals)),
                "p5_final_equity": float(np.quantile(finals, 0.05)),
            },
            details={"n_simulations": int(len(finals)), "confidence": mc.confidence},
        )
