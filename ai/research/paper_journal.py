"""
ai/research/paper_journal.py - Continuous paper trading with full audit trail.

Every prediction stores timestamp, prediction, confidence, actual outcome,
PnL, drawdown, explanation, and feature importance. Mistakes become training data.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from ai.config.settings import AIConfig
from ai.evaluation.importance import feature_importance
from ai.execution.executor import Order, OrderExecutor, create_order_executor
from ai.services.pipeline import AIPipeline
from ai.utils.types import OrderSide, OrderType
from database.repositories.research_repository import ResearchRepository

logger = logging.getLogger(__name__)


def _now_iso() -> str:
    return datetime.now(timezone.utc).replace(tzinfo=None).isoformat(timespec="seconds")


@dataclass
class PaperCycleResult:
    predictions_recorded: int = 0
    predictions_resolved: int = 0
    trades_opened: int = 0
    trades_closed: int = 0
    details: Dict[str, Any] | None = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "predictions_recorded": self.predictions_recorded,
            "predictions_resolved": self.predictions_resolved,
            "trades_opened": self.trades_opened,
            "trades_closed": self.trades_closed,
            "details": self.details or {},
        }


class PaperTradingJournal:
    """Persistent paper-trading journal backed by the research database."""

    def __init__(
        self,
        config: AIConfig,
        research_repo: ResearchRepository,
        pipeline: AIPipeline,
        *,
        executor: OrderExecutor | None = None,
        model_id: int | None = None,
        equity: float = 10_000.0,
    ):
        self.config = config
        self.repo = research_repo
        self.pipeline = pipeline
        self.executor = executor or create_order_executor(config=config, mode="paper")
        self.model_id = model_id
        self.equity = float(equity)
        self.peak_equity = float(equity)

    def run(
        self,
        *,
        symbol: str,
        timeframe: str | None = None,
        candles: Sequence[Dict[str, Any]] | None = None,
    ) -> PaperCycleResult:
        tf = timeframe or self.config.primary_timeframe
        active_candles = list(candles or [])
        if not active_candles:
            active_candles = self.pipeline.load_candles(
                symbol=symbol,
                timeframe=tf,
                limit=min(500, int(getattr(self.config.research, "candle_limit", 2000))),
                auto_download=False,
            )
        if len(active_candles) < 30:
            return PaperCycleResult(details={"status": "skipped", "reason": "insufficient_candles"})

        # Resolve prior unresolved predictions against latest closes
        resolved = self.resolve_pending(symbol=symbol, candles=active_candles)

        prediction_payload = self.pipeline.run_prediction(
            candles=active_candles,
            equity=self.equity,
            auto_download=False,
        )
        pred = prediction_payload.get("prediction") or {}
        signal = prediction_payload.get("signal") or {}
        side = str(signal.get("signal") or "HOLD").upper()

        importance = {}
        explanation = "paper_prediction"
        if self.pipeline.model is not None:
            try:
                frame = self.pipeline.build_features(active_candles)
                importance = feature_importance(
                    getattr(self.pipeline.model, "estimator", self.pipeline.model),
                    feature_names=frame.feature_names,
                )
                top = sorted(importance.items(), key=lambda kv: abs(kv[1]), reverse=True)[:8]
                explanation = "top_features=" + ",".join(f"{k}:{v:.3f}" for k, v in top)
            except Exception as exc:
                explanation = f"importance_unavailable:{exc.__class__.__name__}"

        ts = pred.get("timestamp") or active_candles[-1]["timestamp"]
        if hasattr(ts, "isoformat"):
            ts = ts.isoformat(timespec="seconds")

        self.repo.record_prediction(
            model_id=self.model_id,
            symbol=symbol,
            timeframe=tf,
            timestamp=str(ts),
            prediction=float(pred["prediction"]) if pred.get("prediction") is not None else None,
            signal=side,
            confidence=float(pred.get("confidence") or signal.get("confidence") or 0.0),
            explanation=explanation,
            feature_importance=importance,
            features_snapshot={"n_candles": len(active_candles)},
            metadata={"risk": signal.get("risk"), "model_version": pred.get("model_version")},
        )

        trades_opened = 0
        trades_closed = 0
        if side in {"BUY", "SELL"}:
            qty = float((signal.get("risk") or {}).get("lot_size") or self.config.risk.default_lot_size)
            price = float(active_candles[-1]["close"])
            order = Order(
                symbol=symbol,
                side=OrderSide.BUY if side == "BUY" else OrderSide.SELL,
                quantity=qty,
                order_type=OrderType.MARKET,
                price=price,
            )
            report = self.executor.submit_order(order, market_price=price)
            fill_price = report.fills[0].price if report.fills else price
            self.repo.record_paper_trade(
                model_id=self.model_id,
                symbol=symbol,
                timeframe=tf,
                side=side,
                quantity=qty,
                entry_time=str(ts),
                entry_price=float(fill_price),
                status="open",
                metadata={
                    "executor_status": report.status.value if hasattr(report.status, "value") else str(report.status),
                    "confidence": pred.get("confidence"),
                },
            )
            trades_opened = 1

        return PaperCycleResult(
            predictions_recorded=1,
            predictions_resolved=resolved,
            trades_opened=trades_opened,
            trades_closed=trades_closed,
            details={"signal": side, "equity": self.equity},
        )

    def resolve_pending(
        self,
        *,
        symbol: str,
        candles: Sequence[Dict[str, Any]],
        horizon: int | None = None,
    ) -> int:
        """Mark unresolved predictions with realized forward returns / PnL."""

        if len(candles) < 2:
            return 0
        horizon = int(horizon or self.config.labels.horizon)
        closes = { _ts_key(c["timestamp"]): float(c["close"]) for c in candles }
        ordered = sorted(closes.items(), key=lambda kv: kv[0])
        ts_list = [k for k, _ in ordered]
        close_list = [v for _, v in ordered]
        index = {ts: i for i, ts in enumerate(ts_list)}

        unresolved = self.repo.list_unresolved_predictions(symbol=symbol)
        resolved = 0
        for row in unresolved:
            key = _ts_key(row["timestamp"])
            if key not in index:
                continue
            i = index[key]
            j = i + horizon
            if j >= len(close_list):
                continue
            entry = close_list[i]
            exit_px = close_list[j]
            if entry <= 0:
                continue
            ret = (exit_px - entry) / entry
            signal = str(row.get("signal") or "").upper()
            direction = 1.0 if signal == "BUY" else -1.0 if signal == "SELL" else np.sign(float(row.get("prediction") or 0.0))
            pnl = float(direction * ret * self.equity * float(self.config.risk.risk_per_trade))
            self.equity += pnl
            self.peak_equity = max(self.peak_equity, self.equity)
            drawdown = float((self.equity - self.peak_equity) / max(self.peak_equity, 1e-9))
            self.repo.resolve_prediction(
                int(row["prediction_id"]),
                actual_outcome=float(ret * direction),
                pnl=pnl,
                drawdown=drawdown,
            )
            # Close matching open paper trade if present
            self.repo._execute(
                """
                UPDATE research_paper_trades
                SET exit_time=?, exit_price=?, pnl=?, return_pct=?, drawdown=?, status='closed'
                WHERE symbol=? AND status='open' AND entry_time=?
                """,
                (
                    ts_list[j],
                    exit_px,
                    pnl,
                    float(ret * direction),
                    drawdown,
                    symbol.upper(),
                    str(row["timestamp"]),
                ),
            )
            self.repo._commit()
            resolved += 1
        return resolved


def _ts_key(value: Any) -> str:
    if hasattr(value, "isoformat"):
        return value.replace(tzinfo=None).isoformat(timespec="seconds") if hasattr(value, "replace") else value.isoformat(timespec="seconds")
    text = str(value).replace("Z", "")
    try:
        return datetime.fromisoformat(text).isoformat(timespec="seconds")
    except Exception:
        return text
