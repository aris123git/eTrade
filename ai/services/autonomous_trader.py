"""
ai/services/autonomous_trader.py - Phase 4 autonomous trading system orchestrator.

Wires model training, signal generation, execution, portfolio, risk, live
monitoring, auto-retraining, and explainability into a single paper/live loop.
"""

from __future__ import annotations

import logging
import signal as signal_module
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence

import numpy as np

from ai.config.settings import AIConfig
from ai.data.candle_adapter import CandleRepositoryAdapter
from ai.evaluation.trading_metrics import sharpe as sharpe_ratio
from ai.execution.executor import OrderExecutor, OrderStatus, create_order_executor
from ai.explainability.explainer import ModelExplainer, create_model_explainer
from ai.models.trainer import ModelTrainer, create_model_trainer
from ai.monitoring.live import LiveMonitor, create_live_monitor
from ai.portfolio.manager import PortfolioManager, create_portfolio_manager
from ai.risk.manager import RiskManager, create_risk_manager
from ai.services.pipeline import AIPipeline
from ai.signals.engine import TradeSignal
from ai.signals.generator import SignalGenerator, create_signal_generator
from ai.training.scheduler import TrainingScheduler, create_training_scheduler
from ai.utils.types import CandleDict, SignalType

logger = logging.getLogger(__name__)


@dataclass
class AutonomousTrader:
    """
    End-to-end autonomous trader for paper and live modes.

    All live broker calls go through OrderExecutor; paper mode needs no broker.
    """

    config: AIConfig = field(default_factory=AIConfig)
    mode: str = "paper"
    candle_repository: Any = None
    candle_source: Any = None
    initial_equity: float = 100_000.0
    broker: str = "default"

    trainer: ModelTrainer | None = None
    signals: SignalGenerator | None = None
    executor: OrderExecutor | None = None
    portfolio: PortfolioManager | None = None
    risk: RiskManager | None = None
    monitor: LiveMonitor | None = None
    scheduler: TrainingScheduler | None = None
    pipeline: AIPipeline | None = None
    explainer: ModelExplainer | None = None

    _shutdown: bool = False
    _prev_handler: Any = None
    session_log: List[Dict[str, Any]] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.mode = str(self.mode or "paper").lower().strip()
        if Path(self.config.storage.root_dir) == Path("ai_artifacts"):
            self.config.storage.root_dir = Path("ai/artifacts")
        self.config.ensure_directories()

        if self.candle_source is None and self.candle_repository is not None:
            self.candle_source = CandleRepositoryAdapter(self.candle_repository)

        self.pipeline = self.pipeline or AIPipeline(
            config=self.config,
            candle_source=self.candle_source,
        )
        self.trainer = self.trainer or create_model_trainer(
            self.config,
            candle_repository=self.candle_repository,
        )
        self.trainer.pipeline = self.pipeline
        self.signals = self.signals or create_signal_generator(self.config)
        self.executor = self.executor or create_order_executor(self.config, mode=self.mode)
        self.portfolio = self.portfolio or create_portfolio_manager(
            self.config,
            cash=self.initial_equity,
        )
        self.risk = self.risk or create_risk_manager(self.config, equity=self.initial_equity)
        self.monitor = self.monitor or create_live_monitor(self.config, self.portfolio)
        self.scheduler = self.scheduler or create_training_scheduler(
            self.config,
            candle_repository=self.candle_repository,
            trainer=self.trainer,
        )
        logger.info("AutonomousTrader ready mode=%s equity=%.2f", self.mode, self.initial_equity)

    def install_signal_handlers(self) -> None:
        """Install SIGINT/SIGTERM graceful shutdown hooks."""

        def _handler(signum: int, _frame: Any) -> None:
            logger.warning("received signal %s — shutting down", signum)
            self.request_shutdown()

        try:
            self._prev_handler = signal_module.getsignal(signal_module.SIGINT)
            signal_module.signal(signal_module.SIGINT, _handler)
            signal_module.signal(signal_module.SIGTERM, _handler)
        except Exception:
            logger.debug("signal handlers unavailable in this environment")

    def request_shutdown(self) -> None:
        self._shutdown = True
        if self.executor:
            self.executor.request_shutdown()
        if self.portfolio:
            self.portfolio.request_shutdown()
        if self.monitor:
            self.monitor.request_shutdown()
        if self.scheduler:
            self.scheduler.request_shutdown()
        logger.warning("AutonomousTrader shutdown complete")

    def train(
        self,
        *,
        symbol: str,
        model_type: str = "random_forest",
        candles: Sequence[CandleDict] | None = None,
        limit: int = 5000,
    ) -> Dict[str, Any]:
        assert self.trainer is not None
        result = self.trainer.train(
            symbol=symbol,
            model_type=model_type,
            candles=candles,
            limit=limit,
            register=True,
        )
        # Keep pipeline model in sync for predict / explain.
        if self.pipeline and self.pipeline.registry and result.get("registered"):
            try:
                loaded = self.pipeline.registry.load(result["registered"]["name"])
                self.pipeline.model = loaded.get("model")
                self.pipeline.model_version = loaded.get("version")
            except Exception:
                logger.exception("failed to load registered model into pipeline")
        return result

    def explain_latest(
        self,
        candles: Sequence[CandleDict],
        *,
        top_background: int = 64,
    ) -> Dict[str, Any]:
        assert self.pipeline is not None
        if self.pipeline.model is None:
            raise RuntimeError("No trained model available for explainability")
        frame = self.pipeline.build_features(list(candles))
        matrix = np.asarray(frame.matrix, dtype=float)
        if len(matrix) < 2:
            raise ValueError("Need more feature rows for explanation")
        background = matrix[max(0, len(matrix) - top_background - 1) : -1]
        self.explainer = create_model_explainer(
            self.pipeline.model,
            background,
            feature_names=frame.feature_names,
        )
        report = self.explainer.explain(matrix[-1:])
        return report.to_dict()

    def run_paper_week(
        self,
        candles: Sequence[CandleDict],
        *,
        symbol: str,
        model_type: str = "random_forest",
        train_bars: int = 1500,
        week_bars: int = 168,  # ~1 week of H1
        correlations: Mapping[Any, float] | None = None,
    ) -> Dict[str, Any]:
        """
        Train on history then paper-trade approximately one week of bars.

        ``week_bars`` defaults to 168 H1 candles (7×24). For M15 use 672.
        This is a historical replay — not a real-time 7-day sleep.
        """

        series = list(candles)
        if len(series) < train_bars + 50:
            raise ValueError(
                f"Need >= {train_bars + 50} candles for paper week, got {len(series)}"
            )
        paper_n = min(week_bars, max(1, len(series) - train_bars))
        train_slice = series[-(train_bars + paper_n) : -paper_n]
        paper_slice = series[-paper_n:]

        logger.info(
            "paper week start symbol=%s train=%s paper=%s model=%s",
            symbol,
            len(train_slice),
            len(paper_slice),
            model_type,
        )
        train_result = self.train(symbol=symbol, model_type=model_type, candles=train_slice)

        # Seed monitor reference predictions from a short hold-out of train features.
        assert self.pipeline is not None and self.monitor is not None
        try:
            frame = self.pipeline.build_features(train_slice)
            preds = np.asarray(self.pipeline.model.predict(frame.matrix[-100:]), dtype=float).reshape(-1)
            self.monitor.set_reference_predictions(preds.tolist())
        except Exception:
            logger.exception("failed to seed reference predictions")

        explain_report: Dict[str, Any] = {}
        try:
            explain_report = self.explain_latest(train_slice)
        except Exception as exc:
            logger.exception("explainability failed")
            explain_report = {"error": f"{exc.__class__.__name__}: {exc}"}

        lookback = train_slice[:]
        for i, bar in enumerate(paper_slice):
            if self._shutdown:
                logger.warning("paper week interrupted at bar %s", i)
                break
            lookback.append(bar)
            self._step(
                lookback=lookback,
                bar=bar,
                symbol=symbol,
                correlations=correlations or {},
            )
            if i % 24 == 0 or i == len(paper_slice) - 1:
                self.monitor.snapshot(
                    timestamp=_as_utc(bar.get("timestamp")),
                    metadata={"bar_index": i, "phase": "paper_week"},
                )

        # Flatten remaining positions at last price for clean week PnL.
        assert self.portfolio is not None and self.executor is not None
        last_price = float(paper_slice[-1]["close"])
        for position in list(self.portfolio.positions(symbol=symbol)):
            self.portfolio.close_position(
                position.position_id,
                price=last_price,
                closed_at=_as_utc(paper_slice[-1].get("timestamp")),
                metadata={"reason": "paper_week_end"},
            )

        final = self.monitor.snapshot(
            timestamp=_as_utc(paper_slice[-1].get("timestamp")),
            metadata={"phase": "paper_week_final"},
        )
        metrics = self.portfolio.performance_metrics()
        summary = {
            "mode": self.mode,
            "symbol": symbol.upper(),
            "model_type": model_type,
            "train": train_result,
            "paper_bars": paper_n,
            "paper_days_approx": round(paper_n / 24.0, 2) if self.config.primary_timeframe.upper().startswith("H") else round(paper_n / 96.0, 2),
            "metrics": metrics,
            "live": final.to_dict(),
            "explainability": explain_report,
            "risk": {
                "circuit_breaker_tripped": bool(self.risk.circuit_breaker_tripped) if self.risk else False,
                "drawdown": metrics.get("drawdown"),
            },
            "session_events": len(self.session_log),
            "sharpe": sharpe_ratio(self.portfolio.returns_history) if len(self.portfolio.returns_history) >= 2 else 0.0,
        }
        logger.info(
            "paper week done equity=%.2f trades=%s win=%.1f%% sharpe=%.2f",
            metrics.get("equity", 0.0),
            int(metrics.get("closed_trades", 0)),
            float(metrics.get("win_rate", 0.0)) * 100.0,
            summary["sharpe"],
        )
        return summary

    def _step(
        self,
        *,
        lookback: Sequence[CandleDict],
        bar: CandleDict,
        symbol: str,
        correlations: Mapping[Any, float],
    ) -> None:
        assert self.pipeline and self.signals and self.risk and self.executor and self.portfolio and self.monitor
        price = float(bar["close"])
        ts = _as_utc(bar.get("timestamp"))
        self.portfolio.update_prices({symbol: price})
        self.risk.update_equity(self.portfolio.total_equity(), timestamp=ts)

        # SL/TP automation
        exit_reports = self.executor.manage_exits(
            self.portfolio.positions(symbol=symbol),
            prices={symbol: price},
            timestamp=ts,
        )
        for report in exit_reports:
            if report.status in {OrderStatus.FILLED, OrderStatus.PARTIALLY_FILLED}:
                for fill in report.fills:
                    self.portfolio.apply_fill(fill, broker=self.broker)
                self.risk.update_equity(self.portfolio.total_equity(), timestamp=ts)

        if self._shutdown or self.risk.circuit_breaker_tripped:
            return

        try:
            prediction = self.pipeline.predict(candles=list(lookback), symbol=symbol)
        except Exception:
            logger.exception("prediction failed at %s", ts)
            return

        atr = _estimate_atr(lookback, period=14)
        signal = self.signals.generate(
            prediction,
            market_context={"price": price, "entry": price, "atr": atr, "timestamp": ts},
        )
        # Attach ATR-based protective levels when the engine did not supply them.
        if signal.side in {SignalType.BUY, SignalType.SELL} and atr > 0:
            if signal.sl is None or signal.tp is None:
                if signal.side == SignalType.BUY:
                    signal = TradeSignal(
                        symbol=signal.symbol,
                        side=signal.side,
                        strength=signal.strength,
                        confidence=signal.confidence,
                        entry=signal.entry or price,
                        sl=price - atr * self.config.risk.atr_stop_mult,
                        tp=price + atr * self.config.risk.atr_tp_mult,
                        size_hint=signal.size_hint,
                        metadata=dict(signal.metadata),
                    )
                else:
                    signal = TradeSignal(
                        symbol=signal.symbol,
                        side=signal.side,
                        strength=signal.strength,
                        confidence=signal.confidence,
                        entry=signal.entry or price,
                        sl=price + atr * self.config.risk.atr_stop_mult,
                        tp=price - atr * self.config.risk.atr_tp_mult,
                        size_hint=signal.size_hint,
                        metadata=dict(signal.metadata),
                    )

        # Realized direction vs prior close for accuracy tracking
        if len(lookback) >= 2:
            prev_close = float(lookback[-2]["close"])
            realized = price - prev_close
            self.monitor.record_prediction(prediction=float(prediction.prediction), realized_direction=realized)

        decision = self.risk.pre_trade_validate(
            signal,
            open_positions=self.portfolio.positions(),
            equity=self.portfolio.total_equity(),
            correlations=correlations,
            atr=atr,
        )
        event = {
            "timestamp": ts.isoformat(),
            "prediction": prediction.prediction,
            "confidence_pct": signal.metadata.get("confidence_pct"),
            "side": signal.side.value,
            "risk": decision.to_dict(),
        }
        if not decision.approved or signal.side not in {SignalType.BUY, SignalType.SELL}:
            event["action"] = "skip"
            self.session_log.append(event)
            return

        # Drawdown-aware size adjustment from portfolio
        size = float(decision.size) * self.portfolio.size_multiplier_for_drawdown()
        if size <= 0:
            event["action"] = "skip_size"
            self.session_log.append(event)
            return

        order = self.executor.create_order(
            symbol=symbol,
            side=signal.side,
            quantity=size,
            sl=signal.sl,
            tp=signal.tp,
            metadata={"sl": signal.sl, "tp": signal.tp, "signal": signal.metadata},
        )
        report = self.executor.submit_order(order, market_price=price, timestamp=ts)
        event["action"] = report.status.value
        event["order_id"] = order.client_order_id
        if report.status in {OrderStatus.FILLED, OrderStatus.PARTIALLY_FILLED}:
            for fill in report.fills:
                fill.metadata = {**dict(fill.metadata or {}), "sl": order.sl, "tp": order.tp}
                self.portfolio.apply_fill(fill, broker=self.broker)
            # Ensure SL/TP landed on the open position
            for position in self.portfolio.positions(symbol=symbol):
                if position.sl is None and order.sl is not None:
                    position.sl = float(order.sl)
                if position.tp is None and order.tp is not None:
                    position.tp = float(order.tp)
            self.risk.update_equity(self.portfolio.total_equity(), timestamp=ts)
        self.session_log.append(event)


def create_autonomous_trader(
    config: AIConfig | None = None,
    *,
    mode: str = "paper",
    candle_repository: Any = None,
    candle_source: Any = None,
    initial_equity: float = 100_000.0,
) -> AutonomousTrader:
    return AutonomousTrader(
        config=config or AIConfig(),
        mode=mode,
        candle_repository=candle_repository,
        candle_source=candle_source,
        initial_equity=initial_equity,
    )


def _as_utc(value: Any) -> datetime:
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=timezone.utc)
    return datetime.now(timezone.utc)


def _estimate_atr(candles: Sequence[CandleDict], period: int = 14) -> float:
    if len(candles) < 2:
        return abs(float(candles[-1]["high"]) - float(candles[-1]["low"])) if candles else 0.0001
    trs: List[float] = []
    for i in range(1, len(candles)):
        high = float(candles[i]["high"])
        low = float(candles[i]["low"])
        prev_close = float(candles[i - 1]["close"])
        trs.append(max(high - low, abs(high - prev_close), abs(low - prev_close)))
    window = trs[-period:] if len(trs) >= period else trs
    return float(np.mean(window)) if window else 0.0001
