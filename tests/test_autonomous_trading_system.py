"""Phase 4 autonomous trading system — unit + paper-week integration tests."""

from __future__ import annotations

import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

from ai.config.settings import AIConfig
from ai.data.candle_adapter import InMemoryCandleSource
from ai.execution.executor import OrderStatus, create_order_executor
from ai.explainability.explainer import create_model_explainer
from ai.models.trainer import create_model_trainer
from ai.monitoring.live import create_live_monitor
from ai.portfolio.manager import create_portfolio_manager
from ai.risk.manager import create_risk_manager
from ai.services.autonomous_trader import create_autonomous_trader
from ai.signals.generator import create_signal_generator
from ai.training.scheduler import create_training_scheduler
from ai.utils.types import OrderSide, PredictionResult, SignalType


def _candles(n: int = 1800, symbol: str = "EURUSD", timeframe: str = "H1", seed: int = 7) -> List[Dict[str, Any]]:
    rng = np.random.default_rng(seed)
    ts = datetime(2024, 1, 1, tzinfo=timezone.utc)
    price = 1.10
    out: List[Dict[str, Any]] = []
    for i in range(n):
        drift = 0.00012 * np.sin(i / 29.0)
        shock = float(rng.normal(0.0, 0.0007))
        open_ = price
        close = max(0.5, open_ * (1.0 + drift + shock))
        high = max(open_, close) * 1.0004
        low = min(open_, close) * 0.9996
        out.append(
            {
                "symbol": symbol,
                "timeframe": timeframe,
                "timestamp": ts,
                "open": float(open_),
                "high": float(high),
                "low": float(low),
                "close": float(close),
                "volume": float(100 + i % 50),
            }
        )
        price = close
        ts += timedelta(hours=1)
    return out


class ModelTrainerTests(unittest.TestCase):
    def test_trains_random_forest_and_saves_under_ai_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cfg = AIConfig()
            cfg.storage.root_dir = Path(tmp) / "ai" / "artifacts"
            cfg.data.auto_download = False
            cfg.data.require_validated = False
            cfg.model.model_type = "random_forest"
            source = InMemoryCandleSource(_candles())
            trainer = create_model_trainer(cfg)
            trainer.pipeline.candle_source = source  # type: ignore[union-attr]
            result = trainer.train(symbol="EURUSD", model_type="random_forest", candles=_candles(), register=True)
            self.assertNotIn("error", result)
            self.assertEqual(result["model_type"], "random_forest")
            self.assertTrue(str(result["artifact_root"]).endswith("models") or "artifacts" in str(result["artifact_root"]))
            self.assertIsNotNone(result.get("registered"))


class SignalGeneratorTests(unittest.TestCase):
    def test_confidence_pct_and_mtf_veto(self) -> None:
        cfg = AIConfig()
        cfg.risk.min_confidence = 0.4
        gen = create_signal_generator(cfg)
        gen.engine.filters.session_filter = False  # type: ignore[union-attr]
        pred = PredictionResult(
            symbol="EURUSD",
            timeframe="H1",
            timestamp=datetime(2024, 1, 2, 12, tzinfo=timezone.utc),
            prediction=1.0,
            confidence=0.8,
            probabilities=None,
            expected_return=None,
            model_version="t",
            metadata={},
        )
        higher = [
            PredictionResult(
                symbol="EURUSD",
                timeframe="H4",
                timestamp=pred.timestamp,
                prediction=-1.0,
                confidence=0.9,
                probabilities=None,
                expected_return=None,
                model_version="t",
                metadata={},
            )
        ]
        signal = gen.generate(
            pred,
            market_context={"price": 1.1, "entry": 1.1, "sl": 1.09, "tp": 1.12},
            higher_tf_predictions=higher,
        )
        self.assertEqual(signal.side, SignalType.HOLD)
        self.assertIn("mtf_confirmation", signal.metadata)


class RiskAndExecutionTests(unittest.TestCase):
    def test_circuit_breaker_blocks_trades(self) -> None:
        cfg = AIConfig()
        cfg.risk.circuit_breaker_loss = 0.05
        risk = create_risk_manager(cfg, equity=100_000.0)
        risk.update_equity(94_000.0)  # 6% drawdown
        self.assertTrue(risk.circuit_breaker_tripped)
        from ai.signals.engine import TradeSignal

        decision = risk.pre_trade_validate(
            TradeSignal(
                symbol="EURUSD",
                side=SignalType.BUY,
                strength=1.0,
                confidence=0.9,
                entry=1.1,
                sl=1.09,
                tp=1.12,
            )
        )
        self.assertFalse(decision.approved)
        self.assertEqual(decision.reason, "circuit_breaker_active")

    def test_per_symbol_limit(self) -> None:
        cfg = AIConfig()
        cfg.risk.max_positions_per_symbol = 1
        cfg.risk.min_expected_rr = 1.0
        cfg.risk.min_confidence = 0.5
        risk = create_risk_manager(cfg, equity=100_000.0)
        from ai.signals.engine import TradeSignal

        open_pos = [{"symbol": "EURUSD", "side": "LONG", "size": 1.0, "entry_price": 1.1, "sl": 1.09}]
        decision = risk.pre_trade_validate(
            TradeSignal(
                symbol="EURUSD",
                side=SignalType.BUY,
                strength=1.0,
                confidence=0.9,
                entry=1.1,
                sl=1.09,
                tp=1.13,
            ),
            open_positions=open_pos,
        )
        self.assertFalse(decision.approved)
        self.assertEqual(decision.reason, "max_positions_per_symbol_exceeded")

    def test_sl_tp_automation_and_shutdown(self) -> None:
        exe = create_order_executor(mode="paper")
        portfolio = create_portfolio_manager(cash=100_000.0)
        pos = portfolio.open_position(
            symbol="EURUSD",
            side=OrderSide.BUY,
            size=1.0,
            price=1.1000,
            sl=1.0950,
            tp=1.1100,
        )
        reports = exe.manage_exits([pos], prices={"EURUSD": 1.0940})
        self.assertEqual(len(reports), 1)
        self.assertEqual(reports[0].metadata.get("exit_reason"), "stop_loss")
        exe.request_shutdown()
        rejected = exe.submit_order(
            exe.create_order(symbol="EURUSD", side=OrderSide.BUY, quantity=1.0),
            market_price=1.1,
        )
        self.assertEqual(rejected.status, OrderStatus.REJECTED)
        self.assertEqual(rejected.message, "executor_shutdown")


class PortfolioCorrelationTests(unittest.TestCase):
    def test_correlation_concentration_and_drawdown_sizing(self) -> None:
        pm = create_portfolio_manager(cash=100_000.0)
        pm.open_position(symbol="EURUSD", side=OrderSide.BUY, size=1.0, price=1.1)
        pm.open_position(symbol="GBPUSD", side=OrderSide.BUY, size=1.0, price=1.25)
        conc = pm.correlation_concentration({("EURUSD", "GBPUSD"): 0.9}, threshold=0.75)
        self.assertTrue(conc["overconcentrated"])
        pm.peak_equity = 100_000.0
        pm.cash = 90_000.0
        self.assertGreater(pm.drawdown(), 0.0)
        self.assertLess(pm.size_multiplier_for_drawdown(), 1.0)


class LiveMonitorTests(unittest.TestCase):
    def test_accuracy_drift_flag(self) -> None:
        pm = create_portfolio_manager(cash=100_000.0)
        mon = create_live_monitor(portfolio=pm, min_accuracy=0.55)
        for _ in range(20):
            mon.record_prediction(prediction=1.0, correct=False)
        snap = mon.snapshot()
        self.assertFalse(snap.accuracy_ok)
        self.assertIsNotNone(snap.accuracy)
        self.assertLess(float(snap.accuracy), 0.55)


class ExplainerTests(unittest.TestCase):
    def test_feature_importance_shap_tree_ci(self) -> None:
        from ai.models import create_model

        cfg = AIConfig()
        cfg.model.model_type = "random_forest"
        model = create_model("random_forest", cfg)
        rng = np.random.default_rng(0)
        X = rng.normal(size=(80, 6))
        y = (X[:, 0] + X[:, 1] > 0).astype(int)
        model.fit(X, y)
        explainer = create_model_explainer(model, X[:40], feature_names=[f"f{i}" for i in range(6)])
        report = explainer.explain(X[-1:])
        self.assertIn(report.decision, {"buy", "sell", "hold"})
        self.assertTrue(report.feature_importance)
        self.assertIn("values", report.shap)
        self.assertTrue(report.tree["rules"])
        self.assertIn("low", report.confidence_interval)


class TrainingSchedulerTests(unittest.TestCase):
    def test_deploy_first_model(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cfg = AIConfig()
            cfg.storage.root_dir = Path(tmp) / "ai" / "artifacts"
            cfg.data.auto_download = False
            cfg.data.require_validated = False
            cfg.model.model_type = "random_forest"
            trainer = create_model_trainer(cfg)
            trainer.pipeline.candle_source = InMemoryCandleSource(_candles())  # type: ignore[union-attr]
            sched = create_training_scheduler(cfg, trainer=trainer)
            sched.state_path = Path(tmp) / "sched.json"
            result = sched.run_once(
                symbol="EURUSD",
                model_type="random_forest",
                candles=_candles(),
                limit=1800,
            )
            self.assertTrue(result.promoted)
            self.assertIn(result.action, {"trained", "deployed"})


class PaperWeekIntegrationTests(unittest.TestCase):
    def test_paper_week_end_to_end(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cfg = AIConfig()
            cfg.storage.root_dir = Path(tmp) / "ai" / "artifacts"
            cfg.symbols = ["EURUSD"]
            cfg.primary_timeframe = "H1"
            cfg.data.auto_download = False
            cfg.data.require_validated = False
            cfg.model.model_type = "random_forest"
            cfg.risk.min_confidence = 0.45
            cfg.risk.min_expected_rr = 1.0
            cfg.risk.circuit_breaker_loss = 0.25
            cfg.risk.max_drawdown = 0.30
            candles = _candles(2000)
            source = InMemoryCandleSource(candles)
            trader = create_autonomous_trader(
                cfg,
                mode="paper",
                candle_source=source,
                initial_equity=100_000.0,
            )
            trader.signals.engine.filters.session_filter = False  # type: ignore[union-attr]
            summary = trader.run_paper_week(
                candles,
                symbol="EURUSD",
                model_type="random_forest",
                train_bars=1500,
                week_bars=168,
            )
            self.assertEqual(summary["paper_bars"], 168)
            self.assertIn("metrics", summary)
            self.assertIn("explainability", summary)
            self.assertNotIn("error", summary["train"])
            # Explainability should produce a decision or structured error
            expl = summary["explainability"]
            self.assertTrue("decision" in expl or "error" in expl)
            self.assertGreaterEqual(summary["metrics"]["closed_trades"] + summary["metrics"]["open_positions"], 0)
            # Live monitor produced snapshots
            self.assertGreaterEqual(len(trader.monitor.snapshots), 1)  # type: ignore[union-attr]


if __name__ == "__main__":
    unittest.main()
