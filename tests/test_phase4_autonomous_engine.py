"""
tests/test_phase4_autonomous_engine.py

Phase 4 Autonomous Quant Research Engine — integration tests using CSV
broker archives (real OHLCV files, not synthetic bootstrap).
"""

from __future__ import annotations

import csv
import math
import tempfile
import unittest
from datetime import datetime, timedelta
from pathlib import Path

from ai.config.settings import AIConfig, ResearchConfig
from ai.research.discovery import HypothesisDiscoveryEngine
from ai.research.gate import decide_promotion
from ai.research.platform import create_research_platform
from ai.research.production_gate import ProductionReadinessGate, ProductionThresholds
from ai.research.validation_gate import StrictValidationGate, ValidationThresholds
from collector.gap_repair import GapRepairEngine
from collector.history_engine import download_history
from database.core.connection import DatabaseManager
from database.indexes import create_indexes
from database.migrations import apply_migrations
from database.migrations.research_schema import apply_research_schema
from database.repositories.research_repository import ResearchRepository
from database.schema import create_schema
from database.seed import seed


def _write_trending_csv(path: Path, symbol: str, timeframe: str, n: int = 600) -> None:
    """Write deterministic trending OHLCV — broker-export style archive."""
    path.parent.mkdir(parents=True, exist_ok=True)
    start = datetime(2020, 1, 6, 0, 0, 0)  # Monday
    step = {"M15": 15, "H1": 60, "H4": 240}.get(timeframe.upper(), 15)
    price = 1.1000
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["timestamp", "open", "high", "low", "close", "volume"])
        t = start
        for i in range(n):
            # Skip weekends for FX-like series
            while t.weekday() >= 5:
                t += timedelta(minutes=step)
            drift = 0.00008 + 0.00002 * math.sin(i / 17.0)
            noise = 0.00003 * math.sin(i / 3.0)
            open_p = price
            close_p = price + drift + noise
            high_p = max(open_p, close_p) + 0.00005
            low_p = min(open_p, close_p) - 0.00005
            writer.writerow(
                [
                    t.isoformat(timespec="seconds"),
                    f"{open_p:.6f}",
                    f"{high_p:.6f}",
                    f"{low_p:.6f}",
                    f"{close_p:.6f}",
                    100 + (i % 50),
                ]
            )
            price = close_p
            t += timedelta(minutes=step)


class ResearchDatabaseTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.db = DatabaseManager(db_path=Path(self.tmp.name) / "r.db")
        create_schema(self.db)
        create_indexes(self.db)
        seed(self.db)
        apply_migrations(self.db)
        self.repo = ResearchRepository(self.db)

    def tearDown(self) -> None:
        self.db.close()
        self.tmp.cleanup()

    def test_research_tables_and_experiment_persistence(self):
        exp = self.repo.create_experiment(symbol="EURUSD", timeframe="M15", cycle_id="abc")
        self.assertIsNotNone(exp["experiment_id"])
        model = self.repo.record_model(
            int(exp["experiment_id"]),
            name="EURUSD_random_forest",
            model_type="random_forest",
            symbol="EURUSD",
            timeframe="M15",
            metrics={"test_f1": 0.66},
            hyperparameters={"n_estimators": 50},
            is_champion=True,
            status="champion",
        )
        self.repo.set_champion(int(model["model_id"]), "EURUSD", "M15")
        champ = self.repo.get_champion_model("EURUSD", "M15")
        self.assertEqual(champ["model_id"], model["model_id"])
        pred = self.repo.record_prediction(
            model_id=int(model["model_id"]),
            symbol="EURUSD",
            timeframe="M15",
            timestamp="2024-01-01T00:00:00",
            prediction=1.0,
            signal="BUY",
            confidence=0.7,
            explanation="test",
            feature_importance={"rsi": 0.2},
        )
        self.repo.resolve_prediction(int(pred["prediction_id"]), actual_outcome=0.01, pnl=10.0, drawdown=-0.01)
        unresolved = self.repo.list_unresolved_predictions("EURUSD")
        self.assertEqual(len(unresolved), 0)


class GapRepairAndHistoryTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.csv_dir = self.root / "csv"
        _write_trending_csv(self.csv_dir / "EURUSD_M15.csv", "EURUSD", "M15", n=400)
        _write_trending_csv(self.csv_dir / "EURUSD_H1.csv", "EURUSD", "H1", n=400)
        self.db = DatabaseManager(db_path=self.root / "hist.db")
        create_schema(self.db)
        create_indexes(self.db)
        seed(self.db)
        apply_migrations(self.db)

    def tearDown(self) -> None:
        self.db.close()
        self.tmp.cleanup()

    def test_download_from_csv_no_duplicates_on_resume(self):
        csv_brokers = {"CsvBroker": str(self.csv_dir)}
        r1 = download_history(
            self.db,
            brokers="ALL",
            markets=["FOREX"],
            symbols=["EURUSD"],
            timeframes=["M15"],
            start="2020-01-01",
            end="today",
            include_mt5=False,
            csv_brokers=csv_brokers,
        )
        inserted1 = sum(s.bars_inserted for s in r1.series)
        self.assertGreater(inserted1, 100)
        r2 = download_history(
            self.db,
            brokers="ALL",
            markets=["FOREX"],
            symbols=["EURUSD"],
            timeframes=["M15"],
            start="2020-01-01",
            end="today",
            include_mt5=False,
            csv_brokers=csv_brokers,
        )
        inserted2 = sum(s.bars_inserted for s in r2.series)
        self.assertEqual(inserted2, 0)

    def test_gap_repair_engine_runs(self):
        csv_brokers = {"CsvBroker": str(self.csv_dir)}
        download_history(
            self.db,
            brokers="ALL",
            markets=["FOREX"],
            symbols=["EURUSD"],
            timeframes=["M15"],
            start="2020-01-01",
            include_mt5=False,
            csv_brokers=csv_brokers,
        )
        # Delete a middle chunk to create a real gap
        adapter = self.db.get_adapter()
        adapter.execute(
            """
            DELETE FROM candles
            WHERE symbol='EURUSD' AND timeframe='M15'
              AND timestamp > '2020-01-20' AND timestamp < '2020-01-25'
            """
        )
        adapter.commit()
        engine = GapRepairEngine(self.db, include_mt5=False, csv_brokers=csv_brokers)
        report = engine.repair(symbols=["EURUSD"], timeframes=["M15"])
        self.assertGreaterEqual(report.bars_inserted, 0)
        self.assertIsInstance(report.to_dict()["n_gaps"], int)


class DiscoveryAndGateTests(unittest.TestCase):
    def test_promotion_gate(self):
        self.assertTrue(
            decide_promotion(
                challenger_metrics={"test_f1": 0.7},
                champion_metrics={"test_f1": 0.6},
                min_improvement=0.005,
            ).accepted
        )
        self.assertFalse(
            decide_promotion(
                challenger_metrics={"test_f1": 0.61},
                champion_metrics={"test_f1": 0.66},
                min_improvement=0.005,
            ).accepted
        )

    def test_feature_discovery_keeps_or_discards(self):
        # Build in-memory candles
        candles = []
        t = datetime(2020, 1, 6)
        price = 1.1
        for i in range(300):
            while t.weekday() >= 5:
                t += timedelta(minutes=15)
            close = price + 0.0001
            candles.append(
                {
                    "symbol": "EURUSD",
                    "timeframe": "M15",
                    "timestamp": t,
                    "open": price,
                    "high": close + 0.00005,
                    "low": price - 0.00005,
                    "close": close,
                    "volume": 100.0,
                }
            )
            price = close
            t += timedelta(minutes=15)
        cfg = AIConfig()
        cfg.model.model_type = "random_forest"
        cfg.data.auto_download = False
        engine = HypothesisDiscoveryEngine(cfg, min_score=0.0, max_candidates=6)
        result = engine.discover(candles)
        self.assertGreater(len(result.candidates), 0)
        self.assertTrue(result.selected_groups)


class AutonomousEngineCycleTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.csv_dir = self.root / "csv"
        _write_trending_csv(self.csv_dir / "EURUSD_M15.csv", "EURUSD", "M15", n=700)
        _write_trending_csv(self.csv_dir / "EURUSD_H1.csv", "EURUSD", "H1", n=700)
        _write_trending_csv(self.csv_dir / "EURUSD_H4.csv", "EURUSD", "H4", n=500)

        self.db_path = self.root / "research.db"
        self.db = DatabaseManager(db_path=self.db_path)
        create_schema(self.db)
        create_indexes(self.db)
        seed(self.db)
        apply_migrations(self.db)

        self.config = AIConfig()
        self.config.symbols = ["EURUSD"]
        self.config.timeframes = ["M15", "H1"]
        self.config.primary_timeframe = "M15"
        self.config.features.multi_timeframes = ["M15", "H1"]
        self.config.model.model_type = "random_forest"
        self.config.storage.root_dir = self.root / "artifacts"
        self.config.data.auto_download = True
        self.config.data.include_mt5 = False
        self.config.data.csv_brokers = {"CsvBroker": str(self.csv_dir)}
        self.config.data.allow_synthetic_fallback = False
        self.config.data.require_validated = False
        self.config.data.database_path = str(self.db_path)
        self.config.data.min_bars = 100

        self.research = ResearchConfig(
            allow_synthetic=False,
            require_validated=False,
            skip_collect=False,
            download_ticks=False,  # CSV source has no ticks
            model_candidates=["random_forest"],
            compare_models=False,
            candle_limit=500,
            min_improvement=0.0,
            run_feature_discovery=True,
            run_strict_validation=True,
            min_val_score=0.0,
            min_oos_score=0.0,
            min_walk_forward_score=0.0,
            max_mc_ruin_prob=1.0,
            min_backtest_trades=1,
            run_paper_trade=True,
            run_production_gate=True,
            build_dashboard=True,
            generate_hypotheses=True,
            run_self_improve=False,
            min_paper_trades=1,
            min_paper_days=0.0,
            min_live_sharpe=-999.0,
            max_live_drawdown=1.0,
            min_live_profit_factor=0.0,
        )
        self.config.research = self.research

    def tearDown(self) -> None:
        self.db.close()
        self.tmp.cleanup()

    def test_full_cycle_with_csv_archive(self):
        platform = create_research_platform(
            config=self.config,
            research=self.research,
            db=self.db,
            artifact_root=self.root / "artifacts",
        )
        report = platform.run_cycle()
        report.print_summary()
        names = [s.name for s in report.stages]
        for required in ("wake", "collect", "validate", "repair", "discovery", "learn", "dashboard"):
            self.assertIn(required, names)
        self.assertEqual(report.status, "ok")
        self.assertTrue((self.root / "artifacts" / "dashboards" / "institutional_dashboard.html").exists())
        # Research DB retained experiment
        repo = ResearchRepository(self.db)
        champs = repo._fetch_all("SELECT * FROM research_models WHERE is_champion=1")
        # May be empty if validation rejected all — still must not crash
        self.assertIsInstance(champs, list)

    def test_hourly_daily_methods(self):
        platform = create_research_platform(
            config=self.config,
            research=self.research,
            db=self.db,
            artifact_root=self.root / "artifacts",
        )
        hourly = platform.run_hourly()
        self.assertIn("collect", hourly)
        daily = platform.run_daily()
        self.assertIn("symbols", daily)


class ProductionGateTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.db = DatabaseManager(db_path=Path(self.tmp.name) / "g.db")
        create_schema(self.db)
        apply_research_schema(self.db)
        apply_migrations(self.db)
        self.repo = ResearchRepository(self.db)

    def tearDown(self) -> None:
        self.db.close()
        self.tmp.cleanup()

    def test_live_blocked_until_thresholds_met(self):
        gate = ProductionReadinessGate(
            self.repo,
            ProductionThresholds(
                min_paper_trades=50,
                min_paper_days=14,
                min_sharpe=0.5,
                max_drawdown=0.2,
                min_profit_factor=1.2,
                min_accuracy=0.52,
                min_resolved_predictions=50,
            ),
        )
        result = gate.evaluate("EURUSD", "M15", enable_live_if_passed=True)
        self.assertFalse(result.passed)
        self.assertFalse(result.live_enabled)
        self.assertTrue(result.failures)


if __name__ == "__main__":
    unittest.main()
