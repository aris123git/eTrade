"""
tests/test_autonomous_research.py

Gate/hypothesis unit tests + smoke cycle with CSV archive (no synthetic).
"""

from __future__ import annotations

import csv
import math
import tempfile
import unittest
from datetime import datetime, timedelta
from pathlib import Path

from ai.config.settings import AIConfig, ResearchConfig
from ai.research.gate import decide_promotion
from ai.research.hypotheses import generate_hypotheses
from ai.research.platform import create_research_platform
from database.core.connection import DatabaseManager
from database.indexes import create_indexes
from database.migrations import apply_migrations
from database.schema import create_schema
from database.seed import seed


def _write_csv(path: Path, n: int = 500) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    start = datetime(2020, 1, 6)
    price = 1.1
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["timestamp", "open", "high", "low", "close", "volume"])
        t = start
        for i in range(n):
            while t.weekday() >= 5:
                t += timedelta(minutes=15)
            close = price + 0.00008 + 0.00002 * math.sin(i / 11.0)
            w.writerow([t.isoformat(timespec="seconds"), price, max(price, close) + 1e-5, min(price, close) - 1e-5, close, 100])
            price = close
            t += timedelta(minutes=15)


class GateTests(unittest.TestCase):
    def test_baseline_champion_accepted(self):
        decision = decide_promotion(
            challenger_metrics={"test_f1": 0.61},
            champion_metrics=None,
            metric_name="test_f1",
            min_improvement=0.005,
        )
        self.assertTrue(decision.accepted)

    def test_regression_rejected(self):
        decision = decide_promotion(
            challenger_metrics={"test_f1": 0.60},
            champion_metrics={"test_f1": 0.65},
            metric_name="test_f1",
            min_improvement=0.005,
        )
        self.assertFalse(decision.accepted)

    def test_improvement_accepted(self):
        decision = decide_promotion(
            challenger_metrics={"test_f1": 0.70},
            champion_metrics={"test_f1": 0.65},
            metric_name="test_f1",
            min_improvement=0.005,
        )
        self.assertTrue(decision.accepted)


class HypothesisTests(unittest.TestCase):
    def test_weak_symbol_generates_actions(self):
        hyps = generate_hypotheses(
            per_symbol_metrics={
                "EURUSD": {"test_f1": 0.72},
                "XAUUSD": {"test_f1": 0.41},
            },
            validation_failures=["XAUUSD:H1"],
            primary_metric="test_f1",
            weak_threshold=0.55,
        )
        kinds = {h.kind for h in hyps}
        self.assertIn("weak_performance", kinds)
        self.assertIn("data_coverage", kinds)


class AutonomousCycleTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        csv_dir = self.root / "csv"
        _write_csv(csv_dir / "EURUSD_M15.csv", n=900)
        _write_csv(csv_dir / "EURUSD_H1.csv", n=900)
        self.db_path = self.root / "research.db"
        self.db = DatabaseManager(db_path=self.db_path)
        create_schema(self.db)
        create_indexes(self.db)
        seed(self.db)
        apply_migrations(self.db)

        self.config = AIConfig()
        self.config.symbols = ["EURUSD"]
        self.config.timeframes = ["M15"]
        self.config.primary_timeframe = "M15"
        self.config.features.multi_timeframes = ["M15"]
        self.config.features.enabled_groups = [
            "price", "returns", "momentum", "volatility", "session",
        ]
        self.config.model.model_type = "random_forest"
        self.config.storage.root_dir = self.root / "artifacts"
        self.config.data.auto_download = True
        self.config.data.min_bars = 100
        self.config.data.include_mt5 = False
        self.config.data.csv_brokers = {"CsvBroker": str(csv_dir)}
        self.config.data.allow_synthetic_fallback = False
        self.config.data.database_path = str(self.db_path)
        self.config.data.require_validated = False

        self.research = ResearchConfig(
            skip_collect=False,
            require_validated=False,
            allow_synthetic=False,
            download_ticks=False,
            model_candidates=["random_forest"],
            compare_models=False,
            candle_limit=800,
            min_improvement=0.0,
            run_feature_discovery=False,
            run_strict_validation=False,
            run_backtest=True,
            run_paper_trade=True,
            detect_drift=True,
            generate_hypotheses=True,
            run_self_improve=False,
            run_production_gate=True,
            build_dashboard=True,
            sleep_seconds=0.0,
            min_paper_trades=9999,  # force gate fail (no live)
            min_paper_days=9999,
        )
        self.config.research = self.research

    def tearDown(self) -> None:
        self.db.close()
        self.tmp.cleanup()

    def test_full_cycle_offline(self):
        platform = create_research_platform(
            config=self.config,
            research=self.research,
            db=self.db,
            artifact_root=self.root / "artifacts",
        )
        report = platform.run_cycle()
        stage_names = [s.name for s in report.stages]
        self.assertIn("wake", stage_names)
        self.assertIn("collect", stage_names)
        self.assertIn("learn", stage_names)
        self.assertEqual(report.status, "ok")
        self.assertEqual(platform.state.cycles_completed, 1)

    def test_gate_discards_regression_across_cycles(self):
        platform = create_research_platform(
            config=self.config,
            research=self.research,
            db=self.db,
            artifact_root=self.root / "artifacts",
        )
        platform.run_cycle()
        key = "EURUSD:M15"
        platform.state.champions[key] = {
            "model_type": "random_forest",
            "metrics": {"test_f1": 0.99, "val_f1": 0.99},
            "version": "forced",
            "decision": {"accepted": True, "reason": "baseline_champion"},
        }
        platform.research.min_improvement = 0.01
        report = platform.run_cycle()
        learn = next(s for s in report.stages if s.name == "learn")
        self.assertEqual(learn.status, "ok")


if __name__ == "__main__":
    unittest.main()
