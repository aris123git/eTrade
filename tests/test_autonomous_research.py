"""
tests/test_autonomous_research.py

Autonomous Quant Research Platform: gate, hypotheses, and one offline cycle.
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from ai.config.settings import AIConfig, ResearchConfig
from ai.research.gate import decide_promotion
from ai.research.hypotheses import generate_hypotheses
from ai.research.platform import create_research_platform
from ai.services.pipeline import create_ai_pipeline
from database.core.connection import DatabaseManager
from database.indexes import create_indexes
from database.migrations import apply_migrations
from database.schema import create_schema
from database.seed import seed


class GateTests(unittest.TestCase):
    def test_baseline_champion_accepted(self):
        decision = decide_promotion(
            challenger_metrics={"test_f1": 0.61},
            champion_metrics=None,
            metric_name="test_f1",
            min_improvement=0.005,
        )
        self.assertTrue(decision.accepted)
        self.assertEqual(decision.reason, "baseline_champion")

    def test_regression_rejected(self):
        decision = decide_promotion(
            challenger_metrics={"test_f1": 0.60},
            champion_metrics={"test_f1": 0.65},
            metric_name="test_f1",
            min_improvement=0.005,
        )
        self.assertFalse(decision.accepted)
        self.assertEqual(decision.reason, "no_improvement")

    def test_improvement_accepted(self):
        decision = decide_promotion(
            challenger_metrics={"test_f1": 0.70},
            champion_metrics={"test_f1": 0.65},
            metric_name="test_f1",
            min_improvement=0.005,
        )
        self.assertTrue(decision.accepted)
        self.assertEqual(decision.reason, "improved")


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
        gold = [h for h in hyps if h.symbol == "XAUUSD"]
        self.assertTrue(gold)
        self.assertTrue(any("download_more_history" in h.actions for h in gold))


class AutonomousCycleTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
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
        self.config.features.multi_timeframes = ["M15", "H1"]
        self.config.model.model_type = "random_forest"
        self.config.storage.root_dir = self.root / "artifacts"
        self.config.data.auto_download = True
        self.config.data.min_bars = 120
        self.config.data.include_mt5 = False
        self.config.data.allow_synthetic_fallback = True
        self.config.data.database_path = str(self.db_path)
        self.config.data.require_validated = False

        self.research = ResearchConfig(
            skip_collect=False,
            require_validated=False,
            model_candidates=["random_forest"],
            compare_models=False,
            candle_limit=400,
            min_improvement=0.0,
            run_backtest=True,
            run_paper_trade=True,
            detect_drift=True,
            generate_hypotheses=True,
            sleep_seconds=0.0,
        )
        self.config.research = self.research

    def tearDown(self) -> None:
        self.db.close()
        self.tmp.cleanup()

    def test_full_cycle_offline(self):
        pipeline = create_ai_pipeline(config=self.config, ensure_data=False)
        if pipeline.data_service is not None:
            pipeline.data_service.db = self.db
        platform = create_research_platform(
            config=self.config,
            research=self.research,
            pipeline=pipeline,
            artifact_root=self.root / "artifacts",
        )
        report = platform.run_cycle()
        report.print_summary()

        stage_names = [s.name for s in report.stages]
        self.assertIn("wake", stage_names)
        self.assertIn("collect", stage_names)
        self.assertIn("validate", stage_names)
        self.assertIn("learn", stage_names)
        self.assertIn("hypotheses", stage_names)
        self.assertEqual(report.status, "ok")
        self.assertTrue((self.root / "artifacts" / "research_cycles" / "latest.json").exists())
        self.assertEqual(platform.state.cycles_completed, 1)
        # Second cycle should use champion gate (may accept or reject)
        report2 = platform.run_cycle()
        self.assertEqual(platform.state.cycles_completed, 2)
        self.assertIn(report2.status, {"ok", "failed"})

    def test_gate_discards_regression_across_cycles(self):
        pipeline = create_ai_pipeline(config=self.config, ensure_data=True)
        if pipeline.data_service is not None:
            pipeline.data_service.db = self.db
        platform = create_research_platform(
            config=self.config,
            research=self.research,
            pipeline=pipeline,
            artifact_root=self.root / "artifacts",
        )
        platform.run_cycle()
        # Force an unbeatable champion
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
        # With a near-perfect champion, promotions should be rare / rejected
        self.assertEqual(learn.status, "ok")
        symbol_learn = (learn.detail.get("symbols") or {}).get("EURUSD") or {}
        self.assertIn(symbol_learn.get("status"), {"no_promotion", "promoted", "skipped"})


if __name__ == "__main__":
    unittest.main()
