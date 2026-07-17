"""
tests/test_edge_proof.py — Evidence-first mode smoke test (CSV archive only).
"""

from __future__ import annotations

import csv
import math
import tempfile
import unittest
from datetime import datetime, timedelta
from pathlib import Path

from ai.config.settings import AIConfig, ResearchConfig
from ai.research.edge_proof import create_edge_proof_engine
from ai.research.platform import create_research_platform
from database.core.connection import DatabaseManager
from database.indexes import create_indexes
from database.migrations import apply_migrations
from database.schema import create_schema
from database.seed import seed


def _write_csv(path: Path, n: int = 800) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    t = datetime(2018, 1, 8)
    price = 1.12
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["timestamp", "open", "high", "low", "close", "volume"])
        for i in range(n):
            while t.weekday() >= 5:
                t += timedelta(minutes=15)
            close = price + 0.00005 + 0.00001 * math.sin(i / 9.0)
            w.writerow(
                [
                    t.isoformat(timespec="seconds"),
                    price,
                    max(price, close) + 1e-5,
                    min(price, close) - 1e-5,
                    close,
                    120,
                ]
            )
            price = close
            t += timedelta(minutes=15)


class EdgeProofTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        csv_dir = self.root / "csv"
        _write_csv(csv_dir / "EURUSD_M15.csv")
        _write_csv(csv_dir / "EURUSD_H1.csv", n=800)
        self.db = DatabaseManager(db_path=self.root / "edge.db")
        create_schema(self.db)
        create_indexes(self.db)
        seed(self.db)
        apply_migrations(self.db)

        self.config = AIConfig()
        self.config.symbols = ["EURUSD"]
        self.config.timeframes = ["M15", "H1"]
        self.config.primary_timeframe = "M15"
        self.config.features.multi_timeframes = ["M15", "H1"]
        self.config.features.enabled_groups = ["price", "returns", "momentum", "volatility", "session"]
        self.config.model.model_type = "random_forest"
        self.config.storage.root_dir = self.root / "artifacts"
        self.config.data.database_path = str(self.root / "edge.db")
        self.config.data.include_mt5 = False
        self.config.data.csv_brokers = {"CsvBroker": str(csv_dir)}
        self.config.data.allow_synthetic_fallback = False
        self.config.data.require_validated = False
        self.config.data.min_bars = 100

        self.research = ResearchConfig(
            allow_synthetic=False,
            require_validated=False,
            download_ticks=False,
            model_candidates=["random_forest"],
            candle_limit=600,
            min_improvement=0.0,
            min_val_score=0.0,
            min_oos_score=0.0,
            min_walk_forward_score=0.0,
            max_mc_ruin_prob=1.0,
            min_backtest_trades=1,
            min_paper_trades=9999,
            min_paper_days=9999,
            run_feature_discovery=False,
            build_dashboard=True,
        )
        self.config.research = self.research

    def tearDown(self) -> None:
        self.db.close()
        self.tmp.cleanup()

    def test_run_day_writes_evidence_ledger(self):
        platform = create_research_platform(
            config=self.config,
            research=self.research,
            db=self.db,
            artifact_root=self.root / "artifacts",
        )
        engine = create_edge_proof_engine(platform)
        result = engine.run_day()
        self.assertIn("evidence", result)
        self.assertTrue(engine.evidence_path.exists())
        evidence = result["evidence"]
        self.assertGreaterEqual(evidence["runs_completed"], 1)
        # Early archive must not claim a scientific edge
        self.assertFalse(evidence["edge_demonstrated"])
        self.assertIn("insufficient_evidence", evidence["reason"])
        self.assertGreater(int((evidence.get("archive") or {}).get("candles") or 0), 0)


if __name__ == "__main__":
    unittest.main()
