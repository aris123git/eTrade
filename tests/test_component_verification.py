"""
tests/test_component_verification.py — Continuous component verification suite.
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from ai.config.settings import AIConfig
from ai.research.component_verification import verify_components
from ai.research.edge_proof import create_edge_proof_engine
from ai.research.platform import create_research_platform
from database.core.connection import DatabaseManager
from database.indexes import create_indexes
from database.migrations import apply_migrations
from database.schema import create_schema
from database.seed import seed


class ComponentVerificationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.db = DatabaseManager(db_path=self.root / "verify.db")
        create_schema(self.db)
        create_indexes(self.db)
        seed(self.db)
        apply_migrations(self.db)
        self.config = AIConfig()
        self.config.data.allow_synthetic_fallback = False
        self.config.data.require_validated = True
        self.config.storage.root_dir = self.root / "artifacts"

    def tearDown(self) -> None:
        self.db.close()
        self.tmp.cleanup()

    def test_verify_all_critical_pass(self):
        report = verify_components(db=self.db, config=self.config)
        report.print_summary()
        self.assertTrue(report.critical_ok, msg=[c.to_dict() for c in report.checks if c.status == "FAIL"])
        names = {c.name for c in report.checks}
        for required in (
            "imports",
            "database_schema",
            "research_tables",
            "history_engine",
            "ai_pipeline",
            "walk_forward",
            "monte_carlo",
            "strict_validation_gate",
            "model_promotion_gate",
            "research_repository",
            "edge_proof",
            "production_gate",
        ):
            self.assertIn(required, names)

    def test_config_honesty_fails_when_synthetic_enabled(self):
        bad = self.config.copy()
        bad.data.allow_synthetic_fallback = True
        report = verify_components(db=self.db, config=bad)
        honesty = next(c for c in report.checks if c.name == "config_honesty")
        self.assertEqual(honesty.status, "FAIL")
        self.assertFalse(report.critical_ok)

    def test_edge_day_runs_verification_first(self):
        # Minimal CSV-free path: verification must still run and be recorded
        platform = create_research_platform(
            config=self.config,
            db=self.db,
            artifact_root=self.root / "artifacts",
        )
        # Avoid MT5/tick noise for this unit check
        platform.research.download_ticks = False
        platform.research.skip_collect = True
        platform.research.run_self_improve = False
        platform.research.run_paper_trade = False
        platform.research.build_dashboard = False
        platform.config.data.include_mt5 = False
        engine = create_edge_proof_engine(platform)
        components = engine.verify_components()
        self.assertTrue(components["critical_ok"])
        self.assertTrue((self.root / "artifacts" / "component_verification_latest.json").exists())


if __name__ == "__main__":
    unittest.main()
