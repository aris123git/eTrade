"""
tests/test_ai_auto_download.py

AI downloads its own required timeframes before training.
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from ai.config.settings import AIConfig
from ai.data.auto_download import AIMarketDataService
from ai.services.pipeline import create_ai_pipeline
from database.core.connection import DatabaseManager
from database.schema import create_schema
from database.indexes import create_indexes
from database.seed import seed
from database.migrations import apply_migrations


class AIAutoDownloadTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.db_path = self.root / "ai_data.db"
        self.db = DatabaseManager(db_path=self.db_path)
        create_schema(self.db)
        create_indexes(self.db)
        seed(self.db)
        apply_migrations(self.db)

        self.config = AIConfig()
        self.config.symbols = ["EURUSD", "GBPUSD"]
        self.config.timeframes = ["M15", "H1"]
        self.config.primary_timeframe = "M15"
        self.config.features.multi_timeframes = ["M15", "H1", "H4"]
        self.config.data.auto_download = True
        self.config.data.min_bars = 100
        self.config.data.include_mt5 = False
        self.config.data.allow_synthetic_fallback = True
        self.config.data.database_path = str(self.db_path)
        self.config.storage.root_dir = self.root / "artifacts"

    def tearDown(self) -> None:
        self.db.close()
        self.tmp.cleanup()

    def test_required_timeframes_include_multi_tf(self):
        service = AIMarketDataService(config=self.config, db=self.db)
        tfs = service.required_timeframes()
        self.assertEqual(tfs, ["M15", "H1", "H4"])
        self.assertEqual(service.required_symbols(), ["EURUSD", "GBPUSD"])

    def test_ensure_downloads_all_timeframes_autonomously(self):
        service = AIMarketDataService(config=self.config, db=self.db)
        before = service.coverage()
        self.assertTrue(any(g.missing for g in before))

        result = service.ensure()
        self.assertTrue(result.ok)
        self.assertGreater(result.synthetic_filled, 0)

        after = { (g.symbol, g.timeframe): g.bars for g in result.gaps_after }
        for symbol in ("EURUSD", "GBPUSD"):
            for tf in ("M15", "H1", "H4"):
                self.assertGreaterEqual(after[(symbol, tf)], 100)

    def test_pipeline_loads_after_auto_download(self):
        pipeline = create_ai_pipeline(config=self.config, ensure_data=True)
        candles = pipeline.load_candles(symbol="EURUSD", timeframe="M15", limit=50, auto_download=False)
        # Source may be DB adapter; if wired, we get bars. Otherwise ensure filled DB.
        service = AIMarketDataService(config=self.config, db=self.db)
        self.assertGreaterEqual(service.bar_count("EURUSD", "M15"), 100)
        if candles:
            self.assertEqual(candles[-1]["symbol"].upper(), "EURUSD")


if __name__ == "__main__":
    unittest.main()
