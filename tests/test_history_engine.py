"""
tests/test_history_engine.py

Production historical data engine: resumable download + validator.
"""

from __future__ import annotations

import tempfile
import unittest
from datetime import datetime, timedelta
from pathlib import Path

from database.core.connection import DatabaseManager
from database.schema import create_schema
from database.indexes import create_indexes
from database.seed import seed
from database.migrations import apply_migrations
from collector.history_engine import HistoricalDataEngine, download_history
from collector.history_validator import HistoryValidator


class HistoryEngineTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.db_path = self.root / "history.db"
        self.db = DatabaseManager(db_path=self.db_path)
        create_schema(self.db)
        create_indexes(self.db)
        seed(self.db)
        apply_migrations(self.db)

        self.csv_dir = self.root / "broker_a"
        self.csv_dir.mkdir()
        self._write_csv("EURUSD", "M15", n=200, start=datetime(2024, 1, 1))
        self._write_csv("EURUSD", "H1", n=100, start=datetime(2024, 1, 1))
        self._write_csv("XAUUSD", "M15", n=150, start=datetime(2024, 1, 1))
        self._write_csv("US30", "H1", n=80, start=datetime(2024, 1, 1))

    def tearDown(self) -> None:
        self.db.close()
        self.tmp.cleanup()

    def _write_csv(self, symbol: str, timeframe: str, *, n: int, start: datetime) -> None:
        step = {"M15": 15, "H1": 60, "M5": 5, "D1": 1440}.get(timeframe, 15)
        path = self.csv_dir / f"{symbol}_{timeframe}.csv"
        lines = ["symbol,timeframe,timestamp,open,high,low,close,volume"]
        price = 1.1 if symbol.startswith("EUR") else (2000.0 if symbol.startswith("XAU") else 39000.0)
        for i in range(n):
            ts = start + timedelta(minutes=step * i)
            # Skip weekends roughly for FX continuity in validator
            if ts.weekday() >= 5:
                continue
            o = price
            c = price + 0.0001
            lines.append(
                f"{symbol},{timeframe},{ts.isoformat()},"
                f"{o},{max(o, c) + 0.0002},{min(o, c) - 0.0002},{c},10"
            )
            price = c
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    def test_download_history_all_markets_and_inventory(self):
        result = download_history(
            self.db,
            brokers="ALL",
            markets=["FOREX", "METALS", "INDICES"],
            symbols="ALL",
            timeframes=["M15", "H1"],
            start="2024-01-01",
            end="2024-12-31",
            resume=True,
            include_mt5=False,
            csv_brokers={"DemoBroker": str(self.csv_dir)},
        )
        self.assertGreater(result.total_inserted, 0)
        self.assertTrue(any(s.symbol == "EURUSD" for s in result.series))
        self.assertTrue(any(s.symbol == "XAUUSD" for s in result.series))
        self.assertIn("Forex", result.inventory)
        self.assertIn("EURUSD", result.inventory.get("Forex", {}))

    def test_resume_skips_redownload(self):
        engine = HistoricalDataEngine(
            self.db,
            include_mt5=False,
            csv_brokers={"DemoBroker": str(self.csv_dir)},
        )
        first = engine.download_history(
            markets="FOREX",
            symbols=["EURUSD"],
            timeframes=["M15"],
            start="2024-01-01",
            end="2024-12-31",
            resume=True,
        )
        inserted_first = first.total_inserted
        self.assertGreater(inserted_first, 0)

        second = engine.download_history(
            markets="FOREX",
            symbols=["EURUSD"],
            timeframes=["M15"],
            start="2024-01-01",
            end="2024-12-31",
            resume=True,
        )
        # Second pass should insert nothing (already up to date)
        self.assertEqual(second.total_inserted, 0)
        self.assertTrue(any(s.status == "skipped_uptodate" for s in second.series))

    def test_validator_pass_on_clean_data(self):
        download_history(
            self.db,
            markets="ALL",
            symbols=["EURUSD"],
            timeframes=["M15"],
            start="2024-01-01",
            end="2024-12-31",
            include_mt5=False,
            csv_brokers={"DemoBroker": str(self.csv_dir)},
        )
        report = HistoryValidator(self.db, min_bars=50).validate_all(
            symbols=["EURUSD"],
            timeframes=["M15"],
        )
        self.assertGreaterEqual(report.passed, 1)
        series = report.series[0]
        self.assertIsNotNone(series.first_candle)
        self.assertIsNotNone(series.last_candle)
        self.assertGreater(series.total_candles, 0)
        self.assertEqual(series.duplicates, 0)
        self.assertEqual(series.invalid_ohlc, 0)
        self.assertEqual(series.status, "PASS")


if __name__ == "__main__":
    unittest.main()
