"""
tests/test_multi_broker_identity.py

Canonical symbol identity + multi-broker join/compare.
"""

from __future__ import annotations

import tempfile
import unittest
from datetime import datetime, timedelta
from pathlib import Path
from uuid import uuid4

from core.symbol_identity import canonicalize, group_by_canonical, same_instrument
from database.core.connection import DatabaseManager
from database.schema import create_schema
from database.indexes import create_indexes
from database.seed import seed
from database.migrations import apply_migrations
from collector.broker_sources.csv_source import CsvBrokerSource
from collector.broker_sources.registry import BrokerSourceRegistry
from collector.multi_broker import MultiBrokerCollector


class SymbolIdentityTests(unittest.TestCase):
    def test_fx_suffix_and_separators(self):
        for raw in ("EURUSD", "EURUSD.a", "EURUSDm", "EUR/USD", "eur-usd", "EURUSD.pro"):
            self.assertEqual(canonicalize(raw).canonical_symbol, "EURUSD", raw)

    def test_index_and_metal_aliases(self):
        self.assertTrue(same_instrument("US30", "DJ30"))
        self.assertTrue(same_instrument("GOLD", "XAUUSD"))
        self.assertEqual(canonicalize("USTEC").canonical_symbol, "NAS100")
        self.assertEqual(canonicalize("DE40").canonical_symbol, "GER40")

    def test_group_by_canonical(self):
        groups = group_by_canonical(["EURUSD.a", "EURUSDm", "GBPUSD", "DJ30", "US30"])
        self.assertEqual(sorted(groups["EURUSD"]), ["EURUSD.a", "EURUSDm"])
        self.assertEqual(set(groups["US30"]), {"DJ30", "US30"})


class MultiBrokerCollectorTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.db_path = self.root / "mb.db"
        self.db = DatabaseManager(db_path=self.db_path)
        create_schema(self.db)
        create_indexes(self.db)
        seed(self.db)
        apply_migrations(self.db)

        # Two CSV brokers with differently named same instruments
        self.broker_a = self.root / "broker_a"
        self.broker_b = self.root / "broker_b"
        self.broker_a.mkdir()
        self.broker_b.mkdir()
        self._write_csv(
            self.broker_a / "EURUSD_M15.csv",
            "EURUSD",
            "M15",
            base=1.1000,
        )
        self._write_csv(
            self.broker_b / "EURUSD.a_M15.csv",
            "EURUSD.a",
            "M15",
            base=1.1002,
        )
        self._write_csv(
            self.broker_a / "US30_M15.csv",
            "US30",
            "M15",
            base=39000.0,
        )
        self._write_csv(
            self.broker_b / "DJ30_M15.csv",
            "DJ30",
            "M15",
            base=39005.0,
        )

        registry = BrokerSourceRegistry()
        registry.register(CsvBrokerSource("AlphaFX", self.broker_a))
        registry.register(CsvBrokerSource("BetaMarkets", self.broker_b))
        self.collector = MultiBrokerCollector(self.db, registry)

    def tearDown(self) -> None:
        self.db.close()
        self.tmp.cleanup()

    def _write_csv(self, path: Path, symbol: str, timeframe: str, *, base: float) -> None:
        start = datetime(2024, 1, 1, 0, 0, 0)
        lines = ["symbol,timeframe,timestamp,open,high,low,close,volume"]
        price = base
        for i in range(20):
            ts = start + timedelta(minutes=15 * i)
            o = price
            c = price + 0.0001
            lines.append(
                f"{symbol},{timeframe},{ts.isoformat()},"
                f"{o},{o + 0.0002},{o - 0.0002},{c},10"
            )
            price = c
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    def test_download_join_and_compare(self):
        inserted = self.collector.download_all(["M15"], currency_pairs_only=False, years=1)
        self.assertGreater(inserted["AlphaFX"], 0)
        self.assertGreater(inserted["BetaMarkets"], 0)

        joins = self.collector.join_instruments(min_brokers=2)
        canons = {j.canonical_symbol for j in joins}
        self.assertIn("EURUSD", canons)
        self.assertIn("US30", canons)

        eurusd = next(j for j in joins if j.canonical_symbol == "EURUSD")
        local_names = {m.broker_symbol for m in eurusd.markets}
        self.assertIn("EURUSD", local_names)
        self.assertIn("EURUSD.a", local_names)

        resolved = self.collector.resolve_broker_symbol("BetaMarkets", "EURUSD")
        self.assertEqual(resolved, "EURUSD.a")

        rows = self.collector.compare_closes(
            "EURUSD", "M15", "AlphaFX", "BetaMarkets", limit=100
        )
        self.assertGreaterEqual(len(rows), 10)
        # Broker B was seeded with a small premium
        self.assertNotEqual(rows[0].close_a, rows[0].close_b)

    def test_same_symbol_coexists_per_broker(self):
        # Persist identical local names under two brokers
        a = self.collector.ensure_broker("BrokerOne")
        b = self.collector.ensure_broker("BrokerTwo")
        from database.models.market_model import MarketModel

        mm = MarketModel(self.db)
        mm.add_market(symbol="EURUSD", category="FOREX", broker_id=a, canonical_symbol="EURUSD")
        mm.add_market(symbol="EURUSD", category="FOREX", broker_id=b, canonical_symbol="EURUSD")
        rows = self.db.fetch_all(
            "SELECT broker_id, symbol, canonical_symbol FROM markets WHERE symbol='EURUSD'"
        )
        self.assertEqual(len(rows), 2)
        self.assertEqual({r["broker_id"] for r in rows}, {a, b})


if __name__ == "__main__":
    unittest.main()
