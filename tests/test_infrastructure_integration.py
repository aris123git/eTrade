"""
tests/test_infrastructure_integration.py

Phase 2 integration tests for database infrastructure.
"""

from __future__ import annotations

import tempfile
import unittest
from datetime import datetime, timedelta
from pathlib import Path
from uuid import uuid4

from database.core.connection import DatabaseManager
from database.schema import create_schema
from database.indexes import create_indexes
from database.seed import seed
from database.repositories.factory import create_repository_manager
from database.models.market import MarketType, MarketStatus
from database.migrations import apply_migrations
from ai.data import CandleRepositoryAdapter, TickRepositoryAdapter


class InfrastructureIntegrationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.db_path = Path(self.tmp.name) / "test_market_ai.db"
        self.db = DatabaseManager(db_path=self.db_path)
        create_schema(self.db)
        create_indexes(self.db)
        seed(self.db)
        self.repos = create_repository_manager(self.db)

    def tearDown(self) -> None:
        self.db.close()
        self.tmp.cleanup()

    def test_schema_seed_and_migrations(self):
        tfs = self.db.fetch_all("SELECT name FROM timeframes ORDER BY sort_order")
        self.assertGreaterEqual(len(tfs), 9)
        currencies = self.db.fetch_all("SELECT code FROM currencies")
        self.assertGreaterEqual(len(currencies), 8)
        brokers = self.db.fetch_all("SELECT name FROM brokers")
        self.assertTrue(any(b["name"] == "Default" for b in brokers))
        applied = apply_migrations(self.db)
        self.assertIsInstance(applied, list)

    def test_repository_manager_registration(self):
        names = set(self.repos.get_names())
        for required in {"markets", "candles", "ticks", "brokers", "timeframes", "currencies", "symbols"}:
            self.assertIn(required, names)

    def test_market_and_candle_crud_upsert_stream(self):
        markets = self.repos.markets
        market = markets.create(
            symbol="EURUSD",
            market_type=MarketType.FOREX,
            description="Euro/US Dollar",
            base_currency="EUR",
            quote_currency="USD",
            digits=5,
            point=0.00001,
        )
        self.assertIsNotNone(market.market_id)
        found = markets.find_by_symbol("EURUSD")
        self.assertIsNotNone(found)

        candles = self.repos.candles
        start = datetime(2024, 1, 1, 0, 0, 0)
        payload = []
        for i in range(25):
            ts = start + timedelta(minutes=15 * i)
            payload.append(
                {
                    "symbol": "EURUSD",
                    "timeframe": "M15",
                    "timestamp": ts,
                    "open": 1.1 + i * 0.0001,
                    "high": 1.1 + i * 0.0001 + 0.0002,
                    "low": 1.1 + i * 0.0001 - 0.0002,
                    "close": 1.1 + i * 0.0001 + 0.00005,
                    "volume": 100 + i,
                    "market_id": market.market_id,
                    "tick_volume": 100 + i,
                    "status": "active",
                }
            )
        inserted = candles.bulk_upsert(payload, trusted_source=True)
        self.assertGreaterEqual(inserted, 25)

        # upsert again should not explode
        candles.bulk_upsert(payload[:5], trusted_source=True)

        streamed = list(
            candles.stream_candles(
                "EURUSD",
                "M15",
                start,
                start + timedelta(hours=10),
                batch_size=10,
            )
        )
        self.assertGreaterEqual(len(streamed), 25)
        self.assertTrue(all(c.symbol == "EURUSD" for c in streamed))

    def test_tick_repository_stream(self):
        ticks = self.repos.ticks
        start = datetime(2024, 1, 1, 12, 0, 0)
        rows = []
        for i in range(40):
            ts = start + timedelta(seconds=i)
            rows.append(
                {
                    "symbol": "EURUSD",
                    "timestamp": ts,
                    "bid": 1.1000 + i * 0.00001,
                    "ask": 1.1002 + i * 0.00001,
                    "volume": 1.0,
                }
            )
        count = ticks.bulk_upsert(rows)
        self.assertGreaterEqual(count, 40)
        streamed = list(
            ticks.stream_ticks("EURUSD", start, start + timedelta(minutes=5), batch_size=15)
        )
        self.assertGreaterEqual(len(streamed), 40)

    def test_transactions(self):
        with self.repos.markets.transaction():
            self.repos.markets.create(
                symbol="GBPUSD",
                market_type=MarketType.FOREX,
                description="Cable",
            )
        found = self.repos.markets.find_by_symbol("GBPUSD")
        self.assertIsNotNone(found)

    def test_ai_adapters_stream_without_ai_changes(self):
        markets = self.repos.markets
        market = markets.create(symbol="USDJPY", market_type=MarketType.FOREX)
        candles = self.repos.candles
        start = datetime(2024, 2, 1)
        payload = [
            {
                "symbol": "USDJPY",
                "timeframe": "H1",
                "timestamp": start + timedelta(hours=i),
                "open": 150.0 + i,
                "high": 150.2 + i,
                "low": 149.8 + i,
                "close": 150.1 + i,
                "volume": 10,
                "market_id": market.market_id,
                "status": "active",
            }
            for i in range(12)
        ]
        candles.bulk_upsert(payload, trusted_source=True)

        adapter = CandleRepositoryAdapter(candles)
        out = list(
            adapter.stream_candles(
                "USDJPY",
                "H1",
                start,
                start + timedelta(hours=20),
                batch_size=5,
            )
        )
        self.assertEqual(len(out), 12)
        self.assertIn("open", out[0])
        self.assertIsInstance(out[0]["timestamp"], datetime)

        # ticks adapter
        tick_repo = self.repos.ticks
        tick_repo.bulk_upsert(
            [
                {
                    "symbol": "USDJPY",
                    "timestamp": start + timedelta(seconds=i),
                    "bid": 150.0,
                    "ask": 150.02,
                }
                for i in range(5)
            ]
        )
        tick_adapter = TickRepositoryAdapter(tick_repo)
        ticks = list(
            tick_adapter.stream_ticks("USDJPY", start, start + timedelta(minutes=1))
        )
        self.assertEqual(len(ticks), 5)


if __name__ == "__main__":
    unittest.main()
