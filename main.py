"""
main.py - eTrade collector entrypoint

Initializes the database layer, discovers markets from MT5 when available,
and downloads historical candles. On hosts without MetaTrader5, runs in
offline bootstrap mode so infrastructure can be verified end-to-end.
"""

from __future__ import annotations

import sys
from datetime import datetime, timedelta
from uuid import uuid4

try:
    import MetaTrader5 as mt5
except ImportError:  # pragma: no cover
    mt5 = None

from core.config import TIMEFRAMES, Config
from database.database import Database
from database.schema import create_schema
from database.indexes import create_indexes
from database.seed import seed
from collector.symbol_manager import SymbolManager
from collector.downloader import Downloader


def _seed_demo_market(db: Database) -> None:
    """Insert a demo EURUSD market + synthetic candles when MT5 is unavailable."""
    now = datetime.utcnow().isoformat(timespec="seconds")
    db.execute(
        """
        INSERT OR IGNORE INTO markets (
            symbol, name, category, market_type, status, active,
            description, digits, point, base_currency, quote_currency,
            metadata, created_at, updated_at
        ) VALUES (
            'EURUSD', 'EURUSD', 'FOREX', 'FOREX', 'active', 1,
            'Demo Euro/US Dollar', 5, 0.00001, 'EUR', 'USD',
            '{}', ?, ?
        )
        """,
        (now, now),
    )
    db.commit()
    row = db.fetch_one("SELECT market_id FROM markets WHERE symbol = 'EURUSD'")
    market_id = row["market_id"] if row else None
    start = datetime.utcnow() - timedelta(hours=50)
    price = 1.1000
    for i in range(200):
        ts = start + timedelta(minutes=15 * i)
        o = price
        c = price + ((-1) ** i) * 0.0002
        h = max(o, c) + 0.0001
        l = min(o, c) - 0.0001
        db.execute(
            """
            INSERT OR IGNORE INTO candles (
                candle_uuid, symbol, timeframe, timestamp, open, high, low, close,
                volume, market_id, tick_volume, status, metadata, created_at, updated_at
            ) VALUES (?, 'EURUSD', 'M15', ?, ?, ?, ?, ?, 100, ?, 100, 'active', '{}', ?, ?)
            """,
            (
                str(uuid4()),
                ts.isoformat(timespec="seconds"),
                o,
                h,
                l,
                c,
                market_id,
                now,
                now,
            ),
        )
        price = c
    db.commit()


def main() -> int:
    print("=" * 60)
    print("eTrade Collector")
    print("=" * 60)

    config = Config()
    config.ensure_directories()

    db = Database()
    create_schema(db)
    create_indexes(db)
    seed(db)
    print("Database ready.")

    mt5_ready = False
    if mt5 is not None:
        try:
            mt5_ready = bool(mt5.initialize())
        except Exception as exc:  # pragma: no cover
            print(f"MT5 initialize failed: {exc}")
            mt5_ready = False

    if mt5_ready:
        manager = SymbolManager(db)
        manager.discover()
        print("Markets discovered.")

        collector = Downloader(db, TIMEFRAMES)
        collector.download_all()
        mt5.shutdown()
    else:
        print("MetaTrader5 unavailable — running offline bootstrap mode.")
        _seed_demo_market(db)
        print("Demo market + candles inserted.")

    # Quick verification
    markets = db.fetch_all("SELECT symbol FROM markets WHERE COALESCE(active,1)=1")
    candles = db.fetch_one("SELECT COUNT(*) AS c FROM candles")
    print(f"Active markets: {len(markets)} | Candles: {candles['c'] if candles else 0}")

    db.close()
    print("Finished.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
