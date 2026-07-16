"""
main.py - eTrade collector entrypoint

Discovers ALL currency pairs from MT5, then downloads ALL configured
timeframes (M1..MN1) for each pair. Falls back to offline demo mode when
MetaTrader5 is unavailable (e.g. Linux CI).
"""

from __future__ import annotations

import argparse
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
    pairs = ["EURUSD", "GBPUSD", "USDJPY", "AUDUSD"]
    for symbol in pairs:
        db.execute(
            """
            INSERT OR IGNORE INTO markets (
                symbol, name, category, market_type, status, active,
                description, digits, point, base_currency, quote_currency,
                metadata, created_at, updated_at
            ) VALUES (
                ?, ?, 'FOREX', 'FOREX', 'active', 1,
                ?, 5, 0.00001, ?, ?,
                '{}', ?, ?
            )
            """,
            (
                symbol,
                symbol,
                f"Demo {symbol}",
                symbol[:3],
                symbol[3:6],
                now,
                now,
            ),
        )
    db.commit()

    for symbol in pairs:
        row = db.fetch_one("SELECT market_id FROM markets WHERE symbol = ?", (symbol,))
        market_id = row["market_id"] if row else None
        start = datetime.utcnow() - timedelta(hours=50)
        price = 1.1000
        for i in range(100):
            ts = start + timedelta(minutes=15 * i)
            o = price
            c = price + ((-1) ** i) * 0.0002
            h = max(o, c) + 0.0001
            l = min(o, c) - 0.0001
            for timeframe in ("M15", "H1"):
                db.execute(
                    """
                    INSERT OR IGNORE INTO candles (
                        candle_uuid, symbol, timeframe, timestamp, open, high, low, close,
                        volume, market_id, tick_volume, status, metadata, created_at, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, 100, ?, 100, 'active', '{}', ?, ?)
                    """,
                    (
                        str(uuid4()),
                        symbol,
                        timeframe,
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


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="eTrade MT5 full collector")
    parser.add_argument(
        "--all-symbols",
        action="store_true",
        help="Discover every MT5 symbol (not only currency pairs)",
    )
    parser.add_argument(
        "--currency-pairs-only",
        action="store_true",
        default=True,
        help="Discover only FX currency pairs (default)",
    )
    parser.add_argument(
        "--timeframes",
        type=str,
        default="",
        help="Comma-separated timeframes to download (default: all configured)",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)

    print("=" * 60)
    print("eTrade Collector — ALL currency pairs × ALL timeframes")
    print("=" * 60)

    config = Config()
    config.ensure_directories()

    db = Database()
    create_schema(db)
    create_indexes(db)
    seed(db)
    print("Database ready.")

    # Resolve timeframe map
    if args.timeframes.strip():
        wanted = {t.strip().upper() for t in args.timeframes.split(",") if t.strip()}
        timeframe_map = {k: v for k, v in TIMEFRAMES.items() if k in wanted}
        if not timeframe_map:
            print(f"No valid timeframes in: {args.timeframes}")
            print(f"Available: {', '.join(TIMEFRAMES)}")
            db.close()
            return 2
    else:
        timeframe_map = dict(TIMEFRAMES)

    print(f"Timeframes to download ({len(timeframe_map)}): {', '.join(timeframe_map)}")

    mt5_ready = False
    if mt5 is not None:
        try:
            mt5_ready = bool(mt5.initialize())
        except Exception as exc:  # pragma: no cover
            print(f"MT5 initialize failed: {exc}")
            mt5_ready = False

    if mt5_ready:
        manager = SymbolManager(db)
        currency_only = not args.all_symbols
        saved = manager.discover(currency_pairs_only=currency_only, select_all=True)
        print(f"Markets discovered: {len(saved)}")
        if saved:
            preview = ", ".join(saved[:20])
            more = "" if len(saved) <= 20 else f" ... (+{len(saved) - 20} more)"
            print(f"Examples: {preview}{more}")

        print()
        print("=" * 60)
        print(f"Downloading history for {len(saved)} markets × {len(timeframe_map)} timeframes")
        print("=" * 60)

        collector = Downloader(db, timeframe_map)
        ok = collector.download_all()
        mt5.shutdown()
        print("Download finished." if ok else "Download finished with errors.")
    else:
        print("MetaTrader5 unavailable — running offline bootstrap mode.")
        print("On a Windows host with MT5 terminal open, re-run to download all FX pairs.")
        _seed_demo_market(db)
        print("Demo FX markets + candles inserted.")

    markets = db.fetch_all(
        """
        SELECT symbol, COALESCE(category, market_type, '') AS category
        FROM markets
        WHERE COALESCE(active, 1) = 1
        ORDER BY symbol
        """
    )
    candles = db.fetch_one("SELECT COUNT(*) AS c FROM candles")
    by_tf = db.fetch_all(
        """
        SELECT timeframe, COUNT(*) AS c
        FROM candles
        GROUP BY timeframe
        ORDER BY timeframe
        """
    )
    print()
    print(f"Active markets: {len(markets)} | Candles: {candles['c'] if candles else 0}")
    if by_tf:
        print("Candles by timeframe:")
        for row in by_tf:
            print(f"  {row['timeframe']}: {row['c']}")

    db.close()
    print("Finished.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
