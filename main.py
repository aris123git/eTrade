"""
main.py - eTrade collector entrypoint

Default: discover ALL currency pairs from MT5 and download ALL timeframes.
Also supports multi-broker mode (MT5 accounts + CSV broker exports) with
canonical symbol identity so instruments can be joined across brokers.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timedelta
from pathlib import Path
from uuid import uuid4

try:
    import MetaTrader5 as mt5
except ImportError:  # pragma: no cover
    mt5 = None

from core.config import TIMEFRAMES, Config
from core.symbol_identity import canonicalize
from database.database import Database
from database.schema import create_schema
from database.indexes import create_indexes
from database.seed import seed
from database.migrations import apply_migrations
from collector.symbol_manager import SymbolManager
from collector.downloader import Downloader
from collector.multi_broker import MultiBrokerCollector
from collector.broker_sources.registry import (
    BrokerSourceRegistry,
    build_default_registry,
    load_registry_from_config,
)
from collector.broker_sources.csv_source import CsvBrokerSource
from collector.broker_sources.mt5_source import MT5BrokerSource


def _ensure_broker(db: Database, name: str, **kwargs) -> int:
    collector = MultiBrokerCollector(db, BrokerSourceRegistry())
    return collector.ensure_broker(name, **kwargs)


def _seed_demo_market(db: Database) -> None:
    """Insert demo FX markets for two synthetic brokers when MT5 is unavailable."""
    now = datetime.utcnow().isoformat(timespec="seconds")
    broker_a = _ensure_broker(db, "DemoBrokerA", broker_type="demo", description="Offline demo A")
    broker_b = _ensure_broker(db, "DemoBrokerB", broker_type="demo", description="Offline demo B")

    # Same instruments, different broker naming conventions
    samples = [
        (broker_a, "EURUSD", "EURUSD"),
        (broker_a, "GBPUSD", "GBPUSD"),
        (broker_a, "US30", "US30"),
        (broker_b, "EURUSD.a", "EURUSD"),
        (broker_b, "GBPUSDm", "GBPUSD"),
        (broker_b, "DJ30", "US30"),
    ]

    from database.models.market_model import MarketModel

    mm = MarketModel(db)
    for broker_id, symbol, _canon in samples:
        ident = canonicalize(symbol)
        mm.add_market(
            symbol=symbol,
            category=ident.asset_class,
            description=f"Demo {symbol}",
            digits=5 if ident.asset_class == "FOREX" else 2,
            point=0.00001 if ident.asset_class == "FOREX" else 0.01,
            currency_base=ident.base_currency,
            currency_profit=ident.quote_currency,
            broker_id=broker_id,
            canonical_symbol=ident.canonical_symbol,
        )

    for broker_id, symbol, _ in samples:
        row = db.fetch_one(
            "SELECT market_id FROM markets WHERE broker_id = ? AND symbol = ?",
            (broker_id, symbol),
        )
        market_id = row["market_id"] if row else None
        start = datetime.utcnow() - timedelta(hours=50)
        price = 1.1000 if canonicalize(symbol).asset_class == "FOREX" else 39000.0
        for i in range(100):
            ts = start + timedelta(minutes=15 * i)
            # Slight broker skew so compare is non-zero but joinable
            skew = 0.0 if broker_id == broker_a else (0.00015 if price < 100 else 1.5)
            o = price
            c = price + ((-1) ** i) * (0.0002 if price < 100 else 2.0) + skew
            h = max(o, c) + (0.0001 if price < 100 else 1.0)
            l = min(o, c) - (0.0001 if price < 100 else 1.0)
            for timeframe in ("M15", "H1"):
                db.execute(
                    """
                    INSERT OR IGNORE INTO candles (
                        candle_uuid, symbol, timeframe, timestamp, open, high, low, close,
                        volume, market_id, broker_id, tick_volume, status, metadata, created_at, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, 100, ?, ?, 100, 'active', '{}', ?, ?)
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
                        broker_id,
                        now,
                        now,
                    ),
                )
            price = c - skew
    db.commit()


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="eTrade multi-broker collector")
    parser.add_argument(
        "--all-symbols",
        action="store_true",
        help="Discover every symbol (not only currency pairs)",
    )
    parser.add_argument(
        "--timeframes",
        type=str,
        default="",
        help="Comma-separated timeframes to download (default: all configured)",
    )
    parser.add_argument(
        "--brokers-config",
        type=str,
        default="",
        help="JSON config listing MT5 accounts and/or CSV broker folders",
    )
    parser.add_argument(
        "--csv-broker",
        action="append",
        default=[],
        metavar="NAME=DIR",
        help="Add a CSV broker source (repeatable), e.g. Pepperstone=data/pepperstone",
    )
    parser.add_argument(
        "--join-report",
        action="store_true",
        help="Print cross-broker instrument join report and exit",
    )
    parser.add_argument(
        "--compare",
        type=str,
        default="",
        help="Compare closes: CANONICAL,TIMEFRAME,BROKER_A,BROKER_B",
    )
    return parser.parse_args(argv)


def _build_registry(args: argparse.Namespace) -> BrokerSourceRegistry:
    if args.brokers_config.strip():
        registry = load_registry_from_config(args.brokers_config.strip())
    else:
        registry = build_default_registry(include_mt5=True)
    for item in args.csv_broker:
        if "=" not in item:
            raise SystemExit(f"Invalid --csv-broker value (expected NAME=DIR): {item}")
        name, directory = item.split("=", 1)
        registry.register(CsvBrokerSource(name=name.strip(), data_dir=directory.strip()))
    return registry


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)

    print("=" * 60)
    print("eTrade Collector — multi-broker + canonical symbol identity")
    print("=" * 60)

    config = Config()
    config.ensure_directories()

    db = Database()
    create_schema(db)
    seed(db)
    # Upgrade legacy DBs (canonical_symbol, broker-scoped uniqueness) before indexes
    applied = apply_migrations(db)
    if applied:
        print(f"Migrations applied: {', '.join(applied)}")
    create_indexes(db)
    print("Database ready.")

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

    print(f"Timeframes ({len(timeframe_map)}): {', '.join(timeframe_map)}")

    registry = _build_registry(args)
    multi = MultiBrokerCollector(db, registry)

    if args.join_report:
        report = multi.join_report(min_brokers=2)
        print(json.dumps(report, indent=2))
        db.close()
        return 0

    if args.compare.strip():
        parts = [p.strip() for p in args.compare.split(",")]
        if len(parts) != 4:
            print("Usage: --compare CANONICAL,TIMEFRAME,BROKER_A,BROKER_B")
            db.close()
            return 2
        rows = multi.compare_closes(parts[0], parts[1], parts[2], parts[3], limit=50)
        print(f"Compared {len(rows)} overlapping bars for {parts[0]} {parts[1]}")
        for row in rows[-10:]:
            print(
                f"  {row.timestamp}  {row.broker_a}={row.close_a:.5f}  "
                f"{row.broker_b}={row.close_b:.5f}  diff_bps={row.diff_bps:.2f}"
            )
        db.close()
        return 0

    currency_only = not args.all_symbols
    has_csv = bool(args.csv_broker) or bool(args.brokers_config.strip())

    # Multi-source path when CSV brokers or explicit config provided
    if has_csv:
        print(f"Sources: {', '.join(registry.names())}")
        inserted = multi.download_all(
            list(timeframe_map.keys()),
            currency_pairs_only=currency_only,
            years=5,
        )
        print(f"Inserted candles by broker: {inserted}")
    else:
        # Legacy single-MT5 path (still stamps broker + canonical)
        mt5_ready = False
        if mt5 is not None:
            try:
                mt5_ready = bool(mt5.initialize())
            except Exception as exc:  # pragma: no cover
                print(f"MT5 initialize failed: {exc}")
                mt5_ready = False

        if mt5_ready:
            source = MT5BrokerSource(name="MT5")
            # Reuse already-initialized terminal
            source._account_info = mt5.account_info()
            if source._account_info is not None:
                company = getattr(source._account_info, "company", None) or "MT5"
                server = getattr(source._account_info, "server", None) or ""
                source.name = f"{company}@{server}" if server else str(company)
            broker_id = multi.ensure_broker(
                source.name,
                broker_type="mt5",
                server=(source.broker_metadata() or {}).get("server"),
                metadata=source.broker_metadata(),
            )
            manager = SymbolManager(db, broker_id=broker_id)
            saved = manager.discover(currency_pairs_only=currency_only, select_all=True)
            print(f"Markets discovered: {len(saved)} (broker_id={broker_id})")
            collector = Downloader(db, timeframe_map)
            ok = collector.download_all()
            mt5.shutdown()
            print("Download finished." if ok else "Download finished with errors.")
        else:
            print("MetaTrader5 unavailable — offline multi-broker demo bootstrap.")
            print("Tip: use --csv-broker Name=dir or --brokers-config config.json for real multi-broker imports.")
            _seed_demo_market(db)
            print("Demo brokers + aliased symbols + candles inserted.")

    report = multi.join_report(min_brokers=2)
    markets = db.fetch_all(
        """
        SELECT b.name AS broker, m.symbol, m.canonical_symbol,
               COALESCE(m.category, m.market_type, '') AS category
        FROM markets m
        LEFT JOIN brokers b ON b.broker_id = m.broker_id
        WHERE COALESCE(m.active, 1) = 1
        ORDER BY m.canonical_symbol, b.name
        """
    )
    candles = db.fetch_one("SELECT COUNT(*) AS c FROM candles")
    print()
    print(f"Active markets: {len(markets)} | Candles: {candles['c'] if candles else 0}")
    print(f"Instruments on 2+ brokers: {report['instruments_with_multiple_brokers']}")
    if report["instruments"]:
        print("Cross-broker joins (canonical → broker symbols):")
        for item in report["instruments"][:20]:
            mapping = ", ".join(
                f"{m['broker_name']}:{m['broker_symbol']}" for m in item["markets"]
            )
            print(f"  {item['canonical_symbol']}: {mapping}")

    db.close()
    print("Finished.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
