#!/usr/bin/env python3
"""
scripts/download_history.py - Production historical data engine CLI

Objective:
  "I can download any market, any timeframe, any date range from MT5
   into my database with one command."

Examples:
  python3 scripts/download_history.py \\
      --markets FOREX,METALS,INDICES,CRYPTO,ENERGY \\
      --symbols ALL \\
      --timeframes M1,M5,M15,H1,H4,D1 \\
      --start 2010-01-01 --end today

  python3 scripts/download_history.py --symbols EURUSD,XAUUSD --timeframes M15,H1
  python3 scripts/download_history.py --validate-only
  python3 scripts/download_history.py --csv-broker Demo=data/brokers/demo --no-mt5
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from database.core.connection import DatabaseManager
from collector.history_engine import HistoricalDataEngine, download_history
from collector.history_validator import HistoryValidator


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Production historical market data engine")
    p.add_argument("--brokers", type=str, default="ALL", help="ALL or comma-separated broker names")
    p.add_argument(
        "--markets",
        type=str,
        default="FOREX,METALS,INDICES,CRYPTO,ENERGY",
        help="ALL or FOREX,METALS,INDICES,CRYPTO,ENERGY",
    )
    p.add_argument("--symbols", type=str, default="ALL", help="ALL or EURUSD,XAUUSD,...")
    p.add_argument(
        "--timeframes",
        type=str,
        default="M1,M5,M15,H1,H4,D1",
        help="Comma-separated timeframes",
    )
    p.add_argument("--start", type=str, default="2010-01-01")
    p.add_argument("--end", type=str, default="today")
    p.add_argument("--no-resume", action="store_true", help="Force full re-download window")
    p.add_argument("--brokers-config", type=str, default="")
    p.add_argument("--csv-broker", action="append", default=[], metavar="NAME=DIR")
    p.add_argument("--no-mt5", action="store_true")
    p.add_argument("--db", type=str, default="", help="SQLite path (default from config)")
    p.add_argument("--validate", action="store_true", help="Run validator after download")
    p.add_argument("--validate-only", action="store_true", help="Only validate existing DB")
    p.add_argument("--inventory", action="store_true", help="Print market inventory tree")
    p.add_argument("--min-bars", type=int, default=100)
    p.add_argument("--json-out", type=str, default="", help="Write report JSON to path")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)

    from core.config import DATABASE_PATH

    db_path = Path(args.db) if args.db.strip() else Path(DATABASE_PATH)
    db = DatabaseManager(db_path=db_path)

    csv_brokers = {}
    for item in args.csv_broker:
        if "=" not in item:
            print(f"Invalid --csv-broker: {item}", file=sys.stderr)
            return 2
        name, directory = item.split("=", 1)
        csv_brokers[name.strip()] = directory.strip()

    engine = HistoricalDataEngine(
        db,
        brokers_config=args.brokers_config.strip() or None,
        include_mt5=not args.no_mt5,
        csv_brokers=csv_brokers,
    )

    payload: dict = {}

    if not args.validate_only:
        print("=" * 70)
        print("Historical Data Engine — download_history")
        print("=" * 70)
        print(f"brokers={args.brokers} markets={args.markets}")
        print(f"symbols={args.symbols} timeframes={args.timeframes}")
        print(f"start={args.start} end={args.end} resume={not args.no_resume}")
        print(f"db={db_path}")
        print()

        result = engine.download_history(
            brokers=args.brokers,
            markets=args.markets,
            symbols=args.symbols,
            timeframes=[t.strip() for t in args.timeframes.split(",") if t.strip()],
            start=args.start,
            end=args.end,
            resume=not args.no_resume,
        )
        payload["download"] = result.summary()
        payload["inventory"] = result.inventory
        print()
        print("Download summary:", json.dumps(result.summary(), indent=2))
        if args.inventory or True:
            print()
            print("Inventory:")
            for cat, symbols in result.inventory.items():
                print(f"  {cat}")
                for sym, tfs in list(symbols.items())[:50]:
                    print(f"    {sym}: {', '.join(tfs)}")
                if len(symbols) > 50:
                    print(f"    ... (+{len(symbols) - 50} more symbols)")

    if args.validate or args.validate_only:
        print()
        print("=" * 70)
        print("History Validator")
        print("=" * 70)
        validator = HistoryValidator(db, min_bars=args.min_bars)
        symbols = None if args.symbols.upper() == "ALL" else [s.strip() for s in args.symbols.split(",")]
        timeframes = [t.strip() for t in args.timeframes.split(",") if t.strip()]
        report = validator.validate_all(symbols=symbols, timeframes=timeframes)
        report.print_summary()
        payload["validation"] = report.to_dict()
        if args.json_out.strip():
            Path(args.json_out).write_text(json.dumps(payload, indent=2), encoding="utf-8")
            print(f"Wrote {args.json_out}")
        db.close()
        return 0 if report.ok or (not args.validate_only and report.passed > 0) else 1

    if args.json_out.strip():
        Path(args.json_out).write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"Wrote {args.json_out}")

    # Always validate lightly after download when series exist
    if not args.validate_only and payload.get("download", {}).get("series", 0) > 0:
        validator = HistoryValidator(db, min_bars=max(1, args.min_bars // 10))
        report = validator.validate_all()
        payload["validation_quick"] = {
            "passed": report.passed,
            "failed": report.failed,
            "ok": report.ok,
        }
        print(f"Quick validation: PASS={report.passed} FAIL={report.failed}")

    db.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
