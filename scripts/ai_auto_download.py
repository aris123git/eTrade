#!/usr/bin/env python3
"""
scripts/ai_auto_download.py

Let the AI download every symbol × timeframe it needs by itself.

Uses AIConfig.symbols, AIConfig.timeframes, primary_timeframe, and
features.multi_timeframes. Tries MT5 / CSV brokers first; falls back to
synthetic bars when allow_synthetic_fallback=True (default).

Examples:
  python3 scripts/ai_auto_download.py
  python3 scripts/ai_auto_download.py --symbols EURUSD,GBPUSD --timeframes M15,H1,H4,D1
  python3 scripts/ai_auto_download.py --csv-broker Pepperstone=data/brokers/pepperstone
  python3 scripts/ai_auto_download.py --brokers-config config/brokers.example.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ai.config.settings import AIConfig, create_ai_config
from ai.data.auto_download import AIMarketDataService


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="AI auto-download of required market data")
    p.add_argument("--symbols", type=str, default="", help="Comma-separated symbols")
    p.add_argument("--timeframes", type=str, default="", help="Comma-separated timeframes")
    p.add_argument("--primary-timeframe", type=str, default="")
    p.add_argument("--min-bars", type=int, default=0)
    p.add_argument("--years", type=int, default=0)
    p.add_argument("--brokers-config", type=str, default="")
    p.add_argument(
        "--csv-broker",
        action="append",
        default=[],
        metavar="NAME=DIR",
        help="CSV broker source (repeatable)",
    )
    p.add_argument("--no-synthetic", action="store_true", help="Disable synthetic fallback")
    p.add_argument("--force", action="store_true", help="Re-download even if coverage OK")
    p.add_argument("--db", type=str, default="", help="SQLite database path")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    overrides: dict = {}
    if args.symbols.strip():
        overrides["symbols"] = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
    if args.timeframes.strip():
        overrides["timeframes"] = [t.strip().upper() for t in args.timeframes.split(",") if t.strip()]
    if args.primary_timeframe.strip():
        overrides["primary_timeframe"] = args.primary_timeframe.strip().upper()

    config = create_ai_config(**overrides) if overrides else AIConfig()
    config.data.auto_download = True
    if args.min_bars > 0:
        config.data.min_bars = args.min_bars
    if args.years > 0:
        config.data.years = args.years
    if args.brokers_config.strip():
        config.data.brokers_config = args.brokers_config.strip()
    if args.db.strip():
        config.data.database_path = args.db.strip()
    if args.no_synthetic:
        config.data.allow_synthetic_fallback = False
    for item in args.csv_broker:
        if "=" not in item:
            print(f"Invalid --csv-broker value: {item}", file=sys.stderr)
            return 2
        name, directory = item.split("=", 1)
        config.data.csv_brokers[name.strip()] = directory.strip()

    service = AIMarketDataService(config=config)
    print("AI required symbols:", service.required_symbols())
    print("AI required timeframes:", service.required_timeframes())
    result = service.ensure(force=args.force)
    payload = {
        "source": result.source,
        "ok": result.ok,
        "symbols": result.symbols,
        "timeframes": result.timeframes,
        "downloaded": result.downloaded,
        "synthetic_filled": result.synthetic_filled,
        "gaps_after": [
            {
                "symbol": g.symbol,
                "timeframe": g.timeframe,
                "bars": g.bars,
                "required": g.required,
                "missing": g.missing,
            }
            for g in result.gaps_after
        ],
    }
    print(json.dumps(payload, indent=2))
    return 0 if result.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
