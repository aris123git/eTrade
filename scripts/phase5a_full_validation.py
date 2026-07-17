#!/usr/bin/env python3
"""
scripts/phase5a_full_validation.py - Phase 5a historical validation runner.

Downloads real market history into CandleRepository (Yahoo when MT5 unavailable),
runs purged walk-forward backtests across symbols/timeframes, and writes
ai/validation/phase5a_report.json.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main() -> int:
    parser = argparse.ArgumentParser(description="Phase 5a full historical validation")
    parser.add_argument("--symbols", default="EURUSD,GBPUSD,USDJPY,XAUUSD")
    parser.add_argument("--timeframes", default="M15,H1,H4")
    parser.add_argument("--start", default="2019-01-01")
    parser.add_argument("--end", default="2024-12-31")
    parser.add_argument("--report", default="ai/validation/phase5a_report.json")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-multi-tf", action="store_true")
    parser.add_argument("--no-d1", action="store_true", help="Skip D1 long-horizon series")
    parser.add_argument("--fast", action="store_true", help="Symbols=EURUSD only, skip multi-TF")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    from ai.validation.phase5a_validator import Phase5aValidator, WalkForwardBacktester

    symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
    timeframes = [t.strip().upper() for t in args.timeframes.split(",") if t.strip()]
    if args.fast:
        symbols = symbols[:1]
        timeframes = [t for t in timeframes if t in {"H1", "D1", "H4"}] or ["H1", "D1"]

    backtester = WalkForwardBacktester(random_seed=int(args.seed))
    validator = Phase5aValidator(
        backtester=backtester,
        symbols=symbols,
        timeframes=timeframes,
        start_date=args.start,
        end_date=args.end,
        report_path=Path(args.report),
    )
    report = validator.run(
        include_multi_tf=not args.no_multi_tf and not args.fast,
        include_d1=not args.no_d1,
    )

    summary = report.get("summary") or {}
    print(json.dumps(
        {
            "report": str(args.report),
            "proceed_to_phase_5b": summary.get("proceed_to_phase_5b"),
            "answer": summary.get("answer"),
            "symbols_with_edge": summary.get("symbols_with_edge"),
            "series_count": len(report.get("series") or []),
        },
        indent=2,
    ))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
