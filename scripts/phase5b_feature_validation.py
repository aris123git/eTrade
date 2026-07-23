#!/usr/bin/env python3
"""
scripts/phase5b_feature_validation.py - Phase 5b advanced-feature validation.

Re-runs walk-forward with microstructure / regime / correlation / session /
volatility enabled, compares vs Phase 5a, and writes:
  - ai/validation/phase5b_report.json
  - ai/validation/phase5b_vs_phase5a.json
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
    parser = argparse.ArgumentParser(description="Phase 5b feature-driven edge validation")
    parser.add_argument("--symbols", default="EURUSD,GBPUSD,USDJPY,XAUUSD")
    parser.add_argument("--timeframes", default="M15,H1,H4")
    parser.add_argument("--start", default="2019-01-01")
    parser.add_argument("--end", default="2024-12-31")
    parser.add_argument("--report", default="ai/validation/phase5b_report.json")
    parser.add_argument("--comparison", default="ai/validation/phase5b_vs_phase5a.json")
    parser.add_argument("--phase5a-report", default="ai/validation/phase5a_report.json")
    parser.add_argument(
        "--peers",
        default="US30,SPX500,XAUUSD,USOIL,USDJPY,US10Y",
        help="Cross-asset peer symbols for correlation features",
    )
    parser.add_argument(
        "--model",
        default="random_forest",
        help="Model type (default random_forest to match Phase 5a; lightgbm also supported)",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-multi-tf", action="store_true")
    parser.add_argument("--no-d1", action="store_true")
    parser.add_argument("--fast", action="store_true", help="EURUSD + H1/D1 only")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    from ai.validation.phase5b_validator import (
        Phase5bValidator,
        format_comparison_table,
    )

    symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
    timeframes = [t.strip().upper() for t in args.timeframes.split(",") if t.strip()]
    peers = [s.strip().upper() for s in args.peers.split(",") if s.strip()]
    if args.fast:
        symbols = symbols[:1]
        timeframes = [t for t in timeframes if t in {"H1", "D1", "H4"}] or ["H1", "D1"]

    validator = Phase5bValidator(
        symbols=symbols,
        timeframes=timeframes,
        start_date=args.start,
        end_date=args.end,
        report_path=Path(args.report),
        comparison_path=Path(args.comparison),
        phase5a_report_path=Path(args.phase5a_report),
        peer_symbols=peers,
        compare_vs_phase5a=True,
        model_type=str(args.model).lower(),
        random_seed=int(args.seed),
    )
    report = validator.run(
        include_multi_tf=not args.no_multi_tf and not args.fast,
        include_d1=not args.no_d1,
    )

    table = format_comparison_table(report.get("deltas") or [])
    print(table)
    print()
    decision = report.get("decision") or {}
    print(
        json.dumps(
            {
                "report": str(args.report),
                "comparison": str(args.comparison),
                "decision": decision.get("decision"),
                "proceed_to_phase_5c": decision.get("proceed_to_phase_5c"),
                "symbols_for_phase_5c": decision.get("symbols_for_phase_5c"),
                "answer": decision.get("answer"),
                "series_count": len(report.get("series") or []),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
