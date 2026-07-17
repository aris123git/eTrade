#!/usr/bin/env python3
"""
scripts/validate_and_research.py — Complete validation + autonomous edge research

No new models. No new indicators.

Mission:
  1) Continuously verify every platform component
  2) Download / repair real market history
  3) Train, validate, reject weak models
  4) Paper trade
  5) Accumulate honest edge evidence

Usage:
  # Verify components only
  python3 scripts/validate_and_research.py --verify-only

  # One full evidence day (verify → download → train → paper → store)
  python3 scripts/validate_and_research.py --once

  # Continuous autonomous research
  python3 scripts/validate_and_research.py --forever --interval 86400
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

from ai.config.settings import AIConfig, ResearchConfig
from ai.research.component_verification import verify_components
from ai.research.edge_proof import create_edge_proof_engine
from ai.research.platform import create_research_platform
from database.core.connection import DatabaseManager
from database.indexes import create_indexes
from database.migrations import apply_migrations
from database.schema import create_schema
from database.seed import seed


DEFAULT_SYMBOLS = "EURUSD,GBPUSD,USDJPY,USDCHF,AUDUSD,USDCAD,NZDUSD,XAUUSD,XAGUSD,US30,NAS100,GER40"
DEFAULT_TIMEFRAMES = "M1,M5,M15,H1,H4,D1"


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Validate components and prove statistical edge")
    mode = p.add_mutually_exclusive_group()
    mode.add_argument("--verify-only", action="store_true", help="Component verification only")
    mode.add_argument("--once", action="store_true", help="One evidence day (default)")
    mode.add_argument("--forever", action="store_true", help="Loop forever")
    p.add_argument("--days", type=int, default=None)
    p.add_argument("--interval", type=float, default=86400.0)
    p.add_argument("--symbols", type=str, default=DEFAULT_SYMBOLS)
    p.add_argument("--timeframes", type=str, default=DEFAULT_TIMEFRAMES)
    p.add_argument("--models", type=str, default="random_forest,lightgbm,xgboost")
    p.add_argument("--markets", type=str, default="FOREX,METALS,INDICES,CRYPTO,ENERGY")
    p.add_argument("--history-start", type=str, default="2005-01-01")
    p.add_argument("--candle-limit", type=int, default=20000)
    p.add_argument("--db", type=str, default=None)
    p.add_argument("--artifacts", type=str, default="ai_artifacts/edge_proof")
    p.add_argument("--csv-broker", action="append", default=[])
    p.add_argument("--no-mt5", action="store_true")
    p.add_argument("--no-ticks", action="store_true")
    p.add_argument("--verbose", action="store_true")
    return p.parse_args()


def _build_platform(args: argparse.Namespace):
    symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
    timeframes = [t.strip().upper() for t in args.timeframes.split(",") if t.strip()]
    models = [m.strip() for m in args.models.split(",") if m.strip()]
    markets = [m.strip().upper() for m in args.markets.split(",") if m.strip()]
    csv_brokers = {}
    for item in args.csv_broker:
        if "=" in item:
            name, path = item.split("=", 1)
            csv_brokers[name.strip()] = path.strip()

    artifact_root = Path(args.artifacts)
    artifact_root.mkdir(parents=True, exist_ok=True)
    db_path = Path(args.db) if args.db else artifact_root / "edge_market.db"

    db = DatabaseManager(db_path=db_path)
    create_schema(db)
    create_indexes(db)
    seed(db)
    apply_migrations(db)

    config = AIConfig()
    config.symbols = symbols
    config.timeframes = timeframes
    config.primary_timeframe = "M15" if "M15" in timeframes else timeframes[0]
    config.features.multi_timeframes = [t for t in ("M15", "H1", "H4") if t in timeframes] or timeframes[:3]
    config.model.model_type = models[0]
    config.storage.root_dir = artifact_root
    config.data.database_path = str(db_path)
    config.data.auto_download = True
    config.data.include_mt5 = not args.no_mt5
    config.data.csv_brokers = csv_brokers
    config.data.allow_synthetic_fallback = False
    config.data.require_validated = True
    config.data.min_bars = 500
    config.data.years = 20

    research = ResearchConfig(
        markets=markets,
        history_start=args.history_start,
        model_candidates=models,
        candle_limit=int(args.candle_limit),
        download_ticks=not args.no_ticks,
        allow_synthetic=False,
        require_validated=True,
        sleep_seconds=float(args.interval),
        cycle_interval_seconds=float(args.interval),
        max_cycles=args.days,
    )
    config.research = research

    platform = create_research_platform(
        config=config,
        research=research,
        db=db,
        artifact_root=artifact_root,
    )
    return platform, db, artifact_root


def main() -> int:
    args = _parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    print()
    print("=" * 64)
    print("eTrade Validation + Autonomous Research")
    print("=" * 64)
    print("No new models. No new indicators.")
    print("Verify components → download → train → validate → paper → evidence")
    print()

    platform, db, artifact_root = _build_platform(args)

    if args.verify_only:
        report = verify_components(db=db, config=platform.config)
        report.print_summary()
        out = artifact_root / "component_verification_latest.json"
        out.write_text(json.dumps(report.to_dict(), indent=2, default=str), encoding="utf-8")
        print(f"wrote {out}")
        return 0 if report.critical_ok else 1

    engine = create_edge_proof_engine(platform)
    print(f"db={platform.config.data.database_path}")
    print(f"evidence={engine.evidence_path}")
    print()

    if args.forever:
        evidence = engine.run_forever(max_days=args.days, sleep_seconds=float(args.interval))
        print(json.dumps(evidence.to_dict(), indent=2, default=str)[:5000])
        return 0 if evidence.components_ok else 1

    result = engine.run_day()
    evidence = result["evidence"]
    components = (result.get("day") or {}).get("components") or evidence.get("components") or {}
    if components:
        print(
            f"components: critical_ok={components.get('critical_ok')} "
            f"passed={components.get('passed')} failed={components.get('failed')}"
        )
    print(
        json.dumps(
            {
                "runs_completed": evidence["runs_completed"],
                "components_ok": evidence.get("components_ok"),
                "scientific_claim_allowed": evidence["scientific_claim_allowed"],
                "edge_demonstrated": evidence["edge_demonstrated"],
                "reason": evidence["reason"],
                "archive": {
                    "candles": (evidence.get("archive") or {}).get("candles"),
                    "ticks": (evidence.get("archive") or {}).get("ticks"),
                },
                "experiments": evidence.get("experiments"),
                "evidence_path": str(engine.evidence_path),
            },
            indent=2,
            default=str,
        )
    )
    if result.get("day", {}).get("aborted"):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
