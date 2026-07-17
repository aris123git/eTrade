#!/usr/bin/env python3
"""
scripts/prove_edge.py — Evidence-first entry point for eTrade

Objective: prove whether eTrade has a real statistical edge.

Daily routine (no architecture expansion):
  1. Download every candle/tick the broker allows
  2. Repair gaps
  3. Train / validate / reject weak models
  4. Paper trade
  5. Store results into the research DB + edge_evidence.json
  6. Sleep until the next run

Usage:
  # One evidence day (Windows + MT5 recommended)
  python3 scripts/prove_edge.py --once \\
      --symbols EURUSD,XAUUSD,GBPUSD,USDJPY \\
      --timeframes M1,M5,M15,H1,H4,D1

  # Continuous (resume tomorrow from the same DB)
  python3 scripts/prove_edge.py --forever --interval 86400

  # CSV broker archives (no MT5)
  python3 scripts/prove_edge.py --once --no-mt5 \\
      --csv-broker Export=/path/to/ohlcv

Never invents bars. Never enables live trading.
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
    p = argparse.ArgumentParser(description="Prove whether eTrade has a real edge")
    mode = p.add_mutually_exclusive_group()
    mode.add_argument("--once", action="store_true", help="Run one evidence day (default)")
    mode.add_argument("--forever", action="store_true", help="Loop daily until stopped")
    p.add_argument("--days", type=int, default=None, help="Max days with --forever")
    p.add_argument("--interval", type=float, default=86400.0, help="Seconds between days")
    p.add_argument("--symbols", type=str, default=DEFAULT_SYMBOLS)
    p.add_argument("--timeframes", type=str, default=DEFAULT_TIMEFRAMES)
    p.add_argument("--models", type=str, default="random_forest,lightgbm,xgboost")
    p.add_argument("--markets", type=str, default="FOREX,METALS,INDICES,CRYPTO,ENERGY")
    p.add_argument("--history-start", type=str, default="2005-01-01")
    p.add_argument("--candle-limit", type=int, default=20000)
    p.add_argument("--db", type=str, default=None)
    p.add_argument("--artifacts", type=str, default="ai_artifacts/edge_proof")
    p.add_argument("--csv-broker", action="append", default=[], help="name=dir")
    p.add_argument("--no-mt5", action="store_true")
    p.add_argument("--no-ticks", action="store_true")
    p.add_argument("--verbose", action="store_true")
    return p.parse_args()


def main() -> int:
    args = _parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

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
    engine = create_edge_proof_engine(platform)

    print()
    print("=" * 64)
    print("eTrade Edge Proof Mode")
    print("=" * 64)
    print("Objective: prove a real statistical edge on real market data.")
    print("Loop: download → repair → train → validate → paper → store → sleep")
    print("Live trading: DISABLED")
    print(f"symbols={len(symbols)} timeframes={timeframes}")
    print(f"db={db_path}")
    print(f"evidence={engine.evidence_path}")
    print()

    if args.forever:
        evidence = engine.run_forever(max_days=args.days, sleep_seconds=float(args.interval))
        print(json.dumps(evidence.to_dict(), indent=2, default=str)[:4000])
        return 0 if evidence.scientific_claim_allowed else 2

    result = engine.run_day()
    evidence = result["evidence"]
    print(json.dumps({
        "runs_completed": evidence["runs_completed"],
        "scientific_claim_allowed": evidence["scientific_claim_allowed"],
        "edge_demonstrated": evidence["edge_demonstrated"],
        "reason": evidence["reason"],
        "archive": {
            "candles": (evidence.get("archive") or {}).get("candles"),
            "ticks": (evidence.get("archive") or {}).get("ticks"),
            "symbols": (evidence.get("archive") or {}).get("symbols"),
        },
        "experiments": evidence.get("experiments"),
        "evidence_path": str(engine.evidence_path),
    }, indent=2, default=str))

    # Exit 0 = day ran; 2 = no scientific claim yet (expected early on)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
