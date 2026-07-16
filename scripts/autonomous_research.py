#!/usr/bin/env python3
"""
scripts/autonomous_research.py — Autonomous Quant Research Platform CLI

Runs the self-improving research loop:

  Wake → Collect → Validate → Repair → Learn → Keep improvements →
  Backtest → Paper trade → Hypotheses → Report → Sleep

The AI automates process. It cannot invent market history.
Raw bars must come from MT5 / Dukascopy / Polygon / Binance / IB / CSV.

Examples:
  python3 scripts/autonomous_research.py --once
  python3 scripts/autonomous_research.py --symbols EURUSD,XAUUSD --cycles 3
  python3 scripts/autonomous_research.py --forever --interval 3600
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ai.config.settings import AIConfig, ResearchConfig
from ai.research.platform import create_research_platform
from database.core.connection import DatabaseManager
from database.indexes import create_indexes
from database.migrations import apply_migrations
from database.schema import create_schema
from database.seed import seed


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Autonomous Quant Research Platform")
    p.add_argument("--once", action="store_true", help="Run a single research cycle (default)")
    p.add_argument("--forever", action="store_true", help="Repeat cycles until stopped")
    p.add_argument("--cycles", type=int, default=None, help="Max cycles when using --forever")
    p.add_argument("--interval", type=float, default=None, help="Seconds between cycles")
    p.add_argument("--symbols", type=str, default="EURUSD", help="Comma-separated symbols")
    p.add_argument("--timeframes", type=str, default="M15,H1", help="Comma-separated timeframes")
    p.add_argument("--models", type=str, default="random_forest", help="Comma-separated model candidates")
    p.add_argument("--metric", type=str, default="test_f1", help="Primary promotion metric")
    p.add_argument("--min-improvement", type=float, default=0.005)
    p.add_argument("--candle-limit", type=int, default=2000)
    p.add_argument("--db", type=str, default=None, help="SQLite database path")
    p.add_argument("--artifacts", type=str, default="ai_artifacts", help="Artifact root")
    p.add_argument("--skip-collect", action="store_true", help="Skip download stage")
    p.add_argument("--require-validated", action="store_true", help="Fail if history validation fails")
    p.add_argument("--no-backtest", action="store_true")
    p.add_argument("--no-paper", action="store_true")
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

    artifact_root = Path(args.artifacts)
    artifact_root.mkdir(parents=True, exist_ok=True)
    db_path = Path(args.db) if args.db else artifact_root / "research_market.db"

    db = DatabaseManager(db_path=db_path)
    create_schema(db)
    create_indexes(db)
    seed(db)
    apply_migrations(db)

    config = AIConfig()
    config.symbols = symbols
    config.timeframes = timeframes
    config.primary_timeframe = timeframes[0]
    config.features.multi_timeframes = list(dict.fromkeys([*timeframes, "H4"]))
    config.model.model_type = models[0]
    config.storage.root_dir = artifact_root
    config.data.database_path = str(db_path)
    config.data.auto_download = True
    config.data.include_mt5 = True
    config.data.allow_synthetic_fallback = True
    config.data.min_bars = min(500, max(100, args.candle_limit // 4))
    config.data.require_validated = bool(args.require_validated)

    research = ResearchConfig(
        enabled=True,
        sleep_seconds=float(args.interval or 0.0),
        cycle_interval_seconds=float(args.interval or 86_400.0),
        max_cycles=args.cycles,
        skip_collect=bool(args.skip_collect),
        require_validated=bool(args.require_validated),
        model_candidates=models,
        candle_limit=int(args.candle_limit),
        primary_metric=args.metric,
        min_improvement=float(args.min_improvement),
        run_backtest=not args.no_backtest,
        run_paper_trade=not args.no_paper,
    )
    config.research = research

    platform = create_research_platform(
        config=config,
        research=research,
        artifact_root=artifact_root,
    )
    # Share prepared DB with the data service
    if platform.pipeline.data_service is not None:
        platform.pipeline.data_service.db = db

    print()
    print("Autonomous Quant Research Platform")
    print("  Automates process. Does not invent market history.")
    print(f"  symbols={symbols} timeframes={timeframes} models={models}")
    print(f"  db={db_path}")
    print()

    if args.forever:
        reports = platform.run_forever(max_cycles=args.cycles)
        for report in reports:
            report.print_summary()
        return 0 if reports and reports[-1].status == "ok" else 1

    report = platform.run_cycle()
    report.print_summary()
    return 0 if report.status == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
