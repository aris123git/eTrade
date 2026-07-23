#!/usr/bin/env python3
"""
scripts/autonomous_research.py — Phase 4 Autonomous Quant Research Engine

Cadence:
  --hourly / --daily / --weekly / --monthly / --once (full cycle)
  --scheduler  run calendar jobs in-process

Never invents market history. Uses MT5/CSV broker sources only.
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ai.config.settings import AIConfig, ResearchConfig
from ai.research.autonomous_scheduler import SchedulePlan
from ai.research.platform import create_research_platform
from database.core.connection import DatabaseManager
from database.indexes import create_indexes
from database.migrations import apply_migrations
from database.schema import create_schema
from database.seed import seed


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Phase 4 Autonomous Quant Research Engine")
    mode = p.add_mutually_exclusive_group()
    mode.add_argument("--once", action="store_true", help="Full research cycle (default)")
    mode.add_argument("--hourly", action="store_true")
    mode.add_argument("--daily", action="store_true")
    mode.add_argument("--weekly", action="store_true")
    mode.add_argument("--monthly", action="store_true")
    mode.add_argument("--scheduler", action="store_true", help="Run H/D/W/M scheduler")
    mode.add_argument("--forever", action="store_true")
    p.add_argument("--cycles", type=int, default=None)
    p.add_argument("--interval", type=float, default=None)
    p.add_argument("--symbols", type=str, default="EURUSD")
    p.add_argument("--timeframes", type=str, default="M15,H1")
    p.add_argument("--models", type=str, default="random_forest")
    p.add_argument("--metric", type=str, default="test_f1")
    p.add_argument("--candle-limit", type=int, default=3000)
    p.add_argument("--db", type=str, default=None)
    p.add_argument("--artifacts", type=str, default="ai_artifacts")
    p.add_argument("--csv-broker", action="append", default=[], help="name=dir for CSV broker source")
    p.add_argument("--no-mt5", action="store_true")
    p.add_argument("--allow-synthetic", action="store_true", help="Dev-only; disabled by default")
    p.add_argument("--skip-collect", action="store_true")
    p.add_argument("--no-strict-validation", action="store_true")
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
    csv_brokers = {}
    for item in args.csv_broker:
        if "=" in item:
            name, path = item.split("=", 1)
            csv_brokers[name.strip()] = path.strip()

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
    config.data.include_mt5 = not args.no_mt5
    config.data.csv_brokers = csv_brokers
    config.data.allow_synthetic_fallback = bool(args.allow_synthetic)
    config.data.min_bars = min(500, max(100, args.candle_limit // 4))
    config.data.require_validated = not args.allow_synthetic

    research = ResearchConfig(
        enabled=True,
        sleep_seconds=float(args.interval or 0.0),
        cycle_interval_seconds=float(args.interval or 86_400.0),
        max_cycles=args.cycles,
        skip_collect=bool(args.skip_collect),
        require_validated=not args.allow_synthetic,
        allow_synthetic=bool(args.allow_synthetic),
        model_candidates=models,
        candle_limit=int(args.candle_limit),
        primary_metric=args.metric,
        download_ticks=not args.no_ticks,
        run_strict_validation=not args.no_strict_validation,
    )
    config.research = research

    platform = create_research_platform(
        config=config,
        research=research,
        db=db,
        artifact_root=artifact_root,
    )

    print("Phase 4 Autonomous Quant Research Engine")
    print("  Automates process. Does not invent market history.")
    print(f"  symbols={symbols} models={models} db={db_path}")
    print(f"  synthetic_allowed={research.allow_synthetic} strict_validation={research.run_strict_validation}")

    if args.scheduler:
        sched = platform.start_scheduler(SchedulePlan(run_immediately=True))
        print("Scheduler running (hourly/daily/weekly/monthly). Ctrl+C to stop.")
        try:
            while True:
                time.sleep(5)
                sched.run_pending()
        except KeyboardInterrupt:
            sched.stop()
            return 0

    if args.hourly:
        print(platform.run_hourly())
        return 0
    if args.daily:
        print(platform.run_daily())
        return 0
    if args.weekly:
        print(platform.run_weekly())
        return 0
    if args.monthly:
        report = platform.run_cycle()
        report.print_summary()
        return 0 if report.status == "ok" else 1
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
