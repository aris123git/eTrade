#!/usr/bin/env python3
"""
scripts/run_paper_week.py - Train → signal → paper-trade ~1 week of bars.

Replays historical candles (H1 default: 168 bars ≈ 7 days). Does not sleep
for a calendar week. Uses CandleRepository when enough bars exist; otherwise
seeds a temporary repository with realistic OHLCV for offline verification.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _configure_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )


def _synthetic_candles(
    *,
    symbol: str,
    timeframe: str,
    n: int,
    start: datetime | None = None,
    seed: int = 42,
) -> List[Dict[str, Any]]:
    import numpy as np

    rng = np.random.default_rng(seed)
    tf = timeframe.upper()
    step = {
        "M1": timedelta(minutes=1),
        "M5": timedelta(minutes=5),
        "M15": timedelta(minutes=15),
        "M30": timedelta(minutes=30),
        "H1": timedelta(hours=1),
        "H4": timedelta(hours=4),
        "D1": timedelta(days=1),
    }.get(tf, timedelta(hours=1))
    ts = start or (datetime.now(timezone.utc) - step * n)
    price = 1.1000 if "USD" in symbol.upper() and symbol.upper().endswith("USD") else 100.0
    if symbol.upper() in {"US30", "DJ30"}:
        price = 34000.0
    candles: List[Dict[str, Any]] = []
    for i in range(n):
        # Mild trend + mean-reverting noise so labels are learnable.
        drift = 0.00015 * np.sin(i / 37.0)
        shock = float(rng.normal(0.0, 0.0008 if price < 10 else price * 0.0015))
        open_ = price
        close = max(1e-6, open_ * (1.0 + drift + shock))
        high = max(open_, close) * (1.0 + abs(float(rng.normal(0.0, 0.0003))))
        low = min(open_, close) * (1.0 - abs(float(rng.normal(0.0, 0.0003))))
        candles.append(
            {
                "symbol": symbol.upper(),
                "timeframe": tf,
                "timestamp": ts,
                "open": float(open_),
                "high": float(high),
                "low": float(low),
                "close": float(close),
                "volume": float(rng.integers(100, 2000)),
            }
        )
        price = close
        ts = ts + step
    return candles


def _load_or_seed_candles(args: argparse.Namespace) -> List[Dict[str, Any]]:
    from core.config import DATABASE_PATH
    from database.core.connection import DatabaseManager
    from database.repositories.candle_repository import CandleRepository
    from ai.data.normalizers import candle_entity_to_dict

    need = int(args.train_bars) + int(args.week_bars) + 50
    db = DatabaseManager(str(DATABASE_PATH))
    repo = CandleRepository(db)
    if hasattr(repo, "get_last_n"):
        entities = repo.get_last_n(args.symbol, args.timeframe, need)
    else:
        entities = list(reversed(list(repo.find_latest(args.symbol, args.timeframe, limit=need))))
    candles = [candle_entity_to_dict(e) for e in entities]
    if len(candles) >= need:
        logging.getLogger(__name__).info(
            "using %s real candles from %s", len(candles), DATABASE_PATH
        )
        return candles

    logging.getLogger(__name__).warning(
        "only %s DB candles for %s %s — seeding synthetic series for paper-week verification",
        len(candles),
        args.symbol,
        args.timeframe,
    )
    return _synthetic_candles(
        symbol=args.symbol,
        timeframe=args.timeframe,
        n=need,
        seed=int(args.seed),
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Run a 1-week paper trading replay")
    parser.add_argument("--symbol", default="EURUSD")
    parser.add_argument("--timeframe", default="H1")
    parser.add_argument("--model", default="random_forest")
    parser.add_argument("--train-bars", type=int, default=1500)
    parser.add_argument("--week-bars", type=int, default=168, help="H1 bars in one week")
    parser.add_argument("--equity", type=float, default=100_000.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--artifacts", default="ai/artifacts/paper_week")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    _configure_logging(args.verbose)

    from ai.config.settings import AIConfig
    from ai.data.candle_adapter import InMemoryCandleSource
    from ai.services.autonomous_trader import create_autonomous_trader

    candles = _load_or_seed_candles(args)
    artifact_dir = Path(args.artifacts)
    artifact_dir.mkdir(parents=True, exist_ok=True)

    config = AIConfig()
    config.storage.root_dir = Path("ai/artifacts")
    config.symbols = [args.symbol.upper()]
    config.primary_timeframe = args.timeframe.upper()
    config.model.model_type = args.model
    config.data.auto_download = False
    config.data.require_validated = False
    config.data.allow_synthetic_fallback = False
    # Paper-week verification: keep risk active but allow more trades to exercise the loop.
    config.risk.min_confidence = 0.50
    config.risk.min_expected_rr = 1.2
    config.risk.circuit_breaker_loss = 0.20
    config.risk.max_drawdown = 0.25
    config.ensure_directories()

    source = InMemoryCandleSource(candles)
    trader = create_autonomous_trader(
        config,
        mode="paper",
        candle_source=source,
        initial_equity=float(args.equity),
    )
    trader.install_signal_handlers()

    summary = trader.run_paper_week(
        candles,
        symbol=args.symbol,
        model_type=args.model,
        train_bars=int(args.train_bars),
        week_bars=int(args.week_bars),
    )
    out = artifact_dir / f"paper_week_{args.symbol}_{args.timeframe}.json"
    out.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    print(json.dumps({"passed": True, "summary_path": str(out), "metrics": summary.get("metrics"), "live": summary.get("live")}, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
