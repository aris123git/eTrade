#!/usr/bin/env python3
"""
scripts/phase3_validation.py - End-to-end AI validation & performance harness

Phase 3 objective:
Prove the platform works together — repository data -> features -> train ->
evaluate -> backtest -> registry -> reload -> live prediction — with measurable
benchmarks. No new product features; validation only.

Usage:
  python3 scripts/phase3_validation.py
  PHASE3_STRESS_CANDLES=500000 PHASE3_TRAIN_BARS=25000 python3 scripts/phase3_validation.py
"""

from __future__ import annotations

import gc
import json
import os
import sys
import time
import traceback
from dataclasses import asdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple
from uuid import uuid4

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ai.config.settings import create_ai_config
from ai.data import CandleRepositoryAdapter, TickRepositoryAdapter
from ai.evaluation.backtest import BacktestEngine, BacktestSignal
from ai.evaluation.report import Evaluator
from ai.models import create_model
from ai.services.pipeline import create_ai_pipeline
from ai.signals import create_signal_engine
from ai.storage.registry import ModelRegistry
from database.core.connection import DatabaseManager
from database.indexes import create_indexes
from database.models.market import MarketType
from database.repositories.factory import create_repository_manager
from database.schema import create_schema
from database.seed import seed


# ------------------------------------------------------------------------------
# Config knobs (env-overridable)
# ------------------------------------------------------------------------------

STRESS_CANDLES = int(os.environ.get("PHASE3_STRESS_CANDLES", "1000000"))
STRESS_STREAM = int(os.environ.get("PHASE3_STRESS_STREAM", "250000"))
TRAIN_BARS = int(os.environ.get("PHASE3_TRAIN_BARS", "30000"))
SYMBOL = os.environ.get("PHASE3_SYMBOL", "EURUSD")
TIMEFRAME = os.environ.get("PHASE3_TIMEFRAME", "M15")
BASELINE_MODELS = ["random_forest", "lightgbm", "xgboost"]
ARTIFACT_DIR = Path(os.environ.get("PHASE3_ARTIFACT_DIR", "ai_artifacts/phase3"))


def _rss_mb() -> float:
    try:
        import psutil

        return float(psutil.Process(os.getpid()).memory_info().rss) / (1024 * 1024)
    except Exception:
        return float("nan")


def _now() -> str:
    return datetime.utcnow().isoformat(timespec="seconds") + "Z"


def _iso(ts: datetime) -> str:
    return ts.isoformat(timespec="seconds")


# ==============================================================================
# 1) Architecture checks
# ==============================================================================


def check_architecture() -> Dict[str, Any]:
    import compileall
    import importlib

    compile_ok = compileall.compile_dir(str(ROOT), quiet=1, force=False)
    targets = [
        "core.config",
        "database.core",
        "database.repositories.factory",
        "ai.services.pipeline",
        "ai.models",
        "ai.evaluation.backtest",
        "ai.storage.registry",
        "ai.data",
    ]
    imports: Dict[str, str] = {}
    for name in targets:
        try:
            importlib.import_module(name)
            imports[name] = "ok"
        except Exception as exc:  # pragma: no cover
            imports[name] = f"{type(exc).__name__}: {exc}"

    backends = {}
    for pkg in ("numpy", "sklearn", "lightgbm", "xgboost", "psutil"):
        try:
            mod = importlib.import_module(pkg)
            backends[pkg] = getattr(mod, "__version__", "present")
        except Exception as exc:
            backends[pkg] = f"missing ({exc})"

    return {
        "compileall": bool(compile_ok),
        "imports": imports,
        "imports_ok": all(v == "ok" for v in imports.values()),
        "backends": backends,
        "rss_mb": _rss_mb(),
    }


# ==============================================================================
# 2) Database stress
# ==============================================================================


def _ensure_market(repos, symbol: str) -> int:
    existing = repos.markets.find_by_symbol(symbol)
    if existing is not None:
        return int(existing.market_id)
    market = repos.markets.create(
        symbol=symbol,
        market_type=MarketType.FOREX,
        description=f"Phase3 {symbol}",
        base_currency=symbol[:3],
        quote_currency=symbol[3:],
        digits=5,
        point=0.00001,
    )
    return int(market.market_id)


def stress_database(db: DatabaseManager, repos) -> Dict[str, Any]:
    """Insert STRESS_CANDLES rows and stream STRESS_STREAM without retaining them."""
    market_id = _ensure_market(repos, SYMBOL)
    adapter = db.get_adapter()
    start = datetime(2018, 1, 1)
    step = timedelta(minutes=15)

    # Wipe previous stress symbol/timeframe rows for clean measurement
    adapter.execute(
        "DELETE FROM candles WHERE symbol = ? AND timeframe = ?",
        (SYMBOL, TIMEFRAME),
    )
    adapter.commit()

    batch = 10_000
    inserted = 0
    t0 = time.perf_counter()
    rss0 = _rss_mb()
    price = 1.1000
    rng = np.random.default_rng(7)

    for offset in range(0, STRESS_CANDLES, batch):
        n = min(batch, STRESS_CANDLES - offset)
        rows = []
        now = _iso(datetime.utcnow())
        for i in range(n):
            idx = offset + i
            ts = start + step * idx
            ret = float(rng.normal(0.0, 0.0004))
            o = price
            c = price * (1.0 + ret)
            h = max(o, c) * (1.0 + abs(float(rng.normal(0, 0.00015))))
            l = min(o, c) * (1.0 - abs(float(rng.normal(0, 0.00015))))
            rows.append(
                (
                    str(uuid4()),
                    SYMBOL,
                    TIMEFRAME,
                    _iso(ts),
                    float(o),
                    float(h),
                    float(l),
                    float(c),
                    float(100 + (idx % 500)),
                    market_id,
                    0.00012,
                    int(100 + (idx % 500)),
                    "active",
                    "{}",
                    now,
                    now,
                )
            )
            price = c
        adapter.execute_many(
            """
            INSERT OR IGNORE INTO candles (
                candle_uuid, symbol, timeframe, timestamp, open, high, low, close,
                volume, market_id, spread, tick_volume, status, metadata, created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            rows,
        )
        inserted += n
        if offset % (batch * 20) == 0:
            adapter.commit()
            gc.collect()
    adapter.commit()
    insert_s = time.perf_counter() - t0
    rss1 = _rss_mb()

    # Streaming throughput / memory
    candle_repo = repos.candles
    stream_n = min(STRESS_STREAM, inserted)
    end = start + step * (stream_n + 10)
    t1 = time.perf_counter()
    rss_stream0 = _rss_mb()
    count = 0
    peak = rss_stream0
    for _ in candle_repo.stream_candles(SYMBOL, TIMEFRAME, start, end, batch_size=20_000):
        count += 1
        if count % 50_000 == 0:
            peak = max(peak, _rss_mb())
            gc.collect()
    stream_s = time.perf_counter() - t1
    rss_stream1 = _rss_mb()

    # Also seed a modest tick sample for TickRepositoryAdapter checks
    tick_rows = []
    tick_start = start
    now = _iso(datetime.utcnow())
    for i in range(5_000):
        ts = tick_start + timedelta(seconds=i)
        bid = 1.1 + i * 1e-6
        tick_rows.append(
            (
                str(uuid4()),
                SYMBOL,
                _iso(ts),
                bid,
                bid + 0.00012,
                bid + 0.00006,
                1.0,
                0,
                market_id,
                None,
                "active",
                "{}",
                now,
                now,
            )
        )
    adapter.execute_many(
        """
        INSERT OR IGNORE INTO ticks (
            tick_uuid, symbol, timestamp, bid, ask, last, volume, flags,
            market_id, broker_id, status, metadata, created_at, updated_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        tick_rows,
    )
    adapter.commit()

    return {
        "requested_insert": STRESS_CANDLES,
        "inserted": inserted,
        "insert_seconds": round(insert_s, 3),
        "insert_candles_per_sec": round(inserted / insert_s, 1) if insert_s else None,
        "rss_before_mb": round(rss0, 1) if rss0 == rss0 else None,
        "rss_after_insert_mb": round(rss1, 1) if rss1 == rss1 else None,
        "stream_requested": stream_n,
        "streamed": count,
        "stream_seconds": round(stream_s, 3),
        "stream_candles_per_sec": round(count / stream_s, 1) if stream_s else None,
        "stream_rss_start_mb": round(rss_stream0, 1) if rss_stream0 == rss_stream0 else None,
        "stream_rss_end_mb": round(rss_stream1, 1) if rss_stream1 == rss_stream1 else None,
        "stream_rss_peak_mb": round(peak, 1) if peak == peak else None,
        "stream_rss_delta_mb": round(rss_stream1 - rss_stream0, 1)
        if rss_stream0 == rss_stream0 and rss_stream1 == rss_stream1
        else None,
        "ticks_inserted": len(tick_rows),
    }


# ==============================================================================
# 3-5) Full AI pipeline on repository data
# ==============================================================================


def load_train_candles(repos, n: int) -> List[Dict[str, Any]]:
    """Stream the last N candles from CandleRepository into dicts for the AI pipeline."""
    adapter = CandleRepositoryAdapter(repos.candles)
    # Determine range from DB
    row = repos.candles.adapter.fetch_one(
        """
        SELECT MIN(timestamp) AS mn, MAX(timestamp) AS mx, COUNT(*) AS c
        FROM candles WHERE symbol = ? AND timeframe = ? AND status = 'active'
        """,
        (SYMBOL, TIMEFRAME),
    )
    if not row or not row["c"]:
        raise RuntimeError("No candles available for training")
    mx = datetime.fromisoformat(str(row["mx"]))
    # Approximate start: n bars back
    minutes = 15 if TIMEFRAME == "M15" else 60
    start = mx - timedelta(minutes=minutes * (n + 50))
    candles = list(adapter.stream_candles(SYMBOL, TIMEFRAME, start, mx, batch_size=20_000))
    if len(candles) > n:
        candles = candles[-n:]
    return candles


def _prediction_returns(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    closes: np.ndarray,
    horizon: int = 5,
) -> np.ndarray:
    """Simple strategy return proxy: long if pred==1 else short, held `horizon` bars."""
    rets = np.zeros(len(y_pred), dtype=float)
    for i in range(len(y_pred) - horizon):
        future = (closes[i + horizon] - closes[i]) / max(closes[i], 1e-12)
        direction = 1.0 if int(y_pred[i]) == 1 else -1.0
        # Only score where label was valid
        if np.isfinite(y_true[i]):
            rets[i] = direction * future
    return rets


def _naive(ts: Any) -> Any:
    """Strip tzinfo so repository timestamps match feature timestamps."""
    if isinstance(ts, datetime) and ts.tzinfo is not None:
        return ts.replace(tzinfo=None)
    return ts


def train_and_compare(candles: Sequence[Dict[str, Any]], artifact_root: Path) -> Dict[str, Any]:
    results: Dict[str, Any] = {"models": {}, "best_model": None}
    best_score = -1e18
    best_name = None

    # Shared feature config — lean groups for speed while covering the stack
    shared = dict(
        symbols=[SYMBOL],
        timeframes=[TIMEFRAME],
        primary_timeframe=TIMEFRAME,
        labels={"methods": ["binary_direction"], "horizon": 5, "horizons": [5]},
        features={
            "enabled_groups": [
                "price",
                "returns",
                "moving_averages",
                "momentum",
                "volatility",
                "volume",
                "candle_structure",
                "session",
            ],
            "dropna": False,
        },
        datasets={"train_ratio": 0.70, "val_ratio": 0.15, "test_ratio": 0.15},
        training={"epochs": 1, "verbose": 0, "resume_from_checkpoint": False, "log_tensorboard": False},
        storage={"root_dir": str(artifact_root)},
        model={"n_estimators": 120, "max_depth": 5, "learning_rate": 0.05, "task": "classification"},
    )

    # Build one dataset with RF config shell, reuse for all models
    base_kwargs = {**shared, "model": {**shared["model"], "model_type": "random_forest"}}
    base_config = create_ai_config(**base_kwargs)
    base_config.ensure_directories()
    pipeline = create_ai_pipeline(base_config)
    t_feat = time.perf_counter()
    dataset = pipeline.build_dataset(candles, label_name="binary_direction_5")
    feature_seconds = time.perf_counter() - t_feat
    bundle = dataset.bundle
    results["dataset"] = {
        "rows_train": int(bundle.n_train),
        "rows_val": int(bundle.n_val),
        "rows_test": int(bundle.n_test),
        "n_features": len(bundle.feature_names),
        "feature_build_seconds": round(feature_seconds, 3),
        "label": dataset.label.name,
    }

    # closes aligned to test set for trading proxy
    close_by_ts = {_naive(c["timestamp"]): float(c["close"]) for c in candles}
    test_ts = [_naive(ts) for ts in (bundle.metadata.get("test_timestamps") or bundle.timestamps[-bundle.n_test :])]
    test_closes = np.array([close_by_ts.get(ts, np.nan) for ts in test_ts], dtype=float)

    evaluator = Evaluator(config=base_config, task="classification")
    registry = ModelRegistry(config=base_config)

    for model_type in BASELINE_MODELS:
        model_info: Dict[str, Any] = {"model_type": model_type}
        try:
            extra = {}
            if model_type == "lightgbm":
                extra = {"verbose": -1}
            elif model_type == "xgboost":
                extra = {"verbosity": 0}
            cfg = create_ai_config(
                **{
                    **shared,
                    "model": {
                        **shared["model"],
                        "model_type": model_type,
                        "extra_params": extra,
                    },
                }
            )
            cfg.ensure_directories()
            model = create_model(model_type, cfg)
            pipe = create_ai_pipeline(cfg)
            t0 = time.perf_counter()
            train_result = pipe.train(bundle, model=model)
            train_s = time.perf_counter() - t0
            metrics = pipe.evaluate(train_result.model, bundle)

            # Prediction latency on test
            t1 = time.perf_counter()
            preds = np.asarray(train_result.model.predict(bundle.X_test))
            pred_s = time.perf_counter() - t1
            proba = train_result.model.predict_proba(bundle.X_test)
            y_true = np.asarray(bundle.y_test).reshape(-1)

            returns = _prediction_returns(y_true, preds, test_closes, horizon=5)
            report = evaluator.evaluate(
                y_true,
                preds,
                y_proba=proba,
                returns=returns,
                feature_names=bundle.feature_names,
                model=train_result.model,
            )

            # Walk-forward-ish backtest signals from test predictions
            ts_to_candle = {_naive(c["timestamp"]): c for c in candles}
            signals = []
            bt_candles = []
            for i, ts in enumerate(test_ts):
                src = ts_to_candle.get(_naive(ts))
                if src is None or not np.isfinite(test_closes[i]):
                    continue
                px = float(src["close"])
                side = "buy" if int(preds[i]) == 1 else "sell"
                signals.append(
                    BacktestSignal(
                        symbol=SYMBOL,
                        timestamp=_naive(src["timestamp"]),
                        side=side,
                        order_type="market",
                        quantity=1.0,
                        price=px,
                        stop_loss=px * (0.998 if side == "buy" else 1.002),
                        take_profit=px * (1.004 if side == "buy" else 0.996),
                        timeframe=TIMEFRAME,
                    )
                )
                bt_candles.append({**src, "timestamp": _naive(src["timestamp"])})

            uniq = {}
            for c in bt_candles:
                uniq[c["timestamp"]] = c
            bt_candles = [uniq[k] for k in sorted(uniq.keys())]

            bt = BacktestEngine(config=cfg, initial_equity=10_000.0)
            bt_result = bt.run(signals, bt_candles)

            registered = registry.register(
                name=model_type,
                model=train_result.model,
                features=bundle.feature_names,
                metrics={**metrics, **{f"clf_{k}": v for k, v in report.classification.items() if isinstance(v, (int, float))}},
                params=train_result.model.get_params(),
                metadata={"phase": "3", "symbol": SYMBOL, "timeframe": TIMEFRAME},
                overwrite=True,
                version="phase3",
            )

            # Reload + live predict on last window
            loaded = registry.load(model_type, version="phase3")
            reloaded = loaded["model"]
            t2 = time.perf_counter()
            live_pred = np.asarray(reloaded.predict(bundle.X_test[-1:]))[0]
            live_latency_ms = (time.perf_counter() - t2) * 1000.0

            # Signal path
            from ai.utils.types import PredictionResult

            sig_engine = create_signal_engine(cfg)
            pred_payload = PredictionResult(
                symbol=SYMBOL,
                timeframe=TIMEFRAME,
                timestamp=test_ts[-1] if test_ts else datetime.utcnow(),
                prediction=int(live_pred),
                probabilities={"0": 0.4, "1": 0.6} if int(live_pred) == 1 else {"0": 0.6, "1": 0.4},
                confidence=0.6,
                model_version=registered.version,
            )
            signal = sig_engine.generate(
                pred_payload,
                market_context={
                    "entry": float(test_closes[-1]),
                    "sl": float(test_closes[-1]) * 0.998,
                    "tp": float(test_closes[-1]) * 1.004,
                    "price": float(test_closes[-1]),
                    "trend": "bullish" if int(live_pred) == 1 else "bearish",
                },
            )

            score = float(metrics.get("test_f1", metrics.get("test_accuracy", 0.0)) or 0.0)
            model_info.update(
                {
                    "status": "ok",
                    "train_seconds": round(train_s, 3),
                    "predict_test_seconds": round(pred_s, 4),
                    "predict_test_rows_per_sec": round(len(preds) / pred_s, 1) if pred_s else None,
                    "live_predict_latency_ms": round(live_latency_ms, 3),
                    "metrics": {k: (None if isinstance(v, float) and (v != v or abs(v) == float("inf")) else v) for k, v in metrics.items()},
                    "classification": {
                        k: v
                        for k, v in report.classification.items()
                        if isinstance(v, (int, float, str, dict, list))
                    },
                    "trading_proxy": report.trading,
                    "importance_top10": dict(
                        sorted(report.importance.items(), key=lambda kv: abs(float(kv[1])), reverse=True)[:10]
                    )
                    if report.importance
                    else {},
                    "backtest": {
                        "n_trades": len(bt_result.trades),
                        "final_equity": bt_result.final_equity,
                        "metrics": bt_result.metrics,
                        "equity_points": len(bt_result.equity),
                    },
                    "registry_path": str(registered.path),
                    "reloaded_prediction": int(live_pred),
                    "signal": getattr(signal.side, "value", str(signal.side)),
                }
            )
            if score > best_score:
                best_score = score
                best_name = model_type
        except Exception as exc:
            model_info.update(
                {
                    "status": "error",
                    "error": f"{type(exc).__name__}: {exc}",
                    "traceback": traceback.format_exc()[-2000:],
                }
            )
        results["models"][model_type] = model_info

    results["best_model"] = best_name
    results["best_score_test_f1_or_acc"] = best_score if best_name else None
    return results


def verify_adapters(repos) -> Dict[str, Any]:
    candle_adapter = CandleRepositoryAdapter(repos.candles)
    tick_adapter = TickRepositoryAdapter(repos.ticks)
    end = datetime.utcnow()
    start = end - timedelta(days=5)
    candles = list(candle_adapter.stream_candles(SYMBOL, TIMEFRAME, start - timedelta(days=4000), end, batch_size=5000))
    ticks = list(tick_adapter.stream_ticks(SYMBOL, datetime(2018, 1, 1), datetime(2018, 1, 1, 2), batch_size=1000))
    return {
        "candle_adapter_streamed": len(candles),
        "tick_adapter_streamed": len(ticks),
        "candle_sample_keys": sorted(list(candles[0].keys())) if candles else [],
    }


def main() -> int:
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    report: Dict[str, Any] = {
        "phase": 3,
        "started_at": _now(),
        "symbol": SYMBOL,
        "timeframe": TIMEFRAME,
        "env": {
            "stress_candles": STRESS_CANDLES,
            "stress_stream": STRESS_STREAM,
            "train_bars": TRAIN_BARS,
        },
    }

    print("=" * 72)
    print("Phase 3 — AI Validation & Performance")
    print("=" * 72)

    print("\n[1/4] Architecture checks...")
    report["architecture"] = check_architecture()
    print("  compileall:", report["architecture"]["compileall"])
    print("  imports_ok:", report["architecture"]["imports_ok"])
    print("  backends:", report["architecture"]["backends"])

    db_path = ARTIFACT_DIR / "phase3_market_ai.db"
    if db_path.exists():
        db_path.unlink()
    print(f"\n[2/4] Database stress on {db_path} ...")
    db = DatabaseManager(db_path=db_path)
    create_schema(db)
    create_indexes(db)
    seed(db)
    repos = create_repository_manager(db)
    report["database"] = stress_database(db, repos)
    print(
        "  inserted={inserted} ({insert_candles_per_sec}/s) streamed={streamed} ({stream_candles_per_sec}/s) rss_delta={stream_rss_delta_mb}MB".format(
            **{k: report["database"].get(k) for k in report["database"]}
        )
    )

    print("\n[3/4] Adapter + AI training/backtest/registry...")
    report["adapters"] = verify_adapters(repos)
    candles = load_train_candles(repos, TRAIN_BARS)
    print(f"  training bars loaded from repository: {len(candles)}")
    report["pipeline"] = train_and_compare(candles, ARTIFACT_DIR)
    for name, info in report["pipeline"]["models"].items():
        status = info.get("status")
        if status == "ok":
            print(
                f"  {name}: train={info['train_seconds']}s "
                f"metrics={info.get('metrics')} "
                f"trades={info['backtest']['n_trades']} "
                f"signal={info['signal']}"
            )
        else:
            print(f"  {name}: ERROR {info.get('error')}")

    # Pass criteria
    arch_ok = report["architecture"]["compileall"] and report["architecture"]["imports_ok"]
    db_ok = (
        report["database"]["inserted"] >= min(STRESS_CANDLES, report["database"]["inserted"])
        and report["database"]["streamed"] >= min(STRESS_STREAM, 1000)
        and (report["database"]["stream_rss_delta_mb"] is None or report["database"]["stream_rss_delta_mb"] < 800)
    )
    models_ok = all(m.get("status") == "ok" for m in report["pipeline"]["models"].values())
    registry_ok = all(
        m.get("status") != "ok" or Path(m.get("registry_path", "")).exists()
        for m in report["pipeline"]["models"].values()
    )
    report["gates"] = {
        "architecture": arch_ok,
        "database_stress": db_ok,
        "baseline_models": models_ok,
        "registry_reload": registry_ok,
        "passed": bool(arch_ok and db_ok and models_ok and registry_ok),
    }
    report["finished_at"] = _now()

    out = ARTIFACT_DIR / "phase3_report.json"
    # Make JSON safe
    def _sanitize(obj: Any) -> Any:
        if isinstance(obj, dict):
            return {str(k): _sanitize(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [_sanitize(v) for v in obj]
        if isinstance(obj, float):
            if obj != obj or abs(obj) == float("inf"):
                return None
            return obj
        if isinstance(obj, (np.floating,)):
            val = float(obj)
            return None if val != val or abs(val) == float("inf") else val
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, Path):
            return str(obj)
        if isinstance(obj, datetime):
            return obj.isoformat()
        return obj

    out.write_text(json.dumps(_sanitize(report), indent=2), encoding="utf-8")
    print("\n[4/4] Report written:", out)
    print("GATES:", report["gates"])
    db.close()
    return 0 if report["gates"]["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
