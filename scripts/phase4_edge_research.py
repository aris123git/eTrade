#!/usr/bin/env python3
"""
scripts/phase4_edge_research.py — Quantitative edge research harness

Phase 4 rules:
- Do NOT add new packages/classes under ai/ or database/.
- Reuse existing: CandleRepository, FeatureEngine, TripleBarrier labels,
  FeatureSelector, walk-forward validation, BacktestEngine, ModelRegistry.
- Prefer real MT5 history when available; otherwise research on repository data
  and clearly mark the data source.

Goals:
1) Strict walk-forward (no train/test leakage)
2) Triple-barrier labels (not naive up/down)
3) Feature selection fitted only inside each training window
4) Compare RF / LightGBM / XGBoost on identical folds
5) Report financial metrics with realistic costs
6) Per-symbol / per-timeframe breakdown

Usage:
  python3 scripts/phase4_edge_research.py
  PHASE4_SYMBOLS=EURUSD,GBPUSD PHASE4_TIMEFRAMES=M15,H1 python3 scripts/phase4_edge_research.py
"""

from __future__ import annotations

import json
import os
import sys
import time
import traceback
from copy import deepcopy
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ai.config.settings import create_ai_config
from ai.data import CandleRepositoryAdapter
from ai.evaluation.backtest import BacktestEngine, BacktestSignal
from ai.evaluation.report import Evaluator
from ai.evaluation.trading_metrics import trading_metrics
from ai.features import create_feature_engine
from ai.labels import create_label_generator
from ai.models import create_model
from ai.preprocessing.selector import FeatureSelector
from ai.utils.types import CandleDict
from database.core.connection import DatabaseManager
from database.indexes import create_indexes
from database.models.market import MarketType
from database.repositories.factory import create_repository_manager
from database.schema import create_schema
from database.seed import seed


# ------------------------------------------------------------------------------
# Env knobs
# ------------------------------------------------------------------------------

SYMBOLS = [s.strip().upper() for s in os.environ.get("PHASE4_SYMBOLS", "EURUSD").split(",") if s.strip()]
TIMEFRAMES = [t.strip().upper() for t in os.environ.get("PHASE4_TIMEFRAMES", "M15").split(",") if t.strip()]
MAX_BARS = int(os.environ.get("PHASE4_MAX_BARS", "40000"))
TOP_K = int(os.environ.get("PHASE4_TOP_K", "40"))
FOLDS = int(os.environ.get("PHASE4_FOLDS", "5"))
EMBARGO = int(os.environ.get("PHASE4_EMBARGO", "5"))
HORIZON = int(os.environ.get("PHASE4_HORIZON", "10"))
MODELS = [m.strip() for m in os.environ.get("PHASE4_MODELS", "random_forest,lightgbm,xgboost").split(",") if m.strip()]
ARTIFACT_DIR = Path(os.environ.get("PHASE4_ARTIFACT_DIR", "ai_artifacts/phase4"))
DB_PATH = Path(os.environ.get("PHASE4_DB", str(ARTIFACT_DIR / "phase4_market_ai.db")))
TRY_MT5 = os.environ.get("PHASE4_TRY_MT5", "1") == "1"
# FX research defaults: express friction in basis points of price, not $ / lot.
# (BacktestEngine lot commissions are optional and disabled by default here.)
COMMISSION = float(os.environ.get("PHASE4_COMMISSION", "0.0"))
SLIPPAGE = float(os.environ.get("PHASE4_SLIPPAGE", "0.0"))
COST_BPS = float(os.environ.get("PHASE4_COST_BPS", "1.0"))  # round-turn friction


def _naive(ts: Any) -> Any:
    if isinstance(ts, datetime) and ts.tzinfo is not None:
        return ts.replace(tzinfo=None)
    return ts


def _json_safe(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, datetime):
        return obj.isoformat()
    if isinstance(obj, (np.floating, float)):
        val = float(obj)
        if val != val or abs(val) == float("inf"):
            return None
        return val
    if isinstance(obj, (np.integer, int)):
        return int(obj)
    if isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    return obj


# ==============================================================================
# Data acquisition
# ==============================================================================


def _ensure_db() -> Tuple[DatabaseManager, Any, str]:
    """
    Return (db, repos, data_source_tag).

    data_source_tag:
      - mt5_download: MT5 initialize + download succeeded
      - repository_existing: reused existing DB content
      - synthetic_bootstrap: generated research bars (NOT market edge evidence)
    """
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    db_exists = DB_PATH.exists()
    db = DatabaseManager(db_path=DB_PATH)
    create_schema(db)
    create_indexes(db)
    seed(db)
    repos = create_repository_manager(db)

    source = "repository_existing" if db_exists else "synthetic_bootstrap"

    if TRY_MT5:
        mt5_ok = _try_mt5_download(db, repos)
        if mt5_ok:
            return db, repos, "mt5_download"

    # Ensure each configured market has enough bars for research
    for symbol in SYMBOLS:
        for timeframe in TIMEFRAMES:
            count = db.fetch_one(
                "SELECT COUNT(*) AS c FROM candles WHERE symbol=? AND timeframe=? AND status='active'",
                (symbol, timeframe),
            )
            n = int(count["c"]) if count else 0
            if n < max(MAX_BARS // 2, 5000):
                _bootstrap_symbol(repos, symbol, timeframe, max(MAX_BARS, 10000))
                source = "synthetic_bootstrap" if source != "mt5_download" else source
    return db, repos, source


def _try_mt5_download(db: DatabaseManager, repos) -> bool:
    try:
        import MetaTrader5 as mt5  # type: ignore
    except ImportError:
        print("MT5 package unavailable — research will use repository/synthetic data.")
        return False

    if not mt5.initialize():
        print("MT5 initialize() failed — research will use repository/synthetic data.")
        return False

    try:
        from collector.symbol_manager import SymbolManager
        from collector.downloader import Downloader
        from core.config import TIMEFRAMES as MT5_TF

        # Ensure markets exist for requested symbols
        for symbol in SYMBOLS:
            info = mt5.symbol_info(symbol)
            if info is None:
                mt5.symbol_select(symbol, True)
                info = mt5.symbol_info(symbol)
            if info is None:
                print(f"Symbol not found in MT5: {symbol}")
                continue
            existing = repos.markets.find_by_symbol(symbol)
            if existing is None:
                repos.markets.create(
                    symbol=symbol,
                    market_type=MarketType.FOREX,
                    description=getattr(info, "description", symbol),
                    digits=int(getattr(info, "digits", 5)),
                    point=float(getattr(info, "point", 0.00001)),
                    base_currency=getattr(info, "currency_base", None),
                    quote_currency=getattr(info, "currency_profit", None),
                )

        # Restrict downloader timeframes to requested ones
        tf_map = {k: v for k, v in MT5_TF.items() if k in TIMEFRAMES}
        if not tf_map:
            tf_map = {TIMEFRAMES[0]: MT5_TF.get(TIMEFRAMES[0], list(MT5_TF.values())[0])}

        downloader = Downloader(db, tf_map)
        ok = downloader.download_all()
        print(f"MT5 download_all -> {ok}")
        return bool(ok)
    except Exception as exc:
        print(f"MT5 download path failed: {type(exc).__name__}: {exc}")
        return False
    finally:
        try:
            mt5.shutdown()
        except Exception:
            pass


def _bootstrap_symbol(repos, symbol: str, timeframe: str, n: int) -> None:
    """Generate multi-year-like synthetic bars for pipeline research only."""
    from uuid import uuid4

    market = repos.markets.find_by_symbol(symbol)
    if market is None:
        market = repos.markets.create(
            symbol=symbol,
            market_type=MarketType.FOREX if len(symbol) == 6 else MarketType.UNKNOWN,
            description=f"Phase4 bootstrap {symbol}",
            digits=5,
            point=0.00001,
        )
    market_id = int(market.market_id)
    minutes = {"M1": 1, "M5": 5, "M15": 15, "M30": 30, "H1": 60, "H4": 240, "D1": 1440}.get(timeframe, 15)
    # Seed from symbol hash for reproducibility across assets
    seed_i = abs(hash(symbol + timeframe)) % (2**32)
    rng = np.random.default_rng(seed_i)
    start = datetime(2019, 1, 1)
    price = 1.10 if symbol.endswith("USD") and symbol.startswith("EUR") else 1.25
    if "XAU" in symbol:
        price = 1800.0
    rows = []
    now = datetime.utcnow().isoformat(timespec="seconds")
    for i in range(n):
        ts = start + timedelta(minutes=minutes * i)
        # Mild regime + noise (still not real edge)
        regime = 0.00005 * np.sin(i / 500.0)
        ret = float(rng.normal(regime, 0.00045))
        o = price
        c = max(price * (1.0 + ret), 1e-6)
        h = max(o, c) * (1.0 + abs(float(rng.normal(0, 0.0002))))
        l = min(o, c) * (1.0 - abs(float(rng.normal(0, 0.0002))))
        rows.append(
            {
                "symbol": symbol,
                "timeframe": timeframe,
                "timestamp": ts,
                "open": float(o),
                "high": float(h),
                "low": float(l),
                "close": float(c),
                "volume": float(100 + (i % 400)),
                "market_id": market_id,
                "tick_volume": int(100 + (i % 400)),
                "status": "active",
                "candle_uuid": str(uuid4()),
            }
        )
        price = c
    # bulk via repository
    for i in range(0, len(rows), 10000):
        repos.candles.bulk_upsert(rows[i : i + 10000], trusted_source=True)


def load_candles(repos, symbol: str, timeframe: str, max_bars: int) -> List[CandleDict]:
    adapter = CandleRepositoryAdapter(repos.candles)
    row = repos.candles.adapter.fetch_one(
        """
        SELECT MIN(timestamp) AS mn, MAX(timestamp) AS mx, COUNT(*) AS c
        FROM candles WHERE symbol=? AND timeframe=? AND status='active'
        """,
        (symbol, timeframe),
    )
    if not row or int(row["c"] or 0) == 0:
        raise RuntimeError(f"No candles for {symbol} {timeframe}")
    mx = datetime.fromisoformat(str(row["mx"]))
    mn = datetime.fromisoformat(str(row["mn"]))
    candles = list(adapter.stream_candles(symbol, timeframe, mn, mx, batch_size=20000))
    if len(candles) > max_bars:
        candles = candles[-max_bars:]
    # normalize timestamps
    for c in candles:
        c["timestamp"] = _naive(c["timestamp"])
    return candles


# ==============================================================================
# Feature / label matrix
# ==============================================================================


def build_xy(
    candles: Sequence[CandleDict],
    symbol: str,
    timeframe: str,
) -> Tuple[np.ndarray, np.ndarray, List[datetime], List[str], np.ndarray, Dict[str, Any]]:
    config = create_ai_config(
        symbols=[symbol],
        timeframes=[timeframe],
        primary_timeframe=timeframe,
        labels={
            "methods": ["triple_barrier"],
            "horizon": HORIZON,
            "horizons": [HORIZON],
            "take_profit_atr_mult": 2.0,
            "stop_loss_atr_mult": 1.0,
            "atr_period": 14,
        },
        features={
            "enabled_groups": [
                "price",
                "returns",
                "moving_averages",
                "momentum",
                "volatility",
                "volume",
                "candle_structure",
                "patterns",
                "session",
                "regime",
            ],
            "dropna": False,
        },
        datasets={"feature_selection_k": TOP_K, "walk_forward_folds": FOLDS, "walk_forward_embargo": EMBARGO},
    )
    fe = create_feature_engine(config)
    frame = fe.transform(list(candles), config)
    lg = create_label_generator(config)
    labels = lg.generate(list(candles), config)
    label_key = f"triple_barrier_{HORIZON}"
    if label_key not in labels:
        label_key = next(iter(labels))
    lab = labels[label_key]

    # Align by naive timestamp
    close_by_ts = {_naive(c["timestamp"]): float(c["close"]) for c in candles}
    label_by_idx_ts = {}
    base = list(candles)
    for i, c in enumerate(base):
        label_by_idx_ts[_naive(c["timestamp"])] = i

    X_rows = []
    y_rows = []
    ts_rows = []
    closes = []
    for row, ts in zip(frame.matrix, frame.timestamps):
        ts_n = _naive(ts)
        idx = label_by_idx_ts.get(ts_n)
        if idx is None:
            continue
        if not lab.valid_mask[idx] or not np.isfinite(lab.values[idx]):
            continue
        if not np.isfinite(row).all():
            continue
        X_rows.append(row)
        y_rows.append(float(lab.values[idx]))
        ts_rows.append(ts_n)
        closes.append(close_by_ts.get(ts_n, np.nan))

    X = np.asarray(X_rows, dtype=float)
    y = np.asarray(y_rows, dtype=float)
    closes_arr = np.asarray(closes, dtype=float)
    meta = {
        "label_key": label_key,
        "label_method": lab.method,
        "horizon": int(lab.horizon),
        "n_features_raw": len(frame.feature_names),
        "n_rows": int(len(y)),
        "class_balance": {
            str(k): float(np.mean(y == k)) for k in sorted(set(y.tolist()))
        },
    }
    return X, y, ts_rows, list(frame.feature_names), closes_arr, meta


# ==============================================================================
# Walk-forward research loop
# ==============================================================================


def _model_extra(model_type: str) -> Dict[str, Any]:
    if model_type == "lightgbm":
        return {"verbose": -1}
    if model_type == "xgboost":
        return {"verbosity": 0}
    return {}


def research_symbol_timeframe(
    candles: Sequence[CandleDict],
    symbol: str,
    timeframe: str,
) -> Dict[str, Any]:
    X, y, timestamps, feature_names, closes, meta = build_xy(candles, symbol, timeframe)
    if len(y) < max(2000, FOLDS * 200):
        return {
            "status": "insufficient_data",
            "rows": int(len(y)),
            "meta": meta,
        }

    # Remap labels for classifiers that dislike negatives: -1,0,1 -> 0,1,2
    classes = sorted(set(y.tolist()))
    class_to_idx = {c: i for i, c in enumerate(classes)}
    idx_to_class = {i: c for c, i in class_to_idx.items()}
    y_enc = np.asarray([class_to_idx[v] for v in y], dtype=int)

    result: Dict[str, Any] = {
        "status": "ok",
        "symbol": symbol,
        "timeframe": timeframe,
        "meta": meta,
        "classes": classes,
        "models": {},
    }

    config = create_ai_config(
        symbols=[symbol],
        timeframes=[timeframe],
        primary_timeframe=timeframe,
        datasets={
            "feature_selection_k": TOP_K,
            "walk_forward_folds": FOLDS,
            "walk_forward_embargo": EMBARGO,
        },
        execution={"commission_per_lot": COMMISSION, "slippage_points": SLIPPAGE},
        training={"resume_from_checkpoint": False, "verbose": 0},
        storage={"root_dir": str(ARTIFACT_DIR)},
        model={"task": "classification", "n_estimators": 150, "max_depth": 5, "learning_rate": 0.05},
    )

    from ai.preprocessing.splitter import TimeSeriesSplitter

    splitter = TimeSeriesSplitter(config)
    folds = splitter.walk_forward_indices(len(X), n_folds=FOLDS, embargo=EMBARGO)
    evaluator = Evaluator(config=config, task="classification")

    for model_type in MODELS:
        model_report: Dict[str, Any] = {"model_type": model_type, "folds": []}
        try:
            fold_clf_scores = []
            fold_fin_scores = []
            selected_counts = []
            all_test_pred = []
            all_test_true = []
            all_test_ts = []
            all_test_close = []

            for fold_i, split in enumerate(folds):
                X_train, y_train = X[split.train], y_enc[split.train]
                X_val, y_val = X[split.val], y_enc[split.val]
                X_test, y_test = X[split.test], y_enc[split.test]

                # Feature selection ONLY on train
                selector = FeatureSelector(config=config, method="mutual_info")
                selector.fit(X_train, y_train, top_k=TOP_K)
                X_tr = selector.transform(X_train)
                X_va = selector.transform(X_val)
                X_te = selector.transform(X_test)
                selected_counts.append(int(X_tr.shape[1]))
                selected_names = [feature_names[i] for i in selector.selected_indices_.tolist()]

                cfg = create_ai_config(
                    symbols=[symbol],
                    timeframes=[timeframe],
                    primary_timeframe=timeframe,
                    model={
                        "model_type": model_type,
                        "task": "classification",
                        "n_estimators": 150,
                        "max_depth": 5,
                        "learning_rate": 0.05,
                        "extra_params": _model_extra(model_type),
                    },
                    training={"resume_from_checkpoint": False, "verbose": 0},
                    execution={"commission_per_lot": COMMISSION, "slippage_points": SLIPPAGE},
                    storage={"root_dir": str(ARTIFACT_DIR)},
                )
                model = create_model(model_type, cfg)
                t0 = time.perf_counter()
                model.fit(X_tr, y_train, X_val=X_va, y_val=y_val)
                train_s = time.perf_counter() - t0
                pred = np.asarray(model.predict(X_te)).reshape(-1)
                proba = model.predict_proba(X_te)

                # Classification metrics on encoded labels
                report = evaluator.evaluate(
                    y_test,
                    pred,
                    y_proba=proba,
                    feature_names=selected_names,
                    model=model,
                )

                # Primary financial metrics: barrier-aligned return proxy with bps costs.
                # direction from predicted class; realized move from label horizon closes.
                trade_returns: List[float] = []
                signals: List[BacktestSignal] = []
                cost = COST_BPS / 10000.0
                for local_i, global_i in enumerate(split.test):
                    cls = idx_to_class[int(pred[local_i])]
                    if cls == 0:
                        continue
                    direction = 1.0 if cls > 0 else -1.0
                    j = global_i + HORIZON
                    if j >= len(closes):
                        continue
                    p0 = float(closes[global_i])
                    p1 = float(closes[j])
                    if not (np.isfinite(p0) and np.isfinite(p1) and p0 > 0):
                        continue
                    ret = direction * ((p1 - p0) / p0) - cost
                    trade_returns.append(float(ret))
                    ts = timestamps[global_i]
                    side = "buy" if direction > 0 else "sell"
                    signals.append(
                        BacktestSignal(
                            symbol=symbol,
                            timestamp=ts,
                            side=side,
                            order_type="market",
                            quantity=1.0,
                            price=p0,
                            stop_loss=p0 * (0.998 if side == "buy" else 1.002),
                            take_profit=p0 * (1.004 if side == "buy" else 0.996),
                            timeframe=timeframe,
                        )
                    )

                fin = trading_metrics(trade_returns, initial_equity=1.0, periods=252)
                fin["total_trades"] = len(trade_returns)
                fin["cost_bps"] = COST_BPS

                # Optional engine path (secondary) using OHLC path for the fold
                candle_map = {_naive(c["timestamp"]): c for c in candles}
                dense = []
                for global_i in split.test:
                    ts = timestamps[global_i]
                    src = candle_map.get(ts)
                    if src:
                        dense.append({**src, "timestamp": ts})
                if dense and signals:
                    engine = BacktestEngine(config=cfg, initial_equity=10_000.0)
                    bt = engine.run(signals, dense)
                    fin["engine_final_equity"] = bt.final_equity
                    fin["engine_total_return"] = bt.metrics.get("total_return")
                    fin["engine_max_drawdown"] = bt.metrics.get("max_drawdown")

                fold_clf_scores.append(
                    {
                        "accuracy": report.classification.get("accuracy"),
                        "f1": report.classification.get("f1")
                        or report.classification.get("f1_weighted")
                        or report.classification.get("f1_macro"),
                    }
                )
                fold_fin_scores.append(fin)
                all_test_pred.extend(pred.tolist())
                all_test_true.extend(y_test.tolist())
                all_test_ts.extend([timestamps[i] for i in split.test])
                all_test_close.extend([closes[i] for i in split.test])

                model_report["folds"].append(
                    {
                        "fold": fold_i,
                        "train_rows": int(len(split.train)),
                        "val_rows": int(len(split.val)),
                        "test_rows": int(len(split.test)),
                        "selected_features": int(X_tr.shape[1]),
                        "train_seconds": round(train_s, 4),
                        "classification": {
                            k: v
                            for k, v in report.classification.items()
                            if isinstance(v, (int, float, str))
                        },
                        "financial": fin,
                        "top_features": selected_names[:15],
                    }
                )

            # Aggregate
            model_report["aggregate"] = {
                "mean_selected_features": float(np.mean(selected_counts)) if selected_counts else 0,
                "classification_mean": {
                    "accuracy": float(np.nanmean([s.get("accuracy") or np.nan for s in fold_clf_scores])),
                    "f1": float(np.nanmean([s.get("f1") or np.nan for s in fold_clf_scores])),
                },
                "financial_mean": {
                    key: float(np.nanmean([f.get(key) if f.get(key) is not None else np.nan for f in fold_fin_scores]))
                    for key in (
                        "profit_factor",
                        "sharpe",
                        "sortino",
                        "max_drawdown",
                        "win_rate",
                        "expectancy",
                        "total_trades",
                        "total_return",
                    )
                },
            }
            model_report["status"] = "ok"
        except Exception as exc:
            model_report["status"] = "error"
            model_report["error"] = f"{type(exc).__name__}: {exc}"
            model_report["traceback"] = traceback.format_exc()[-2500:]
        result["models"][model_type] = model_report

    # Best model by mean walk-forward Sharpe (financial, not accuracy)
    best = None
    best_sharpe = -1e18
    for name, mr in result["models"].items():
        if mr.get("status") != "ok":
            continue
        sharpe = mr.get("aggregate", {}).get("financial_mean", {}).get("sharpe")
        if sharpe is None or sharpe != sharpe:
            continue
        if sharpe > best_sharpe:
            best_sharpe = sharpe
            best = name
    result["best_model_by_sharpe"] = best
    result["best_mean_sharpe"] = best_sharpe if best else None
    return result


def main() -> int:
    print("=" * 72)
    print("Phase 4 — Edge Research (walk-forward + triple-barrier)")
    print("=" * 72)
    print("Symbols:", SYMBOLS)
    print("Timeframes:", TIMEFRAMES)
    print("Models:", MODELS)
    print("Folds/Embargo/Horizon/TopK:", FOLDS, EMBARGO, HORIZON, TOP_K)

    db, repos, source = _ensure_db()
    report: Dict[str, Any] = {
        "phase": 4,
        "started_at": datetime.utcnow().isoformat() + "Z",
        "data_source": source,
        "scientific_claim_allowed": source == "mt5_download",
        "config": {
            "symbols": SYMBOLS,
            "timeframes": TIMEFRAMES,
            "models": MODELS,
            "max_bars": MAX_BARS,
            "top_k": TOP_K,
            "folds": FOLDS,
            "embargo": EMBARGO,
            "horizon": HORIZON,
            "commission_per_lot": COMMISSION,
            "slippage_points": SLIPPAGE,
        },
        "pairs": {},
        "notes": [],
    }

    if source != "mt5_download":
        report["notes"].append(
            "Data source is NOT live MT5 history. Results validate the research "
            "methodology but do NOT demonstrate a real market edge."
        )
    else:
        report["notes"].append(
            "MT5 download path succeeded. Walk-forward financial metrics may be "
            "interpreted as preliminary out-of-sample evidence (still verify costs/spread)."
        )

    for symbol in SYMBOLS:
        for timeframe in TIMEFRAMES:
            key = f"{symbol}_{timeframe}"
            print(f"\n>>> Research {key}")
            try:
                candles = load_candles(repos, symbol, timeframe, MAX_BARS)
                print(f"  bars={len(candles)} source={source}")
                pair_result = research_symbol_timeframe(candles, symbol, timeframe)
                report["pairs"][key] = pair_result
                if pair_result.get("status") == "ok":
                    for m_name, m_res in pair_result["models"].items():
                        if m_res.get("status") == "ok":
                            agg = m_res["aggregate"]
                            print(
                                f"  {m_name}: acc={agg['classification_mean']['accuracy']:.4f} "
                                f"sharpe={agg['financial_mean']['sharpe']} "
                                f"pf={agg['financial_mean']['profit_factor']} "
                                f"mdd={agg['financial_mean']['max_drawdown']} "
                                f"trades={agg['financial_mean']['total_trades']}"
                            )
                        else:
                            print(f"  {m_name}: ERROR {m_res.get('error')}")
                else:
                    print(f"  skipped: {pair_result.get('status')} rows={pair_result.get('rows')}")
            except Exception as exc:
                report["pairs"][key] = {
                    "status": "error",
                    "error": f"{type(exc).__name__}: {exc}",
                    "traceback": traceback.format_exc()[-2500:],
                }
                print(f"  ERROR: {type(exc).__name__}: {exc}")

    # Cross-pair summary
    summary_rows = []
    for key, pair in report["pairs"].items():
        if pair.get("status") != "ok":
            continue
        for m_name, m_res in pair.get("models", {}).items():
            if m_res.get("status") != "ok":
                continue
            fin = m_res["aggregate"]["financial_mean"]
            clf = m_res["aggregate"]["classification_mean"]
            summary_rows.append(
                {
                    "pair": key,
                    "model": m_name,
                    "accuracy": clf.get("accuracy"),
                    "f1": clf.get("f1"),
                    "sharpe": fin.get("sharpe"),
                    "sortino": fin.get("sortino"),
                    "profit_factor": fin.get("profit_factor"),
                    "max_drawdown": fin.get("max_drawdown"),
                    "win_rate": fin.get("win_rate"),
                    "expectancy": fin.get("expectancy"),
                    "total_trades": fin.get("total_trades"),
                    "total_return": fin.get("total_return"),
                }
            )
    report["leaderboard"] = sorted(
        summary_rows,
        key=lambda r: (r.get("sharpe") is not None, r.get("sharpe") or -1e18),
        reverse=True,
    )
    report["finished_at"] = datetime.utcnow().isoformat() + "Z"
    report["gates"] = {
        "methodology_ready": True,
        "real_mt5_history": source == "mt5_download",
        "walk_forward": True,
        "triple_barrier": True,
        "feature_selection_train_only": True,
        "financial_metrics": True,
        "edge_demonstrated": False,  # never true on synthetic; set manually after real MT5 review
    }

    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    out = ARTIFACT_DIR / "phase4_edge_report.json"
    out.write_text(json.dumps(_json_safe(report), indent=2), encoding="utf-8")
    print("\nReport:", out)
    print("Data source:", source)
    print("Scientific claim allowed:", report["scientific_claim_allowed"])
    print("Leaderboard top:")
    for row in report["leaderboard"][:5]:
        print(" ", row)
    db.close()

    # Exit 0 if methodology completed; edge claim is separate
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
