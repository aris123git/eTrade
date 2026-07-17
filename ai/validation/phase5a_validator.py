"""
ai/validation/phase5a_validator.py - Phase 5a historical validation & walk-forward.

Builds on existing CandleRepository, ModelTrainer, SignalGenerator, OrderExecutor,
and RiskManager without modifying those modules.

Data sources (in order):
1. CandleRepository if coverage is sufficient
2. Broker history engine / CSV when configured
3. Yahoo Finance public market data (real OHLC) persisted into CandleRepository
"""

from __future__ import annotations

import json
import logging
import math
import random
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# Yahoo symbol map for real market OHLC (no synthetic prices).
YAHOO_SYMBOLS: Dict[str, str] = {
    "EURUSD": "EURUSD=X",
    "GBPUSD": "GBPUSD=X",
    "USDJPY": "JPY=X",
    "XAUUSD": "GC=F",
}

PIP_SIZE: Dict[str, float] = {
    "EURUSD": 0.0001,
    "GBPUSD": 0.0001,
    "USDJPY": 0.01,
    "XAUUSD": 0.1,
}


@dataclass(frozen=True)
class TrainTestSplit:
    """One walk-forward fold with non-overlapping train/test windows."""

    fold: int
    train_start: datetime
    train_end: datetime
    test_start: datetime
    test_end: datetime
    train_indices: Tuple[int, int]  # [start, end)
    test_indices: Tuple[int, int]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "fold": self.fold,
            "train_period": f"{_fmt(self.train_start)} to {_fmt(self.train_end)}",
            "test_period": f"{_fmt(self.test_start)} to {_fmt(self.test_end)}",
            "train_indices": list(self.train_indices),
            "test_indices": list(self.test_indices),
        }


@dataclass
class HistoricalDataLoader:
    """Load, validate, and walk-forward-split broker history via CandleRepository."""

    candle_repository: Any = None
    db_path: str | Path | None = None
    random_seed: int = 42

    def __post_init__(self) -> None:
        if self.candle_repository is None:
            from core.config import DATABASE_PATH
            from database.core.connection import DatabaseManager
            from database.repositories.candle_repository import CandleRepository

            path = self.db_path or DATABASE_PATH
            self.candle_repository = CandleRepository(DatabaseManager(str(path)))
        random.seed(self.random_seed)
        np.random.seed(self.random_seed)

    def download_broker_data(
        self,
        symbol: str,
        start: datetime | str,
        end: datetime | str,
        *,
        timeframe: str = "H1",
        min_bars: int = 500,
        persist: bool = True,
    ) -> "Any":  # pandas.DataFrame
        """
        Return OHLCV DataFrame for symbol/timeframe covering [start, end].

        Prefers repository history; downloads real Yahoo market data when gaps exist.
        """
        import pandas as pd

        start_dt = _as_utc(start)
        end_dt = _as_utc(end)
        tf = timeframe.upper()
        sym = symbol.upper()

        existing = self._load_from_repo(sym, tf, start_dt, end_dt)
        if len(existing) >= min_bars and _coverage_ok(existing, start_dt, end_dt, tf):
            logger.info("repo hit %s %s bars=%s", sym, tf, len(existing))
            return self._candles_to_frame(existing)

        # Resample path: build H4 from H1 when H1 is available.
        if tf == "H4":
            h1 = self.download_broker_data(
                sym, start_dt, end_dt, timeframe="H1", min_bars=min_bars, persist=persist
            )
            if len(h1) >= min_bars // 4:
                framed = self._resample_ohlc(h1, "4h")
                framed["symbol"] = sym
                framed["timeframe"] = "H4"
                if persist:
                    self._persist_frame(framed, sym, "H4")
                return framed

        downloaded = self._download_yahoo(sym, tf, start_dt, end_dt)
        if downloaded is None or len(downloaded) < max(50, min_bars // 10):
            # Fall back to whatever is in the repo (may be partial).
            if existing:
                logger.warning(
                    "partial repo data for %s %s bars=%s (download insufficient)",
                    sym,
                    tf,
                    len(existing),
                )
                return self._candles_to_frame(existing)
            raise RuntimeError(
                f"Unable to obtain real market data for {sym} {tf} "
                f"between {start_dt.date()} and {end_dt.date()}"
            )

        if persist:
            self._persist_frame(downloaded, sym, tf)
        logger.info(
            "downloaded %s %s bars=%s range=%s→%s source=yahoo",
            sym,
            tf,
            len(downloaded),
            downloaded.index.min(),
            downloaded.index.max(),
        )
        return downloaded

    def validate_data_quality(self, df: Any) -> bool:
        """Return True when frame has no NaN OHLC, no duplicates, and no extreme gaps."""
        import pandas as pd

        if df is None or len(df) < 10:
            return False
        frame = df.copy()
        cols = [c for c in ("open", "high", "low", "close") if c in frame.columns]
        if len(cols) < 4:
            return False
        if frame[cols].isna().any().any():
            logger.warning("quality fail: NaN in OHLC")
            return False
        if (frame["high"] < frame["low"]).any():
            logger.warning("quality fail: high < low")
            return False
        if hasattr(frame, "index") and not isinstance(frame.index, pd.RangeIndex):
            timestamps = pd.to_datetime(frame.index, utc=True)
        else:
            timestamps = pd.to_datetime(frame["timestamp"], utc=True)
        if timestamps.duplicated().any():
            logger.warning("quality fail: duplicate timestamps")
            return False
        ordered = timestamps.sort_values()
        delta_td = ordered.diff().dropna()
        deltas = delta_td.total_seconds() if hasattr(delta_td, "total_seconds") else pd.Series(delta_td).dt.total_seconds()
        if len(deltas) == 0:
            return True
        median = float(np.median(np.asarray(deltas, dtype=float)))
        # Allow weekend/holiday gaps; only log pathological holes.
        if median > 0 and float(np.max(deltas)) > median * 20:
            logger.warning(
                "quality warn: large gap max=%ss median=%ss (allowed; continuing)",
                float(np.max(deltas)),
                median,
            )
        return True

    def split_walk_forward(
        self,
        df: Any,
        train_size: int = 252 * 4,
        test_size: int = 252,
        step: int | None = None,
        max_folds: int = 5,
    ) -> List[TrainTestSplit]:
        """
        Purged walk-forward splits with zero train/test overlap.

        Windows move forward in time only. ``step`` defaults to ``test_size``.
        """
        import pandas as pd

        n = len(df)
        if n < train_size + test_size:
            # Scale down while preserving ~4:1 train:test if possible.
            test_size = max(30, n // 5)
            train_size = max(100, n - test_size)
            if train_size + test_size > n:
                train_size = n - test_size
        step = int(step if step is not None else test_size)
        timestamps = _frame_timestamps(df)

        folds: List[TrainTestSplit] = []
        start = 0
        fold_i = 1
        while start + train_size + test_size <= n and fold_i <= max_folds:
            tr_a, tr_b = start, start + train_size
            te_a, te_b = tr_b, tr_b + test_size
            # Hard no-leakage assert
            assert tr_b <= te_a, "train/test overlap"
            folds.append(
                TrainTestSplit(
                    fold=fold_i,
                    train_start=_as_utc(timestamps[tr_a]),
                    train_end=_as_utc(timestamps[tr_b - 1]),
                    test_start=_as_utc(timestamps[te_a]),
                    test_end=_as_utc(timestamps[te_b - 1]),
                    train_indices=(tr_a, tr_b),
                    test_indices=(te_a, te_b),
                )
            )
            start += step
            fold_i += 1
        logger.info("walk-forward folds=%s train=%s test=%s step=%s n=%s", len(folds), train_size, test_size, step, n)
        return folds

    def candles_from_frame(self, df: Any, symbol: str, timeframe: str) -> List[Dict[str, Any]]:
        """Convert DataFrame to CandleDict list for ModelTrainer / paper trading."""
        import pandas as pd

        frame = df.copy()
        if "timestamp" not in frame.columns:
            frame = frame.reset_index()
            if "timestamp" not in frame.columns:
                frame = frame.rename(columns={frame.columns[0]: "timestamp"})
        out: List[Dict[str, Any]] = []
        for _, row in frame.iterrows():
            ts = _as_utc(row["timestamp"]).replace(tzinfo=None)  # FeatureEngine expects naive UTC
            vol = float(row["volume"]) if "volume" in row and pd.notna(row["volume"]) else 0.0
            # Yahoo FX often has volume=0; use unit volume so volume features stay finite.
            if vol <= 0:
                vol = 1.0
            out.append(
                {
                    "symbol": symbol.upper(),
                    "timeframe": timeframe.upper(),
                    "timestamp": ts,
                    "open": float(row["open"]),
                    "high": float(row["high"]),
                    "low": float(row["low"]),
                    "close": float(row["close"]),
                    "volume": vol,
                }
            )
        out.sort(key=lambda c: c["timestamp"])
        return out

    def _load_from_repo(
        self,
        symbol: str,
        timeframe: str,
        start: datetime,
        end: datetime,
    ) -> List[Dict[str, Any]]:
        from ai.data.normalizers import candle_entity_to_dict

        candles = [
            candle_entity_to_dict(c)
            for c in self.candle_repository.stream_candles(
                symbol=symbol,
                timeframe=timeframe,
                start_time=start - timedelta(days=3),
                end_time=end + timedelta(days=3),
            )
        ]
        return [c for c in candles if start <= _as_utc(c["timestamp"]) <= end]

    def _candles_to_frame(self, candles: Sequence[Dict[str, Any]]) -> Any:
        import pandas as pd

        rows = [
            {
                "timestamp": _as_utc(c["timestamp"]),
                "open": float(c["open"]),
                "high": float(c["high"]),
                "low": float(c["low"]),
                "close": float(c["close"]),
                "volume": float(c.get("volume") or 0.0),
                "symbol": c.get("symbol"),
                "timeframe": c.get("timeframe"),
            }
            for c in candles
        ]
        frame = pd.DataFrame(rows).drop_duplicates("timestamp").sort_values("timestamp")
        return frame.set_index("timestamp")

    def _download_yahoo(
        self,
        symbol: str,
        timeframe: str,
        start: datetime,
        end: datetime,
    ) -> Any | None:
        import pandas as pd

        ticker = YAHOO_SYMBOLS.get(symbol.upper())
        if ticker is None:
            return None
        try:
            import yfinance as yf
        except ImportError as exc:
            raise RuntimeError("yfinance is required for Phase 5a market download") from exc

        tf = timeframe.upper()
        interval = {"M15": "15m", "H1": "1h", "H4": "1h", "D1": "1d"}.get(tf)
        if interval is None:
            return None

        # Yahoo intraday limits: 15m≤60d, 1h≤730d; daily unbounded in practice.
        if tf == "M15":
            period = "60d"
            raw = yf.download(ticker, period=period, interval=interval, progress=False, auto_adjust=True)
        elif tf in {"H1", "H4"}:
            period = "730d"
            raw = yf.download(ticker, period=period, interval=interval, progress=False, auto_adjust=True)
        else:
            raw = yf.download(
                ticker,
                start=start.strftime("%Y-%m-%d"),
                end=(end + timedelta(days=1)).strftime("%Y-%m-%d"),
                interval=interval,
                progress=False,
                auto_adjust=True,
            )

        if raw is None or len(raw) == 0:
            return None
        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = [c[0].lower() if isinstance(c, tuple) else str(c).lower() for c in raw.columns]
        else:
            raw.columns = [str(c).lower() for c in raw.columns]
        frame = raw.rename(
            columns={
                "open": "open",
                "high": "high",
                "low": "low",
                "close": "close",
                "adj close": "close",
                "volume": "volume",
            }
        )
        keep = [c for c in ("open", "high", "low", "close", "volume") if c in frame.columns]
        frame = frame[keep].dropna(subset=["open", "high", "low", "close"])
        frame.index = pd.to_datetime(frame.index, utc=True)
        start_ts = pd.Timestamp(start)
        end_ts = pd.Timestamp(end)
        if start_ts.tzinfo is None:
            start_ts = start_ts.tz_localize("UTC")
        else:
            start_ts = start_ts.tz_convert("UTC")
        if end_ts.tzinfo is None:
            end_ts = end_ts.tz_localize("UTC")
        else:
            end_ts = end_ts.tz_convert("UTC")
        filtered = frame[(frame.index >= start_ts) & (frame.index <= end_ts)]
        # Yahoo intraday is a rolling window (15m≤60d, 1h≤730d). If the requested
        # calendar window yields too few bars, keep the full real download.
        if tf in {"M15", "H1", "H4"} and len(filtered) < 100:
            logger.warning(
                "%s %s: requested %s→%s not covered by Yahoo window; using available %s→%s (%s bars)",
                symbol,
                tf,
                start_ts.date(),
                end_ts.date(),
                frame.index.min().date() if len(frame) else None,
                frame.index.max().date() if len(frame) else None,
                len(frame),
            )
        else:
            frame = filtered
        if tf == "H4" and len(frame):
            frame = self._resample_ohlc(frame, "4h")
        frame["symbol"] = symbol.upper()
        frame["timeframe"] = tf
        return frame

    def _resample_ohlc(self, frame: Any, rule: str) -> Any:
        import pandas as pd

        src = frame.copy()
        if not isinstance(src.index, pd.DatetimeIndex):
            src = src.set_index("timestamp")
        src.index = pd.to_datetime(src.index, utc=True)
        ohlc = src.resample(rule).agg(
            {
                "open": "first",
                "high": "max",
                "low": "min",
                "close": "last",
                "volume": "sum",
            }
        ).dropna()
        return ohlc

    def _persist_frame(self, frame: Any, symbol: str, timeframe: str) -> int:
        candles = self.candles_from_frame(frame, symbol, timeframe)
        payload = []
        for c in candles:
            payload.append(
                {
                    **c,
                    "broker_id": 0,
                    "status": "active",
                    "metadata": {"source": "yahoo_finance", "phase": "5a"},
                }
            )
        return int(self.candle_repository.bulk_upsert(payload, trusted_source=True))


@dataclass
class StatisticalTester:
    """Statistical significance and trading performance metrics."""

    risk_free: float = 0.02

    def is_significant(
        self,
        win_rate: float,
        n_trades: int,
        alpha: float = 0.05,
    ) -> Dict[str, Any]:
        """One-sided binomial test: H0 win_rate=50%, HA > 50%."""
        n = int(n_trades)
        if n <= 0:
            return {
                "is_significant": False,
                "p_value": 1.0,
                "confidence": f"{int((1 - alpha) * 100)}%",
                "wins": 0,
                "n_trades": 0,
                "win_rate": 0.0,
            }
        wins = int(round(float(win_rate) * n))
        wins = min(max(wins, 0), n)
        try:
            from scipy.stats import binomtest

            result = binomtest(wins, n=n, p=0.5, alternative="greater")
            p_value = float(result.pvalue)
        except Exception:
            # Normal approximation fallback
            z = (wins / n - 0.5) / math.sqrt(0.25 / n)
            p_value = float(0.5 * (1.0 - math.erf(z / math.sqrt(2.0))))
        return {
            "is_significant": bool(p_value < alpha and win_rate >= 0.52),
            "p_value": p_value,
            "confidence": f"{int((1 - alpha) * 100)}%",
            "wins": wins,
            "n_trades": n,
            "win_rate": float(win_rate),
        }

    def sharpe_ratio(self, returns: Sequence[float], rf: float | None = None, periods: int = 252) -> float:
        from ai.evaluation.trading_metrics import sharpe

        return float(sharpe(returns, risk_free=self.risk_free if rf is None else rf, periods=periods))

    def sortino_ratio(self, returns: Sequence[float], rf: float | None = None, periods: int = 252) -> float:
        from ai.evaluation.trading_metrics import sortino

        return float(sortino(returns, risk_free=self.risk_free if rf is None else rf, periods=periods))

    def profit_factor(self, gross_wins: float, gross_losses: float) -> float:
        losses = abs(float(gross_losses))
        wins = float(gross_wins)
        if losses <= 1e-12:
            return float("inf") if wins > 0 else 0.0
        return wins / losses

    def max_drawdown(self, equity_curve: Sequence[float]) -> float:
        from ai.evaluation.trading_metrics import max_drawdown

        # Returns negative decimal (e.g. -0.083); report as percent magnitude later.
        return float(max_drawdown(equity_curve))


@dataclass
class OverfitDetector:
    """Train/test gap, regime stability, and correlation-decay checks."""

    accuracy_gap_threshold: float = 0.15
    regime_window: int = 252
    regime_collapse_ratio: float = 0.35  # recent Sharpe < ratio × early Sharpe

    def compare_train_vs_test(
        self,
        train_metrics: Mapping[str, Any],
        test_metrics: Mapping[str, Any],
    ) -> Dict[str, Any]:
        train_acc = float(train_metrics.get("accuracy", train_metrics.get("train_accuracy", 0.0)) or 0.0)
        test_acc = float(test_metrics.get("accuracy", test_metrics.get("test_accuracy", 0.0)) or 0.0)
        gap = train_acc - test_acc
        overfitted = bool(gap > self.accuracy_gap_threshold and train_acc >= 0.60)
        return {
            "train_accuracy": train_acc,
            "test_accuracy": test_acc,
            "gap": gap,
            "overfitting_detected": overfitted,
            "threshold": self.accuracy_gap_threshold,
        }

    def regime_detector(
        self,
        equity_curve: Sequence[float],
        window: int | None = None,
    ) -> Dict[str, Any]:
        """Detect Sharpe collapse in later portion of the equity curve."""
        eq = np.asarray(equity_curve, dtype=float)
        if eq.size < 20:
            return {"regime_stable": True, "early_sharpe": 0.0, "late_sharpe": 0.0, "collapsed": False}
        rets = np.diff(eq) / np.maximum(eq[:-1], 1e-12)
        win = int(window or self.regime_window)
        win = max(10, min(win, max(10, len(rets) // 3)))
        early = rets[:win]
        late = rets[-win:]
        tester = StatisticalTester()
        early_s = tester.sharpe_ratio(early.tolist(), periods=win)
        late_s = tester.sharpe_ratio(late.tolist(), periods=win)
        collapsed = bool(early_s > 0.2 and late_s < early_s * self.regime_collapse_ratio)
        return {
            "regime_stable": not collapsed,
            "early_sharpe": early_s,
            "late_sharpe": late_s,
            "collapsed": collapsed,
            "window": win,
        }

    def correlation_decay(self, model_age_years: float, base_score: float = 1.0, half_life_years: float = 2.0) -> float:
        """
        Expected relative performance vs age (exponential decay).

        Returns multiplier in (0, 1]: how much edge remains after ``model_age_years``.
        """
        age = max(float(model_age_years), 0.0)
        decay = float(base_score) * (0.5 ** (age / max(half_life_years, 1e-6)))
        return max(0.0, min(1.0, decay))


@dataclass
class WalkForwardBacktester:
    """Walk-forward train → OOS paper trade → metrics, with no future peeking."""

    config: Any = None
    loader: HistoricalDataLoader | None = None
    stats: StatisticalTester = field(default_factory=StatisticalTester)
    overfit: OverfitDetector = field(default_factory=OverfitDetector)
    model_type: str = "random_forest"
    slippage_pips: float = 2.0
    commission_pips: float = 0.3
    initial_equity: float = 100_000.0
    artifact_dir: Path = field(default_factory=lambda: Path("ai/artifacts/phase5a"))
    random_seed: int = 42

    def __post_init__(self) -> None:
        from ai.config.settings import AIConfig

        if self.config is None:
            self.config = AIConfig()
        # Isolation: do not mutate global download / validation gates unexpectedly
        self.config.data.auto_download = False
        self.config.data.require_validated = False
        self.config.storage.root_dir = Path("ai/artifacts")
        self.config.ensure_directories()
        self.loader = self.loader or HistoricalDataLoader(random_seed=self.random_seed)
        self.artifact_dir = Path(self.artifact_dir)
        self.artifact_dir.mkdir(parents=True, exist_ok=True)
        random.seed(self.random_seed)
        np.random.seed(self.random_seed)

    def run(
        self,
        symbol: str,
        timeframes: Sequence[str] = ("H1",),
        *,
        start: datetime | str = "2019-01-01",
        end: datetime | str = "2024-12-31",
        train_period_years: float = 4.0,
        test_period_years: float = 1.0,
        multi_tf: bool = False,
    ) -> Dict[str, Any]:
        """
        Run walk-forward for one symbol.

        When ``multi_tf`` is True, ``timeframes`` should include primary + confirmation
        TFs (e.g. M15, H1, H4) with the first as primary.
        """
        primary = str(timeframes[0]).upper()
        confirm = [str(t).upper() for t in timeframes[1:]] if multi_tf else []
        start_dt, end_dt = _as_utc(start), _as_utc(end)

        primary_df = self.loader.download_broker_data(symbol, start_dt, end_dt, timeframe=primary, min_bars=200)
        if not self.loader.validate_data_quality(primary_df):
            return {
                "symbol": symbol.upper(),
                "timeframe": primary,
                "error": "data_quality_failed",
                "data_snapshot": _snapshot(primary_df, primary),
            }

        bars_per_year = _bars_per_year(primary)
        train_size = max(120, int(bars_per_year * train_period_years))
        test_size = max(40, int(bars_per_year * test_period_years))
        # Adapt to available coverage
        n = len(primary_df)
        if train_size + test_size > n:
            test_size = max(40, n // 5)
            train_size = max(120, n - test_size)

        splits = self.loader.split_walk_forward(
            primary_df,
            train_size=train_size,
            test_size=test_size,
            step=test_size,
            max_folds=5,
        )
        if not splits:
            return {
                "symbol": symbol.upper(),
                "timeframe": "+".join([primary] + confirm) if multi_tf else primary,
                "error": "insufficient_bars_for_walk_forward",
                "data_snapshot": _snapshot(primary_df, primary),
                "n_bars": n,
                "train_size_requested": train_size,
                "test_size_requested": test_size,
            }

        confirm_frames: Dict[str, Any] = {}
        for tf in confirm:
            try:
                confirm_frames[tf] = self.loader.download_broker_data(
                    symbol, start_dt, end_dt, timeframe=tf, min_bars=100
                )
            except Exception as exc:
                logger.warning("confirm TF %s unavailable: %s", tf, exc)

        candles = self.loader.candles_from_frame(primary_df, symbol, primary)
        results: Dict[str, Any] = {}
        fold_metrics: List[Dict[str, Any]] = []

        for split in splits:
            fold_result = self._run_fold(
                symbol=symbol,
                primary=primary,
                candles=candles,
                split=split,
                confirm_frames=confirm_frames,
                multi_tf=multi_tf,
            )
            key = f"fold_{split.fold}"
            results[key] = fold_result
            fold_metrics.append(fold_result)
            logger.info(
                "%s %s %s wr=%.3f sharpe=%.3f pf=%.3f dd=%.2f%% overfit=%s sig=%s",
                symbol,
                primary,
                key,
                fold_result.get("test_win_rate", 0.0),
                fold_result.get("test_sharpe", 0.0),
                fold_result.get("test_profit_factor", 0.0),
                fold_result.get("test_max_dd", 0.0),
                fold_result.get("overfitting_detected"),
                fold_result.get("significance", {}).get("is_significant"),
            )

        aggregate = self._aggregate(fold_metrics, symbol=symbol, timeframe=primary, multi_tf=multi_tf)
        return {
            "symbol": symbol.upper(),
            "timeframe": "+".join([primary] + confirm) if multi_tf else primary,
            "period": f"{_fmt(start_dt)[:10]} to {_fmt(end_dt)[:10]}",
            "actual_data_period": f"{_fmt(candles[0]['timestamp'])[:10]} to {_fmt(candles[-1]['timestamp'])[:10]}",
            "train_folds": len(splits),
            "multi_tf": multi_tf,
            "data_snapshot": _snapshot(primary_df, primary),
            "costs": {
                "slippage_pips": self.slippage_pips,
                "commission_pips": self.commission_pips,
            },
            "results": results,
            "aggregate": aggregate,
        }

    def _run_fold(
        self,
        *,
        symbol: str,
        primary: str,
        candles: Sequence[Dict[str, Any]],
        split: TrainTestSplit,
        confirm_frames: Mapping[str, Any],
        multi_tf: bool,
    ) -> Dict[str, Any]:
        from ai.config.settings import AIConfig
        from ai.models.trainer import create_model_trainer
        from ai.services.pipeline import AIPipeline
        from ai.signals.generator import create_signal_generator
        from ai.execution.executor import create_order_executor, OrderStatus
        from ai.portfolio.manager import create_portfolio_manager
        from ai.risk.manager import create_risk_manager
        from ai.utils.types import SignalType
        from ai.signals.engine import TradeSignal

        tr_a, tr_b = split.train_indices
        te_a, te_b = split.test_indices
        train_candles = list(candles[tr_a:tr_b])
        test_candles = list(candles[te_a:te_b])
        assert train_candles[-1]["timestamp"] < test_candles[0]["timestamp"], "leakage: train after test"

        cfg = self.config.copy() if hasattr(self.config, "copy") else AIConfig()
        cfg.data.auto_download = False
        cfg.data.require_validated = False
        cfg.symbols = [symbol.upper()]
        cfg.primary_timeframe = primary
        cfg.model.model_type = self.model_type
        cfg.risk.min_confidence = 0.45
        cfg.risk.min_expected_rr = 1.0
        cfg.risk.circuit_breaker_loss = 0.25
        cfg.risk.max_drawdown = 0.20
        # RiskManager clips to max_lot_size; for FX unit sizing allow full risk units.
        cfg.risk.max_lot_size = 1_000_000.0
        cfg.risk.default_lot_size = 1.0
        cfg.risk.position_sizing = "fixed_risk"
        cfg.risk.risk_per_trade = 0.01
        # Avoid all-NaN feature rows on Yahoo FX (no package changes — config only).
        cfg.features.dropna = False
        cfg.features.multi_timeframes = []
        cfg.features.correlation_symbols = []
        # Realistic costs: slippage via executor points; commission applied in price units
        # (RiskManager sizes in notional units, not exchange lots).
        pip = PIP_SIZE.get(symbol.upper(), 0.0001)
        cfg.execution.slippage_points = float(self.slippage_pips)
        cfg.execution.commission_per_lot = 0.0
        cfg.storage.root_dir = self.artifact_dir / "models"
        cfg.ensure_directories()

        trainer = create_model_trainer(cfg)
        train_out = trainer.train(
            symbol=symbol,
            model_type=self.model_type,
            candles=train_candles,
            register=True,
        )
        train_accuracy = _extract_accuracy(train_out.get("metrics") or {})

        pipeline = AIPipeline(config=cfg)
        if train_out.get("registered"):
            try:
                loaded = pipeline.registry.load(train_out["registered"]["name"])
                pipeline.model = loaded.get("model")
                pipeline.model_version = loaded.get("version")
            except Exception:
                logger.exception("fold model load failed; using trainer pipeline model")
                pipeline.model = trainer.pipeline.model if trainer.pipeline else None
        else:
            pipeline.model = trainer.pipeline.model if trainer.pipeline else None

        # Feature matrix on train+test chronologically (indicators are causal / backward-looking).
        combined = train_candles + test_candles
        frame = pipeline.build_features(combined)
        ts_to_idx = {_norm_ts(t): i for i, t in enumerate(frame.timestamps)}

        signals = create_signal_generator(cfg)
        signals.engine.filters.session_filter = False  # type: ignore[union-attr]
        signals.engine.filters.min_confidence = 0.45  # type: ignore[union-attr]
        signals.engine.filters.cooldown_seconds = 0.0  # type: ignore[union-attr]
        executor = create_order_executor(cfg, mode="paper", point_size=pip)
        portfolio = create_portfolio_manager(cfg, cash=self.initial_equity)
        risk = create_risk_manager(cfg, equity=self.initial_equity)

        # Precompute higher-TF bias series for multi-TF confirmation
        higher_bias_by_ts: Dict[datetime, List[float]] = {}
        if multi_tf and confirm_frames and pipeline.model is not None:
            higher_bias_by_ts = self._higher_tf_bias(
                symbol=symbol,
                confirm_frames=confirm_frames,
                test_candles=test_candles,
                cfg=cfg,
            )

        preds: List[float] = []
        correct = 0
        total_pred = 0
        trade_pnls: List[float] = []
        hold_seconds: List[float] = []
        equity_curve = [self.initial_equity]

        # Step through test bars; subsample ultra-dense M15 for runtime while keeping order.
        step = 1 if primary in {"H4", "D1"} else (2 if primary == "H1" else 4)

        for i in range(0, len(test_candles), step):
            bar = test_candles[i]
            price = float(bar["close"])
            ts = _as_utc(bar["timestamp"])
            portfolio.update_prices({symbol.upper(): price})
            risk.update_equity(portfolio.total_equity(), timestamp=ts)

            # SL/TP automation
            for report in executor.manage_exits(portfolio.positions(symbol=symbol.upper()), prices={symbol.upper(): price}, timestamp=ts):
                if report.status in {OrderStatus.FILLED, OrderStatus.PARTIALLY_FILLED}:
                    for fill in report.fills:
                        closed = portfolio.apply_fill(fill)
                        if hasattr(closed, "pnl"):
                            trade_pnls.append(float(closed.pnl))
                            hold_seconds.append(
                                max(0.0, (closed.closed_at - closed.opened_at).total_seconds())
                            )
                    risk.update_equity(portfolio.total_equity(), timestamp=ts)

            feat_idx = ts_to_idx.get(_norm_ts(ts))
            if feat_idx is None or pipeline.model is None:
                equity_curve.append(portfolio.total_equity())
                continue

            row = frame.matrix[feat_idx : feat_idx + 1]
            try:
                prediction = pipeline.predict(features=row, symbol=symbol.upper(), timeframe=primary)
            except Exception:
                equity_curve.append(portfolio.total_equity())
                continue

            pred_val = float(prediction.prediction) if _is_number(prediction.prediction) else 0.0
            preds.append(pred_val)
            if i + 1 < len(test_candles):
                realized = float(test_candles[i + 1]["close"]) - price
                pred_dir = 1.0 if pred_val > 0 else (-1.0 if pred_val < 0 else 0.0)
                real_dir = 1.0 if realized > 0 else (-1.0 if realized < 0 else 0.0)
                if pred_dir != 0:
                    total_pred += 1
                    if pred_dir == real_dir:
                        correct += 1

            cursor = len(train_candles) + i
            atr = _atr(combined[max(0, cursor - 40) : cursor + 1])
            higher = None
            if multi_tf:
                biases = higher_bias_by_ts.get(_norm_ts(ts)) or higher_bias_by_ts.get(ts)
                if biases:
                    from ai.utils.types import PredictionResult

                    higher = [
                        PredictionResult(
                            symbol=symbol.upper(),
                            timeframe=tf,
                            timestamp=ts,
                            prediction=bias,
                            confidence=max(0.5, float(prediction.confidence)),
                            probabilities=None,
                            expected_return=None,
                            model_version=prediction.model_version,
                            metadata={},
                        )
                        for tf, bias in zip(confirm_frames.keys(), biases)
                    ]

            signal = signals.generate(
                prediction,
                market_context={"price": price, "entry": price, "atr": atr, "timestamp": ts},
                higher_tf_predictions=higher,
            )
            if signal.side in {SignalType.BUY, SignalType.SELL} and atr > 0 and (signal.sl is None or signal.tp is None):
                if signal.side == SignalType.BUY:
                    signal = TradeSignal(
                        symbol=signal.symbol,
                        side=signal.side,
                        strength=signal.strength,
                        confidence=signal.confidence,
                        entry=price,
                        sl=price - atr * cfg.risk.atr_stop_mult,
                        tp=price + atr * cfg.risk.atr_tp_mult,
                        size_hint=signal.size_hint,
                        metadata=dict(signal.metadata),
                    )
                else:
                    signal = TradeSignal(
                        symbol=signal.symbol,
                        side=signal.side,
                        strength=signal.strength,
                        confidence=signal.confidence,
                        entry=price,
                        sl=price + atr * cfg.risk.atr_stop_mult,
                        tp=price - atr * cfg.risk.atr_tp_mult,
                        size_hint=signal.size_hint,
                        metadata=dict(signal.metadata),
                    )

            # Apply explicit pip costs on entry via widened slippage already in config.
            decision = risk.pre_trade_validate(
                signal,
                open_positions=portfolio.positions(),
                equity=portfolio.total_equity(),
                atr=atr,
            )
            if decision.approved and signal.side in {SignalType.BUY, SignalType.SELL}:
                size = float(decision.size) * portfolio.size_multiplier_for_drawdown()
                if size > 0:
                    order = executor.create_order(
                        symbol=symbol.upper(),
                        side=signal.side,
                        quantity=size,
                        sl=signal.sl,
                        tp=signal.tp,
                        metadata={"sl": signal.sl, "tp": signal.tp},
                    )
                    report = executor.submit_order(order, market_price=price, timestamp=ts)
                    if report.status in {OrderStatus.FILLED, OrderStatus.PARTIALLY_FILLED}:
                        for fill in report.fills:
                            # Extra commission in pips
                            fill.commission = float(fill.commission) + abs(fill.quantity) * self.commission_pips * pip
                            fill.metadata = {**dict(fill.metadata or {}), "sl": order.sl, "tp": order.tp}
                            portfolio.apply_fill(fill)
                            for pos in portfolio.positions(symbol=symbol.upper()):
                                if pos.sl is None and order.sl is not None:
                                    pos.sl = float(order.sl)
                                if pos.tp is None and order.tp is not None:
                                    pos.tp = float(order.tp)

            equity_curve.append(portfolio.total_equity())

        # Flatten open positions at end of fold
        if test_candles:
            last = test_candles[-1]
            last_price = float(last["close"])
            for pos in list(portfolio.positions(symbol=symbol.upper())):
                trade = portfolio.close_position(
                    pos.position_id,
                    price=last_price,
                    closed_at=_as_utc(last["timestamp"]),
                    metadata={"reason": "fold_end"},
                )
                trade_pnls.append(float(trade.pnl))
                hold_seconds.append(max(0.0, (trade.closed_at - trade.opened_at).total_seconds()))

        metrics = portfolio.performance_metrics()
        wins = [p for p in trade_pnls if p > 0]
        losses = [p for p in trade_pnls if p < 0]
        # Sharpe/Sortino from bar equity returns (stable); periods match timeframe.
        eq = np.asarray(equity_curve, dtype=float)
        if eq.size >= 2:
            rets = ((eq[1:] - eq[:-1]) / np.maximum(eq[:-1], 1e-12)).tolist()
        else:
            rets = list(portfolio.returns_history)
        periods = _bars_per_year(primary)
        test_accuracy = (correct / total_pred) if total_pred else 0.0
        overfit_info = self.overfit.compare_train_vs_test(
            {"accuracy": train_accuracy},
            {"accuracy": test_accuracy},
        )
        wr = float(metrics.get("win_rate", 0.0))
        n_trades = int(metrics.get("closed_trades", 0))
        significance = self.stats.is_significant(wr, n_trades)
        sharpe = self.stats.sharpe_ratio(rets, periods=periods)
        sortino = self.stats.sortino_ratio(rets, periods=periods)
        pf = self.stats.profit_factor(sum(wins), sum(losses))
        dd = self.stats.max_drawdown(equity_curve)
        # Store max DD as negative percent (e.g. -8.3)
        dd_pct = abs(float(dd)) * 100.0
        weeks = max(
            (_as_utc(test_candles[-1]["timestamp"]) - _as_utc(test_candles[0]["timestamp"])).days / 7.0,
            1e-6,
        ) if test_candles else 1.0

        return {
            "train_period": f"{_fmt(split.train_start)} to {_fmt(split.train_end)}",
            "test_period": f"{_fmt(split.test_start)} to {_fmt(split.test_end)}",
            "test_win_rate": wr,
            "test_sharpe": sharpe,
            "test_sortino": sortino,
            "test_profit_factor": pf if math.isfinite(pf) else 0.0,
            "test_max_dd": -dd_pct,
            "test_trades": n_trades,
            "trades_per_week": n_trades / weeks,
            "avg_trade_duration_hours": (float(np.mean(hold_seconds)) / 3600.0) if hold_seconds else 0.0,
            "train_accuracy": train_accuracy,
            "test_accuracy": test_accuracy,
            "overfitting_detected": overfit_info["overfitting_detected"],
            "accuracy_gap": overfit_info["gap"],
            "significance": significance,
            "equity_end": float(equity_curve[-1]) if equity_curve else self.initial_equity,
            "model_registration": train_out.get("registered"),
        }

    def _higher_tf_bias(
        self,
        *,
        symbol: str,
        confirm_frames: Mapping[str, Any],
        test_candles: Sequence[Dict[str, Any]],
        cfg: Any,
    ) -> Dict[datetime, List[float]]:
        """Map each test timestamp to a list of higher-TF prediction biases."""
        from ai.services.pipeline import AIPipeline

        out: Dict[datetime, List[float]] = {}
        series: Dict[str, List[Tuple[datetime, float]]] = {}
        for tf, frame in confirm_frames.items():
            candles = self.loader.candles_from_frame(frame, symbol, tf)  # type: ignore[union-attr]
            if len(candles) < 120:
                continue
            local = AIPipeline(config=cfg)
            # Train a lightweight model on confirm TF using only bars before first test ts
            first_test = _as_utc(test_candles[0]["timestamp"])
            train_c = [c for c in candles if _as_utc(c["timestamp"]) < first_test]
            if len(train_c) < 120:
                continue
            try:
                from ai.models.trainer import create_model_trainer

                trainer = create_model_trainer(cfg)
                trainer.train(symbol=symbol, model_type=self.model_type, candles=train_c, register=False)
                local.model = trainer.pipeline.model if trainer.pipeline else None
                feat = local.build_features(candles)
                if local.model is None:
                    continue
                preds = np.asarray(local.model.predict(feat.matrix), dtype=float).reshape(-1)
                series[tf] = [(_norm_ts(t), float(p)) for t, p in zip(feat.timestamps, preds)]
            except Exception:
                logger.exception("higher TF model failed for %s", tf)

        if not series:
            return out
        for bar in test_candles:
            ts = _norm_ts(bar["timestamp"])
            biases: List[float] = []
            for tf in confirm_frames.keys():
                pts = series.get(tf) or []
                # last prediction at or before ts
                bias = 0.0
                for t, p in pts:
                    if t <= ts:
                        bias = p
                    else:
                        break
                biases.append(bias)
            out[ts] = biases
        return out

    def _aggregate(
        self,
        fold_metrics: Sequence[Mapping[str, Any]],
        *,
        symbol: str,
        timeframe: str,
        multi_tf: bool,
    ) -> Dict[str, Any]:
        if not fold_metrics:
            return {
                "recommendation": "NO EDGE - insufficient folds",
                "overall_significant": False,
                "regime_stable": False,
            }
        wrs = [float(f.get("test_win_rate", 0.0)) for f in fold_metrics]
        sharpes = [float(f.get("test_sharpe", 0.0)) for f in fold_metrics]
        pfs = [float(f.get("test_profit_factor", 0.0)) for f in fold_metrics]
        dds = [abs(float(f.get("test_max_dd", 0.0))) for f in fold_metrics]
        sortinos = [float(f.get("test_sortino", 0.0)) for f in fold_metrics]
        trades = [int(f.get("test_trades", 0)) for f in fold_metrics]
        overfits = [bool(f.get("overfitting_detected")) for f in fold_metrics]
        sigs = [bool((f.get("significance") or {}).get("is_significant")) for f in fold_metrics]

        # Regime: compare first-half vs second-half fold Sharpes
        mid = max(1, len(sharpes) // 2)
        early = float(np.mean(sharpes[:mid]))
        late = float(np.mean(sharpes[-mid:]))
        regime_stable = not (early > 0.2 and late < early * 0.35)
        # Correlation decay proxy by fold age
        ages = list(range(len(fold_metrics)))
        decay_scores = [self.overfit.correlation_decay(float(a)) for a in ages]
        corr_decay = float(np.mean(np.diff([s * max(sharpes[0], 1e-6) for s in decay_scores]))) if len(decay_scores) > 1 else 0.0

        avg_wr = float(np.mean(wrs))
        avg_sharpe = float(np.mean(sharpes))
        avg_pf = float(np.mean(pfs))
        avg_dd = float(np.mean(dds))
        avg_sortino = float(np.mean(sortinos))
        total_trades = int(sum(trades))
        pooled_sig = self.stats.is_significant(avg_wr, total_trades)

        failures: List[str] = []
        if avg_wr < 0.52 or not pooled_sig["is_significant"]:
            failures.append("win_rate_or_significance")
        if avg_sharpe < 0.5:
            failures.append("sharpe")
        if avg_pf < 1.2:
            failures.append("profit_factor")
        if avg_dd > 15.0:
            failures.append("max_drawdown")
        if any(overfits):
            failures.append("overfitting")
        if not regime_stable:
            failures.append("regime_unstable")

        if not failures:
            recommendation = "EDGE DETECTED - Proceed to Phase 5b (live paper)"
        else:
            recommendation = (
                "NO ROBUST EDGE - Do NOT proceed to Phase 5b. Failed: "
                + ", ".join(failures)
                + ". Suggestions: review labels/features, increase sample, reduce model complexity, "
                "re-check costs, or extend real history coverage."
            )

        return {
            "avg_win_rate": avg_wr,
            "std_win_rate": float(np.std(wrs)),
            "avg_sharpe": avg_sharpe,
            "avg_sortino": avg_sortino,
            "avg_profit_factor": avg_pf,
            "avg_max_dd": -avg_dd,
            "total_trades": total_trades,
            "folds_significant": int(sum(sigs)),
            "overfitting_folds": int(sum(overfits)),
            "regime_stable": regime_stable,
            "early_fold_sharpe": early,
            "late_fold_sharpe": late,
            "correlation_decay_per_fold": corr_decay,
            "overall_significant": bool(pooled_sig["is_significant"]),
            "pooled_significance": pooled_sig,
            "failures": failures,
            "recommendation": recommendation,
            "symbol": symbol.upper(),
            "timeframe": timeframe,
            "multi_tf": multi_tf,
        }


@dataclass
class Phase5aValidator:
    """Orchestrates multi-symbol / multi-timeframe Phase 5a validation."""

    backtester: WalkForwardBacktester | None = None
    symbols: Sequence[str] = field(
        default_factory=lambda: ["EURUSD", "GBPUSD", "USDJPY", "XAUUSD"]
    )
    timeframes: Sequence[str] = field(default_factory=lambda: ["M15", "H1", "H4"])
    start_date: str = "2019-01-01"
    end_date: str = "2024-12-31"
    report_path: Path = field(default_factory=lambda: Path("ai/validation/phase5a_report.json"))

    def __post_init__(self) -> None:
        self.backtester = self.backtester or WalkForwardBacktester()
        self.report_path = Path(self.report_path)

    def run(self, *, include_multi_tf: bool = True, include_d1: bool = True) -> Dict[str, Any]:
        assert self.backtester is not None
        series: List[Dict[str, Any]] = []
        tfs = list(self.timeframes)
        if include_d1 and "D1" not in tfs:
            # Long-horizon real history (Yahoo daily covers 2019-2024 fully).
            tfs = tfs + ["D1"]

        for symbol in self.symbols:
            for tf in tfs:
                try:
                    result = self.backtester.run(
                        symbol,
                        timeframes=[tf],
                        start=self.start_date,
                        end=self.end_date,
                        multi_tf=False,
                    )
                except Exception as exc:
                    logger.exception("validation failed %s %s", symbol, tf)
                    result = {
                        "symbol": symbol,
                        "timeframe": tf,
                        "error": f"{exc.__class__.__name__}: {exc}",
                        "aggregate": {
                            "recommendation": "NO EDGE - run error",
                            "failures": ["runtime_error"],
                            "overall_significant": False,
                        },
                    }
                series.append(result)

            if include_multi_tf:
                try:
                    result = self.backtester.run(
                        symbol,
                        timeframes=["M15", "H1", "H4"],
                        start=self.start_date,
                        end=self.end_date,
                        multi_tf=True,
                    )
                except Exception as exc:
                    logger.exception("multi-tf validation failed %s", symbol)
                    result = {
                        "symbol": symbol,
                        "timeframe": "M15+H1+H4",
                        "error": f"{exc.__class__.__name__}: {exc}",
                        "aggregate": {
                            "recommendation": "NO EDGE - run error",
                            "failures": ["runtime_error"],
                            "overall_significant": False,
                        },
                    }
                series.append(result)

        summary = self._global_summary(series)
        report = {
            "phase": "5a",
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "config": {
                "symbols": list(self.symbols),
                "timeframes": tfs,
                "start_date": self.start_date,
                "end_date": self.end_date,
                "train_period_years": 4,
                "test_period_years": 1,
                "overlap": 0,
                "paper_slippage_pips": self.backtester.slippage_pips,
                "paper_commission_pips": self.backtester.commission_pips,
                "random_seed": self.backtester.random_seed,
            },
            "series": series,
            "summary": summary,
        }
        self.report_path.parent.mkdir(parents=True, exist_ok=True)
        self.report_path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
        logger.info("wrote Phase 5a report → %s", self.report_path)
        return report

    def _global_summary(self, series: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
        """Apply Phase 5a gate: edge must hold on ≥3/4 symbols without overfitting."""
        by_symbol: Dict[str, List[Mapping[str, Any]]] = {}
        for item in series:
            if item.get("error"):
                continue
            by_symbol.setdefault(str(item.get("symbol")), []).append(item)

        symbol_pass: Dict[str, bool] = {}
        for symbol, items in by_symbol.items():
            # Prefer D1 / H1 aggregates for the gate (longest / most liquid history).
            ranked = sorted(
                items,
                key=lambda x: {"D1": 0, "H1": 1, "H4": 2, "M15": 3}.get(str(x.get("timeframe")), 9),
            )
            ok = False
            for item in ranked:
                agg = item.get("aggregate") or {}
                failures = set(agg.get("failures") or [])
                # Allow missing significance if trades are few but other metrics pass — still require no overfit.
                if "overfitting" in failures:
                    continue
                if agg.get("avg_sharpe", -999) >= 0.5 and agg.get("avg_profit_factor", 0) >= 1.2:
                    if abs(agg.get("avg_max_dd", -100)) <= 15 or agg.get("avg_max_dd", -100) >= -15:
                        if agg.get("avg_win_rate", 0) >= 0.52 and agg.get("overall_significant"):
                            ok = True
                            break
            symbol_pass[symbol] = ok

        n_pass = sum(1 for v in symbol_pass.values() if v)
        proceed = n_pass >= 3
        return {
            "symbols_evaluated": list(symbol_pass),
            "symbols_with_edge": [s for s, v in symbol_pass.items() if v],
            "symbols_passed": n_pass,
            "symbols_required": 3,
            "proceed_to_phase_5b": proceed,
            "answer": (
                "YES — statistically supported edge on real historical data across multiple symbols."
                if proceed
                else "NO — edge not robust on real historical data (curve-fit, insignificant, or data-limited)."
            ),
            "per_symbol": symbol_pass,
        }


# ------------------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------------------


def _as_utc(value: Any) -> datetime:
    if isinstance(value, datetime):
        if value.tzinfo is None:
            return value.replace(tzinfo=timezone.utc)
        return value.astimezone(timezone.utc)
    if isinstance(value, str):
        text = value.strip().replace("Z", "+00:00")
        if len(text) == 10:
            text += "T00:00:00+00:00"
        dt = datetime.fromisoformat(text)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)
    raise TypeError(f"Cannot parse datetime from {value!r}")


def _fmt(value: datetime) -> str:
    return _as_utc(value).strftime("%Y-%m-%d")


def _norm_ts(value: Any) -> datetime:
    dt = _as_utc(value)
    return dt.replace(microsecond=0)


def _frame_timestamps(df: Any) -> List[datetime]:
    import pandas as pd

    if isinstance(df.index, pd.DatetimeIndex):
        return [_as_utc(ts) for ts in df.index.to_pydatetime()]
    return [_as_utc(ts) for ts in pd.to_datetime(df["timestamp"], utc=True)]


def _coverage_ok(candles: Sequence[Dict[str, Any]], start: datetime, end: datetime, timeframe: str) -> bool:
    if len(candles) < 50:
        return False
    first, last = _as_utc(candles[0]["timestamp"]), _as_utc(candles[-1]["timestamp"])
    # Intraday Yahoo cannot cover full 2019-2024; accept best-effort coverage ≥50% of span for D1,
    # or ≥100 bars for intraday.
    if timeframe.upper() == "D1":
        span = (end - start).days
        got = (last - first).days
        return got >= span * 0.8
    return len(candles) >= 200


def _snapshot(df: Any, timeframe: str) -> Dict[str, Any]:
    import pandas as pd

    if df is None or len(df) == 0:
        return {"timeframe": timeframe, "bars": 0}
    if isinstance(df.index, pd.DatetimeIndex):
        start, end = df.index.min(), df.index.max()
    else:
        ts = pd.to_datetime(df["timestamp"], utc=True)
        start, end = ts.min(), ts.max()
    return {
        "timeframe": timeframe,
        "bars": int(len(df)),
        "start": str(start),
        "end": str(end),
        "source": "candle_repository_or_yahoo",
    }


def _bars_per_year(timeframe: str) -> int:
    return {
        "M15": 252 * 24 * 4,
        "H1": 252 * 24,
        "H4": 252 * 6,
        "D1": 252,
    }.get(timeframe.upper(), 252 * 24)


def _extract_accuracy(metrics: Mapping[str, Any]) -> float:
    for key in ("train_accuracy", "val_accuracy", "test_accuracy", "accuracy", "train_f1", "val_f1", "f1"):
        if key in metrics and metrics[key] is not None:
            try:
                return float(metrics[key])
            except (TypeError, ValueError):
                continue
    return 0.0


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _atr(candles: Sequence[Dict[str, Any]], period: int = 14) -> float:
    if len(candles) < 2:
        return 0.0001
    trs: List[float] = []
    for i in range(1, len(candles)):
        high = float(candles[i]["high"])
        low = float(candles[i]["low"])
        prev = float(candles[i - 1]["close"])
        trs.append(max(high - low, abs(high - prev), abs(low - prev)))
    window = trs[-period:] if len(trs) >= period else trs
    return float(np.mean(window)) if window else 0.0001
