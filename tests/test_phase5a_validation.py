"""Unit tests for Phase 5a walk-forward validation (no leakage, significance, overfit)."""

from __future__ import annotations

import unittest
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List

import numpy as np
import pandas as pd

from ai.validation.phase5a_validator import (
    HistoricalDataLoader,
    OverfitDetector,
    StatisticalTester,
    TrainTestSplit,
    WalkForwardBacktester,
)


def _synthetic_frame(n: int = 1500, start: str = "2019-01-01") -> pd.DataFrame:
    rng = np.random.default_rng(0)
    ts = pd.date_range(start=start, periods=n, freq="D", tz="UTC")
    price = 1.10
    rows: List[Dict[str, Any]] = []
    for i, t in enumerate(ts):
        shock = float(rng.normal(0, 0.002))
        open_ = price
        close = max(0.5, open_ * (1 + shock + 0.0001 * np.sin(i / 20)))
        high = max(open_, close) * 1.001
        low = min(open_, close) * 0.999
        rows.append(
            {
                "open": open_,
                "high": high,
                "low": low,
                "close": close,
                "volume": 1000 + i,
            }
        )
        price = close
    return pd.DataFrame(rows, index=ts)


class WalkForwardSplitTests(unittest.TestCase):
    def test_no_train_test_overlap(self) -> None:
        loader = HistoricalDataLoader(candle_repository=object())  # unused
        # Bypass __post_init__ repo creation
        loader.candle_repository = object()
        df = _synthetic_frame(1400)
        splits = loader.split_walk_forward(df, train_size=800, test_size=200, step=200, max_folds=5)
        self.assertGreaterEqual(len(splits), 2)
        for split in splits:
            self.assertIsInstance(split, TrainTestSplit)
            tr_a, tr_b = split.train_indices
            te_a, te_b = split.test_indices
            self.assertLessEqual(tr_b, te_a)
            self.assertLess(split.train_end, split.test_start)
            # chronologically increasing
            self.assertLess(tr_a, tr_b)
            self.assertLess(te_a, te_b)

    def test_windows_move_forward_only(self) -> None:
        loader = HistoricalDataLoader.__new__(HistoricalDataLoader)
        loader.candle_repository = object()
        df = _synthetic_frame(1600)
        splits = loader.split_walk_forward(df, train_size=800, test_size=200, step=200, max_folds=4)
        for a, b in zip(splits, splits[1:]):
            self.assertLess(a.train_start, b.train_start)
            self.assertLess(a.test_start, b.test_start)


class StatisticalTesterTests(unittest.TestCase):
    def test_significance_high_win_rate(self) -> None:
        stats = StatisticalTester()
        # 140/250 = 56% should be significant
        out = stats.is_significant(0.56, 250, alpha=0.05)
        self.assertTrue(out["is_significant"])
        self.assertLess(out["p_value"], 0.05)

    def test_significance_coin_flip(self) -> None:
        stats = StatisticalTester()
        out = stats.is_significant(0.50, 200, alpha=0.05)
        self.assertFalse(out["is_significant"])

    def test_metric_helpers(self) -> None:
        stats = StatisticalTester()
        rets = [0.01, -0.005, 0.02, -0.01, 0.015]
        self.assertTrue(np.isfinite(stats.sharpe_ratio(rets)))
        self.assertTrue(np.isfinite(stats.sortino_ratio(rets)))
        self.assertGreater(stats.profit_factor(100, 50), 1.0)
        dd = stats.max_drawdown([100, 110, 90, 95])
        self.assertLess(dd, 0.0)


class OverfitDetectorTests(unittest.TestCase):
    def test_detects_large_train_test_gap(self) -> None:
        det = OverfitDetector(accuracy_gap_threshold=0.15)
        out = det.compare_train_vs_test({"accuracy": 0.78}, {"accuracy": 0.51})
        self.assertTrue(out["overfitting_detected"])

    def test_no_overfit_when_close(self) -> None:
        det = OverfitDetector()
        out = det.compare_train_vs_test({"accuracy": 0.58}, {"accuracy": 0.54})
        self.assertFalse(out["overfitting_detected"])

    def test_regime_collapse(self) -> None:
        det = OverfitDetector(regime_window=50)
        # Strong early growth, flat/negative late
        early = np.linspace(100, 150, 80)
        late = np.linspace(150, 140, 80)
        eq = np.concatenate([early, late])
        out = det.regime_detector(eq.tolist(), window=50)
        self.assertIn("regime_stable", out)

    def test_correlation_decay(self) -> None:
        det = OverfitDetector()
        self.assertGreater(det.correlation_decay(0.0), det.correlation_decay(3.0))


class DataQualityTests(unittest.TestCase):
    def test_validate_rejects_nan(self) -> None:
        loader = HistoricalDataLoader.__new__(HistoricalDataLoader)
        df = _synthetic_frame(100)
        df.iloc[10, df.columns.get_loc("close")] = np.nan
        self.assertFalse(loader.validate_data_quality(df))

    def test_validate_accepts_clean(self) -> None:
        loader = HistoricalDataLoader.__new__(HistoricalDataLoader)
        self.assertTrue(loader.validate_data_quality(_synthetic_frame(100)))


class LeakageIntegrationTests(unittest.TestCase):
    def test_fold_uses_only_past_for_train(self) -> None:
        """Ensure combined train/test slicing never lets test timestamps into train."""
        df = _synthetic_frame(1200)
        loader = HistoricalDataLoader.__new__(HistoricalDataLoader)
        loader.candle_repository = object()
        splits = loader.split_walk_forward(df, train_size=700, test_size=150, step=150, max_folds=3)
        candles = [
            {
                "symbol": "EURUSD",
                "timeframe": "D1",
                "timestamp": ts.to_pydatetime(),
                "open": float(row.open),
                "high": float(row.high),
                "low": float(row.low),
                "close": float(row.close),
                "volume": float(row.volume),
            }
            for ts, row in df.iterrows()
        ]
        for split in splits:
            train = candles[split.train_indices[0] : split.train_indices[1]]
            test = candles[split.test_indices[0] : split.test_indices[1]]
            self.assertTrue(all(t["timestamp"] < test[0]["timestamp"] for t in train))


if __name__ == "__main__":
    unittest.main()
