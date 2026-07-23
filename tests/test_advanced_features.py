"""Tests for microstructure, session, regime (FX VIX), and cross-asset correlation features."""

from __future__ import annotations

import unittest
from datetime import datetime, timedelta, timezone

import numpy as np

from ai.config.settings import AIConfig
from ai.features.correlation import asset_class_for, compute_correlation_features
from ai.features.engine import FeatureEngine, FeatureGroup
from ai.features.microstructure import compute_microstructure_features
from ai.features.regime import compute_regime_features
from ai.features.session import compute_session_features


def _ohlcv(n: int = 300, seed: int = 1):
    rng = np.random.default_rng(seed)
    close = 100 + np.cumsum(rng.normal(0, 0.2, n))
    open_ = np.roll(close, 1)
    open_[0] = close[0]
    high = np.maximum(open_, close) + rng.uniform(0.05, 0.3, n)
    low = np.minimum(open_, close) - rng.uniform(0.05, 0.3, n)
    volume = rng.uniform(100, 1000, n)
    spread = rng.uniform(0.01, 0.05, n)
    return open_, high, low, close, volume, spread


def _candles(symbol: str, n: int = 300, timeframe: str = "H1", seed: int = 1):
    open_, high, low, close, volume, spread = _ohlcv(n, seed=seed)
    start = datetime(2024, 1, 2, 0, 0, tzinfo=timezone.utc)
    out = []
    for i in range(n):
        out.append(
            {
                "symbol": symbol,
                "timeframe": timeframe,
                "timestamp": (start + timedelta(hours=i)).replace(tzinfo=None),
                "open": float(open_[i]),
                "high": float(high[i]),
                "low": float(low[i]),
                "close": float(close[i]),
                "volume": float(volume[i]),
                "spread": float(spread[i]),
            }
        )
    return out


class MicrostructureTests(unittest.TestCase):
    def test_order_flow_and_spread_features(self) -> None:
        open_, high, low, close, volume, spread = _ohlcv()
        feats = compute_microstructure_features(
            open_, high, low, close, volume, spread, rolling_windows=(5, 20)
        )
        self.assertIn("micro_order_flow_imbalance", feats)
        self.assertIn("micro_bid_ask_ratio", feats)
        self.assertIn("micro_spread_zscore_20", feats)
        self.assertIn("micro_spread_widening_20", feats)
        self.assertEqual(feats["micro_has_l2_book"][0], 0.0)  # proxy mode
        self.assertTrue(np.isfinite(feats["micro_order_flow_imbalance"]).any())

    def test_uses_real_bid_ask_when_present(self) -> None:
        open_, high, low, close, volume, spread = _ohlcv(80)
        bid = volume * 0.6
        ask = volume * 0.4
        feats = compute_microstructure_features(
            open_, high, low, close, volume, spread, bid_volume=bid, ask_volume=ask
        )
        self.assertEqual(feats["micro_has_l2_book"][0], 1.0)
        self.assertTrue(np.all(feats["micro_bid_ask_ratio"] > 1.0))


class SessionTests(unittest.TestCase):
    def test_asian_european_american_and_opens(self) -> None:
        # 01:00 Asia/Tokyo open, 08:00 London open, 13:00 NY open
        stamps = [
            datetime(2024, 6, 3, 1, 0),
            datetime(2024, 6, 3, 8, 0),
            datetime(2024, 6, 3, 13, 0),
            datetime(2024, 6, 3, 18, 0),
        ]
        feats = compute_session_features(stamps)
        self.assertEqual(feats["session_asian"][0], 1.0)
        self.assertEqual(feats["session_european"][1], 1.0)
        self.assertEqual(feats["tod_tokyo_open_window"][0], 1.0)
        self.assertEqual(feats["tod_london_open_window"][1], 1.0)
        self.assertEqual(feats["tod_new_york_open_window"][2], 1.0)
        self.assertEqual(feats["session_american"][2], 1.0)

    def test_session_price_analysis(self) -> None:
        stamps = [datetime(2024, 6, 3, h, 0) for h in range(0, 10)]
        close = np.linspace(1.0, 1.01, len(stamps))
        open_ = close.copy()
        high = close + 0.001
        low = close - 0.001
        feats = compute_session_features(stamps, open_=open_, high=high, low=low, close=close)
        self.assertIn("session_return", feats)
        self.assertIn("session_range_pct", feats)
        self.assertTrue(np.isfinite(feats["session_return"][-1]))


class RegimeTests(unittest.TestCase):
    def test_trend_mean_revert_and_fx_vix(self) -> None:
        # Strong trend
        close = np.linspace(100, 130, 400)
        high = close + 0.2
        low = close - 0.2
        feats = compute_regime_features(high, low, close, rolling_windows=(20, 50))
        self.assertIn("regime_trending_20", feats)
        self.assertIn("regime_mean_reverting_20", feats)
        self.assertIn("fx_vix", feats)
        self.assertIn("fx_vix_realized_vol", feats)
        self.assertIn("fx_vix_vol_of_vol", feats)
        # Trending series should light trending flag more often than mean-reverting
        self.assertGreater(np.nansum(feats["regime_trending_50"]), np.nansum(feats["regime_mean_reverting_50"]))


class CorrelationTests(unittest.TestCase):
    def test_asset_class_tags_and_class_aggregates(self) -> None:
        self.assertEqual(asset_class_for("US30"), "equity")
        self.assertEqual(asset_class_for("XAUUSD"), "commodity")
        self.assertEqual(asset_class_for("US10Y"), "bond")
        base = _candles("EURUSD", n=200, seed=1)
        peers = {
            "US30": _candles("US30", n=200, seed=2),
            "XAUUSD": _candles("XAUUSD", n=200, seed=3),
        }
        feats = compute_correlation_features(base, peers, window=30)
        self.assertIn("corr_class_equity_30", feats)
        self.assertIn("corr_class_commodity_30", feats)
        self.assertIn("corr_risk_on_proxy", feats)
        self.assertTrue(np.isfinite(feats["corr_us30_30"]).any())


class FeatureEngineIntegrationTests(unittest.TestCase):
    def test_engine_emits_new_groups(self) -> None:
        cfg = AIConfig()
        cfg.features.dropna = False
        cfg.features.multi_timeframes = []
        cfg.primary_timeframe = "H1"
        cfg.symbols = ["EURUSD"]
        # Pack base + cross-asset peers in one batch
        candles = (
            _candles("EURUSD", n=250, seed=1)
            + _candles("US30", n=250, seed=2)
            + _candles("XAUUSD", n=250, seed=3)
        )
        frame = FeatureEngine(config=cfg).transform(candles)
        names = set(frame.feature_names)
        self.assertTrue(any(n.startswith("micro_") for n in names))
        self.assertTrue(any(n.startswith("fx_vix") for n in names))
        self.assertTrue(any(n.startswith("session_asian") or n == "session_asian" for n in names))
        self.assertTrue(any(n.startswith("corr_class_") for n in names))
        self.assertTrue(any(n.startswith("tod_") for n in names))
        self.assertIn(FeatureGroup.MICROSTRUCTURE.value, cfg.features.enabled_groups)
        self.assertIn(FeatureGroup.CORRELATION.value, cfg.features.enabled_groups)
        self.assertGreater(frame.shape[0], 0)
        self.assertGreater(frame.shape[1], 50)


if __name__ == "__main__":
    unittest.main()
