"""Unit tests for Phase 5b feature validation (config, deltas, gate/decision)."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from typing import Any, Dict

from ai.config.settings import AIConfig, FeatureConfig
from ai.validation.phase5a_validator import (
    PHASE5B_ADVANCED_GROUPS,
    PHASE5B_FEATURE_GROUPS,
    WalkForwardBacktester,
)
from ai.validation.phase5b_validator import (
    Phase5bDecisionEngine,
    Phase5bDeltaCalculator,
    Phase5bValidator,
    aggregate_feature_importance,
    classify_feature_group,
    format_comparison_table,
    load_phase5a_baselines,
)


class FeatureConfigTests(unittest.TestCase):
    def test_phase5b_groups_include_required_advanced(self) -> None:
        enabled = {g.lower() for g in PHASE5B_FEATURE_GROUPS}
        for group in PHASE5B_ADVANCED_GROUPS:
            self.assertIn(group, enabled)

    def test_validator_verify_feature_config_ok(self) -> None:
        v = Phase5bValidator(symbols=["EURUSD"])
        check = v.verify_feature_config()
        self.assertTrue(check["ok"])
        self.assertEqual(set(check["required_advanced_groups"]), set(PHASE5B_ADVANCED_GROUPS))
        for peer in ("US30", "SPX500", "XAUUSD", "USOIL", "USDJPY", "US10Y"):
            self.assertIn(peer, check["peer_symbols"])

    def test_validator_detects_missing_groups(self) -> None:
        v = Phase5bValidator(feature_groups=["price", "returns", "momentum"])
        check = v.verify_feature_config()
        self.assertFalse(check["ok"])
        self.assertIn("microstructure", check["missing_advanced_groups"])

    def test_backtester_enables_advanced_groups(self) -> None:
        bt = WalkForwardBacktester(
            enable_advanced_features=True,
            feature_groups=list(PHASE5B_FEATURE_GROUPS),
            peer_symbols=["US30", "SPX500"],
        )
        self.assertTrue(bt.enable_advanced_features)
        enabled = {g.lower() for g in bt.feature_groups}
        for group in PHASE5B_ADVANCED_GROUPS:
            self.assertIn(group, enabled)

    def test_feature_config_defaults_include_advanced(self) -> None:
        cfg = FeatureConfig()
        enabled = {g.lower() for g in cfg.enabled_groups}
        for group in ("microstructure", "regime", "correlation", "session", "volatility"):
            self.assertIn(group, enabled)


class DeltaCalculationTests(unittest.TestCase):
    def test_delta_metrics(self) -> None:
        calc = Phase5bDeltaCalculator()
        row = calc.compare_symbol(
            symbol="EURUSD",
            phase5a={
                "avg_win_rate": 0.502,
                "avg_sharpe": 0.21,
                "avg_profit_factor": 1.05,
                "avg_max_dd": -8.0,
            },
            phase5b={
                "avg_win_rate": 0.531,
                "avg_sharpe": 0.38,
                "avg_profit_factor": 1.25,
                "avg_max_dd": -6.0,
                "timeframe": "D1",
            },
        )
        self.assertAlmostEqual(row["delta_wr"], 0.029, places=5)
        self.assertAlmostEqual(row["delta_sharpe"], 0.17, places=5)
        self.assertAlmostEqual(row["delta_pf"], 0.20, places=5)
        self.assertAlmostEqual(row["delta_max_dd"], 2.0, places=5)
        self.assertIn("notable_wr_improvement", row["flags"])

    def test_warning_flag_on_regression(self) -> None:
        calc = Phase5bDeltaCalculator()
        row = calc.compare_symbol(
            symbol="GBPUSD",
            phase5a={"avg_win_rate": 0.55, "avg_sharpe": 0.4, "avg_profit_factor": 1.3, "avg_max_dd": -5.0},
            phase5b={"avg_win_rate": 0.53, "avg_sharpe": 0.3, "avg_profit_factor": 1.1, "avg_max_dd": -7.0},
        )
        self.assertIn("warning_wr_regression", row["flags"])
        self.assertNotIn("notable_wr_improvement", row["flags"])


class GateAndDecisionTests(unittest.TestCase):
    def _passing_agg(self, **overrides: Any) -> Dict[str, Any]:
        base = {
            "avg_win_rate": 0.55,
            "avg_sharpe": 0.8,
            "avg_profit_factor": 1.4,
            "avg_max_dd": -6.0,
            "overall_significant": True,
            "regime_stable": True,
            "failures": [],
        }
        base.update(overrides)
        return base

    def test_symbol_passes_gate(self) -> None:
        eng = Phase5bDecisionEngine()
        self.assertTrue(eng.symbol_passes_gate(self._passing_agg()))
        self.assertFalse(eng.symbol_passes_gate(self._passing_agg(avg_sharpe=0.2)))
        self.assertFalse(eng.symbol_passes_gate(self._passing_agg(avg_win_rate=0.50)))
        self.assertFalse(eng.symbol_passes_gate(self._passing_agg(failures=["overfitting"])))
        self.assertFalse(eng.symbol_passes_gate(self._passing_agg(regime_stable=False)))

    def test_decision_edge_found(self) -> None:
        eng = Phase5bDecisionEngine()
        out = eng.decide(
            symbol_pass={"EURUSD": True, "GBPUSD": True, "USDJPY": True, "XAUUSD": True},
            deltas=[{"symbol": "EURUSD", "delta_sharpe": 0.3}],
        )
        self.assertEqual(out["decision"], "EDGE_FOUND")
        self.assertTrue(out["proceed_to_phase_5c"])

    def test_decision_partial_edge(self) -> None:
        eng = Phase5bDecisionEngine()
        out = eng.decide(
            symbol_pass={"EURUSD": True, "GBPUSD": True, "USDJPY": True, "XAUUSD": False},
            deltas=[{"symbol": "XAUUSD", "delta_sharpe": 0.05}],
        )
        self.assertEqual(out["decision"], "PARTIAL_EDGE")
        self.assertTrue(out["proceed_to_phase_5c"])
        self.assertEqual(len(out["symbols_for_phase_5c"]), 3)

    def test_decision_promising(self) -> None:
        eng = Phase5bDecisionEngine()
        out = eng.decide(
            symbol_pass={"EURUSD": False, "GBPUSD": False, "USDJPY": False, "XAUUSD": False},
            deltas=[
                {"symbol": "USDJPY", "delta_sharpe": 0.25},
                {"symbol": "EURUSD", "delta_sharpe": 0.01},
            ],
        )
        self.assertEqual(out["decision"], "PROMISING")
        self.assertFalse(out["proceed_to_phase_5c"])
        self.assertIn("USDJPY", out["symbols_with_promising_delta"])

    def test_decision_no_edge(self) -> None:
        eng = Phase5bDecisionEngine()
        out = eng.decide(
            symbol_pass={"EURUSD": False, "GBPUSD": False},
            deltas=[{"symbol": "EURUSD", "delta_sharpe": 0.01}],
        )
        self.assertEqual(out["decision"], "NO_EDGE_DETECTED")
        self.assertFalse(out["proceed_to_phase_5c"])


class BaselineAndImportanceTests(unittest.TestCase):
    def test_load_phase5a_baselines_prefers_d1(self) -> None:
        payload = {
            "series": [
                {
                    "symbol": "EURUSD",
                    "timeframe": "H1",
                    "aggregate": {"avg_win_rate": 0.4, "avg_sharpe": -1.0, "failures": ["sharpe"]},
                },
                {
                    "symbol": "EURUSD",
                    "timeframe": "D1",
                    "aggregate": {"avg_win_rate": 0.5, "avg_sharpe": 0.3, "failures": ["sharpe"]},
                },
            ]
        }
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "phase5a_report.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            baselines = load_phase5a_baselines(path, preferred_timeframe="D1")
        self.assertAlmostEqual(baselines["EURUSD"]["avg_win_rate"], 0.5)
        self.assertEqual(baselines["EURUSD"]["timeframe"], "D1")

    def test_classify_and_aggregate_importance(self) -> None:
        self.assertEqual(classify_feature_group("micro_ofi_sma_20"), "microstructure")
        self.assertEqual(classify_feature_group("regime_trending_20"), "regime")
        self.assertEqual(classify_feature_group("corr_us30_50"), "correlation")
        self.assertEqual(classify_feature_group("session_london_open"), "session")
        self.assertEqual(classify_feature_group("atr_14"), "volatility")

        series = [
            {
                "symbol": "USDJPY",
                "results": {
                    "fold_1": {
                        "feature_importance_top10": {
                            "corr_us30_50": 0.2,
                            "regime_trending_20": 0.15,
                            "rsi_14": 0.05,
                        }
                    }
                },
            }
        ]
        agg = aggregate_feature_importance(series, passing_symbols=["USDJPY"])
        self.assertIn("corr_us30_50", agg["top10_aggregate"])
        self.assertGreater(agg["advanced_group_importance"]["correlation"], 0.0)

    def test_format_comparison_table(self) -> None:
        text = format_comparison_table(
            [
                {
                    "symbol": "EURUSD",
                    "phase5a_wr": 0.502,
                    "phase5b_wr": 0.531,
                    "delta_wr": 0.029,
                    "phase5a_sharpe": 0.21,
                    "phase5b_sharpe": 0.38,
                    "delta_sharpe": 0.17,
                }
            ]
        )
        self.assertIn("EURUSD", text)
        self.assertIn("Phase5a WR", text)


class PipelinePeerFeaturesTests(unittest.TestCase):
    def test_build_dataset_keeps_peer_candles_for_features(self) -> None:
        """Regression: peers must not be stripped before FeatureEngine.transform."""
        from datetime import datetime, timedelta

        from ai.services.pipeline import AIPipeline

        cfg = AIConfig()
        cfg.symbols = ["EURUSD"]
        cfg.primary_timeframe = "D1"
        cfg.features.dropna = False
        cfg.features.multi_timeframes = []
        cfg.features.correlation_symbols = ["US30"]
        cfg.features.enabled_groups = list(PHASE5B_FEATURE_GROUPS)
        cfg.labels.methods = ["binary_direction"]

        start = datetime(2020, 1, 1)
        candles = []
        price = 1.1
        peer_price = 28000.0
        for i in range(180):
            ts = start + timedelta(days=i)
            shock = 0.001 * ((i % 7) - 3)
            candles.append(
                {
                    "symbol": "EURUSD",
                    "timeframe": "D1",
                    "timestamp": ts,
                    "open": price,
                    "high": price * 1.001,
                    "low": price * 0.999,
                    "close": price * (1 + shock),
                    "volume": 1000.0,
                }
            )
            candles.append(
                {
                    "symbol": "US30",
                    "timeframe": "D1",
                    "timestamp": ts,
                    "open": peer_price,
                    "high": peer_price * 1.002,
                    "low": peer_price * 0.998,
                    "close": peer_price * (1 + shock * 0.5),
                    "volume": 2000.0,
                }
            )
            price *= 1 + shock
            peer_price *= 1 + shock * 0.5

        pipe = AIPipeline(config=cfg)
        frame = pipe.build_features(candles)
        names = " ".join(frame.feature_names).lower()
        self.assertTrue(any("corr_" in n for n in frame.feature_names), msg=names[:500])
        groups = set((frame.metadata or {}).get("generated_groups") or [])
        self.assertIn("correlation", groups)
        self.assertIn("microstructure", groups)
        self.assertIn("regime", groups)
        self.assertIn("session", groups)


if __name__ == "__main__":
    unittest.main()
