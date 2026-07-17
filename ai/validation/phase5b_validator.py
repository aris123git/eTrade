"""
ai/validation/phase5b_validator.py - Phase 5b feature-driven edge validation.

Re-runs Phase 5a walk-forward with advanced feature groups enabled, compares
deltas vs Phase 5a baseline, applies the same statistical gate, and decides
whether to proceed to Phase 5c (live paper) or pivot.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from ai.validation.phase5a_validator import (
    DEFAULT_PEER_SYMBOLS,
    PHASE5B_ADVANCED_GROUPS,
    PHASE5B_FEATURE_GROUPS,
    Phase5aValidator,
    WalkForwardBacktester,
)

logger = logging.getLogger(__name__)

FEATURE_CONFIG_VERSION = "phase5b-v1"


@dataclass
class Phase5bDeltaCalculator:
    """Compute Phase 5b − Phase 5a metric deltas and flags."""

    notable_wr_delta: float = 0.02
    warning_wr_delta: float = -0.01

    def compare_symbol(
        self,
        *,
        symbol: str,
        phase5a: Mapping[str, Any],
        phase5b: Mapping[str, Any],
    ) -> Dict[str, Any]:
        a_wr = float(phase5a.get("avg_win_rate", 0.0))
        b_wr = float(phase5b.get("avg_win_rate", 0.0))
        a_sh = float(phase5a.get("avg_sharpe", 0.0))
        b_sh = float(phase5b.get("avg_sharpe", 0.0))
        a_pf = float(phase5a.get("avg_profit_factor", 0.0))
        b_pf = float(phase5b.get("avg_profit_factor", 0.0))
        a_dd = float(phase5a.get("avg_max_dd", 0.0))
        b_dd = float(phase5b.get("avg_max_dd", 0.0))

        d_wr = b_wr - a_wr
        d_sh = b_sh - a_sh
        d_pf = b_pf - a_pf
        d_dd = b_dd - a_dd  # less negative / smaller magnitude is better

        flags: List[str] = []
        if d_wr > self.notable_wr_delta:
            flags.append("notable_wr_improvement")
        if d_wr < self.warning_wr_delta:
            flags.append("warning_wr_regression")

        return {
            "symbol": symbol.upper(),
            "phase5a_wr": a_wr,
            "phase5b_wr": b_wr,
            "delta_wr": d_wr,
            "phase5a_sharpe": a_sh,
            "phase5b_sharpe": b_sh,
            "delta_sharpe": d_sh,
            "phase5a_pf": a_pf,
            "phase5b_pf": b_pf,
            "delta_pf": d_pf,
            "phase5a_max_dd": a_dd,
            "phase5b_max_dd": b_dd,
            "delta_max_dd": d_dd,
            "flags": flags,
            "timeframe": phase5b.get("timeframe") or phase5a.get("timeframe"),
        }

    def comparison_table(self, rows: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
        return [dict(r) for r in rows]


@dataclass
class Phase5bDecisionEngine:
    """Apply Phase 5b decision tree from gate passes and deltas."""

    sharpe_gate: float = 0.5
    wr_gate: float = 0.52
    pf_gate: float = 1.2
    promising_sharpe_delta: float = 0.2

    def decide(
        self,
        *,
        symbol_pass: Mapping[str, bool],
        deltas: Sequence[Mapping[str, Any]],
    ) -> Dict[str, Any]:
        winners = [s for s, ok in symbol_pass.items() if ok]
        n_pass = len(winners)
        n_symbols = len(symbol_pass)
        max_delta_sharpe = max((float(d.get("delta_sharpe", 0.0)) for d in deltas), default=0.0)
        improved = [
            str(d.get("symbol"))
            for d in deltas
            if float(d.get("delta_sharpe", 0.0)) > self.promising_sharpe_delta
        ]

        if n_symbols > 0 and n_pass == n_symbols:
            decision = "EDGE_FOUND"
            proceed_5c = True
            symbols_for_5c = winners
            rationale = "All symbols pass the statistical gate with advanced features enabled."
        elif n_pass >= 3:
            decision = "PARTIAL_EDGE"
            proceed_5c = True
            symbols_for_5c = winners
            rationale = f"{n_pass} of {n_symbols} symbols pass the gate; proceed on winners only."
        elif improved:
            decision = "PROMISING"
            proceed_5c = False
            symbols_for_5c = []
            rationale = (
                "No robust gate majority, but ΔSharpe > +0.2 on "
                + ", ".join(improved)
                + ". Investigate feature drivers and re-run Phase 5b v2."
            )
        else:
            decision = "NO_EDGE_DETECTED"
            proceed_5c = False
            symbols_for_5c = []
            rationale = (
                "Advanced features did not create a robust edge. "
                "Consider Option C (different markets) or Option D (pivot)."
            )

        return {
            "decision": decision,
            "proceed_to_phase_5c": proceed_5c,
            "symbols_for_phase_5c": symbols_for_5c,
            "symbols_passed": n_pass,
            "symbols_required_full": n_symbols,
            "symbols_required_partial": 3,
            "max_delta_sharpe": max_delta_sharpe,
            "symbols_with_promising_delta": improved,
            "rationale": rationale,
            "answer": _answer_for(decision, winners, improved),
        }

    def symbol_passes_gate(self, aggregate: Mapping[str, Any]) -> bool:
        failures = set(aggregate.get("failures") or [])
        if "overfitting" in failures:
            return False
        wr = float(aggregate.get("avg_win_rate", 0.0))
        sharpe = float(aggregate.get("avg_sharpe", -999.0))
        pf = float(aggregate.get("avg_profit_factor", 0.0))
        dd = float(aggregate.get("avg_max_dd", -100.0))
        sig = bool(aggregate.get("overall_significant"))
        if wr < self.wr_gate or not sig:
            return False
        if sharpe < self.sharpe_gate:
            return False
        if pf < self.pf_gate:
            return False
        if abs(dd) > 15.0:
            return False
        if not aggregate.get("regime_stable", True):
            return False
        return True


def _answer_for(decision: str, winners: Sequence[str], improved: Sequence[str]) -> str:
    if decision == "EDGE_FOUND":
        return "YES — advanced features create an exploitable edge across all symbols."
    if decision == "PARTIAL_EDGE":
        return (
            "PARTIAL — advanced features create an exploitable edge on "
            + ", ".join(winners)
            + "."
        )
    if decision == "PROMISING":
        return (
            "PROMISING — advanced features helped ("
            + ", ".join(improved)
            + ") but not enough for a robust gate pass."
        )
    return "NO — advanced features did not create an exploitable edge on these symbols."


def load_phase5a_baselines(
    report_path: Path | str,
    *,
    preferred_timeframe: str = "D1",
) -> Dict[str, Dict[str, Any]]:
    """Extract per-symbol aggregate metrics from a Phase 5a report (prefer D1)."""
    path = Path(report_path)
    report = json.loads(path.read_text(encoding="utf-8"))
    by_symbol: Dict[str, List[Mapping[str, Any]]] = {}
    for item in report.get("series") or []:
        if item.get("error"):
            continue
        symbol = str(item.get("symbol", "")).upper()
        if not symbol:
            continue
        by_symbol.setdefault(symbol, []).append(item)

    out: Dict[str, Dict[str, Any]] = {}
    rank = {"D1": 0, "H1": 1, "H4": 2, "M15": 3}
    for symbol, items in by_symbol.items():
        preferred = [i for i in items if str(i.get("timeframe")) == preferred_timeframe]
        pool = preferred or sorted(
            items,
            key=lambda x: rank.get(str(x.get("timeframe")), 9),
        )
        chosen = pool[0]
        agg = dict(chosen.get("aggregate") or {})
        agg["timeframe"] = chosen.get("timeframe")
        agg["symbol"] = symbol
        out[symbol] = agg
    return out


def classify_feature_group(feature_name: str) -> str:
    name = str(feature_name).lower()
    if name.startswith("micro_") or "ofi" in name:
        return "microstructure"
    if name.startswith("regime_") or name.startswith("fx_vix"):
        return "regime"
    if name.startswith("corr_") or "correlation" in name:
        return "correlation"
    if name.startswith("session_") or name.startswith("tod_") or "asian" in name or "london" in name:
        return "session"
    if any(k in name for k in ("atr", "bb_", "bollinger", "volatility", "realized_vol", "keltner", "donchian")):
        return "volatility"
    return "baseline"


def aggregate_feature_importance(
    series: Sequence[Mapping[str, Any]],
    *,
    passing_symbols: Sequence[str] | None = None,
) -> Dict[str, Any]:
    """Aggregate top features / advanced-group share across folds and symbols."""
    allowed = {s.upper() for s in (passing_symbols or [])} or None
    totals: Dict[str, float] = {}
    group_totals: Dict[str, float] = {}
    per_symbol: Dict[str, Dict[str, float]] = {}

    for item in series:
        symbol = str(item.get("symbol", "")).upper()
        if allowed is not None and symbol not in allowed:
            continue
        results = item.get("results") or {}
        for fold in results.values():
            top = fold.get("feature_importance_top10") or {}
            for name, score in top.items():
                val = abs(float(score))
                totals[name] = totals.get(name, 0.0) + val
                group = classify_feature_group(name)
                group_totals[group] = group_totals.get(group, 0.0) + val
                bucket = per_symbol.setdefault(symbol, {})
                bucket[name] = bucket.get(name, 0.0) + val

    top10 = dict(sorted(totals.items(), key=lambda kv: kv[1], reverse=True)[:10])
    per_symbol_top = {
        sym: dict(sorted(scores.items(), key=lambda kv: kv[1], reverse=True)[:10])
        for sym, scores in per_symbol.items()
    }
    advanced_share = {
        g: float(group_totals.get(g, 0.0))
        for g in ("microstructure", "regime", "correlation", "session", "volatility")
    }
    return {
        "top10_aggregate": top10,
        "advanced_group_importance": advanced_share,
        "per_symbol_top10": per_symbol_top,
        "new_features_helped_most": sorted(
            advanced_share.items(), key=lambda kv: kv[1], reverse=True
        ),
    }


@dataclass
class Phase5bValidator:
    """Orchestrate Phase 5b advanced-feature walk-forward + Phase 5a comparison."""

    backtester: WalkForwardBacktester | None = None
    symbols: Sequence[str] = field(
        default_factory=lambda: ["EURUSD", "GBPUSD", "USDJPY", "XAUUSD"]
    )
    timeframes: Sequence[str] = field(default_factory=lambda: ["M15", "H1", "H4"])
    start_date: str = "2019-01-01"
    end_date: str = "2024-12-31"
    report_path: Path = field(default_factory=lambda: Path("ai/validation/phase5b_report.json"))
    comparison_path: Path = field(
        default_factory=lambda: Path("ai/validation/phase5b_vs_phase5a.json")
    )
    phase5a_report_path: Path = field(
        default_factory=lambda: Path("ai/validation/phase5a_report.json")
    )
    peer_symbols: Sequence[str] = field(default_factory=lambda: list(DEFAULT_PEER_SYMBOLS))
    feature_groups: Sequence[str] = field(default_factory=lambda: list(PHASE5B_FEATURE_GROUPS))
    compare_vs_phase5a: bool = True
    # Match Phase 5a (random_forest) so deltas isolate feature impact; use --model lightgbm to swap.
    model_type: str = "random_forest"
    random_seed: int = 42
    preferred_compare_timeframe: str = "D1"

    def __post_init__(self) -> None:
        self.report_path = Path(self.report_path)
        self.comparison_path = Path(self.comparison_path)
        self.phase5a_report_path = Path(self.phase5a_report_path)
        if self.backtester is None:
            self.backtester = WalkForwardBacktester(
                model_type=self.model_type,
                random_seed=self.random_seed,
                enable_advanced_features=True,
                peer_symbols=list(self.peer_symbols),
                feature_groups=list(self.feature_groups),
                artifact_dir=Path("ai/artifacts/phase5b"),
            )
        else:
            self.backtester.enable_advanced_features = True
            self.backtester.peer_symbols = list(self.peer_symbols)
            self.backtester.feature_groups = list(self.feature_groups)
            self.backtester.model_type = self.model_type
            self.backtester.random_seed = self.random_seed
            self.backtester.artifact_dir = Path(self.backtester.artifact_dir)

    def verify_feature_config(self) -> Dict[str, Any]:
        """Ensure required advanced groups are present in the enabled set."""
        enabled = {g.lower() for g in self.feature_groups}
        required = set(PHASE5B_ADVANCED_GROUPS)
        missing = sorted(required - enabled)
        return {
            "feature_config_version": FEATURE_CONFIG_VERSION,
            "enabled_groups": sorted(enabled),
            "required_advanced_groups": sorted(required),
            "missing_advanced_groups": missing,
            "ok": not missing,
            "peer_symbols": [str(s).upper() for s in self.peer_symbols],
        }

    def run(
        self,
        *,
        include_multi_tf: bool = True,
        include_d1: bool = True,
    ) -> Dict[str, Any]:
        assert self.backtester is not None
        cfg_check = self.verify_feature_config()
        if not cfg_check["ok"]:
            raise RuntimeError(f"Phase 5b feature config missing groups: {cfg_check['missing_advanced_groups']}")

        # Reuse Phase5aValidator orchestration with the advanced-feature backtester.
        runner = Phase5aValidator(
            backtester=self.backtester,
            symbols=self.symbols,
            timeframes=self.timeframes,
            start_date=self.start_date,
            end_date=self.end_date,
            report_path=self.report_path,
        )
        raw = runner.run(include_multi_tf=include_multi_tf, include_d1=include_d1)

        decision_engine = Phase5bDecisionEngine()
        delta_calc = Phase5bDeltaCalculator()

        # Prefer D1 aggregates for symbol-level gate + comparison (longest history).
        baselines = (
            load_phase5a_baselines(
                self.phase5a_report_path,
                preferred_timeframe=self.preferred_compare_timeframe,
            )
            if self.compare_vs_phase5a and self.phase5a_report_path.exists()
            else {}
        )

        symbol_aggs = _best_aggregates_by_symbol(
            raw.get("series") or [],
            preferred_timeframe=self.preferred_compare_timeframe,
        )
        symbol_pass = {
            sym: decision_engine.symbol_passes_gate(agg) for sym, agg in symbol_aggs.items()
        }

        delta_rows: List[Dict[str, Any]] = []
        for symbol in self.symbols:
            sym = symbol.upper()
            b_agg = symbol_aggs.get(sym) or {}
            a_agg = baselines.get(sym) or {
                "avg_win_rate": 0.0,
                "avg_sharpe": 0.0,
                "avg_profit_factor": 0.0,
                "avg_max_dd": 0.0,
                "timeframe": self.preferred_compare_timeframe,
            }
            row = delta_calc.compare_symbol(symbol=sym, phase5a=a_agg, phase5b=b_agg)
            row["phase5b_passes_gate"] = bool(symbol_pass.get(sym))
            row["phase5a_failures"] = list((a_agg or {}).get("failures") or [])
            row["phase5b_failures"] = list((b_agg or {}).get("failures") or [])
            delta_rows.append(row)

        decision = decision_engine.decide(symbol_pass=symbol_pass, deltas=delta_rows)
        importance = aggregate_feature_importance(
            raw.get("series") or [],
            passing_symbols=decision.get("symbols_for_phase_5c")
            or [s for s, ok in symbol_pass.items() if ok]
            or list(symbol_pass.keys()),
        )

        comparison = {
            "phase": "5b_vs_5a",
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "feature_config": cfg_check,
            "preferred_compare_timeframe": self.preferred_compare_timeframe,
            "model_type_phase5b": self.model_type,
            "note": (
                "Phase 5a baseline used random_forest with advanced features disabled; "
                "Phase 5b enables microstructure/regime/correlation/session/volatility "
                f"with model_type={self.model_type}."
            ),
            "comparison_table": delta_rows,
            "decision": decision,
            "feature_importance": importance,
        }

        report = {
            "phase": "5b",
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "config": {
                **(raw.get("config") or {}),
                "feature_groups": list(self.feature_groups),
                "advanced_groups": list(PHASE5B_ADVANCED_GROUPS),
                "peer_symbols": list(self.peer_symbols),
                "feature_config_version": FEATURE_CONFIG_VERSION,
                "compare_vs_phase5a": self.compare_vs_phase5a,
                "model_type": self.model_type,
                "enable_advanced_features": True,
            },
            "feature_config_check": cfg_check,
            "series": raw.get("series") or [],
            "summary": {
                **(raw.get("summary") or {}),
                "proceed_to_phase_5c": decision["proceed_to_phase_5c"],
                "decision": decision["decision"],
                "symbols_with_edge": decision["symbols_for_phase_5c"],
                "answer": decision["answer"],
                "comparison_table": delta_rows,
                "feature_importance": importance,
            },
            "deltas": delta_rows,
            "decision": decision,
        }

        self.report_path.parent.mkdir(parents=True, exist_ok=True)
        self.report_path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
        self.comparison_path.write_text(
            json.dumps(comparison, indent=2, default=str), encoding="utf-8"
        )
        logger.info("wrote Phase 5b report → %s", self.report_path)
        logger.info("wrote Phase 5b comparison → %s", self.comparison_path)
        return report


def _best_aggregates_by_symbol(
    series: Sequence[Mapping[str, Any]],
    *,
    preferred_timeframe: str = "D1",
) -> Dict[str, Dict[str, Any]]:
    by_symbol: Dict[str, List[Mapping[str, Any]]] = {}
    for item in series:
        if item.get("error"):
            continue
        symbol = str(item.get("symbol", "")).upper()
        if symbol:
            by_symbol.setdefault(symbol, []).append(item)
    rank = {"D1": 0, "H1": 1, "H4": 2, "M15": 3}
    out: Dict[str, Dict[str, Any]] = {}
    for symbol, items in by_symbol.items():
        preferred = [i for i in items if str(i.get("timeframe")) == preferred_timeframe]
        pool = preferred or sorted(items, key=lambda x: rank.get(str(x.get("timeframe")), 9))
        chosen = pool[0]
        agg = dict(chosen.get("aggregate") or {})
        agg["timeframe"] = chosen.get("timeframe")
        agg["symbol"] = symbol
        out[symbol] = agg
    return out


def format_comparison_table(rows: Sequence[Mapping[str, Any]]) -> str:
    """Human-readable markdown-ish table for logs / CLI."""
    header = (
        f"{'Symbol':<8} | {'Phase5a WR':>10} | {'Phase5b WR':>10} | {'Δ WR':>7} | "
        f"{'Phase5a Sharpe':>14} | {'Phase5b Sharpe':>14} | {'Δ Sharpe':>8}"
    )
    lines = [header, "-" * len(header)]
    for row in rows:
        lines.append(
            f"{str(row.get('symbol', '')):<8} | "
            f"{100 * float(row.get('phase5a_wr', 0)):9.1f}% | "
            f"{100 * float(row.get('phase5b_wr', 0)):9.1f}% | "
            f"{100 * float(row.get('delta_wr', 0)):+6.1f}% | "
            f"{float(row.get('phase5a_sharpe', 0)):14.2f} | "
            f"{float(row.get('phase5b_sharpe', 0)):14.2f} | "
            f"{float(row.get('delta_sharpe', 0)):+8.2f}"
        )
    return "\n".join(lines)
