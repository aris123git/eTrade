"""
ai/research/platform.py - Autonomous Quant Research Platform.

Orchestrates:

  Wake → Check MT5/sources → Find missing history → Download → Validate →
  Repair → Features → Train → Evaluate → Reject weak models → Keep best →
  Backtest → Paper trade → Report → Sleep

Philosophy:
  Automate the *process*. Never invent market history that does not exist.
  Prediction models are one component; the advantage is a loop that continuously
  acquires data, tests ideas, measures results, and evolves automatically.
"""

from __future__ import annotations

import json
import logging
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from ai.config.settings import AIConfig
from ai.research.config import ResearchConfig
from ai.research.gate import decide_promotion, extract_metric
from ai.research.hypotheses import Hypothesis, generate_hypotheses
from ai.research.report import CycleReport, StageResult, utc_now_iso
from ai.services.pipeline import AIPipeline, create_ai_pipeline

logger = logging.getLogger(__name__)


@dataclass
class ResearchState:
    """Persisted champion metrics and pending hypotheses across cycles."""

    champions: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    pending_hypotheses: List[Dict[str, Any]] = field(default_factory=list)
    cycles_completed: int = 0
    last_cycle_id: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ResearchState":
        return cls(
            champions=dict(data.get("champions") or {}),
            pending_hypotheses=list(data.get("pending_hypotheses") or []),
            cycles_completed=int(data.get("cycles_completed") or 0),
            last_cycle_id=data.get("last_cycle_id"),
        )


class AutonomousResearchPlatform:
    """
    Self-improving quant research loop for eTrade.

    What it can do alone: connect, discover, download, validate, repair,
    feature, train, compare, backtest, paper trade, report, retrain on drift.

    What it cannot do: invent years of history a broker never supplied.
    """

    def __init__(
        self,
        config: AIConfig | None = None,
        research: ResearchConfig | None = None,
        pipeline: AIPipeline | None = None,
        *,
        artifact_root: Path | str | None = None,
    ) -> None:
        self.config = config or AIConfig()
        self.research = research or getattr(self.config, "research", None) or ResearchConfig()
        if hasattr(self.config, "research") and research is not None:
            self.config.research = research
        root = Path(artifact_root) if artifact_root is not None else Path(self.config.storage.root_dir)
        self.artifact_root = root
        self.reports_dir = root / self.research.reports_dir
        self.reports_dir.mkdir(parents=True, exist_ok=True)
        self.state_path = root / self.research.state_filename
        self.state = self._load_state()
        self.pipeline = pipeline or create_ai_pipeline(
            config=self.config,
            ensure_data=False,
        )
        self._last_report: CycleReport | None = None
        self._current_report: CycleReport | None = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run_cycle(self) -> CycleReport:
        """Execute one full autonomous research cycle."""

        cycle_id = uuid.uuid4().hex[:12]
        report = CycleReport(cycle_id=cycle_id, started_at=utc_now_iso())
        self._current_report = report
        logger.info("research cycle %s wake", cycle_id)

        try:
            self._run_stage(report, "wake", self._stage_wake)
            if not self.research.skip_collect:
                self._run_stage(report, "collect", self._stage_collect)
            self._run_stage(report, "validate", self._stage_validate)
            if self.research.repair_failed_series:
                self._run_stage(report, "repair", self._stage_repair)
            self._run_stage(report, "learn", lambda: self._stage_learn(report))
            if self.research.run_backtest:
                self._run_stage(report, "backtest", lambda: self._stage_backtest(report))
            if self.research.run_paper_trade:
                self._run_stage(report, "paper_trade", lambda: self._stage_paper_trade(report))
            if self.research.generate_hypotheses:
                self._run_stage(report, "hypotheses", lambda: self._stage_hypotheses(report))
            self._run_stage(report, "report", lambda: self._stage_persist(report))
            failed = [s for s in report.stages if s.status == "failed" and s.name in {"learn"}]
            report.status = "failed" if failed else "ok"
        except Exception as exc:
            report.status = "failed"
            report.notes.append(f"cycle_aborted: {exc.__class__.__name__}: {exc}")
            logger.exception("research cycle %s failed", cycle_id)
        finally:
            self._current_report = None

        report.finished_at = utc_now_iso()
        path = self.reports_dir / f"cycle_{cycle_id}.json"
        report.save(path)
        self.state.cycles_completed += 1
        self.state.last_cycle_id = cycle_id
        self._save_state()
        self._last_report = report
        logger.info("research cycle %s finished status=%s path=%s", cycle_id, report.status, path)
        return report

    def run_forever(self, *, max_cycles: int | None = None) -> List[CycleReport]:
        """Wake → cycle → sleep → repeat until max_cycles or disabled."""

        limit = max_cycles if max_cycles is not None else self.research.max_cycles
        reports: List[CycleReport] = []
        cycle_count = 0
        while self.research.enabled:
            report = self.run_cycle()
            reports.append(report)
            cycle_count += 1
            if limit is not None and cycle_count >= int(limit):
                break
            sleep_for = float(self.research.sleep_seconds or self.research.cycle_interval_seconds)
            if sleep_for <= 0:
                break
            logger.info("research sleep %.1fs", sleep_for)
            time.sleep(sleep_for)
        return reports

    def apply_hypotheses(self, hypotheses: Sequence[Hypothesis | Dict[str, Any]] | None = None) -> Dict[str, Any]:
        """
        Best-effort application of pending hypotheses before the next cycle.

        Currently implements data-oriented actions (download more history).
        Feature/model actions are recorded for the next learn stage.
        """

        items = list(hypotheses) if hypotheses is not None else list(self.state.pending_hypotheses)
        applied: List[str] = []
        symbols: List[str] = []
        for item in items:
            payload = item.to_dict() if isinstance(item, Hypothesis) else dict(item)
            symbol = str(payload.get("symbol") or "").upper()
            actions = list(payload.get("actions") or [])
            if symbol and any(
                a in actions
                for a in (
                    "download_more_history",
                    "download_missing_history",
                    "expand_history_for_symbol",
                    "repair_gaps",
                )
            ):
                symbols.append(symbol)
                applied.append(payload.get("id", symbol))
        detail: Dict[str, Any] = {"applied": applied, "symbols": sorted(set(symbols))}
        if symbols:
            detail["collect"] = self._download_for_symbols(sorted(set(symbols)))
        return detail

    # ------------------------------------------------------------------
    # Stages
    # ------------------------------------------------------------------

    def _stage_wake(self) -> Dict[str, Any]:
        pending = len(self.state.pending_hypotheses)
        if pending:
            applied = self.apply_hypotheses()
        else:
            applied = {"applied": [], "symbols": []}
        return {
            "cycles_completed": self.state.cycles_completed,
            "champions": list(self.state.champions),
            "pending_hypotheses": pending,
            "hypothesis_application": applied,
            "symbols": list(self.config.symbols),
            "timeframes": list(self.config.timeframes),
            "note": "Platform automates process; raw history must exist at a source.",
        }

    def _stage_collect(self) -> Dict[str, Any]:
        """Download / resume history from configured brokers and sources."""

        # Prefer production history engine when a DB is available
        detail: Dict[str, Any] = {"method": None}
        bars_inserted = 0
        try:
            from collector.history_engine import download_history
            from database.core.connection import DatabaseManager
            from core.config import DATABASE_PATH

            db_path = self.config.data.database_path or str(DATABASE_PATH)
            db = DatabaseManager(db_path=Path(db_path))
            result = download_history(
                db=db,
                brokers="ALL",
                markets=list(self.research.markets),
                symbols=list(self.config.symbols) if self.config.symbols else "ALL",
                timeframes=list(
                    dict.fromkeys(
                        [
                            self.config.primary_timeframe,
                            *self.config.timeframes,
                            *self.config.features.multi_timeframes,
                        ]
                    )
                ),
                start=self.research.history_start,
                end="today",
                include_mt5=bool(self.config.data.include_mt5),
                csv_brokers=dict(self.config.data.csv_brokers or {}),
            )
            bars_inserted = int(
                getattr(result, "bars_inserted", None)
                or sum(int(getattr(s, "bars_inserted", 0) or 0) for s in getattr(result, "series", []) or [])
            )
            detail = {
                "method": "history_engine",
                "series": len(getattr(result, "series", []) or []),
                "bars_inserted": bars_inserted,
                "status": getattr(result, "status", "ok"),
            }
            if self.pipeline.data_service is not None:
                self.pipeline.data_service.db = db
            self.pipeline._ensure_candle_source()
            if bars_inserted > 0:
                return detail
            detail["note"] = "history_engine inserted 0 bars; falling back to AIMarketDataService"
        except Exception as exc:
            logger.warning("history_engine collect failed, falling back to AIMarketDataService: %s", exc)
            detail["history_engine_error"] = f"{exc.__class__.__name__}: {exc}"

        ensure = self.pipeline.ensure_market_data(force=False)
        self.pipeline._ensure_candle_source()
        detail["method"] = "ai_market_data_service" if bars_inserted == 0 else detail.get("method")
        detail["ensure"] = _safe_dict(ensure)
        detail["fallback"] = True
        return detail

    def _stage_validate(self) -> Dict[str, Any]:
        from collector.history_validator import HistoryValidator
        from database.core.connection import DatabaseManager
        from core.config import DATABASE_PATH

        db = self._db()
        if db is None:
            path = self.config.data.database_path or str(DATABASE_PATH)
            db = DatabaseManager(db_path=Path(path))
        validator = HistoryValidator(db, min_bars=max(50, int(self.config.data.min_bars) // 10))
        timeframes = list(
            dict.fromkeys(
                [
                    self.config.primary_timeframe,
                    *self.config.timeframes,
                    *self.config.features.multi_timeframes,
                ]
            )
        )
        report = validator.validate_all(symbols=list(self.config.symbols), timeframes=timeframes)
        failed = [f"{s.canonical_symbol}:{s.timeframe}" for s in report.series if not s.ok]
        if self.research.require_validated and not report.ok:
            raise RuntimeError("Validation failed and require_validated=True: " + ", ".join(failed[:20]))
        return {
            "ok": report.ok,
            "passed": report.passed,
            "failed": report.failed,
            "failed_series": failed,
        }

    def _stage_repair(self) -> Dict[str, Any]:
        """Re-download incomplete series. Cannot fabricate missing vendor history."""

        validate_stage = self._find_stage_detail("validate")
        failed = list((validate_stage or {}).get("failed_series") or [])
        if not failed:
            return {"repaired": 0, "note": "nothing_to_repair"}
        symbols = sorted({item.split(":")[0] for item in failed})
        detail = self._download_for_symbols(symbols)
        detail["requested_from_failures"] = failed
        detail["note"] = (
            "Repair re-requests history from brokers/sources. "
            "If the source has no older bars, gaps remain."
        )
        return detail

    def _stage_learn(self, report: CycleReport) -> Dict[str, Any]:
        """Train / compare models, promote only improvements."""

        symbols = list(self.config.symbols) or ["EURUSD"]
        candidates = list(self.research.model_candidates) if self.research.compare_models else [
            self.config.model.model_type
        ]
        # Drop unavailable boosting backends gracefully
        candidates = [m for m in candidates if _model_available(m)]
        if not candidates:
            candidates = ["random_forest"]

        learned: Dict[str, Any] = {"symbols": {}, "candidates": candidates}
        for symbol in symbols:
            symbol_result = self._learn_symbol(symbol, candidates, report)
            learned["symbols"][symbol] = symbol_result
            if symbol_result.get("metrics"):
                report.metrics_by_symbol[symbol] = dict(symbol_result["metrics"])
        return learned

    def _learn_symbol(
        self,
        symbol: str,
        candidates: Sequence[str],
        report: CycleReport,
    ) -> Dict[str, Any]:
        self.pipeline._ensure_candle_source()
        candles: List[Any] = []
        try:
            candles = self.pipeline.load_candles(
                symbol=symbol,
                timeframe=self.config.primary_timeframe,
                limit=int(self.research.candle_limit),
                auto_download=False,
            )
        except Exception:
            candles = []
        if len(candles) < 100:
            # Offline / thin DB: allow pipeline auto path once
            self.pipeline.ensure_market_data(force=False)
            self.pipeline._ensure_candle_source()
            candles = self.pipeline.load_candles(
                symbol=symbol,
                timeframe=self.config.primary_timeframe,
                limit=int(self.research.candle_limit),
                auto_download=False,
            )
        if len(candles) < 50:
            return {"status": "skipped", "reason": "insufficient_candles", "n": len(candles)}

        champion_key = f"{symbol}:{self.config.primary_timeframe}"
        champion_meta = self.state.champions.get(champion_key)
        best_accepted: Dict[str, Any] | None = None
        comparisons: List[Dict[str, Any]] = []

        for model_type in candidates:
            cfg = self.config.copy()
            cfg.model.model_type = model_type
            cfg.data.auto_download = False
            local_pipeline = AIPipeline(
                config=cfg,
                candle_source=self.pipeline.candle_source,
                data_service=self.pipeline.data_service,
            )
            try:
                run = local_pipeline.run_training(
                    candles=candles,
                    register=False,
                    auto_download=False,
                )
                metrics = {**run.train.metrics, **run.evaluation}
                decision = decide_promotion(
                    challenger_metrics=metrics,
                    champion_metrics=(champion_meta or {}).get("metrics"),
                    metric_name=self.research.primary_metric,
                    minimize=self.research.metric_minimize,
                    min_improvement=self.research.min_improvement,
                )
                entry = {
                    "model_type": model_type,
                    "metrics": metrics,
                    "decision": decision.to_dict(),
                }
                comparisons.append(entry)
                if decision.accepted:
                    if best_accepted is None or _better(
                        metrics,
                        best_accepted["metrics"],
                        self.research.primary_metric,
                        self.research.metric_minimize,
                    ):
                        best_accepted = {
                            "model_type": model_type,
                            "metrics": metrics,
                            "decision": decision.to_dict(),
                            "feature_names": list(run.dataset.bundle.feature_names),
                            "model": run.train.model,
                            "candles": candles,
                        }
                else:
                    report.rejected.append(
                        {
                            "symbol": symbol,
                            "model_type": model_type,
                            "decision": decision.to_dict(),
                        }
                    )
            except Exception as exc:
                comparisons.append(
                    {
                        "model_type": model_type,
                        "error": f"{exc.__class__.__name__}: {exc}",
                        "decision": {"accepted": False, "reason": "train_error"},
                    }
                )

        if best_accepted is None:
            return {"status": "no_promotion", "comparisons": comparisons}

        # Keep the promoted model live on the shared pipeline for paper/backtest
        self.pipeline.model = best_accepted["model"]
        self.pipeline.config.model.model_type = best_accepted["model_type"]

        # Register only improvements (or baseline)
        registration = None
        if self.research.register_only_improvements:
            assert self.pipeline.registry is not None
            registration = self.pipeline.registry.register(
                name=f"{symbol}_{best_accepted['model_type']}",
                model=best_accepted["model"],
                features=best_accepted.get("feature_names"),
                metrics=best_accepted["metrics"],
                metadata={
                    "symbol": symbol,
                    "timeframe": self.config.primary_timeframe,
                    "promoted_by": "autonomous_research",
                    "cycle": report.cycle_id,
                },
            )
            self.pipeline.model_version = registration.version
        champion_payload = {
            "model_type": best_accepted["model_type"],
            "metrics": best_accepted["metrics"],
            "version": getattr(registration, "version", None),
            "decision": best_accepted["decision"],
            "updated_at": utc_now_iso(),
        }
        self.state.champions[champion_key] = champion_payload
        report.champions[champion_key] = {
            "model_type": champion_payload["model_type"],
            "metrics": {
                self.research.primary_metric: extract_metric(
                    champion_payload["metrics"], self.research.primary_metric
                )
            },
            "version": champion_payload["version"],
        }
        # Drift check vs previous train fold if enabled
        drift_flag = False
        if self.research.detect_drift and len(candles) > 200:
            drift_flag = self._detect_simple_drift(candles)
        return {
            "status": "promoted",
            "champion": report.champions[champion_key],
            "comparisons": comparisons,
            "drift": drift_flag,
            "n_candles": len(candles),
        }

    def _stage_backtest(self, report: CycleReport) -> Dict[str, Any]:
        from ai.evaluation.backtest import BacktestEngine, BacktestSignal, Candle

        results: Dict[str, Any] = {}
        for symbol, metrics in report.metrics_by_symbol.items():
            candles_raw = self.pipeline.load_candles(
                symbol=symbol,
                timeframe=self.config.primary_timeframe,
                limit=min(1500, int(self.research.candle_limit)),
                auto_download=False,
            )
            if len(candles_raw) < 30:
                results[symbol] = {"status": "skipped", "reason": "insufficient_candles"}
                continue
            # Lightweight signal: use recent prediction polarity when model exists
            signals: List[BacktestSignal] = []
            try:
                pred = self.pipeline.predict(candles=candles_raw[-200:], symbol=symbol)
                side = "buy" if float(pred.prediction) > 0 else "sell"
                ts = candles_raw[-50]["timestamp"]
                signals.append(
                    BacktestSignal(
                        symbol=symbol,
                        timestamp=ts,
                        side=side,
                        quantity=0.1,
                        timeframe=self.config.primary_timeframe,
                    )
                )
                # flatten near end
                signals.append(
                    BacktestSignal(
                        symbol=symbol,
                        timestamp=candles_raw[-5]["timestamp"],
                        side="close",
                        quantity=0.1,
                        timeframe=self.config.primary_timeframe,
                    )
                )
            except Exception:
                # Directional proxy from close momentum if no live model
                c0 = float(candles_raw[-50]["close"])
                c1 = float(candles_raw[-10]["close"])
                side = "buy" if c1 >= c0 else "sell"
                signals.append(
                    BacktestSignal(
                        symbol=symbol,
                        timestamp=candles_raw[-50]["timestamp"],
                        side=side,
                        quantity=0.1,
                        timeframe=self.config.primary_timeframe,
                    )
                )
                signals.append(
                    BacktestSignal(
                        symbol=symbol,
                        timestamp=candles_raw[-5]["timestamp"],
                        side="close",
                        quantity=0.1,
                        timeframe=self.config.primary_timeframe,
                    )
                )

            bt_candles = [
                Candle(
                    symbol=symbol,
                    timeframe=self.config.primary_timeframe,
                    timestamp=c["timestamp"],
                    open=float(c["open"]),
                    high=float(c["high"]),
                    low=float(c["low"]),
                    close=float(c["close"]),
                    volume=float(c.get("volume") or 0.0),
                )
                for c in candles_raw
            ]
            engine = BacktestEngine(config=self.config)
            bt = engine.run(candles=bt_candles, signals=signals)
            results[symbol] = {
                "status": "ok",
                "n_trades": len(getattr(bt, "trades", []) or []),
                "metrics": getattr(bt, "metrics", None) or _safe_dict(bt),
                "model_metric": extract_metric(metrics, self.research.primary_metric),
            }
        return results

    def _stage_paper_trade(self, report: CycleReport) -> Dict[str, Any]:
        from ai.execution.executor import create_order_executor

        executor = create_order_executor(config=self.config, mode="paper")
        outcomes: Dict[str, Any] = {}
        for symbol in report.metrics_by_symbol:
            try:
                candles = self.pipeline.load_candles(
                    symbol=symbol,
                    timeframe=self.config.primary_timeframe,
                    limit=min(300, int(self.research.candle_limit)),
                    auto_download=False,
                )
                prediction = self.pipeline.run_prediction(
                    candles=candles or None,
                    equity=float(self.research.paper_equity),
                    auto_download=False,
                )
                signal = prediction.get("signal") or {}
                side = str(signal.get("signal") or "HOLD").upper()
                outcomes[symbol] = {
                    "status": "ok",
                    "signal": side,
                    "confidence": signal.get("confidence"),
                    "risk": signal.get("risk"),
                    "executor_mode": getattr(executor, "mode", "paper"),
                }
            except Exception as exc:
                outcomes[symbol] = {
                    "status": "failed",
                    "error": f"{exc.__class__.__name__}: {exc}",
                }
        return outcomes

    def _stage_hypotheses(self, report: CycleReport) -> Dict[str, Any]:
        validate_detail = self._find_stage_detail("validate") or {}
        drift_by_symbol = {}
        learn_detail = self._find_stage_detail("learn") or {}
        for symbol, payload in (learn_detail.get("symbols") or {}).items():
            if isinstance(payload, dict) and payload.get("drift"):
                drift_by_symbol[symbol] = True
        hyps = generate_hypotheses(
            per_symbol_metrics=report.metrics_by_symbol,
            validation_failures=list(validate_detail.get("failed_series") or []),
            drift_by_symbol=drift_by_symbol,
            primary_metric=self.research.primary_metric,
        )
        report.hypotheses = [h.to_dict() for h in hyps]
        self.state.pending_hypotheses = report.hypotheses
        return {"count": len(hyps), "top": report.hypotheses[:5]}

    def _stage_persist(self, report: CycleReport) -> Dict[str, Any]:
        path = self.reports_dir / f"cycle_{report.cycle_id}.json"
        # Final save happens in run_cycle; here we also write latest pointer
        latest = self.reports_dir / "latest.json"
        report.save(latest)
        return {"report_path": str(path), "latest_path": str(latest)}

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _run_stage(self, report: CycleReport, name: str, fn) -> StageResult:
        started = utc_now_iso()
        try:
            detail = fn() or {}
            stage = StageResult(
                name=name,
                status="ok",
                started_at=started,
                finished_at=utc_now_iso(),
                detail=detail if isinstance(detail, dict) else {"result": detail},
            )
        except Exception as exc:
            logger.exception("stage %s failed", name)
            stage = StageResult(
                name=name,
                status="failed",
                started_at=started,
                finished_at=utc_now_iso(),
                detail={},
                error=f"{exc.__class__.__name__}: {exc}",
            )
            if name in {"learn"}:
                raise
        report.stages.append(stage)
        # stash detail lookup via stages list
        return stage

    def _find_stage_detail(self, name: str) -> Dict[str, Any] | None:
        for report in (self._current_report, self._last_report):
            if report is None:
                continue
            for stage in reversed(report.stages):
                if stage.name == name:
                    return stage.detail
        return None

    def _download_for_symbols(self, symbols: Sequence[str]) -> Dict[str, Any]:
        try:
            from collector.history_engine import download_history

            db = self._db()
            if db is None:
                return {"status": "skipped", "reason": "no_db"}
            result = download_history(
                db=db,
                brokers="ALL",
                markets=list(self.research.markets),
                symbols=list(symbols),
                timeframes=list(
                    dict.fromkeys(
                        [
                            self.config.primary_timeframe,
                            *self.config.timeframes,
                            *self.config.features.multi_timeframes,
                        ]
                    )
                ),
                start=self.research.history_start,
                end="today",
            )
            return {
                "status": "ok",
                "symbols": list(symbols),
                "bars_inserted": sum(
                    int(getattr(s, "bars_inserted", 0) or 0) for s in getattr(result, "series", []) or []
                ),
            }
        except Exception as exc:
            # Fallback synthetic/AI service fill for offline environments
            try:
                ensure = self.pipeline.ensure_market_data(force=False)
                return {
                    "status": "fallback_ensure",
                    "error": f"{exc.__class__.__name__}: {exc}",
                    "ensure": _safe_dict(ensure),
                }
            except Exception as exc2:
                return {
                    "status": "failed",
                    "error": f"{exc.__class__.__name__}: {exc}",
                    "fallback_error": f"{exc2.__class__.__name__}: {exc2}",
                }

    def _db(self):
        if self.pipeline.data_service is not None and getattr(self.pipeline.data_service, "db", None) is not None:
            return self.pipeline.data_service.db
        return None

    def _detect_simple_drift(self, candles: Sequence[Dict[str, Any]]) -> bool:
        from ai.monitoring.drift import mean_shift_score

        closes = np.asarray([float(c["close"]) for c in candles], dtype=float)
        if closes.size < 100:
            return False
        mid = closes.size // 2
        score = mean_shift_score(closes[:mid], closes[mid:])
        return bool(score >= float(self.config.monitoring.drift_threshold) * 10.0)

    def _load_state(self) -> ResearchState:
        if not self.state_path.exists():
            return ResearchState()
        try:
            return ResearchState.from_dict(json.loads(self.state_path.read_text(encoding="utf-8")))
        except Exception:
            return ResearchState()

    def _save_state(self) -> None:
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        self.state_path.write_text(json.dumps(self.state.to_dict(), indent=2, default=str), encoding="utf-8")


def create_research_platform(
    config: AIConfig | None = None,
    research: ResearchConfig | None = None,
    **kwargs: Any,
) -> AutonomousResearchPlatform:
    """Factory for AutonomousResearchPlatform."""

    return AutonomousResearchPlatform(config=config, research=research, **kwargs)


# ------------------------------------------------------------------
# Module helpers
# ------------------------------------------------------------------


def _safe_dict(obj: Any) -> Dict[str, Any]:
    if obj is None:
        return {}
    if isinstance(obj, dict):
        return obj
    if hasattr(obj, "to_dict"):
        try:
            return dict(obj.to_dict())
        except Exception:
            pass
    out: Dict[str, Any] = {}
    for key in (
        "ok",
        "status",
        "bars_inserted",
        "synthetic_filled",
        "n_trades",
        "metrics",
        "total_return",
        "sharpe",
    ):
        if hasattr(obj, key):
            value = getattr(obj, key)
            if isinstance(value, (str, int, float, bool)) or value is None:
                out[key] = value
    return out or {"repr": repr(obj)}


def _model_available(model_type: str) -> bool:
    name = str(model_type).lower().strip()
    if name in {"random_forest", "logistic_regression", "decision_tree", "extra_trees", "gradient_boosting"}:
        return True
    if name == "lightgbm":
        try:
            import lightgbm  # noqa: F401

            return True
        except Exception:
            return False
    if name == "xgboost":
        try:
            import xgboost  # noqa: F401

            return True
        except Exception:
            return False
    if name == "catboost":
        try:
            import catboost  # noqa: F401

            return True
        except Exception:
            return False
    return True


def _better(
    left: Dict[str, Any],
    right: Dict[str, Any],
    metric: str,
    minimize: bool,
) -> bool:
    a = extract_metric(left, metric)
    b = extract_metric(right, metric)
    if a is None:
        return False
    if b is None:
        return True
    return a < b if minimize else a > b
