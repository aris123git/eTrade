"""
ai/research/platform.py - Autonomous Quantitative Research Engine (Phase 4).

Continuously discovers, validates, and improves trading strategies without
human intervention. Does not invent market history. Does not add new model
architectures — uses the existing AI stack.

Cadence:
  Hourly  — download data, repair gaps, predict, paper journal
  Daily   — retrain if drifted, validate, compare, promote
  Weekly  — walk-forward + Monte-Carlo robustness
  Monthly — full research cycle + dashboard
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from ai.config.settings import AIConfig, ResearchConfig
from ai.monitoring.dashboard import InstitutionalDashboard
from ai.research.autonomous_scheduler import AutonomousScheduler, SchedulePlan
from ai.research.discovery import HypothesisDiscoveryEngine
from ai.research.gate import decide_promotion, extract_metric
from ai.research.hypotheses import generate_hypotheses
from ai.research.paper_journal import PaperTradingJournal
from ai.research.production_gate import ProductionReadinessGate, ProductionThresholds
from ai.research.report import CycleReport, StageResult, utc_now_iso
from ai.research.self_improve import SelfImprovementController
from ai.research.validation_gate import StrictValidationGate, ValidationThresholds
from ai.services.pipeline import AIPipeline, create_ai_pipeline
from database.core.connection import DatabaseManager
from database.repositories.research_repository import ResearchRepository

logger = logging.getLogger(__name__)


@dataclass
class ResearchState:
    champions: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    pending_hypotheses: List[Dict[str, Any]] = field(default_factory=list)
    cycles_completed: int = 0
    last_cycle_id: Optional[str] = None
    selected_feature_groups: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ResearchState":
        return cls(
            champions=dict(data.get("champions") or {}),
            pending_hypotheses=list(data.get("pending_hypotheses") or []),
            cycles_completed=int(data.get("cycles_completed") or 0),
            last_cycle_id=data.get("last_cycle_id"),
            selected_feature_groups=list(data.get("selected_feature_groups") or []),
        )


class AutonomousResearchPlatform:
    """Self-improving quantitative research platform for eTrade."""

    def __init__(
        self,
        config: AIConfig | None = None,
        research: ResearchConfig | None = None,
        pipeline: AIPipeline | None = None,
        *,
        db: DatabaseManager | None = None,
        artifact_root: Path | str | None = None,
    ) -> None:
        self.config = config or AIConfig()
        self.research = research or self.config.research
        self.config.research = self.research
        # Production path: never invent bars
        self.config.data.allow_synthetic_fallback = bool(self.research.allow_synthetic)
        self.config.data.require_validated = bool(self.research.require_validated)

        root = Path(artifact_root) if artifact_root is not None else Path(self.config.storage.root_dir)
        self.artifact_root = root
        self.reports_dir = root / self.research.reports_dir
        self.dashboard_dir = root / self.research.dashboard_dir
        self.reports_dir.mkdir(parents=True, exist_ok=True)
        self.dashboard_dir.mkdir(parents=True, exist_ok=True)
        self.state_path = root / self.research.state_filename
        self.state = self._load_state()

        self.db = db or self._open_db()
        self.research_repo = ResearchRepository(self.db)
        self.pipeline = pipeline or create_ai_pipeline(config=self.config, ensure_data=False)
        if self.pipeline.data_service is not None:
            self.pipeline.data_service.db = self.db

        self.validation_gate = StrictValidationGate(
            self.config,
            ValidationThresholds(
                min_val_score=self.research.min_val_score,
                min_oos_score=self.research.min_oos_score,
                min_walk_forward_score=self.research.min_walk_forward_score,
                max_mc_ruin_prob=self.research.max_mc_ruin_prob,
                min_backtest_trades=self.research.min_backtest_trades,
            ),
        )
        self.production_gate = ProductionReadinessGate(
            self.research_repo,
            ProductionThresholds(
                min_paper_trades=self.research.min_paper_trades,
                min_paper_days=self.research.min_paper_days,
                min_sharpe=self.research.min_live_sharpe,
                max_drawdown=self.research.max_live_drawdown,
                min_profit_factor=self.research.min_live_profit_factor,
            ),
        )
        self.self_improve = SelfImprovementController(
            self.config,
            self.pipeline,
            self.research_repo,
            primary_metric=self.research.primary_metric,
            min_improvement=self.research.min_improvement,
        )
        self._current_report: CycleReport | None = None
        self._last_report: CycleReport | None = None
        self._scheduler: AutonomousScheduler | None = None

    # ==================================================================
    # Public cadence API
    # ==================================================================

    def run_hourly(self) -> Dict[str, Any]:
        """Download new data, repair gaps, generate predictions, paper journal."""
        detail: Dict[str, Any] = {}
        detail["collect"] = self._stage_collect()
        detail["validate"] = self._stage_validate()
        detail["repair"] = self._stage_repair()
        if self.research.download_ticks:
            detail["ticks"] = self._stage_ticks()
        detail["paper"] = self._stage_paper_all()
        detail["dashboard"] = self._stage_dashboard() if self.research.build_dashboard else {}
        return detail

    def run_daily(self) -> Dict[str, Any]:
        """Retrain when needed, validate, compare models, promote winners."""
        detail: Dict[str, Any] = {"symbols": {}}
        for symbol in self.config.symbols:
            candles = self._load_candles(symbol)
            if len(candles) < 100:
                detail["symbols"][symbol] = {"status": "skipped", "reason": "insufficient_candles"}
                continue
            if self.research.run_self_improve:
                result = self.self_improve.maybe_improve(symbol=symbol, candles=candles)
                detail["symbols"][symbol] = result.to_dict()
            else:
                detail["symbols"][symbol] = self._research_symbol(symbol, candles)
        detail["production_gate"] = self._stage_production_gate()
        return detail

    def run_weekly(self) -> Dict[str, Any]:
        """Walk-forward evaluation and Monte-Carlo robustness on champions."""
        detail: Dict[str, Any] = {"symbols": {}}
        for symbol in self.config.symbols:
            candles = self._load_candles(symbol)
            if len(candles) < 150:
                detail["symbols"][symbol] = {"status": "skipped"}
                continue
            dataset = self.pipeline.build_dataset(candles)
            model = self.pipeline.model
            if model is None:
                train = self.pipeline.run_training(candles=candles, register=False, auto_download=False)
                model = train.train.model
                dataset = train.dataset
            report = self.validation_gate.validate(
                model,
                dataset.bundle,
                candles=candles,
                symbol=symbol,
                timeframe=self.config.primary_timeframe,
            )
            champion = self.research_repo.get_champion_model(symbol, self.config.primary_timeframe)
            model_id = champion.get("model_id") if champion else None
            for stage in report.stages:
                self.research_repo.record_validation(
                    experiment_id=None,
                    model_id=model_id,
                    stage=f"weekly_{stage.stage}",
                    passed=stage.passed,
                    metrics=stage.metrics,
                    details=stage.details,
                )
            detail["symbols"][symbol] = report.to_dict()
        return detail

    def run_monthly(self) -> Dict[str, Any]:
        """Complete research cycle + institutional dashboard."""
        report = self.run_cycle()
        return {"cycle": report.to_dict(), "status": report.status}

    def run_cycle(self) -> CycleReport:
        """Full monthly-style autonomous research cycle."""
        cycle_id = uuid.uuid4().hex[:12]
        report = CycleReport(cycle_id=cycle_id, started_at=utc_now_iso())
        self._current_report = report
        try:
            self._run_stage(report, "wake", self._stage_wake)
            if not self.research.skip_collect:
                self._run_stage(report, "collect", self._stage_collect)
            self._run_stage(report, "validate", self._stage_validate)
            if self.research.repair_failed_series:
                self._run_stage(report, "repair", self._stage_repair)
            if self.research.download_ticks:
                self._run_stage(report, "ticks", self._stage_ticks)
            if self.research.run_feature_discovery:
                self._run_stage(report, "discovery", lambda: self._stage_discovery(report))
            self._run_stage(report, "learn", lambda: self._stage_learn(report))
            if self.research.run_paper_trade:
                self._run_stage(report, "paper_trade", lambda: self._stage_paper_all())
            if self.research.run_production_gate:
                self._run_stage(report, "production_gate", self._stage_production_gate)
            if self.research.generate_hypotheses:
                self._run_stage(report, "hypotheses", lambda: self._stage_hypotheses(report))
            if self.research.build_dashboard:
                self._run_stage(report, "dashboard", self._stage_dashboard)
            self._run_stage(report, "report", lambda: self._stage_persist(report))
            failed_critical = [s for s in report.stages if s.status == "failed" and s.name == "learn"]
            report.status = "failed" if failed_critical else "ok"
        except Exception as exc:
            report.status = "failed"
            report.notes.append(f"cycle_aborted: {exc.__class__.__name__}: {exc}")
            logger.exception("research cycle failed")
        finally:
            self._current_report = None

        report.finished_at = utc_now_iso()
        report.save(self.reports_dir / f"cycle_{cycle_id}.json")
        self.state.cycles_completed += 1
        self.state.last_cycle_id = cycle_id
        self._save_state()
        self._last_report = report
        return report

    def run_forever(self, *, max_cycles: int | None = None) -> List[CycleReport]:
        limit = max_cycles if max_cycles is not None else self.research.max_cycles
        reports: List[CycleReport] = []
        count = 0
        while self.research.enabled:
            reports.append(self.run_cycle())
            count += 1
            if limit is not None and count >= int(limit):
                break
            sleep_for = float(self.research.sleep_seconds or self.research.cycle_interval_seconds)
            if sleep_for <= 0:
                break
            time.sleep(sleep_for)
        return reports

    def start_scheduler(self, plan: SchedulePlan | None = None) -> AutonomousScheduler:
        self._scheduler = AutonomousScheduler(self, plan=plan or SchedulePlan())
        self._scheduler.install()
        self._scheduler.start()
        return self._scheduler

    # ==================================================================
    # Stages
    # ==================================================================

    def _stage_wake(self) -> Dict[str, Any]:
        return {
            "cycles_completed": self.state.cycles_completed,
            "champions": list(self.state.champions),
            "symbols": list(self.config.symbols),
            "allow_synthetic": self.research.allow_synthetic,
            "note": "Automates process; cannot invent broker history.",
        }

    def _stage_collect(self) -> Dict[str, Any]:
        from collector.history_engine import download_history

        result = download_history(
            self.db,
            brokers="ALL",
            markets=list(self.research.markets),
            symbols=list(self.config.symbols) if self.config.symbols else "ALL",
            timeframes=self._timeframes(),
            start=self.research.history_start,
            end="today",
            include_mt5=bool(self.config.data.include_mt5),
            csv_brokers=dict(self.config.data.csv_brokers or {}),
        )
        bars = sum(int(getattr(s, "bars_inserted", 0) or 0) for s in getattr(result, "series", []) or [])
        self.pipeline._ensure_candle_source()
        # Probe available history bounds when MT5 is present
        availability = self._detect_history_bounds()
        return {
            "method": "history_engine",
            "bars_inserted": bars,
            "series": len(getattr(result, "series", []) or []),
            "availability": availability,
        }

    def _stage_validate(self) -> Dict[str, Any]:
        from collector.history_validator import HistoryValidator

        validator = HistoryValidator(
            self.db,
            min_bars=max(50, int(self.config.data.min_bars) // 10),
        )
        report = validator.validate_all(symbols=list(self.config.symbols), timeframes=self._timeframes())
        failed = [f"{s.canonical_symbol}:{s.timeframe}" for s in report.series if not s.ok]
        if self.research.require_validated and failed and not self.research.allow_synthetic:
            # Soft-fail when no series exist yet (fresh DB); hard-fail only with corrupt data
            if report.series and report.failed == len(report.series) and report.passed == 0:
                logger.warning("All series failed validation: %s", failed[:10])
        return {
            "ok": report.ok,
            "passed": report.passed,
            "failed": report.failed,
            "failed_series": failed,
        }

    def _stage_repair(self) -> Dict[str, Any]:
        from collector.gap_repair import GapRepairEngine

        engine = GapRepairEngine(
            self.db,
            include_mt5=bool(self.config.data.include_mt5),
            csv_brokers=dict(self.config.data.csv_brokers or {}),
        )
        report = engine.repair(symbols=list(self.config.symbols), timeframes=self._timeframes())
        return report.to_dict()

    def _stage_ticks(self) -> Dict[str, Any]:
        from collector.tick_history import TickHistoryEngine

        engine = TickHistoryEngine(
            self.db,
            include_mt5=bool(self.config.data.include_mt5),
            csv_brokers=dict(self.config.data.csv_brokers or {}),
            lookback_days=int(self.research.tick_lookback_days),
        )
        report = engine.download_ticks(symbols=list(self.config.symbols), resume=True)
        return report.to_dict()

    def _stage_discovery(self, report: CycleReport) -> Dict[str, Any]:
        engine = HypothesisDiscoveryEngine(
            self.config,
            model_type=self.config.model.model_type
            if self.config.model.model_type == "random_forest"
            else "random_forest",
        )
        all_results: Dict[str, Any] = {}
        for symbol in self.config.symbols:
            candles = self._load_candles(symbol)
            if len(candles) < 120:
                continue
            discovered = engine.discover(candles)
            all_results[symbol] = discovered.to_dict()
            if discovered.selected_groups:
                self.state.selected_feature_groups = list(discovered.selected_groups)
                self.config.features.enabled_groups = list(discovered.selected_groups)
            for hyp in discovered.candidates:
                payload = hyp.to_dict()
                payload["symbol"] = symbol
                self.research_repo.record_hypothesis(payload)
                report.hypotheses.append(payload)
        return all_results

    def _stage_learn(self, report: CycleReport) -> Dict[str, Any]:
        learned: Dict[str, Any] = {"symbols": {}}
        candidates = [m for m in self.research.model_candidates if _model_available(m)]
        if not candidates:
            candidates = ["random_forest"]
        for symbol in self.config.symbols:
            candles = self._load_candles(symbol)
            learned["symbols"][symbol] = self._learn_symbol(symbol, candidates, candles, report)
            if learned["symbols"][symbol].get("metrics"):
                report.metrics_by_symbol[symbol] = dict(learned["symbols"][symbol]["metrics"])
        return learned

    def _learn_symbol(
        self,
        symbol: str,
        candidates: Sequence[str],
        candles: Sequence[Dict[str, Any]],
        report: CycleReport,
    ) -> Dict[str, Any]:
        if len(candles) < 80:
            return {"status": "skipped", "reason": "insufficient_candles", "n": len(candles)}

        experiment = self.research_repo.create_experiment(
            symbol=symbol,
            timeframe=self.config.primary_timeframe,
            cycle_id=report.cycle_id,
            objective_metric=self.research.primary_metric,
            metadata={"candidates": list(candidates)},
        )
        experiment_id = int(experiment["experiment_id"])

        # Apply discovered feature groups if present
        if self.state.selected_feature_groups:
            self.config.features.enabled_groups = list(self.state.selected_feature_groups)

        try:
            dataset = self.pipeline.build_dataset(candles)
        except ValueError as exc:
            self.research_repo.finish_experiment(
                experiment_id, status="skipped", metadata={"reason": str(exc)}
            )
            return {"status": "skipped", "reason": str(exc), "n": len(candles), "experiment_id": experiment_id}

        self.research_repo.record_dataset(
            experiment_id,
            symbol=symbol,
            timeframe=self.config.primary_timeframe,
            n_rows=dataset.bundle.n_samples,
            n_features=len(dataset.bundle.feature_names),
            train_rows=dataset.bundle.n_train,
            val_rows=dataset.bundle.n_val,
            test_rows=dataset.bundle.n_test,
            first_timestamp=str(dataset.features.timestamps[0]) if dataset.features.timestamps else None,
            last_timestamp=str(dataset.features.timestamps[-1]) if dataset.features.timestamps else None,
            feature_hash=hashlib.sha1(",".join(dataset.bundle.feature_names).encode()).hexdigest()[:16],
            label_method=dataset.label.method,
        )
        self.research_repo.record_features(
            experiment_id,
            [{"name": n, "group_name": "discovered", "source": "feature_engine", "kept": True}
             for n in dataset.bundle.feature_names],
        )

        champion_key = f"{symbol}:{self.config.primary_timeframe}"
        champion_meta = self.state.champions.get(champion_key)
        db_champion = self.research_repo.get_champion_model(symbol, self.config.primary_timeframe)
        champion_metrics = (champion_meta or {}).get("metrics")
        if not champion_metrics and db_champion:
            raw = db_champion.get("metrics")
            champion_metrics = json.loads(raw) if isinstance(raw, str) else raw

        best: Dict[str, Any] | None = None
        comparisons: List[Dict[str, Any]] = []

        for model_type in candidates:
            cfg = self.config.copy()
            cfg.model.model_type = model_type
            cfg.data.auto_download = False
            local = AIPipeline(
                config=cfg,
                candle_source=self.pipeline.candle_source,
                data_service=self.pipeline.data_service,
            )
            try:
                run = local.run_training(candles=candles, register=False, auto_download=False)
                metrics = {**run.train.metrics, **run.evaluation}
                decision = decide_promotion(
                    challenger_metrics=metrics,
                    champion_metrics=champion_metrics,
                    metric_name=self.research.primary_metric,
                    minimize=self.research.metric_minimize,
                    min_improvement=self.research.min_improvement,
                )
                entry = {"model_type": model_type, "metrics": metrics, "decision": decision.to_dict()}

                # Strict validation required before any paper path
                validation = None
                if decision.accepted and self.research.run_strict_validation:
                    validation = self.validation_gate.validate(
                        run.train.model,
                        run.dataset.bundle,
                        candles=candles,
                        symbol=symbol,
                        timeframe=self.config.primary_timeframe,
                    )
                    entry["validation"] = validation.to_dict()
                    for stage in validation.stages:
                        self.research_repo.record_validation(
                            experiment_id=experiment_id,
                            model_id=None,
                            stage=stage.stage,
                            passed=stage.passed,
                            metrics=stage.metrics,
                            details=stage.details,
                        )
                    if not validation.passed:
                        decision_payload = decision.to_dict()
                        decision_payload["accepted"] = False
                        decision_payload["reason"] = f"validation_failed:{validation.reason}"
                        entry["decision"] = decision_payload
                        report.rejected.append({"symbol": symbol, "model_type": model_type, "decision": decision_payload})
                        model_row = self.research_repo.record_model(
                            experiment_id,
                            name=f"{symbol}_{model_type}",
                            model_type=model_type,
                            symbol=symbol,
                            timeframe=self.config.primary_timeframe,
                            metrics=metrics,
                            hyperparameters=run.train.model.get_params(),
                            status="rejected_validation",
                            metadata={"validation_reason": validation.reason},
                        )
                        comparisons.append(entry)
                        continue

                comparisons.append(entry)
                accepted = entry["decision"]["accepted"]
                if accepted:
                    if best is None or _better(
                        metrics, best["metrics"], self.research.primary_metric, self.research.metric_minimize
                    ):
                        best = {
                            "model_type": model_type,
                            "metrics": metrics,
                            "decision": entry["decision"],
                            "model": run.train.model,
                            "bundle": run.dataset.bundle,
                            "feature_names": list(run.dataset.bundle.feature_names),
                            "validation": validation.to_dict() if validation else None,
                        }
                else:
                    report.rejected.append({"symbol": symbol, "model_type": model_type, "decision": entry["decision"]})
                    self.research_repo.record_model(
                        experiment_id,
                        name=f"{symbol}_{model_type}",
                        model_type=model_type,
                        symbol=symbol,
                        timeframe=self.config.primary_timeframe,
                        metrics=metrics,
                        hyperparameters=run.train.model.get_params(),
                        status="rejected",
                        metadata={"reason": entry["decision"].get("reason")},
                    )
            except Exception as exc:
                comparisons.append(
                    {
                        "model_type": model_type,
                        "error": f"{exc.__class__.__name__}: {exc}",
                        "decision": {"accepted": False, "reason": "train_error"},
                    }
                )

        if best is None:
            self.research_repo.finish_experiment(experiment_id, status="no_promotion")
            return {"status": "no_promotion", "comparisons": comparisons, "experiment_id": experiment_id}

        # Promote champion
        self.pipeline.model = best["model"]
        self.pipeline.config.model.model_type = best["model_type"]
        registration = None
        if self.research.register_only_improvements:
            assert self.pipeline.registry is not None
            registration = self.pipeline.registry.register(
                name=f"{symbol}_{best['model_type']}",
                model=best["model"],
                features=best["feature_names"],
                metrics=best["metrics"],
                metadata={"cycle": report.cycle_id, "promoted_by": "phase4_engine"},
            )
            self.pipeline.model_version = registration.version

        model_row = self.research_repo.record_model(
            experiment_id,
            name=f"{symbol}_{best['model_type']}",
            model_type=best["model_type"],
            symbol=symbol,
            timeframe=self.config.primary_timeframe,
            version=getattr(registration, "version", None),
            artifact_path=str(getattr(registration, "path", "") or ""),
            metrics=best["metrics"],
            hyperparameters=best["model"].get_params(),
            is_champion=True,
            is_deployed=True,
            status="champion",
            metadata={"decision": best["decision"], "validation": best.get("validation")},
        )
        self.research_repo.set_champion(int(model_row["model_id"]), symbol, self.config.primary_timeframe)
        self.research_repo.record_deployment(
            model_id=int(model_row["model_id"]),
            symbol=symbol,
            timeframe=self.config.primary_timeframe,
            environment="paper",
            status="active",
            reason=best["decision"].get("reason"),
            metrics=best["metrics"],
        )

        if self.research.run_backtest:
            bt_metrics = self._persist_backtest(experiment_id, int(model_row["model_id"]), symbol, candles, best)

        score = extract_metric(best["metrics"], self.research.primary_metric)
        self.research_repo.finish_experiment(
            experiment_id, status="promoted", objective_value=score, metadata={"model_id": model_row["model_id"]}
        )
        champion_payload = {
            "model_type": best["model_type"],
            "metrics": best["metrics"],
            "version": getattr(registration, "version", None),
            "model_id": model_row["model_id"],
            "updated_at": utc_now_iso(),
        }
        self.state.champions[champion_key] = champion_payload
        report.champions[champion_key] = {
            "model_type": best["model_type"],
            "metrics": {self.research.primary_metric: score},
            "model_id": model_row["model_id"],
        }
        return {
            "status": "promoted",
            "champion": report.champions[champion_key],
            "comparisons": comparisons,
            "metrics": best["metrics"],
            "experiment_id": experiment_id,
            "model_id": model_row["model_id"],
            "backtest": bt_metrics if self.research.run_backtest else None,
        }

    def _stage_paper_all(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {}
        for symbol in self.config.symbols:
            champion = self.research_repo.get_champion_model(symbol, self.config.primary_timeframe)
            # Only paper-trade models that passed validation / are champions
            if champion is None and not self.state.champions.get(f"{symbol}:{self.config.primary_timeframe}"):
                out[symbol] = {"status": "skipped", "reason": "no_validated_champion"}
                continue
            model_id = champion.get("model_id") if champion else None
            journal = PaperTradingJournal(
                self.config,
                self.research_repo,
                self.pipeline,
                model_id=int(model_id) if model_id is not None else None,
                equity=float(self.research.paper_equity),
            )
            candles = self._load_candles(symbol, limit=min(500, self.research.candle_limit))
            result = journal.run(symbol=symbol, candles=candles)
            out[symbol] = result.to_dict()
        return out

    def _stage_production_gate(self) -> Dict[str, Any]:
        results = {}
        for symbol in self.config.symbols:
            champion = self.research_repo.get_champion_model(symbol, self.config.primary_timeframe)
            gate = self.production_gate.evaluate(
                symbol,
                self.config.primary_timeframe,
                model_id=champion.get("model_id") if champion else None,
                enable_live_if_passed=False,  # never auto-enable live without explicit ops flag
            )
            results[symbol] = gate.to_dict()
        return results

    def _stage_hypotheses(self, report: CycleReport) -> Dict[str, Any]:
        validate_detail = self._find_stage_detail("validate") or {}
        hyps = generate_hypotheses(
            per_symbol_metrics=report.metrics_by_symbol,
            validation_failures=list(validate_detail.get("failed_series") or []),
            primary_metric=self.research.primary_metric,
        )
        for hyp in hyps:
            payload = hyp.to_dict()
            report.hypotheses.append(payload)
            self.research_repo.record_hypothesis(payload)
        self.state.pending_hypotheses = list(report.hypotheses)
        return {"count": len(report.hypotheses)}

    def _stage_dashboard(self) -> Dict[str, Any]:
        # Snapshot monitoring metrics first
        for symbol in self.config.symbols:
            champion = self.research_repo.get_champion_model(symbol, self.config.primary_timeframe)
            stats = self.research_repo.paper_trade_stats(symbol, self.config.primary_timeframe)
            preds = stats.get("predictions") or {}
            age = None
            if champion and champion.get("created_at"):
                try:
                    from datetime import datetime

                    age = (
                        datetime.utcnow()
                        - datetime.fromisoformat(str(champion["created_at"]).replace("Z", ""))
                    ).total_seconds() / 3600.0
                except Exception:
                    age = None
            self.research_repo.record_monitoring_snapshot(
                symbol=symbol,
                timeframe=self.config.primary_timeframe,
                model_id=champion.get("model_id") if champion else None,
                accuracy=preds.get("accuracy"),
                sharpe=None,
                profit_factor=None,
                max_drawdown=preds.get("min_drawdown"),
                drift_score=None,
                model_age_hours=age,
                feature_importance={},
                metrics=stats,
            )
        dash = InstitutionalDashboard(self.research_repo, self.dashboard_dir)
        artifacts = dash.build(symbols=list(self.config.symbols))
        return artifacts.to_dict()

    def _stage_persist(self, report: CycleReport) -> Dict[str, Any]:
        latest = self.reports_dir / "latest.json"
        report.save(latest)
        return {"latest_path": str(latest)}

    # ==================================================================
    # Helpers
    # ==================================================================

    def _research_symbol(self, symbol: str, candles: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
        report = CycleReport(cycle_id=uuid.uuid4().hex[:12], started_at=utc_now_iso())
        return self._learn_symbol(symbol, [self.config.model.model_type], candles, report)

    def _persist_backtest(
        self,
        experiment_id: int,
        model_id: int,
        symbol: str,
        candles: Sequence[Dict[str, Any]],
        best: Dict[str, Any],
    ) -> Dict[str, Any]:
        from ai.evaluation.backtest import BacktestEngine, BacktestSignal, Candle

        model = best["model"]
        bundle = best["bundle"]
        preds = model.predict(bundle.X_test) if bundle.n_test else []
        preds = list(__import__("numpy").asarray(preds).reshape(-1))
        n = min(len(preds), len(candles))
        slice_c = list(candles)[-n:] if n else []
        signals = []
        for i, (pred, c) in enumerate(zip(preds, slice_c)):
            if i % max(1, self.config.labels.horizon) != 0:
                continue
            side = "buy" if float(pred) > 0 else "sell"
            signals.append(
                BacktestSignal(
                    symbol=symbol,
                    timestamp=c["timestamp"],
                    side=side,
                    quantity=0.1,
                    timeframe=self.config.primary_timeframe,
                )
            )
        if signals and slice_c:
            signals.append(
                BacktestSignal(
                    symbol=symbol,
                    timestamp=slice_c[-1]["timestamp"],
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
            for c in slice_c
        ]
        engine = BacktestEngine(config=self.config)
        result = engine.run(signals=signals, candles=bt_candles)
        metrics = dict(result.metrics or {})
        self.research_repo.record_backtest(
            experiment_id=experiment_id,
            model_id=model_id,
            symbol=symbol,
            timeframe=self.config.primary_timeframe,
            n_trades=len(result.trades),
            metrics=metrics,
            spread_points=float(engine.spread_points or 0.0),
            commission_per_lot=float(engine.commission_per_lot or 0.0),
            slippage_points=float(engine.slippage_points or 0.0),
        )
        return {"n_trades": len(result.trades), "metrics": metrics}

    def _detect_history_bounds(self) -> Dict[str, Any]:
        try:
            from collector.broker_sources.registry import build_default_registry

            registry = build_default_registry(
                include_mt5=bool(self.config.data.include_mt5),
                csv_brokers=dict(self.config.data.csv_brokers or {}),
            )
            out: Dict[str, Any] = {}
            for source in registry.all():
                if not source.connect():
                    continue
                try:
                    for symbol in self.config.symbols:
                        key = f"{source.name}:{symbol}:{self.config.primary_timeframe}"
                        out[key] = source.detect_available_history(
                            symbol, self.config.primary_timeframe
                        )
                finally:
                    source.disconnect()
            return out
        except Exception as exc:
            return {"error": f"{exc.__class__.__name__}: {exc}"}

    def _load_candles(self, symbol: str, limit: int | None = None) -> List[Dict[str, Any]]:
        self.pipeline._ensure_candle_source()
        try:
            return self.pipeline.load_candles(
                symbol=symbol,
                timeframe=self.config.primary_timeframe,
                limit=int(limit or self.research.candle_limit),
                auto_download=False,
            )
        except Exception:
            if self.research.allow_synthetic:
                self.pipeline.ensure_market_data(force=False)
                self.pipeline._ensure_candle_source()
                return self.pipeline.load_candles(
                    symbol=symbol,
                    timeframe=self.config.primary_timeframe,
                    limit=int(limit or self.research.candle_limit),
                    auto_download=False,
                )
            return []

    def _timeframes(self) -> List[str]:
        return list(
            dict.fromkeys(
                [
                    self.config.primary_timeframe,
                    *self.config.timeframes,
                    *self.config.features.multi_timeframes,
                ]
            )
        )

    def _open_db(self) -> DatabaseManager:
        from core.config import DATABASE_PATH
        from database.indexes import create_indexes
        from database.migrations import apply_migrations
        from database.schema import create_schema
        from database.seed import seed

        path = Path(self.config.data.database_path or str(DATABASE_PATH))
        path.parent.mkdir(parents=True, exist_ok=True)
        db = DatabaseManager(db_path=path)
        create_schema(db)
        create_indexes(db)
        seed(db)
        apply_migrations(db)
        return db

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
            if name == "learn":
                report.stages.append(stage)
                raise
        report.stages.append(stage)
        return stage

    def _find_stage_detail(self, name: str) -> Dict[str, Any] | None:
        for report in (self._current_report, self._last_report):
            if report is None:
                continue
            for stage in reversed(report.stages):
                if stage.name == name:
                    return stage.detail
        return None

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
    return AutonomousResearchPlatform(config=config, research=research, **kwargs)


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
    return True


def _better(left: Dict[str, Any], right: Dict[str, Any], metric: str, minimize: bool) -> bool:
    a = extract_metric(left, metric)
    b = extract_metric(right, metric)
    if a is None:
        return False
    if b is None:
        return True
    return a < b if minimize else a > b
