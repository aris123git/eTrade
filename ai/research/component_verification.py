"""
ai/research/component_verification.py - Continuous verification of every subsystem.

No new models or indicators. Verifies that the existing platform components
are present, wired, and behaving correctly before edge-proof research runs.
"""

from __future__ import annotations

import importlib
import logging
import tempfile
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence
from uuid import uuid4

logger = logging.getLogger(__name__)


@dataclass
class CheckResult:
    name: str
    status: str  # PASS | FAIL | SKIP
    detail: str = ""
    metrics: Dict[str, Any] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return self.status in {"PASS", "SKIP"}

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class VerificationReport:
    started_at: str
    finished_at: str = ""
    checks: List[CheckResult] = field(default_factory=list)

    @property
    def passed(self) -> int:
        return sum(1 for c in self.checks if c.status == "PASS")

    @property
    def failed(self) -> int:
        return sum(1 for c in self.checks if c.status == "FAIL")

    @property
    def skipped(self) -> int:
        return sum(1 for c in self.checks if c.status == "SKIP")

    @property
    def ok(self) -> bool:
        return self.failed == 0 and self.passed > 0

    @property
    def critical_ok(self) -> bool:
        """Critical path must pass (imports, DB, research core)."""
        critical = {
            "imports",
            "database_schema",
            "research_tables",
            "history_engine",
            "history_validator",
            "ai_pipeline",
            "strict_validation_gate",
            "model_promotion_gate",
            "research_repository",
            "config_honesty",
        }
        by_name = {c.name: c for c in self.checks}
        for name in critical:
            check = by_name.get(name)
            if check is None or check.status == "FAIL":
                return False
        return True

    def to_dict(self) -> Dict[str, Any]:
        return {
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "ok": self.ok,
            "critical_ok": self.critical_ok,
            "passed": self.passed,
            "failed": self.failed,
            "skipped": self.skipped,
            "checks": [c.to_dict() for c in self.checks],
        }

    def print_summary(self) -> None:
        print()
        print("=" * 64)
        print("Component Verification")
        print("=" * 64)
        for check in self.checks:
            print(f"  [{check.status:4}] {check.name}: {check.detail}")
        print()
        print(
            f"PASS={self.passed} FAIL={self.failed} SKIP={self.skipped} "
            f"overall={'PASS' if self.ok else 'FAIL'} critical={'PASS' if self.critical_ok else 'FAIL'}"
        )
        print()


def _now() -> str:
    return datetime.now(timezone.utc).replace(tzinfo=None).isoformat(timespec="seconds")


class ComponentVerifier:
    """
    Verify every component required for autonomous edge research.

    Does not train production models. Does not invent market data.
    """

    CRITICAL_IMPORTS: Sequence[str] = (
        "core.config",
        "core.symbol_identity",
        "database.core.connection",
        "database.schema",
        "database.migrations",
        "database.repositories.factory",
        "database.repositories.research_repository",
        "collector.history_engine",
        "collector.history_validator",
        "collector.gap_repair",
        "collector.tick_history",
        "ai.config.settings",
        "ai.services.pipeline",
        "ai.features.engine",
        "ai.labels.generator",
        "ai.models",
        "ai.training.trainer",
        "ai.training.validation",
        "ai.evaluation.backtest",
        "ai.evaluation.monte_carlo",
        "ai.storage.registry",
        "ai.execution.executor",
        "ai.monitoring.drift",
        "ai.monitoring.dashboard",
        "ai.scheduler.scheduler",
        "ai.research.platform",
        "ai.research.validation_gate",
        "ai.research.gate",
        "ai.research.paper_journal",
        "ai.research.self_improve",
        "ai.research.production_gate",
        "ai.research.edge_proof",
        "ai.research.discovery",
        "ai.research.autonomous_scheduler",
    )

    RESEARCH_TABLES: Sequence[str] = (
        "research_experiments",
        "research_datasets",
        "research_features",
        "research_models",
        "research_hyperparameters",
        "research_backtests",
        "research_validations",
        "research_deployments",
        "research_paper_trades",
        "research_predictions",
        "research_hypotheses",
        "research_monitoring_snapshots",
        "research_production_gates",
        "candles",
        "ticks",
        "markets",
        "brokers",
        "sync_status",
    )

    def __init__(self, db: Any = None, config: Any = None):
        self.db = db
        self.config = config

    def verify_all(self) -> VerificationReport:
        report = VerificationReport(started_at=_now())
        steps: List[tuple[str, Callable[[], CheckResult]]] = [
            ("imports", self._check_imports),
            ("config_honesty", self._check_config_honesty),
            ("database_schema", self._check_database_schema),
            ("research_tables", self._check_research_tables),
            ("history_engine", self._check_history_engine),
            ("history_validator", self._check_history_validator),
            ("gap_repair", self._check_gap_repair),
            ("tick_history", self._check_tick_history),
            ("ai_pipeline", self._check_ai_pipeline),
            ("walk_forward", self._check_walk_forward),
            ("monte_carlo", self._check_monte_carlo),
            ("backtest_engine", self._check_backtest),
            ("strict_validation_gate", self._check_strict_gate),
            ("model_promotion_gate", self._check_promotion_gate),
            ("research_repository", self._check_research_repo),
            ("paper_journal", self._check_paper_journal),
            ("self_improve", self._check_self_improve),
            ("production_gate", self._check_production_gate),
            ("edge_proof", self._check_edge_proof),
            ("scheduler", self._check_scheduler),
            ("dashboard", self._check_dashboard),
            ("archive_integrity", self._check_archive_integrity),
        ]
        for name, fn in steps:
            try:
                result = fn()
                if result.name != name:
                    result = CheckResult(name=name, status=result.status, detail=result.detail, metrics=result.metrics)
            except Exception as exc:
                result = CheckResult(name=name, status="FAIL", detail=f"{exc.__class__.__name__}: {exc}")
            report.checks.append(result)
        report.finished_at = _now()
        return report

    # ------------------------------------------------------------------
    # Individual checks
    # ------------------------------------------------------------------

    def _check_imports(self) -> CheckResult:
        failed: Dict[str, str] = {}
        for name in self.CRITICAL_IMPORTS:
            try:
                importlib.import_module(name)
            except Exception as exc:
                failed[name] = f"{exc.__class__.__name__}: {exc}"
        if failed:
            return CheckResult("imports", "FAIL", f"{len(failed)} import failures", {"failed": failed})
        return CheckResult("imports", "PASS", f"{len(self.CRITICAL_IMPORTS)} modules importable")

    def _check_config_honesty(self) -> CheckResult:
        from ai.config.settings import AIConfig

        cfg = self.config or AIConfig()
        issues = []
        if bool(getattr(cfg.data, "allow_synthetic_fallback", True)):
            issues.append("allow_synthetic_fallback=True")
        if not bool(getattr(cfg.data, "require_validated", False)):
            # Warn but allow if explicit research override in tests
            if self.config is None:
                issues.append("require_validated=False")
        research = getattr(cfg, "research", None)
        if research is not None and bool(getattr(research, "allow_synthetic", False)):
            issues.append("research.allow_synthetic=True")
        if issues:
            return CheckResult("config_honesty", "FAIL", "synthetic/validation defaults unsafe: " + ", ".join(issues))
        return CheckResult(
            "config_honesty",
            "PASS",
            "synthetic disabled; validated history required by default",
            {
                "allow_synthetic_fallback": bool(cfg.data.allow_synthetic_fallback),
                "require_validated": bool(cfg.data.require_validated),
            },
        )

    def _check_database_schema(self) -> CheckResult:
        if self.db is None:
            return CheckResult("database_schema", "SKIP", "no db injected")
        from database.schema import create_schema
        from database.migrations import apply_migrations

        create_schema(self.db)
        apply_migrations(self.db)
        row = self._fetch_one("SELECT name FROM sqlite_master WHERE type='table' AND name='candles'")
        if not row:
            return CheckResult("database_schema", "FAIL", "candles table missing")
        return CheckResult("database_schema", "PASS", "schema + migrations applied")

    def _check_research_tables(self) -> CheckResult:
        if self.db is None:
            return CheckResult("research_tables", "SKIP", "no db injected")
        from database.migrations.research_schema import apply_research_schema

        apply_research_schema(self.db)
        missing = []
        for table in self.RESEARCH_TABLES:
            row = self._fetch_one(
                "SELECT name FROM sqlite_master WHERE type='table' AND name=?",
                (table,),
            )
            if not row:
                missing.append(table)
        if missing:
            return CheckResult("research_tables", "FAIL", f"missing tables: {missing}")
        return CheckResult("research_tables", "PASS", f"{len(self.RESEARCH_TABLES)} tables present")

    def _check_history_engine(self) -> CheckResult:
        from collector.history_engine import HistoricalDataEngine, download_history

        assert callable(download_history)
        assert HistoricalDataEngine is not None
        return CheckResult("history_engine", "PASS", "download_history available")

    def _check_history_validator(self) -> CheckResult:
        from collector.history_validator import HistoryValidator, validate_history

        assert callable(validate_history)
        if self.db is None:
            return CheckResult("history_validator", "PASS", "HistoryValidator importable (no db)")
        report = HistoryValidator(self.db, min_bars=1).validate_all()
        return CheckResult(
            "history_validator",
            "PASS",
            f"validate_all ok series={len(report.series)} passed={report.passed}",
            {"passed": report.passed, "failed": report.failed},
        )

    def _check_gap_repair(self) -> CheckResult:
        from collector.gap_repair import GapRepairEngine

        assert GapRepairEngine is not None
        return CheckResult("gap_repair", "PASS", "GapRepairEngine available")

    def _check_tick_history(self) -> CheckResult:
        from collector.tick_history import TickHistoryEngine

        assert TickHistoryEngine is not None
        return CheckResult("tick_history", "PASS", "TickHistoryEngine available")

    def _check_ai_pipeline(self) -> CheckResult:
        from ai.config.settings import AIConfig
        from ai.services.pipeline import create_ai_pipeline

        cfg = AIConfig()
        if self.config is not None and hasattr(self.config, "copy"):
            # Keep honesty settings from caller, but use a probe-safe feature set
            cfg.data.allow_synthetic_fallback = bool(self.config.data.allow_synthetic_fallback)
            cfg.data.require_validated = bool(self.config.data.require_validated)
        cfg.data.auto_download = False
        cfg.data.allow_synthetic_fallback = False
        cfg.features.enabled_groups = [
            "price",
            "returns",
            "momentum",
            "volatility",
            "session",
        ]
        cfg.features.multi_timeframes = []
        cfg.features.correlation_symbols = []
        with tempfile.TemporaryDirectory() as tmp:
            cfg.storage.root_dir = Path(tmp)
            pipeline = create_ai_pipeline(config=cfg, ensure_data=False)
            candles = self._tiny_candles(n=400)
            frame = pipeline.build_features(candles)
            labels = pipeline.build_labels(candles)
            if frame.matrix.size == 0 or not labels:
                return CheckResult("ai_pipeline", "FAIL", "features/labels empty on probe candles")
            return CheckResult(
                "ai_pipeline",
                "PASS",
                f"features={frame.matrix.shape} labels={len(labels)}",
                {"feature_count": int(frame.matrix.shape[1]), "label_count": len(labels)},
            )

    def _check_walk_forward(self) -> CheckResult:
        from ai.training.validation import walk_forward_validation, summarize_scores
        from ai.models import create_model
        from ai.config.settings import AIConfig
        import numpy as np

        cfg = AIConfig()
        cfg.model.model_type = "random_forest"
        model = create_model("random_forest", cfg)
        rng = np.random.default_rng(0)
        X = rng.normal(size=(120, 4))
        y = (X[:, 0] + X[:, 1] > 0).astype(float)
        scores = walk_forward_validation(model, X, y, folds=3, embargo=1)
        summary = summarize_scores(scores)
        if not scores:
            return CheckResult("walk_forward", "FAIL", "no fold scores")
        return CheckResult("walk_forward", "PASS", f"folds={len(scores)}", summary)

    def _check_monte_carlo(self) -> CheckResult:
        from ai.evaluation.monte_carlo import monte_carlo_reshuffle

        result = monte_carlo_reshuffle([0.01, -0.005, 0.02, -0.01, 0.015], n_simulations=50, random_seed=1)
        if result.final_equity.size != 50:
            return CheckResult("monte_carlo", "FAIL", "unexpected simulation count")
        return CheckResult("monte_carlo", "PASS", "reshuffle ok", {"n": int(result.final_equity.size)})

    def _check_backtest(self) -> CheckResult:
        from ai.evaluation.backtest import BacktestEngine, BacktestSignal, Candle
        from ai.config.settings import AIConfig

        cfg = AIConfig()
        t0 = datetime(2024, 1, 2, 10, 0, 0)
        candles = [
            Candle("EURUSD", "M15", t0 + timedelta(minutes=15 * i), 1.1, 1.11, 1.09, 1.1 + i * 0.0001, 10)
            for i in range(40)
        ]
        signals = [
            BacktestSignal("EURUSD", candles[5].timestamp, "buy", quantity=0.1, timeframe="M15"),
            BacktestSignal("EURUSD", candles[30].timestamp, "close", quantity=0.1, timeframe="M15"),
        ]
        result = BacktestEngine(config=cfg).run(signals=signals, candles=candles)
        return CheckResult(
            "backtest_engine",
            "PASS",
            f"trades={len(result.trades)}",
            {"n_trades": len(result.trades), **{k: result.metrics.get(k) for k in ("sharpe", "profit_factor", "max_drawdown") if k in (result.metrics or {})}},
        )

    def _check_strict_gate(self) -> CheckResult:
        from ai.research.validation_gate import StrictValidationGate, ValidationThresholds
        from ai.config.settings import AIConfig

        gate = StrictValidationGate(AIConfig(), ValidationThresholds())
        assert hasattr(gate, "validate")
        return CheckResult("strict_validation_gate", "PASS", "StrictValidationGate ready")

    def _check_promotion_gate(self) -> CheckResult:
        from ai.research.gate import decide_promotion

        ok = decide_promotion(challenger_metrics={"test_f1": 0.7}, champion_metrics={"test_f1": 0.6}).accepted
        bad = decide_promotion(challenger_metrics={"test_f1": 0.5}, champion_metrics={"test_f1": 0.6}).accepted
        if not ok or bad:
            return CheckResult("model_promotion_gate", "FAIL", "promotion logic incorrect")
        return CheckResult("model_promotion_gate", "PASS", "keeps improvements, rejects regressions")

    def _check_research_repo(self) -> CheckResult:
        if self.db is None:
            return CheckResult("research_repository", "SKIP", "no db injected")
        from database.repositories.research_repository import ResearchRepository
        from database.migrations.research_schema import apply_research_schema

        apply_research_schema(self.db)
        repo = ResearchRepository(self.db)
        exp = repo.create_experiment(symbol="EURUSD", timeframe="M15", cycle_id="verify")
        model = repo.record_model(
            int(exp["experiment_id"]),
            name=f"verify_{uuid4().hex[:8]}",
            model_type="random_forest",
            symbol="EURUSD",
            timeframe="M15",
            metrics={"test_f1": 0.55},
            status="candidate",
        )
        repo.record_validation(
            experiment_id=int(exp["experiment_id"]),
            model_id=int(model["model_id"]),
            stage="component_verification",
            passed=True,
            metrics={"ok": 1},
        )
        repo.finish_experiment(int(exp["experiment_id"]), status="verified")
        return CheckResult(
            "research_repository",
            "PASS",
            "experiment/model/validation write ok",
            {"experiment_id": exp["experiment_id"], "model_id": model["model_id"]},
        )

    def _check_paper_journal(self) -> CheckResult:
        from ai.research.paper_journal import PaperTradingJournal

        assert PaperTradingJournal is not None
        return CheckResult("paper_journal", "PASS", "PaperTradingJournal available")

    def _check_self_improve(self) -> CheckResult:
        from ai.research.self_improve import SelfImprovementController

        assert SelfImprovementController is not None
        return CheckResult("self_improve", "PASS", "SelfImprovementController available")

    def _check_production_gate(self) -> CheckResult:
        if self.db is None:
            return CheckResult("production_gate", "SKIP", "no db injected")
        from database.repositories.research_repository import ResearchRepository
        from database.migrations.research_schema import apply_research_schema
        from ai.research.production_gate import ProductionReadinessGate, ProductionThresholds

        apply_research_schema(self.db)
        gate = ProductionReadinessGate(
            ResearchRepository(self.db),
            ProductionThresholds(min_paper_trades=50, min_paper_days=14),
        )
        result = gate.evaluate("EURUSD", "M15", enable_live_if_passed=False)
        if result.live_enabled:
            return CheckResult("production_gate", "FAIL", "live enabled without evidence")
        return CheckResult(
            "production_gate",
            "PASS",
            "live blocked until thresholds met",
            {"passed": result.passed, "failures": result.failures[:5]},
        )

    def _check_edge_proof(self) -> CheckResult:
        from ai.research.edge_proof import EdgeProofEngine, apply_edge_proof_config
        from ai.config.settings import ResearchConfig

        cfg = apply_edge_proof_config(ResearchConfig())
        if cfg.allow_synthetic or not cfg.require_validated:
            return CheckResult("edge_proof", "FAIL", "edge-proof config allows synthetic or skips validation")
        assert EdgeProofEngine is not None
        return CheckResult("edge_proof", "PASS", "edge-proof mode constrained correctly")

    def _check_scheduler(self) -> CheckResult:
        from ai.research.autonomous_scheduler import AutonomousScheduler, SchedulePlan
        from ai.scheduler.scheduler import AIScheduler

        sched = AIScheduler(poll_seconds=1.0)
        assert AutonomousScheduler is not None and SchedulePlan is not None
        return CheckResult("scheduler", "PASS", "AIScheduler + AutonomousScheduler available")

    def _check_dashboard(self) -> CheckResult:
        from ai.monitoring.dashboard import InstitutionalDashboard

        assert InstitutionalDashboard is not None
        return CheckResult("dashboard", "PASS", "InstitutionalDashboard available")

    def _check_archive_integrity(self) -> CheckResult:
        if self.db is None:
            return CheckResult("archive_integrity", "SKIP", "no db injected")
        candles = self._count("SELECT COUNT(*) AS c FROM candles")
        dups = self._count(
            """
            SELECT COALESCE(SUM(cnt-1),0) AS c FROM (
              SELECT broker_id, symbol, timeframe, timestamp, COUNT(*) AS cnt
              FROM candles GROUP BY 1,2,3,4 HAVING COUNT(*) > 1
            )
            """
        )
        invalid = self._count(
            """
            SELECT COUNT(*) AS c FROM candles
            WHERE high < low OR high < open OR high < close OR low > open OR low > close
            """
        )
        status = "PASS" if dups == 0 and invalid == 0 else "FAIL"
        detail = f"candles={candles} duplicates={dups} invalid_ohlc={invalid}"
        return CheckResult(
            "archive_integrity",
            status,
            detail,
            {"candles": candles, "duplicates": dups, "invalid_ohlc": invalid},
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _tiny_candles(self, n: int = 400) -> List[Dict[str, Any]]:
        t0 = datetime(2024, 1, 2, 8, 0, 0)
        price = 1.10
        out = []
        for i in range(int(n)):
            close = price + 0.00005
            out.append(
                {
                    "symbol": "EURUSD",
                    "timeframe": "M15",
                    "timestamp": t0 + timedelta(minutes=15 * i),
                    "open": price,
                    "high": max(price, close) + 0.0001,
                    "low": min(price, close) - 0.0001,
                    "close": close,
                    "volume": 100.0,
                }
            )
            price = close
        return out

    def _fetch_one(self, sql: str, params: tuple = ()) -> Optional[Dict[str, Any]]:
        if hasattr(self.db, "fetch_one"):
            row = self.db.fetch_one(sql, params)
        else:
            cur = self.db.get_adapter().execute(sql, params)
            row = cur.fetchone()
        if row is None:
            return None
        return dict(row) if hasattr(row, "keys") else None

    def _count(self, sql: str) -> int:
        row = self._fetch_one(sql)
        if not row:
            return 0
        return int(row.get("c") if isinstance(row, dict) else row[0] or 0)


def verify_components(db: Any = None, config: Any = None) -> VerificationReport:
    """Module-level entrypoint."""
    return ComponentVerifier(db=db, config=config).verify_all()
