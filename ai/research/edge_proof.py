"""
ai/research/edge_proof.py - Evidence-first mode.

Stop expanding architecture. The only objective is answering:

  Can eTrade demonstrate a real statistical edge on real market data?

Daily routine:
  download candles/ticks → repair gaps → train → validate →
  paper trade → store results → sleep

No synthetic data. No live trading. Accumulate evidence over time.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from ai.config.settings import AIConfig, ResearchConfig
from ai.research.platform import AutonomousResearchPlatform
from ai.research.report import utc_now_iso

logger = logging.getLogger(__name__)


@dataclass
class EdgeEvidence:
    """Accumulating ledger of whether a real edge exists."""

    updated_at: str = ""
    runs_completed: int = 0
    scientific_claim_allowed: bool = False
    edge_demonstrated: bool = False
    reason: str = "insufficient_evidence"
    components_ok: bool = False
    components: Dict[str, Any] = field(default_factory=dict)
    archive: Dict[str, Any] = field(default_factory=dict)
    experiments: Dict[str, Any] = field(default_factory=dict)
    paper: Dict[str, Any] = field(default_factory=dict)
    champions: Dict[str, Any] = field(default_factory=dict)
    history: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def apply_edge_proof_config(research: ResearchConfig) -> ResearchConfig:
    """Constrain the platform to download → train → validate → paper → store."""
    research.allow_synthetic = False
    research.require_validated = True
    research.skip_collect = False
    research.repair_failed_series = True
    research.download_ticks = True
    research.run_strict_validation = True
    research.run_backtest = True
    research.run_paper_trade = True
    research.run_self_improve = True
    research.register_only_improvements = True
    # Keep discovery light; do not expand architecture elsewhere
    research.run_feature_discovery = True
    research.generate_hypotheses = True
    research.run_production_gate = True  # track readiness; never auto-live
    research.build_dashboard = True
    research.history_start = research.history_start or "2010-01-01"
    return research


class EdgeProofEngine:
    """
    Thin evidence loop over AutonomousResearchPlatform.

    Does not invent history. Does not enable live trading.
    Writes an accumulating edge_evidence.json after every run.
    """

    def __init__(
        self,
        platform: AutonomousResearchPlatform,
        *,
        evidence_path: Path | str | None = None,
    ):
        self.platform = platform
        self.research = apply_edge_proof_config(platform.research)
        platform.research = self.research
        platform.config.research = self.research
        platform.config.data.allow_synthetic_fallback = False
        platform.config.data.require_validated = True
        root = Path(platform.artifact_root)
        self.evidence_path = Path(evidence_path) if evidence_path else root / "edge_evidence.json"
        self.evidence = self._load()

    def verify_components(self) -> Dict[str, Any]:
        """Verify every subsystem before / during evidence collection."""
        from ai.research.component_verification import verify_components

        report = verify_components(db=self.platform.db, config=self.platform.config)
        path = Path(self.platform.artifact_root) / "component_verification_latest.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(report.to_dict(), indent=2, default=str), encoding="utf-8")
        # Persist into research validations for audit
        try:
            self.platform.research_repo.record_validation(
                experiment_id=None,
                model_id=None,
                stage="component_verification",
                passed=report.critical_ok,
                metrics={"passed": report.passed, "failed": report.failed, "skipped": report.skipped},
                details=report.to_dict(),
            )
        except Exception as exc:
            logger.warning("could not persist component verification: %s", exc)
        return report.to_dict()

    def run_day(self) -> Dict[str, Any]:
        """
        One evidence day:

          verify components → download → repair → ticks → train/validate → paper → store
        """
        started = utc_now_iso()
        logger.info("edge-proof day start")

        components = self.verify_components()
        if not components.get("critical_ok"):
            day = {
                "started_at": started,
                "finished_at": utc_now_iso(),
                "components": components,
                "aborted": True,
                "reason": "critical_component_verification_failed",
            }
            self._update_evidence(day)
            logger.error("edge-proof aborted: critical component verification failed")
            return {"day": day, "evidence": self.evidence.to_dict()}

        # Maximise archive growth
        collect = self.platform._stage_collect()
        validate = self.platform._stage_validate()
        repair = self.platform._stage_repair()
        ticks = self.platform._stage_ticks() if self.research.download_ticks else {"skipped": True}

        # Train / reject / keep only better models (daily path)
        train = self.platform.run_daily()

        # Continuous paper trading
        paper = self.platform._stage_paper_all()

        # Persist monitoring snapshot / dashboard for human review
        dashboard = self.platform._stage_dashboard() if self.research.build_dashboard else {}

        day = {
            "started_at": started,
            "finished_at": utc_now_iso(),
            "components": components,
            "collect": collect,
            "validate": validate,
            "repair": repair,
            "ticks": ticks,
            "train": train,
            "paper": paper,
            "dashboard": dashboard,
        }
        self._update_evidence(day)
        logger.info(
            "edge-proof day done claim_allowed=%s edge=%s runs=%s",
            self.evidence.scientific_claim_allowed,
            self.evidence.edge_demonstrated,
            self.evidence.runs_completed,
        )
        return {"day": day, "evidence": self.evidence.to_dict()}

    def run_forever(self, *, max_days: int | None = None, sleep_seconds: float | None = None) -> EdgeEvidence:
        """Resumeable forever loop: run_day → sleep → continue."""
        limit = max_days
        sleep_for = float(
            sleep_seconds
            if sleep_seconds is not None
            else (self.research.sleep_seconds or self.research.cycle_interval_seconds or 86400.0)
        )
        days = 0
        while True:
            self.run_day()
            days += 1
            if limit is not None and days >= int(limit):
                break
            if sleep_for <= 0:
                break
            logger.info("edge-proof sleep %.0fs", sleep_for)
            time.sleep(sleep_for)
        return self.evidence

    def evidence_summary(self) -> Dict[str, Any]:
        return self.evidence.to_dict()

    # ------------------------------------------------------------------
    # Evidence ledger
    # ------------------------------------------------------------------

    def _update_evidence(self, day: Dict[str, Any]) -> None:
        repo = self.platform.research_repo
        archive = self._archive_stats()
        exp_count = self._count("SELECT COUNT(*) AS c FROM research_experiments")
        model_count = self._count("SELECT COUNT(*) AS c FROM research_models")
        champ_count = self._count("SELECT COUNT(*) AS c FROM research_models WHERE is_champion=1")
        pred_count = self._count("SELECT COUNT(*) AS c FROM research_predictions")
        resolved = self._count("SELECT COUNT(*) AS c FROM research_predictions WHERE resolved=1")
        paper_closed = self._count("SELECT COUNT(*) AS c FROM research_paper_trades WHERE status='closed'")

        paper_metrics = {}
        for symbol in self.platform.config.symbols:
            paper_metrics[symbol] = repo.paper_trade_stats(symbol, self.platform.config.primary_timeframe)

        # Honest claim rules: real archive + enough resolved paper outcomes
        bars = int(archive.get("candles") or 0)
        ticks = int(archive.get("ticks") or 0)
        claim_allowed = bars >= 50_000 and resolved >= 50 and not self.research.allow_synthetic
        edge = False
        reason = "insufficient_evidence"
        if claim_allowed:
            # Edge only if production-style paper stats clear thresholds for ≥1 symbol
            for symbol, stats in paper_metrics.items():
                preds = stats.get("predictions") or {}
                accuracy = float(preds.get("accuracy") or 0.0)
                gate = self.platform.production_gate.evaluate(
                    symbol,
                    self.platform.config.primary_timeframe,
                    enable_live_if_passed=False,
                )
                if gate.passed and accuracy >= float(self.research.min_val_score):
                    edge = True
                    reason = f"paper_thresholds_met:{symbol}"
                    break
            if not edge:
                reason = "archive_and_sample_ok_but_no_persistent_edge"
        else:
            missing = []
            if bars < 50_000:
                missing.append(f"candles={bars}<50000")
            if resolved < 50:
                missing.append(f"resolved_preds={resolved}<50")
            if self.research.allow_synthetic:
                missing.append("synthetic_not_allowed_for_claims")
            reason = "insufficient_evidence:" + ",".join(missing)

        components = day.get("components") or {}
        # Abort days never allow scientific claims
        if day.get("aborted"):
            claim_allowed = False
            edge = False
            reason = str(day.get("reason") or "aborted")

        self.evidence.updated_at = utc_now_iso()
        self.evidence.runs_completed += 1
        self.evidence.scientific_claim_allowed = claim_allowed
        self.evidence.edge_demonstrated = edge
        self.evidence.reason = reason
        self.evidence.components_ok = bool(components.get("critical_ok"))
        self.evidence.components = {
            "ok": components.get("ok"),
            "critical_ok": components.get("critical_ok"),
            "passed": components.get("passed"),
            "failed": components.get("failed"),
            "finished_at": components.get("finished_at"),
        }
        self.evidence.archive = archive
        self.evidence.experiments = {
            "experiments": exp_count,
            "models": model_count,
            "champions": champ_count,
            "predictions": pred_count,
            "resolved_predictions": resolved,
            "closed_paper_trades": paper_closed,
        }
        self.evidence.paper = paper_metrics
        self.evidence.champions = {
            f"{s}:{self.platform.config.primary_timeframe}": repo.get_champion_model(
                s, self.platform.config.primary_timeframe
            )
            for s in self.platform.config.symbols
        }
        self.evidence.history.append(
            {
                "finished_at": day.get("finished_at"),
                "components_ok": bool(components.get("critical_ok")),
                "bars_inserted": (day.get("collect") or {}).get("bars_inserted"),
                "ticks_inserted": (day.get("ticks") or {}).get("ticks_inserted"),
                "claim_allowed": claim_allowed,
                "edge": edge,
                "reason": reason,
            }
        )
        # Keep ledger bounded
        self.evidence.history = self.evidence.history[-365:]
        self._save()

    def _archive_stats(self) -> Dict[str, Any]:
        candles = self._count("SELECT COUNT(*) AS c FROM candles")
        ticks = self._count("SELECT COUNT(*) AS c FROM ticks")
        symbols = self._count(
            "SELECT COUNT(DISTINCT COALESCE(canonical_symbol, symbol)) AS c FROM markets"
        )
        # Per-symbol coverage
        rows = self.platform.research_repo._fetch_all(
            """
            SELECT COALESCE(m.canonical_symbol, m.symbol) AS symbol,
                   c.timeframe,
                   COUNT(*) AS bars,
                   MIN(c.timestamp) AS first_ts,
                   MAX(c.timestamp) AS last_ts
            FROM candles c
            JOIN markets m ON m.market_id = c.market_id
            GROUP BY 1, 2
            ORDER BY 1, 2
            """
        )
        return {
            "candles": candles,
            "ticks": ticks,
            "symbols": symbols,
            "series": rows,
        }

    def _count(self, sql: str) -> int:
        row = self.platform.research_repo._fetch_one(sql)
        if not row:
            return 0
        return int(row.get("c") if isinstance(row, dict) else row[0] or 0)

    def _load(self) -> EdgeEvidence:
        if not self.evidence_path.exists():
            return EdgeEvidence(updated_at=utc_now_iso())
        try:
            data = json.loads(self.evidence_path.read_text(encoding="utf-8"))
            return EdgeEvidence(
                updated_at=str(data.get("updated_at") or ""),
                runs_completed=int(data.get("runs_completed") or 0),
                scientific_claim_allowed=bool(data.get("scientific_claim_allowed")),
                edge_demonstrated=bool(data.get("edge_demonstrated")),
                reason=str(data.get("reason") or ""),
                components_ok=bool(data.get("components_ok")),
                components=dict(data.get("components") or {}),
                archive=dict(data.get("archive") or {}),
                experiments=dict(data.get("experiments") or {}),
                paper=dict(data.get("paper") or {}),
                champions=dict(data.get("champions") or {}),
                history=list(data.get("history") or []),
            )
        except Exception:
            return EdgeEvidence(updated_at=utc_now_iso())

    def _save(self) -> None:
        self.evidence_path.parent.mkdir(parents=True, exist_ok=True)
        self.evidence_path.write_text(
            json.dumps(self.evidence.to_dict(), indent=2, default=str),
            encoding="utf-8",
        )


def create_edge_proof_engine(
    platform: AutonomousResearchPlatform,
    **kwargs: Any,
) -> EdgeProofEngine:
    return EdgeProofEngine(platform, **kwargs)
