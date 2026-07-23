"""
ai/research/self_improve.py - Drift monitoring and champion replacement.

If performance decreases:
  detect drift → determine causes → retrain → compare → deploy only if superior.
Never replace a model unless the new one is statistically better.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from ai.config.settings import AIConfig
from ai.monitoring.drift import FeatureDriftDetector, PredictionDriftDetector
from ai.research.gate import decide_promotion, extract_metric
from ai.services.pipeline import AIPipeline
from database.repositories.research_repository import ResearchRepository

logger = logging.getLogger(__name__)


@dataclass
class DriftDiagnosis:
    drifted: bool
    score: float
    causes: List[str] = field(default_factory=list)
    details: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "drifted": self.drifted,
            "score": self.score,
            "causes": list(self.causes),
            "details": self.details,
        }


@dataclass
class SelfImproveResult:
    action: str
    drift: DriftDiagnosis
    promoted: bool = False
    challenger_metrics: Dict[str, Any] = field(default_factory=dict)
    champion_metrics: Dict[str, Any] = field(default_factory=dict)
    reason: str = ""
    model_id: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "action": self.action,
            "drift": self.drift.to_dict(),
            "promoted": self.promoted,
            "challenger_metrics": self.challenger_metrics,
            "champion_metrics": self.champion_metrics,
            "reason": self.reason,
            "model_id": self.model_id,
        }


class SelfImprovementController:
    """Monitor production/paper models and retrain only when justified."""

    def __init__(
        self,
        config: AIConfig,
        pipeline: AIPipeline,
        research_repo: ResearchRepository,
        *,
        primary_metric: str = "test_f1",
        min_improvement: float = 0.005,
    ):
        self.config = config
        self.pipeline = pipeline
        self.repo = research_repo
        self.primary_metric = primary_metric
        self.min_improvement = min_improvement
        self.feature_drift = FeatureDriftDetector(
            threshold=float(config.monitoring.drift_threshold),
        )
        self.pred_drift = PredictionDriftDetector(
            threshold=float(config.monitoring.drift_threshold),
        )

    def diagnose(
        self,
        candles: Sequence[Dict[str, Any]],
        *,
        reference_fraction: float = 0.5,
    ) -> DriftDiagnosis:
        if len(candles) < 120:
            return DriftDiagnosis(False, 0.0, causes=["insufficient_history"])

        frame = self.pipeline.build_features(list(candles))
        matrix = np.asarray(frame.matrix, dtype=float)
        if len(matrix) < 60:
            return DriftDiagnosis(False, 0.0, causes=["insufficient_features"])

        split = max(20, int(len(matrix) * reference_fraction))
        ref, cur = matrix[:split], matrix[split:]
        feat = self.feature_drift.detect(ref, cur, feature_names=frame.feature_names)

        causes: List[str] = []
        score = float(feat.score)
        if feat.drifted:
            causes.append("feature_distribution_shift")

        # Prediction drift if model available
        if self.pipeline.model is not None and len(cur) > 10:
            try:
                ref_pred = np.asarray(self.pipeline.model.predict(ref[-min(len(ref), len(cur)):])).reshape(-1)
                cur_pred = np.asarray(self.pipeline.model.predict(cur)).reshape(-1)
                n = min(len(ref_pred), len(cur_pred))
                pred = self.pred_drift.detect(ref_pred[:n], cur_pred[:n])
                score = max(score, float(pred.score))
                if pred.drifted:
                    causes.append("prediction_distribution_shift")
            except Exception as exc:
                causes.append(f"prediction_drift_error:{exc.__class__.__name__}")

        # Paper performance degradation
        for symbol in self.config.symbols:
            stats = self.repo.paper_trade_stats(symbol, self.config.primary_timeframe)
            preds = stats.get("predictions") or {}
            accuracy = preds.get("accuracy")
            if accuracy is not None and float(accuracy) < 0.45:
                causes.append(f"paper_accuracy_low:{symbol}:{float(accuracy):.3f}")

        drifted = bool(feat.drifted) or any(c.startswith("paper_accuracy") for c in causes) or any(
            c.startswith("prediction_distribution") for c in causes
        )
        return DriftDiagnosis(
            drifted=drifted,
            score=score,
            causes=causes or ["stable"],
            details={"feature_drift": feat.details if hasattr(feat, "details") else {}},
        )

    def maybe_improve(
        self,
        *,
        symbol: str,
        candles: Sequence[Dict[str, Any]],
        force_retrain: bool = False,
    ) -> SelfImproveResult:
        drift = self.diagnose(candles)
        champion = self.repo.get_champion_model(symbol, self.config.primary_timeframe)
        champion_metrics = {}
        if champion and champion.get("metrics"):
            import json

            raw = champion["metrics"]
            champion_metrics = json.loads(raw) if isinstance(raw, str) else dict(raw)

        if not force_retrain and not drift.drifted and champion is not None:
            return SelfImproveResult(
                action="hold",
                drift=drift,
                promoted=False,
                champion_metrics=champion_metrics,
                reason="no_drift_keep_champion",
                model_id=champion.get("model_id"),
            )

        # Retrain challenger on current candles
        run = self.pipeline.run_training(
            candles=list(candles),
            register=True,
            auto_download=False,
            model_name=f"{symbol}_{self.config.model.model_type}_challenger",
        )
        challenger_metrics = {**run.train.metrics, **run.evaluation}
        decision = decide_promotion(
            challenger_metrics=challenger_metrics,
            champion_metrics=champion_metrics or None,
            metric_name=self.primary_metric,
            min_improvement=self.min_improvement,
        )

        if not decision.accepted:
            model_row = self.repo.record_model(
                run.registration.meta.get("experiment_id") if run.registration else None,
                name=f"{symbol}_{self.config.model.model_type}",
                model_type=self.config.model.model_type,
                symbol=symbol,
                timeframe=self.config.primary_timeframe,
                version=getattr(run.registration, "version", None),
                artifact_path=str(getattr(run.registration, "path", "") or ""),
                metrics=challenger_metrics,
                hyperparameters=self.pipeline.model.get_params() if self.pipeline.model else {},
                is_champion=False,
                status="rejected",
                metadata={"reason": decision.reason, "drift": drift.to_dict()},
            )
            return SelfImproveResult(
                action="retrain_rejected",
                drift=drift,
                promoted=False,
                challenger_metrics=challenger_metrics,
                champion_metrics=champion_metrics,
                reason=decision.reason,
                model_id=model_row.get("model_id"),
            )

        model_row = self.repo.record_model(
            None,
            name=f"{symbol}_{self.config.model.model_type}",
            model_type=self.config.model.model_type,
            symbol=symbol,
            timeframe=self.config.primary_timeframe,
            version=getattr(run.registration, "version", None),
            artifact_path=str(getattr(run.registration, "path", "") or ""),
            metrics=challenger_metrics,
            hyperparameters=self.pipeline.model.get_params() if self.pipeline.model else {},
            is_champion=True,
            is_deployed=True,
            status="champion",
            metadata={"reason": decision.reason, "drift": drift.to_dict()},
        )
        self.repo.set_champion(int(model_row["model_id"]), symbol, self.config.primary_timeframe)
        self.repo.record_deployment(
            model_id=int(model_row["model_id"]),
            symbol=symbol,
            timeframe=self.config.primary_timeframe,
            environment="paper",
            status="active",
            previous_model_id=champion.get("model_id") if champion else None,
            reason=decision.reason,
            metrics=challenger_metrics,
        )
        return SelfImproveResult(
            action="deployed",
            drift=drift,
            promoted=True,
            challenger_metrics=challenger_metrics,
            champion_metrics=champion_metrics,
            reason=decision.reason,
            model_id=int(model_row["model_id"]),
        )
