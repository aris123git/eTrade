"""
ai/training/scheduler.py - Weekly auto-retraining with deploy / rollback.

Compares challenger models against the champion and only deploys when
performance improves. Rolls back if a deployed model later degrades.
"""

from __future__ import annotations

import json
import logging
import threading
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from ai.config.settings import AIConfig
from ai.models.trainer import ModelTrainer, create_model_trainer
from ai.research.gate import decide_promotion, extract_metric

logger = logging.getLogger(__name__)


@dataclass
class RetrainCycleResult:
    """Outcome of one scheduled retrain cycle."""

    symbol: str
    model_type: str
    action: str  # skipped | trained | deployed | rolled_back | kept_champion
    promoted: bool
    champion_metrics: Dict[str, Any] = field(default_factory=dict)
    challenger_metrics: Dict[str, Any] = field(default_factory=dict)
    reason: str = ""
    champion_version: str | None = None
    challenger_version: str | None = None
    trained_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> Dict[str, Any]:
        return {
            "symbol": self.symbol,
            "model_type": self.model_type,
            "action": self.action,
            "promoted": self.promoted,
            "champion_metrics": self.champion_metrics,
            "challenger_metrics": self.challenger_metrics,
            "reason": self.reason,
            "champion_version": self.champion_version,
            "challenger_version": self.challenger_version,
            "trained_at": self.trained_at,
        }


@dataclass
class TrainingScheduler:
    """
    Schedule periodic model retraining (default: weekly).

    Can run once via ``run_once`` (preferred in tests) or via a background
    timer with graceful shutdown.
    """

    config: AIConfig = field(default_factory=AIConfig)
    trainer: ModelTrainer | None = None
    candle_repository: Any = None
    interval: timedelta = field(default_factory=lambda: timedelta(days=7))
    primary_metric: str = "test_f1"
    min_improvement: float = 0.005
    model_types: Sequence[str] = field(default_factory=lambda: ("random_forest",))
    state_path: Path | None = None
    _stop: threading.Event = field(default_factory=threading.Event, repr=False)
    _thread: threading.Thread | None = field(default=None, repr=False)
    history: List[RetrainCycleResult] = field(default_factory=list)
    # Deployed champion state: model_name -> {version, metrics, previous_version, previous_metrics}
    champions: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.trainer = self.trainer or create_model_trainer(
            self.config,
            candle_repository=self.candle_repository,
        )
        root = Path(self.config.storage.root_dir)
        self.state_path = self.state_path or (root / "training_scheduler_state.json")
        self._load_state()
        logger.info(
            "TrainingScheduler ready interval=%s metric=%s models=%s",
            self.interval,
            self.primary_metric,
            list(self.model_types),
        )

    def request_shutdown(self) -> None:
        self._stop.set()
        logger.warning("TrainingScheduler shutdown requested")

    def start_background(self) -> None:
        """Start weekly background loop (daemon). Prefer run_once in tests."""

        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, name="ai-training-scheduler", daemon=True)
        self._thread.start()
        logger.info("TrainingScheduler background thread started")

    def _loop(self) -> None:
        while not self._stop.is_set():
            try:
                # Symbols must be provided via run_once in autonomous mode;
                # background loop only ticks if champions exist.
                for model_name in list(self.champions):
                    symbol = str(self.champions[model_name].get("symbol") or model_name.split("_")[0])
                    model_type = str(self.champions[model_name].get("model_type") or "random_forest")
                    self.run_once(symbol=symbol, model_type=model_type)
            except Exception:
                logger.exception("scheduled retrain cycle failed")
            self._stop.wait(self.interval.total_seconds())

    def run_once(
        self,
        *,
        symbol: str,
        model_type: str = "random_forest",
        timeframe: str | None = None,
        limit: int = 5000,
        candles: Sequence[Any] | None = None,
        live_metrics: Dict[str, Any] | None = None,
    ) -> RetrainCycleResult:
        """
        Train a challenger, compare to champion, deploy or keep.

        If ``live_metrics`` show the current champion degraded vs its stored
        metrics, roll back to the previous version when available.
        """

        assert self.trainer is not None
        model_name = f"{symbol.upper()}_{model_type}"
        champion = dict(self.champions.get(model_name) or {})

        # Rollback path: live performance degraded vs stored champion metrics.
        if live_metrics and champion:
            gate = decide_promotion(
                challenger_metrics=live_metrics,
                champion_metrics=champion.get("metrics") or {},
                metric_name=self.primary_metric,
                min_improvement=0.0,
            )
            live_score = extract_metric(live_metrics, self.primary_metric)
            champ_score = extract_metric(champion.get("metrics") or {}, self.primary_metric)
            if (
                live_score is not None
                and champ_score is not None
                and live_score + self.min_improvement < champ_score
                and champion.get("previous_version")
            ):
                result = RetrainCycleResult(
                    symbol=symbol.upper(),
                    model_type=model_type,
                    action="rolled_back",
                    promoted=False,
                    champion_metrics=champion.get("metrics") or {},
                    challenger_metrics=live_metrics,
                    reason=f"live metric {self.primary_metric} degraded; restoring previous",
                    champion_version=str(champion.get("previous_version")),
                    challenger_version=str(champion.get("version")),
                )
                self.champions[model_name] = {
                    "symbol": symbol.upper(),
                    "model_type": model_type,
                    "version": champion.get("previous_version"),
                    "metrics": champion.get("previous_metrics") or {},
                    "previous_version": None,
                    "previous_metrics": {},
                }
                self.history.append(result)
                self._save_state()
                logger.error("ROLLBACK %s → %s", model_name, champion.get("previous_version"))
                return result

        train_result = self.trainer.train(
            symbol=symbol,
            model_type=model_type,
            timeframe=timeframe,
            candles=candles,
            limit=limit,
            register=True,
        )
        if "error" in train_result:
            result = RetrainCycleResult(
                symbol=symbol.upper(),
                model_type=model_type,
                action="skipped",
                promoted=False,
                reason=str(train_result["error"]),
            )
            self.history.append(result)
            return result

        challenger_metrics = dict(train_result.get("metrics") or {})
        registration = train_result.get("registered") or {}
        challenger_version = registration.get("version")

        decision = decide_promotion(
            challenger_metrics=challenger_metrics,
            champion_metrics=champion.get("metrics") if champion else None,
            metric_name=self.primary_metric,
            min_improvement=self.min_improvement,
        )

        if decision.accepted:
            self.champions[model_name] = {
                "symbol": symbol.upper(),
                "model_type": model_type,
                "version": challenger_version,
                "metrics": challenger_metrics,
                "previous_version": champion.get("version"),
                "previous_metrics": champion.get("metrics") or {},
                "path": registration.get("path"),
            }
            result = RetrainCycleResult(
                symbol=symbol.upper(),
                model_type=model_type,
                action="deployed" if champion else "trained",
                promoted=True,
                champion_metrics=champion.get("metrics") or {},
                challenger_metrics=challenger_metrics,
                reason=decision.reason,
                champion_version=str(champion.get("version")) if champion else None,
                challenger_version=str(challenger_version) if challenger_version else None,
            )
            logger.info("DEPLOY %s version=%s reason=%s", model_name, challenger_version, decision.reason)
        else:
            result = RetrainCycleResult(
                symbol=symbol.upper(),
                model_type=model_type,
                action="kept_champion",
                promoted=False,
                champion_metrics=champion.get("metrics") or {},
                challenger_metrics=challenger_metrics,
                reason=decision.reason,
                champion_version=str(champion.get("version")) if champion else None,
                challenger_version=str(challenger_version) if challenger_version else None,
            )
            logger.info("KEEP champion %s — %s", model_name, decision.reason)

        self.history.append(result)
        self._save_state()
        return result

    def run_weekly(
        self,
        symbols: Sequence[str],
        *,
        model_types: Sequence[str] | None = None,
        limit: int = 5000,
    ) -> List[RetrainCycleResult]:
        """Run one weekly retrain cycle across symbols × model types."""

        types = list(model_types or self.model_types)
        results: List[RetrainCycleResult] = []
        for symbol in symbols:
            for model_type in types:
                results.append(self.run_once(symbol=symbol, model_type=model_type, limit=limit))
        return results

    def _save_state(self) -> None:
        assert self.state_path is not None
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "champions": self.champions,
            "history": [h.to_dict() for h in self.history[-50:]],
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }
        self.state_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def _load_state(self) -> None:
        assert self.state_path is not None
        if not self.state_path.exists():
            return
        try:
            payload = json.loads(self.state_path.read_text(encoding="utf-8"))
            self.champions = dict(payload.get("champions") or {})
            logger.info("loaded scheduler state champions=%s", list(self.champions))
        except Exception:
            logger.exception("failed to load scheduler state")


def create_training_scheduler(
    config: AIConfig | None = None,
    candle_repository: Any = None,
    **kwargs: Any,
) -> TrainingScheduler:
    return TrainingScheduler(
        config=config or AIConfig(),
        candle_repository=candle_repository,
        **kwargs,
    )
