"""
ai/training/trainer.py - Generic model trainer.

RESPONSIBILITY:
Coordinate model fitting, validation metrics, early stopping, checkpointing,
device selection, and artifact tracking for DatasetBundle inputs.

VERSION: 1.0.0
"""

from __future__ import annotations

from dataclasses import dataclass, field
from importlib import import_module
from pathlib import Path
from typing import Any, Dict
import time

import numpy as np
from numpy.typing import NDArray

from ai.config.settings import AIConfig
from ai.datasets.schema import DatasetBundle
from ai.models.base import BaseModel, flatten_features, flatten_target
from ai.training.checkpointing import latest_checkpoint, load_checkpoint, save_checkpoint
from ai.training.experiments import ExperimentTracker
from ai.training.metrics_tracker import TrainingHistory
from ai.training.validation import default_metrics


# ==============================================================================
# RESULTS
# ==============================================================================


@dataclass
class TrainResult:
    """Return payload from Trainer.train."""

    model: BaseModel
    metrics: Dict[str, float]
    history: TrainingHistory
    artifact_paths: Dict[str, str] = field(default_factory=dict)


# ==============================================================================
# TRAINER
# ==============================================================================


@dataclass
class Trainer:
    """Generic trainer for BaseModel implementations."""

    config: AIConfig = field(default_factory=AIConfig)

    @staticmethod
    def train(model: BaseModel, bundle: DatasetBundle, config: AIConfig) -> TrainResult:
        """Convenience entry point matching the production training contract."""
        return Trainer(config=config).fit(model, bundle)

    def fit(self, model: BaseModel, bundle: DatasetBundle, config: AIConfig | None = None) -> TrainResult:
        """Fit a model against a DatasetBundle and write training artifacts."""
        active_config = config or self.config
        active_config.ensure_directories()
        model = self._resume_model_if_requested(model, active_config)
        device = self.detect_device(active_config)
        history = TrainingHistory()
        artifact_paths: Dict[str, str] = {}
        start = time.time()

        if hasattr(model, "fit_epoch"):
            model, best_metric = self._fit_epoch_model(model, bundle, active_config, history, artifact_paths, device)
        else:
            model.fit(bundle.X_train, bundle.y_train, X_val=bundle.X_val, y_val=bundle.y_val)
            train_metrics = self._metrics(model, bundle.X_train, bundle.y_train)
            val_metrics = self._metrics(model, bundle.X_val, bundle.y_val) if bundle.n_val else {}
            history.add_epoch(1, train_metrics=train_metrics, val_metrics=val_metrics, metadata={"device": device})
            best_metric = self._select_metric(val_metrics or train_metrics, active_config)
            checkpoint = self._checkpoint_path(active_config, epoch=1)
            save_checkpoint(
                checkpoint,
                model=model,
                meta={
                    "epoch": 1,
                    "metric": best_metric,
                    "device": device,
                    "experiment_name": active_config.training.experiment_name,
                },
            )
            artifact_paths["checkpoint"] = str(checkpoint)

        test_metrics = self._metrics(model, bundle.X_test, bundle.y_test) if bundle.has_test else {}
        final_metrics = dict(history.latest().get("val") if history.latest() else {})
        if not final_metrics and history.latest():
            final_metrics = dict(history.latest().get("train", {}))
        final_metrics.update({f"test_{key}": value for key, value in test_metrics.items()})
        final_metrics["best_metric"] = float(best_metric)
        final_metrics["duration_seconds"] = float(time.time() - start)

        tracker = ExperimentTracker(active_config)
        artifact_paths.update(
            tracker.write_metrics(
                final_metrics,
                history=history,
                metadata={
                    "model": model.__class__.__name__,
                    "device": device,
                    "dataset": bundle.summary(),
                },
            )
        )
        return TrainResult(model=model, metrics=final_metrics, history=history, artifact_paths=artifact_paths)

    def detect_device(self, config: AIConfig | None = None) -> str:
        """Return cuda when requested and available, otherwise cpu."""
        active_config = config or self.config
        requested = str(active_config.training.device).lower()
        if requested == "cpu":
            return "cpu"
        try:
            torch = import_module("torch")
        except ModuleNotFoundError:
            return "cpu"
        if requested in {"auto", "cuda"} and torch.cuda.is_available():
            return "cuda"
        return "cpu"

    def _fit_epoch_model(
        self,
        model: BaseModel,
        bundle: DatasetBundle,
        config: AIConfig,
        history: TrainingHistory,
        artifact_paths: Dict[str, str],
        device: str,
    ) -> tuple[BaseModel, float]:
        best_value = np.inf if config.training.minimize_metric else -np.inf
        best_model = model
        stale_epochs = 0
        epochs = max(1, int(config.training.epochs))
        for epoch in range(1, epochs + 1):
            getattr(model, "fit_epoch")(bundle.X_train, bundle.y_train, epoch=epoch, device=device)
            train_metrics = self._metrics(model, bundle.X_train, bundle.y_train)
            val_metrics = self._metrics(model, bundle.X_val, bundle.y_val) if bundle.n_val else {}
            history.add_epoch(epoch, train_metrics=train_metrics, val_metrics=val_metrics, metadata={"device": device})
            current = self._select_metric(val_metrics or train_metrics, config)
            improved = current < best_value if config.training.minimize_metric else current > best_value
            if improved:
                best_value = current
                best_model = model
                stale_epochs = 0
            else:
                stale_epochs += 1
            if epoch % max(1, int(config.training.checkpoint_every_n)) == 0:
                checkpoint = self._checkpoint_path(config, epoch=epoch)
                save_checkpoint(
                    checkpoint,
                    model=model,
                    meta={"epoch": epoch, "metric": current, "device": device},
                )
                artifact_paths[f"checkpoint_epoch_{epoch}"] = str(checkpoint)
            if stale_epochs >= int(config.training.early_stopping_patience):
                break
        return best_model, float(best_value)

    def _metrics(
        self,
        model: BaseModel,
        X: NDArray[np.floating],
        y: NDArray[np.floating],
    ) -> Dict[str, float]:
        if len(X) == 0 or len(y) == 0:
            return {}
        prediction = model.predict(X)
        return default_metrics(flatten_target(y), prediction, task=model.task)

    def _select_metric(self, metrics: Dict[str, float], config: AIConfig) -> float:
        metric = config.training.validation_metric
        if metric in metrics:
            return float(metrics[metric])
        if metrics:
            return float(next(iter(metrics.values())))
        return float("inf") if config.training.minimize_metric else float("-inf")

    def _checkpoint_root(self, config: AIConfig) -> Path:
        return Path(config.storage.root_dir) / config.storage.checkpoints_dir / config.training.experiment_name

    def _checkpoint_path(self, config: AIConfig, epoch: int) -> Path:
        return self._checkpoint_root(config) / f"epoch_{int(epoch):04d}"

    def _resume_model_if_requested(self, model: BaseModel, config: AIConfig) -> BaseModel:
        if not config.training.resume_from_checkpoint:
            return model
        checkpoint_path = latest_checkpoint(self._checkpoint_root(config))
        if checkpoint_path is None:
            return model
        return load_checkpoint(checkpoint_path).model
