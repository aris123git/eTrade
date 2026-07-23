"""
ai/training/experiments.py - JSON experiment tracking.

RESPONSIBILITY:
Persist training metrics, history, and run metadata under storage.experiments_dir.

VERSION: 1.0.0
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional
import json
import time

from ai.config.settings import AIConfig
from ai.training.metrics_tracker import TrainingHistory


# ==============================================================================
# EXPERIMENT TRACKER
# ==============================================================================


@dataclass
class ExperimentTracker:
    """Small file-backed experiment tracker."""

    config: AIConfig = field(default_factory=AIConfig)
    experiment_name: str | None = None

    def __post_init__(self) -> None:
        self.experiment_name = self.experiment_name or self.config.training.experiment_name
        self.root_dir.mkdir(parents=True, exist_ok=True)

    @property
    def root_dir(self) -> Path:
        """Return this experiment directory."""
        return Path(self.config.storage.root_dir) / self.config.storage.experiments_dir / str(self.experiment_name)

    def run_dir(self, run_id: str | None = None) -> Path:
        """Return a run directory, creating it when needed."""
        name = run_id or time.strftime("%Y%m%d-%H%M%S")
        target = self.root_dir / name
        target.mkdir(parents=True, exist_ok=True)
        return target

    def write_metrics(
        self,
        metrics: Dict[str, float],
        history: TrainingHistory | None = None,
        metadata: Optional[Dict[str, Any]] = None,
        run_id: str | None = None,
    ) -> Dict[str, str]:
        """Write metrics, history, and metadata JSON files."""
        directory = self.run_dir(run_id)
        paths: Dict[str, str] = {}
        metrics_path = directory / "metrics.json"
        with metrics_path.open("w", encoding="utf-8") as handle:
            json.dump(dict(metrics), handle, indent=2, default=str)
        paths["metrics"] = str(metrics_path)
        if history is not None:
            history_path = directory / "history.json"
            history.save(history_path)
            paths["history"] = str(history_path)
        meta_path = directory / "metadata.json"
        payload = dict(metadata or {})
        payload.setdefault("experiment_name", self.experiment_name)
        payload.setdefault("created_at", time.time())
        with meta_path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, default=str)
        paths["metadata"] = str(meta_path)
        return paths

    def read_run(self, run_id: str) -> Dict[str, Any]:
        """Read all JSON files for a run."""
        directory = self.run_dir(run_id)
        result: Dict[str, Any] = {}
        for name in ("metrics", "history", "metadata"):
            path = directory / f"{name}.json"
            if path.exists():
                with path.open("r", encoding="utf-8") as handle:
                    result[name] = json.load(handle)
        return result
