"""
ai/training - Production training orchestration.

RESPONSIBILITY:
Expose trainer, validation, checkpointing, hyperparameter search, experiment,
and history utilities.

VERSION: 1.0.0
"""

from ai.training.checkpointing import Checkpoint, latest_checkpoint, load_checkpoint, save_checkpoint
from ai.training.experiments import ExperimentTracker
from ai.training.hyperparam import HyperparameterSearch, SearchResult, grid_search, random_search
from ai.training.metrics_tracker import TrainingHistory
from ai.training.trainer import TrainResult, Trainer
from ai.training.validation import (
    cross_val,
    default_metrics,
    kfold_indices,
    summarize_scores,
    time_series_split,
    walk_forward_validation,
)

# NOTE: TrainingScheduler is imported from ai.training.scheduler directly to
# avoid circular imports with ai.models.trainer / ai.services.pipeline.

__all__ = [
    "Trainer",
    "TrainResult",
    "TrainingHistory",
    "Checkpoint",
    "save_checkpoint",
    "load_checkpoint",
    "latest_checkpoint",
    "cross_val",
    "default_metrics",
    "kfold_indices",
    "time_series_split",
    "walk_forward_validation",
    "summarize_scores",
    "HyperparameterSearch",
    "SearchResult",
    "grid_search",
    "random_search",
    "ExperimentTracker",
]
