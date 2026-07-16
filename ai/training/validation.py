"""
ai/training/validation.py - Model validation helpers.

RESPONSIBILITY:
Provide cross-validation, time-series split, and walk-forward evaluation for
BaseModel implementations.

VERSION: 1.0.0
"""

from __future__ import annotations

from typing import Callable, Dict, Iterator, List, Sequence
import copy

import numpy as np
from numpy.typing import NDArray

from ai.datasets.walk_forward import WalkForwardDataset
from ai.models.base import BaseModel, ModelTask, flatten_features, flatten_target


# ==============================================================================
# METRICS
# ==============================================================================


MetricFn = Callable[[NDArray[np.floating], NDArray[np.floating]], Dict[str, float]]


def default_metrics(
    y_true: NDArray[np.floating],
    y_pred: NDArray[np.floating],
    task: ModelTask | str = ModelTask.CLASSIFICATION,
) -> Dict[str, float]:
    """Compute lightweight validation metrics without external dependencies."""
    task_value = ModelTask.from_value(task)
    truth = flatten_target(y_true)
    pred = np.asarray(y_pred).reshape(-1)
    if len(truth) == 0:
        return {}
    if task_value == ModelTask.CLASSIFICATION:
        accuracy = float(np.mean(truth == pred))
        labels = np.unique(np.concatenate([truth, pred]))
        f1_values: list[float] = []
        for label in labels:
            tp = float(np.sum((truth == label) & (pred == label)))
            fp = float(np.sum((truth != label) & (pred == label)))
            fn = float(np.sum((truth == label) & (pred != label)))
            precision = tp / (tp + fp) if tp + fp > 0.0 else 0.0
            recall = tp / (tp + fn) if tp + fn > 0.0 else 0.0
            f1 = 2.0 * precision * recall / (precision + recall) if precision + recall > 0.0 else 0.0
            f1_values.append(f1)
        return {"accuracy": accuracy, "f1": float(np.mean(f1_values))}
    error = truth.astype(float) - pred.astype(float)
    mse = float(np.mean(error * error))
    mae = float(np.mean(np.abs(error)))
    denom = float(np.sum((truth.astype(float) - float(np.mean(truth.astype(float)))) ** 2))
    r2 = 1.0 - float(np.sum(error * error)) / denom if denom > 0.0 else 0.0
    return {"mse": mse, "mae": mae, "r2": r2}


# ==============================================================================
# SPLITS
# ==============================================================================


def kfold_indices(n_samples: int, folds: int = 5, shuffle: bool = False, random_seed: int = 42) -> Iterator[tuple[NDArray[np.integer], NDArray[np.integer]]]:
    """Yield train/validation indices for standard k-fold validation."""
    if folds <= 1:
        raise ValueError("folds must be greater than 1")
    indices = np.arange(n_samples)
    if shuffle:
        rng = np.random.default_rng(random_seed)
        rng.shuffle(indices)
    fold_sizes = np.full(folds, n_samples // folds, dtype=int)
    fold_sizes[: n_samples % folds] += 1
    start = 0
    for fold_size in fold_sizes:
        stop = start + int(fold_size)
        val_idx = indices[start:stop]
        train_idx = np.concatenate([indices[:start], indices[stop:]])
        if len(train_idx) and len(val_idx):
            yield train_idx, val_idx
        start = stop


def time_series_split(
    n_samples: int,
    splits: int = 5,
    min_train_size: int | None = None,
    test_size: int | None = None,
    gap: int = 0,
) -> List[tuple[NDArray[np.integer], NDArray[np.integer]]]:
    """Return expanding-window time-series train/validation index splits."""
    if splits <= 0:
        raise ValueError("splits must be positive")
    if n_samples <= 2:
        return []
    test = int(test_size or max(1, n_samples // (splits + 1)))
    min_train = int(min_train_size or max(1, n_samples - test * splits - gap))
    result: list[tuple[NDArray[np.integer], NDArray[np.integer]]] = []
    for idx in range(splits):
        train_end = min_train + idx * test
        val_start = train_end + gap
        val_end = min(n_samples, val_start + test)
        if train_end <= 0 or val_end <= val_start or val_end > n_samples:
            continue
        result.append((np.arange(0, train_end), np.arange(val_start, val_end)))
    return result


# ==============================================================================
# VALIDATION
# ==============================================================================


def cross_val(
    model: BaseModel,
    X: NDArray[np.floating],
    y: NDArray[np.floating],
    folds: int = 5,
    metric_fn: MetricFn | None = None,
    shuffle: bool = False,
    random_seed: int = 42,
) -> List[Dict[str, float]]:
    """Evaluate a model with k-fold validation."""
    x = flatten_features(X)
    target = flatten_target(y)
    scorer = metric_fn or (lambda yt, yp: default_metrics(yt, yp, model.task))
    scores: list[Dict[str, float]] = []
    for train_idx, val_idx in kfold_indices(len(x), folds=folds, shuffle=shuffle, random_seed=random_seed):
        candidate = copy.deepcopy(model)
        candidate.fit(x[train_idx], target[train_idx], X_val=x[val_idx], y_val=target[val_idx])
        scores.append(scorer(target[val_idx], candidate.predict(x[val_idx])))
    return scores


def walk_forward_validation(
    model: BaseModel,
    X: NDArray[np.floating],
    y: NDArray[np.floating],
    folds: int = 5,
    embargo: int = 0,
    metric_fn: MetricFn | None = None,
) -> List[Dict[str, float]]:
    """Evaluate a model with walk-forward splits."""
    splitter = WalkForwardDataset(folds=folds, embargo=embargo)
    scorer = metric_fn or (lambda yt, yp: default_metrics(yt, yp, model.task))
    scores: list[Dict[str, float]] = []
    for fold in splitter.split(flatten_features(X), flatten_target(y)):
        candidate = copy.deepcopy(model)
        candidate.fit(fold.X_train, fold.y_train, X_val=fold.X_val, y_val=fold.y_val)
        scores.append(scorer(fold.y_val, candidate.predict(fold.X_val)))
    return scores


def summarize_scores(scores: Sequence[Dict[str, float]]) -> Dict[str, float]:
    """Aggregate fold metrics by mean."""
    if not scores:
        return {}
    keys = sorted({key for score in scores for key in score})
    return {key: float(np.mean([score[key] for score in scores if key in score])) for key in keys}
