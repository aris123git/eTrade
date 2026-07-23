"""
ai/datasets/walk_forward.py - Walk-forward dataset splitting.

VERSION: 1.0.0
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Iterator, List, Optional, Sequence

import numpy as np
from numpy.typing import NDArray

from ai.config.settings import AIConfig, DatasetConfig
from ai.utils.validation import AIValidationError


@dataclass(frozen=True)
class WalkForwardFold:
    index: int
    train_start: int
    train_end: int
    val_start: int
    val_end: int
    X_train: NDArray[np.floating]
    y_train: NDArray[np.floating]
    X_val: NDArray[np.floating]
    y_val: NDArray[np.floating]
    train_timestamps: List[datetime]
    val_timestamps: List[datetime]


class WalkForwardDataset:
    """
    Generate successive train/validation folds with an embargo gap.

    The default mode uses expanding training windows followed by fixed-size
    validation windows.
    """

    def __init__(
        self,
        config: AIConfig | DatasetConfig | None = None,
        folds: Optional[int] = None,
        embargo: Optional[int] = None,
        train_size: Optional[int] = None,
        val_size: Optional[int] = None,
        min_train_size: Optional[int] = None,
        expanding: bool = True,
    ) -> None:
        dataset_config = config.datasets if isinstance(config, AIConfig) else config
        self.folds = int(folds if folds is not None else getattr(dataset_config, "walk_forward_folds", 5))
        self.embargo = int(embargo if embargo is not None else getattr(dataset_config, "walk_forward_embargo", 0))
        self.train_size = train_size
        self.val_size = val_size
        self.min_train_size = min_train_size
        self.expanding = bool(expanding)
        if self.folds <= 0:
            raise AIValidationError("folds must be > 0")
        if self.embargo < 0:
            raise AIValidationError("embargo must be >= 0")

    def split_indices(self, n_samples: int) -> Iterator[tuple[int, int, int, int]]:
        if n_samples <= self.embargo + 2:
            return

        val_size = self.val_size
        if val_size is None:
            val_size = max(1, n_samples // (self.folds + 2))

        min_train = self.min_train_size
        if min_train is None:
            remaining = n_samples - (val_size * self.folds) - self.embargo
            min_train = max(val_size, remaining)
        if self.train_size is not None:
            min_train = max(min_train, min(self.train_size, n_samples))

        max_possible = n_samples - self.embargo - val_size
        if min_train > max_possible:
            val_size = max(1, (n_samples - self.embargo) // (self.folds + 1))
            max_possible = n_samples - self.embargo - val_size
            min_train = max(1, min(min_train, max_possible))

        produced = 0
        for fold_idx in range(self.folds):
            train_end = min_train + fold_idx * val_size
            val_start = train_end + self.embargo
            val_end = val_start + val_size
            if val_end > n_samples:
                break
            train_start = 0 if self.expanding else max(0, train_end - (self.train_size or min_train))
            if train_end <= train_start or val_end <= val_start:
                continue
            produced += 1
            yield train_start, train_end, val_start, val_end

        if produced == 0 and n_samples > self.embargo + 2:
            train_end = max(1, n_samples - self.embargo - val_size)
            val_start = train_end + self.embargo
            val_end = n_samples
            yield 0, train_end, val_start, val_end

    def split(
        self,
        X: NDArray[np.floating],
        y: NDArray[np.floating],
        timestamps: Optional[Sequence[datetime]] = None,
    ) -> Iterator[WalkForwardFold]:
        if len(X) != len(y):
            raise AIValidationError(f"X/y length mismatch: {len(X)} != {len(y)}")
        if timestamps is not None and len(timestamps) != len(X):
            raise AIValidationError(f"X/timestamps length mismatch: {len(X)} != {len(timestamps)}")
        ts = list(timestamps or [])

        for idx, (train_start, train_end, val_start, val_end) in enumerate(self.split_indices(len(X))):
            yield WalkForwardFold(
                index=idx,
                train_start=train_start,
                train_end=train_end,
                val_start=val_start,
                val_end=val_end,
                X_train=np.asarray(X[train_start:train_end], dtype=float),
                y_train=np.asarray(y[train_start:train_end], dtype=float),
                X_val=np.asarray(X[val_start:val_end], dtype=float),
                y_val=np.asarray(y[val_start:val_end], dtype=float),
                train_timestamps=ts[train_start:train_end] if ts else [],
                val_timestamps=ts[val_start:val_end] if ts else [],
            )
