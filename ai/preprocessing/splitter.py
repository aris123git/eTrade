"""
ai/preprocessing/splitter.py - Time-series dataset splits

RESPONSIBILITY:
Create chronological train/validation/test and walk-forward splits with embargo.

VERSION: 1.0.0
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

import numpy as np
from numpy.typing import NDArray

from ai.config.settings import AIConfig
from ai.utils.validation import AIValidationError


# ==============================================================================
# SPLIT CONTRACT
# ==============================================================================


@dataclass(frozen=True)
class SplitIndices:
    """Chronological split indices for train, validation, and test partitions."""

    train: NDArray[np.integer]
    val: NDArray[np.integer]
    test: NDArray[np.integer]
    embargo: int = 0


# ==============================================================================
# SPLITTER
# ==============================================================================


@dataclass
class TimeSeriesSplitter:
    """Time-series aware splitter that never shuffles row order."""

    config: AIConfig = field(default_factory=AIConfig)

    def train_val_test_indices(self, n_rows: int) -> SplitIndices:
        """Return chronological train/validation/test indices with embargo gaps removed."""
        self._validate_n_rows(n_rows)
        train_ratio = float(self.config.datasets.train_ratio)
        val_ratio = float(self.config.datasets.val_ratio)
        test_ratio = float(self.config.datasets.test_ratio)
        if min(train_ratio, val_ratio, test_ratio) < 0.0:
            raise AIValidationError("split ratios must be non-negative")
        ratio_total = train_ratio + val_ratio + test_ratio
        if ratio_total <= 0.0:
            raise AIValidationError("at least one split ratio must be positive")

        normalized_train = train_ratio / ratio_total
        normalized_val = val_ratio / ratio_total
        train_end = int(n_rows * normalized_train)
        val_end = int(n_rows * (normalized_train + normalized_val))
        embargo = max(int(self.config.datasets.walk_forward_embargo), 0)

        train = np.arange(0, train_end, dtype=int)
        val = np.arange(min(train_end + embargo, n_rows), val_end, dtype=int)
        test = np.arange(min(val_end + embargo, n_rows), n_rows, dtype=int)
        return SplitIndices(train=train, val=val, test=test, embargo=embargo)

    def split(
        self,
        features: NDArray[np.floating],
        target: NDArray[np.floating] | None = None,
    ) -> tuple[NDArray[np.floating], NDArray[np.floating], NDArray[np.floating]] | tuple[
        tuple[NDArray[np.floating], NDArray[np.floating]],
        tuple[NDArray[np.floating], NDArray[np.floating]],
        tuple[NDArray[np.floating], NDArray[np.floating]],
    ]:
        """Split feature arrays, and optionally target arrays, by chronological indices."""
        x = np.asarray(features)
        indices = self.train_val_test_indices(len(x))
        if target is None:
            return x[indices.train], x[indices.val], x[indices.test]

        y = np.asarray(target)
        if len(y) != len(x):
            raise AIValidationError(f"target length {len(y)} does not match feature rows {len(x)}")
        return (
            (x[indices.train], y[indices.train]),
            (x[indices.val], y[indices.val]),
            (x[indices.test], y[indices.test]),
        )

    def walk_forward_indices(
        self,
        n_rows: int,
        n_folds: int | None = None,
        embargo: int | None = None,
    ) -> list[SplitIndices]:
        """Return expanding-window walk-forward folds with validation and test windows."""
        self._validate_n_rows(n_rows)
        folds = int(n_folds if n_folds is not None else self.config.datasets.walk_forward_folds)
        if folds <= 0:
            raise AIValidationError("walk-forward folds must be > 0")
        gap = max(int(embargo if embargo is not None else self.config.datasets.walk_forward_embargo), 0)
        window = max(n_rows // (folds + 2), 1)
        results: list[SplitIndices] = []

        for fold in range(folds):
            test_end = n_rows - (folds - fold - 1) * window
            test_start = max(test_end - window, 0)
            val_end = max(test_start - gap, 0)
            val_start = max(val_end - window, 0)
            train_end = max(val_start - gap, 0)

            split = SplitIndices(
                train=np.arange(0, train_end, dtype=int),
                val=np.arange(val_start, val_end, dtype=int),
                test=np.arange(test_start, min(test_end, n_rows), dtype=int),
                embargo=gap,
            )
            if len(split.train) > 0 and len(split.val) > 0 and len(split.test) > 0:
                results.append(split)
        return results

    def walk_forward_split(
        self,
        features: NDArray[np.floating],
        target: NDArray[np.floating] | None = None,
        n_folds: int | None = None,
        embargo: int | None = None,
    ) -> Iterable[tuple[SplitIndices, tuple[NDArray[np.floating], NDArray[np.floating], NDArray[np.floating]]]]:
        """Yield walk-forward indices and feature partitions."""
        x = np.asarray(features)
        y = np.asarray(target) if target is not None else None
        if y is not None and len(y) != len(x):
            raise AIValidationError(f"target length {len(y)} does not match feature rows {len(x)}")

        for split in self.walk_forward_indices(len(x), n_folds=n_folds, embargo=embargo):
            x_parts = (x[split.train], x[split.val], x[split.test])
            if y is None:
                yield split, x_parts
            else:
                y_parts = (y[split.train], y[split.val], y[split.test])
                yield split, (x_parts, y_parts)  # type: ignore[misc]

    @staticmethod
    def _validate_n_rows(n_rows: int) -> None:
        if int(n_rows) <= 0:
            raise AIValidationError("n_rows must be > 0")


def create_time_series_splitter(config: AIConfig | None = None) -> TimeSeriesSplitter:
    """Factory for time-series splitters."""
    return TimeSeriesSplitter(config=config or AIConfig())
