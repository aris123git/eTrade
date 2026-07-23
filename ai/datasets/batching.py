"""
ai/datasets/batching.py - Time-series mini-batch iteration.

VERSION: 1.0.0
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from math import ceil
from typing import Any, Dict, Iterator, List, Optional, Sequence

import numpy as np
from numpy.typing import NDArray

from ai.utils.validation import AIValidationError


@dataclass(frozen=True)
class TimeSeriesBatch:
    X: NDArray[np.floating]
    y: NDArray[np.floating]
    timestamps: List[datetime] = field(default_factory=list)
    start: int = 0
    end: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __iter__(self) -> Iterator[NDArray[np.floating]]:
        yield self.X
        yield self.y

    @property
    def size(self) -> int:
        return int(len(self.X))


class BatchIterator:
    """
    Deterministic iterator over time-series batches.

    By default batches preserve chronological order. When shuffle is enabled,
    rows are shuffled within a full pass using a local random generator.
    """

    def __init__(
        self,
        X: NDArray[np.floating],
        y: NDArray[np.floating],
        batch_size: int,
        timestamps: Optional[Sequence[datetime]] = None,
        shuffle: bool = False,
        drop_incomplete: bool = False,
        random_seed: Optional[int] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        if batch_size <= 0:
            raise AIValidationError("batch_size must be > 0")
        if len(X) != len(y):
            raise AIValidationError(f"X/y length mismatch: {len(X)} != {len(y)}")
        if timestamps is not None and len(timestamps) != len(X):
            raise AIValidationError(f"X/timestamps length mismatch: {len(X)} != {len(timestamps)}")

        self.X = np.asarray(X, dtype=float)
        self.y = np.asarray(y, dtype=float)
        self.batch_size = int(batch_size)
        self.timestamps = list(timestamps) if timestamps is not None else []
        self.shuffle = bool(shuffle)
        self.drop_incomplete = bool(drop_incomplete)
        self.random_seed = random_seed
        self.metadata = dict(metadata or {})

    def __len__(self) -> int:
        if self.drop_incomplete:
            return len(self.X) // self.batch_size
        return ceil(len(self.X) / self.batch_size) if len(self.X) else 0

    def __iter__(self) -> Iterator[TimeSeriesBatch]:
        indices = np.arange(len(self.X))
        if self.shuffle:
            rng = np.random.default_rng(self.random_seed)
            rng.shuffle(indices)

        for start in range(0, len(indices), self.batch_size):
            end = min(start + self.batch_size, len(indices))
            if self.drop_incomplete and end - start < self.batch_size:
                break
            batch_idx = indices[start:end]
            batch_timestamps = [self.timestamps[int(i)] for i in batch_idx] if self.timestamps else []
            yield TimeSeriesBatch(
                X=self.X[batch_idx],
                y=self.y[batch_idx],
                timestamps=batch_timestamps,
                start=start,
                end=end,
                metadata=self.metadata,
            )

    def iter_xy(self) -> Iterator[tuple[NDArray[np.floating], NDArray[np.floating]]]:
        for batch in self:
            yield batch.X, batch.y
