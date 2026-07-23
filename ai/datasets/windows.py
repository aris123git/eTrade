"""
ai/datasets/windows.py - Sliding-window and sequence generation utilities.

VERSION: 1.0.0
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Iterator, List, Optional, Sequence, Tuple

import numpy as np
from numpy.typing import NDArray

from ai.utils.validation import AIValidationError


@dataclass(frozen=True)
class WindowSpec:
    window_size: int
    stride: int = 1
    label_offset: int = 0
    drop_incomplete: bool = True

    def validate(self) -> None:
        if self.window_size <= 0:
            raise AIValidationError("window_size must be > 0")
        if self.stride <= 0:
            raise AIValidationError("stride must be > 0")
        if self.label_offset < 0:
            raise AIValidationError("label_offset must be >= 0")


@dataclass(frozen=True)
class SequenceData:
    X: NDArray[np.floating]
    y: Optional[NDArray[np.floating]]
    timestamps: List[datetime]
    source_indices: NDArray[np.integer]


def sliding_window_indices(
    n_rows: int,
    window_size: int,
    stride: int = 1,
    label_offset: int = 0,
    drop_incomplete: bool = True,
) -> NDArray[np.integer]:
    """
    Return [start, stop, target_index] triples for valid windows.

    stop is exclusive. target_index points at the label timestamp and defaults
    to the last row of each window.
    """

    spec = WindowSpec(window_size, stride, label_offset, drop_incomplete)
    spec.validate()
    if n_rows < window_size:
        if drop_incomplete:
            return np.empty((0, 3), dtype=int)
        return np.array([[0, n_rows, min(n_rows - 1 + label_offset, n_rows - 1)]], dtype=int)

    rows: List[Tuple[int, int, int]] = []
    last_start = n_rows - window_size
    for start in range(0, last_start + 1, stride):
        stop = start + window_size
        target = stop - 1 + label_offset
        if target >= n_rows:
            if drop_incomplete:
                break
            target = n_rows - 1
        rows.append((start, stop, target))
    return np.asarray(rows, dtype=int) if rows else np.empty((0, 3), dtype=int)


def iter_sliding_windows(
    values: NDArray[np.floating],
    window_size: int,
    stride: int = 1,
    drop_incomplete: bool = True,
) -> Iterator[NDArray[np.floating]]:
    arr = np.asarray(values, dtype=float)
    for start, stop, _ in sliding_window_indices(
        len(arr),
        window_size=window_size,
        stride=stride,
        drop_incomplete=drop_incomplete,
    ):
        yield arr[int(start) : int(stop)]


def make_sliding_windows(
    values: NDArray[np.floating],
    window_size: int,
    stride: int = 1,
    drop_incomplete: bool = True,
) -> NDArray[np.floating]:
    arr = np.asarray(values, dtype=float)
    indices = sliding_window_indices(
        len(arr),
        window_size=window_size,
        stride=stride,
        drop_incomplete=drop_incomplete,
    )
    if len(indices) == 0:
        return np.empty((0, window_size, *arr.shape[1:]), dtype=float)
    return np.stack([arr[int(start) : int(stop)] for start, stop, _ in indices]).astype(float, copy=False)


def generate_sequences(
    features: NDArray[np.floating],
    labels: Optional[NDArray[np.floating]] = None,
    timestamps: Optional[Sequence[datetime]] = None,
    sequence_length: int = 1,
    stride: int = 1,
    label_offset: int = 0,
    drop_incomplete: bool = True,
) -> SequenceData:
    """
    Convert aligned row-wise features into 3D sequence samples.

    The label for each sequence is taken from the sequence target index, which
    defaults to the last row in the sequence.
    """

    X = np.asarray(features, dtype=float)
    if X.ndim != 2:
        raise AIValidationError(f"features must be 2D, got shape {X.shape}")
    y_arr = None if labels is None else np.asarray(labels, dtype=float)
    if y_arr is not None and len(y_arr) != len(X):
        raise AIValidationError(f"features/labels length mismatch: {len(X)} != {len(y_arr)}")
    if timestamps is not None and len(timestamps) != len(X):
        raise AIValidationError(f"features/timestamps length mismatch: {len(X)} != {len(timestamps)}")

    indices = sliding_window_indices(
        len(X),
        window_size=sequence_length,
        stride=stride,
        label_offset=label_offset,
        drop_incomplete=drop_incomplete,
    )
    if len(indices) == 0:
        empty_y = None
        if y_arr is not None:
            empty_y = np.empty((0, *y_arr.shape[1:]), dtype=float)
        return SequenceData(
            X=np.empty((0, sequence_length, X.shape[1]), dtype=float),
            y=empty_y,
            timestamps=[],
            source_indices=np.empty((0,), dtype=int),
        )

    seq_X = np.stack([X[int(start) : int(stop)] for start, stop, _ in indices]).astype(float, copy=False)
    target_indices = indices[:, 2].astype(int)
    seq_y = None if y_arr is None else y_arr[target_indices].astype(float, copy=False)
    seq_timestamps = [] if timestamps is None else [timestamps[int(i)] for i in target_indices]
    return SequenceData(X=seq_X, y=seq_y, timestamps=seq_timestamps, source_indices=target_indices)


def apply_stride(
    features: NDArray[np.floating],
    labels: NDArray[np.floating],
    timestamps: Sequence[datetime],
    stride: int,
) -> tuple[NDArray[np.floating], NDArray[np.floating], List[datetime]]:
    if stride <= 0:
        raise AIValidationError("stride must be > 0")
    if stride == 1:
        return features, labels, list(timestamps)
    return features[::stride], labels[::stride], list(timestamps)[::stride]
