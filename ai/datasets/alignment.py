"""
ai/datasets/alignment.py - Multi-timeframe alignment helpers.

VERSION: 1.0.0
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
from numpy.typing import NDArray

from ai.utils.time_ops import align_timestamps
from ai.utils.validation import AIValidationError


@dataclass(frozen=True)
class AlignedFeatureBlock:
    matrix: NDArray[np.floating]
    feature_names: List[str]
    source_indices: NDArray[np.integer]


def _validate_sorted(name: str, timestamps: Sequence[datetime]) -> None:
    if any(timestamps[i] > timestamps[i + 1] for i in range(len(timestamps) - 1)):
        raise AIValidationError(f"{name} timestamps must be sorted ascending")


def asof_indices(
    primary_timestamps: Sequence[datetime],
    secondary_timestamps: Sequence[datetime],
    max_lag: Optional[timedelta] = None,
) -> NDArray[np.integer]:
    """
    For each primary timestamp, locate the latest secondary timestamp <= it.
    Returns -1 when no secondary row is available or max_lag is exceeded.
    """

    primary = list(primary_timestamps)
    secondary = list(secondary_timestamps)
    _validate_sorted("primary", primary)
    _validate_sorted("secondary", secondary)
    raw = np.asarray(align_timestamps(primary, secondary), dtype=int)
    if max_lag is None:
        return raw
    for pos, sec_idx in enumerate(raw):
        if sec_idx < 0:
            continue
        if primary[pos] - secondary[int(sec_idx)] > max_lag:
            raw[pos] = -1
    return raw


def align_timeframe_to_primary(
    primary_timestamps: Sequence[datetime],
    secondary_timestamps: Sequence[datetime],
    secondary_values: NDArray[np.floating],
    fill_value: float = np.nan,
    max_lag: Optional[timedelta] = None,
) -> AlignedFeatureBlock:
    values = np.asarray(secondary_values, dtype=float)
    if len(secondary_timestamps) != len(values):
        raise AIValidationError(
            f"secondary timestamps/value length mismatch: {len(secondary_timestamps)} != {len(values)}"
        )
    if values.ndim == 1:
        values = values.reshape(-1, 1)

    indices = asof_indices(primary_timestamps, secondary_timestamps, max_lag=max_lag)
    aligned = np.full((len(primary_timestamps), values.shape[1]), fill_value, dtype=float)
    valid = indices >= 0
    if valid.any():
        aligned[valid] = values[indices[valid]]
    names = [f"aligned_{i}" for i in range(values.shape[1])]
    return AlignedFeatureBlock(matrix=aligned, feature_names=names, source_indices=indices)


def align_feature_blocks(
    primary_timestamps: Sequence[datetime],
    blocks: Mapping[str, Tuple[Sequence[datetime], NDArray[np.floating], Sequence[str]]],
    fill_value: float = np.nan,
    max_lag: Optional[timedelta] = None,
    prefix_names: bool = True,
) -> AlignedFeatureBlock:
    """
    Align named timeframe feature matrices onto primary timestamps and concatenate.

    blocks maps a timeframe/source name to (timestamps, matrix, feature_names).
    """

    matrices: List[NDArray[np.floating]] = []
    names: List[str] = []
    index_columns: List[NDArray[np.integer]] = []
    for source_name, (timestamps, matrix, feature_names) in blocks.items():
        aligned = align_timeframe_to_primary(
            primary_timestamps,
            timestamps,
            matrix,
            fill_value=fill_value,
            max_lag=max_lag,
        )
        matrices.append(aligned.matrix)
        index_columns.append(aligned.source_indices)
        for feature_name in feature_names:
            names.append(f"{source_name}_{feature_name}" if prefix_names else str(feature_name))

    if not matrices:
        return AlignedFeatureBlock(
            matrix=np.empty((len(primary_timestamps), 0), dtype=float),
            feature_names=[],
            source_indices=np.empty((len(primary_timestamps), 0), dtype=int),
        )
    return AlignedFeatureBlock(
        matrix=np.hstack(matrices).astype(float, copy=False),
        feature_names=names,
        source_indices=np.vstack(index_columns).T.astype(int, copy=False),
    )


def merge_aligned_features(
    primary_matrix: NDArray[np.floating],
    primary_feature_names: Sequence[str],
    aligned_blocks: Mapping[str, Tuple[Sequence[datetime], NDArray[np.floating], Sequence[str]]],
    primary_timestamps: Sequence[datetime],
    fill_value: float = np.nan,
    max_lag: Optional[timedelta] = None,
) -> tuple[NDArray[np.floating], List[str], Dict[str, NDArray[np.integer]]]:
    primary = np.asarray(primary_matrix, dtype=float)
    if primary.ndim == 1:
        primary = primary.reshape(-1, 1)
    if len(primary) != len(primary_timestamps):
        raise AIValidationError(f"primary matrix/timestamps length mismatch: {len(primary)} != {len(primary_timestamps)}")

    aligned = align_feature_blocks(
        primary_timestamps,
        aligned_blocks,
        fill_value=fill_value,
        max_lag=max_lag,
    )
    matrix = np.hstack([primary, aligned.matrix]).astype(float, copy=False)
    names = list(primary_feature_names) + aligned.feature_names
    metadata = {"source_indices": aligned.source_indices}
    return matrix, names, metadata
