"""
ai/utils/validation.py - Input validation helpers

VERSION: 1.0.0
"""

from __future__ import annotations

from typing import Iterable, Sequence
import numpy as np
from numpy.typing import NDArray


class AIValidationError(ValueError):
    """Raised when AI pipeline inputs fail validation."""


def require_columns(rows: Sequence[dict], required: Iterable[str]) -> None:
    required_list = list(required)
    if not rows:
        raise AIValidationError("Empty dataset")
    missing = [col for col in required_list if col not in rows[0]]
    if missing:
        raise AIValidationError(f"Missing required columns: {missing}")


def validate_finite(array: NDArray[np.floating], name: str = "array") -> NDArray[np.floating]:
    arr = np.asarray(array, dtype=float)
    if not np.isfinite(arr).all():
        bad = int(np.size(arr) - np.isfinite(arr).sum())
        raise AIValidationError(f"{name} contains {bad} non-finite values")
    return arr


def ensure_2d(array: NDArray[np.floating]) -> NDArray[np.floating]:
    arr = np.asarray(array, dtype=float)
    if arr.ndim == 1:
        return arr.reshape(-1, 1)
    if arr.ndim != 2:
        raise AIValidationError(f"Expected 1D or 2D array, got shape {arr.shape}")
    return arr


def ensure_1d(array: NDArray[np.floating]) -> NDArray[np.floating]:
    arr = np.asarray(array, dtype=float).reshape(-1)
    return arr


def assert_same_length(*arrays: NDArray[np.floating]) -> None:
    lengths = [len(np.asarray(a)) for a in arrays]
    if len(set(lengths)) != 1:
        raise AIValidationError(f"Array length mismatch: {lengths}")
