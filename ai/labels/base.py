"""
ai/labels/base.py - Shared contracts for label generation

RESPONSIBILITY:
Define stable label result and labeler interfaces used by supervised AI datasets.

VERSION: 1.0.0
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, Mapping, Sequence

import numpy as np
from numpy.typing import NDArray

from ai.config.settings import AIConfig
from ai.utils.validation import AIValidationError, assert_same_length


# ==============================================================================
# RESULT CONTRACT
# ==============================================================================


@dataclass(frozen=True)
class LabelResult:
    """Container returned by every production labeler."""

    values: NDArray[np.floating]
    name: str
    method: str
    horizon: int
    valid_mask: NDArray[np.bool_]
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        values = np.asarray(self.values, dtype=float)
        valid_mask = np.asarray(self.valid_mask, dtype=bool)
        assert_same_length(values, valid_mask)
        object.__setattr__(self, "values", values)
        object.__setattr__(self, "valid_mask", valid_mask)


# ==============================================================================
# BASE LABELER
# ==============================================================================


class BaseLabeler(ABC):
    """Abstract base class for deterministic, horizon-aware labelers."""

    method: str

    def __init__(self, config: AIConfig | None = None):
        self.config = config or AIConfig()

    @abstractmethod
    def label(self, candles: Any, horizon: int | None = None, name: str | None = None) -> LabelResult:
        """Generate labels aligned to the input candle rows."""

    def _resolve_horizon(self, horizon: int | None) -> int:
        resolved = int(horizon if horizon is not None else self.config.labels.horizon)
        if resolved <= 0:
            raise AIValidationError("label horizon must be > 0")
        return resolved

    def _result(
        self,
        values: NDArray[np.floating],
        name: str,
        method: str,
        horizon: int,
        valid_mask: NDArray[np.bool_],
        metadata: Dict[str, Any] | None = None,
    ) -> LabelResult:
        return LabelResult(
            values=np.asarray(values, dtype=float),
            name=name,
            method=method,
            horizon=horizon,
            valid_mask=np.asarray(valid_mask, dtype=bool),
            metadata=metadata or {},
        )

    @staticmethod
    def _empty_result_arrays(length: int) -> tuple[NDArray[np.floating], NDArray[np.bool_]]:
        return np.full(length, np.nan, dtype=float), np.zeros(length, dtype=bool)

    @classmethod
    def _ohlc(cls, candles: Any) -> tuple[NDArray[np.floating], NDArray[np.floating], NDArray[np.floating], NDArray[np.floating]]:
        return (
            cls._column(candles, "open"),
            cls._column(candles, "high"),
            cls._column(candles, "low"),
            cls._column(candles, "close"),
        )

    @staticmethod
    def _column(candles: Any, field_name: str) -> NDArray[np.floating]:
        """Extract a numeric candle column from mappings, objects, dict-of-arrays, or dataframes."""
        if isinstance(candles, Mapping):
            if field_name not in candles:
                raise AIValidationError(f"candles missing '{field_name}'")
            return np.asarray(candles[field_name], dtype=float).reshape(-1)

        if hasattr(candles, "columns") and field_name in getattr(candles, "columns"):
            return np.asarray(candles[field_name], dtype=float).reshape(-1)

        if isinstance(candles, np.ndarray):
            if candles.dtype.names and field_name in candles.dtype.names:
                return np.asarray(candles[field_name], dtype=float).reshape(-1)
            raise AIValidationError("numpy candle arrays must be structured with OHLC fields")

        rows = list(candles) if not isinstance(candles, Sequence) else candles
        if len(rows) == 0:
            return np.asarray([], dtype=float)

        values: list[float] = []
        for row in rows:
            if isinstance(row, Mapping):
                if field_name not in row:
                    raise AIValidationError(f"candle row missing '{field_name}'")
                raw = row[field_name]
            elif hasattr(row, field_name):
                raw = getattr(row, field_name)
            else:
                raise AIValidationError(f"candle row missing '{field_name}'")
            values.append(float(raw))
        return np.asarray(values, dtype=float)

    @classmethod
    def _optional_column(cls, candles: Any, *field_names: str) -> NDArray[np.floating] | None:
        for field_name in field_names:
            try:
                return cls._column(candles, field_name)
            except (AIValidationError, KeyError, TypeError, ValueError):
                continue
        return None

    @staticmethod
    def _future_returns(close: NDArray[np.floating], horizon: int) -> NDArray[np.floating]:
        values = np.full(len(close), np.nan, dtype=float)
        if len(close) <= horizon:
            return values
        base = close[:-horizon]
        future = close[horizon:]
        with np.errstate(divide="ignore", invalid="ignore"):
            values[:-horizon] = (future - base) / base
        return values

    @staticmethod
    def _base_valid(close: NDArray[np.floating], horizon: int) -> NDArray[np.bool_]:
        valid = np.zeros(len(close), dtype=bool)
        if len(close) > horizon:
            valid[:-horizon] = np.isfinite(close[:-horizon]) & np.isfinite(close[horizon:])
        return valid
