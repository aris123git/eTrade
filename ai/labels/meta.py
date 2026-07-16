"""
ai/labels/meta.py - Secondary model labels

RESPONSIBILITY:
Create meta-labels that mark whether a primary model prediction was profitable.

VERSION: 1.0.0
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ai.labels.base import BaseLabeler, LabelResult


# ==============================================================================
# META LABELS
# ==============================================================================


class MetaLabels(BaseLabeler):
    """
    Label primary predictions as successful (1) or unsuccessful (0).

    Primary predictions can be injected explicitly or supplied on candle rows via
    primary_prediction/prediction and optional primary_probability/confidence.
    """

    method = "meta_labels"

    def label(
        self,
        candles: object,
        horizon: int | None = None,
        name: str | None = None,
        primary_predictions: Any | None = None,
        primary_probabilities: Any | None = None,
    ) -> LabelResult:
        horizon = self._resolve_horizon(horizon)
        close = self._column(candles, "close")
        values, valid = self._empty_result_arrays(len(close))
        future = self._future_returns(close, horizon)

        predictions = (
            np.asarray(primary_predictions, dtype=object).reshape(-1)
            if primary_predictions is not None
            else self._prediction_column(candles)
        )
        probabilities = (
            np.asarray(primary_probabilities, dtype=float).reshape(-1)
            if primary_probabilities is not None
            else self._optional_column(candles, "primary_probability", "primary_confidence", "confidence")
        )

        if predictions is None or len(predictions) != len(close):
            return self._result(
                values=values,
                name=name or f"{self.method}_{horizon}",
                method=self.method,
                horizon=horizon,
                valid_mask=valid,
                metadata={"reason": "primary predictions unavailable"},
            )

        threshold = float(self.config.labels.meta_primary_threshold)
        direction = np.asarray([self._prediction_direction(item) for item in predictions], dtype=float)
        valid = self._base_valid(close, horizon) & np.isfinite(future) & np.isfinite(direction) & (direction != 0.0)
        if probabilities is not None and len(probabilities) == len(close):
            valid &= np.isfinite(probabilities) & (probabilities >= threshold)

        values[valid] = (np.sign(future[valid]) == direction[valid]).astype(float)
        return self._result(
            values=values,
            name=name or f"{self.method}_{horizon}",
            method=self.method,
            horizon=horizon,
            valid_mask=valid,
            metadata={"primary_threshold": threshold},
        )

    @classmethod
    def _prediction_column(cls, candles: object) -> np.ndarray | None:
        for field_name in ("primary_prediction", "prediction", "signal"):
            try:
                if hasattr(candles, "columns") and field_name in getattr(candles, "columns"):
                    return np.asarray(candles[field_name], dtype=object).reshape(-1)
                if isinstance(candles, dict) and field_name in candles:
                    return np.asarray(candles[field_name], dtype=object).reshape(-1)

                rows = list(candles)  # type: ignore[arg-type]
                if rows and isinstance(rows[0], dict) and field_name in rows[0]:
                    return np.asarray([row[field_name] for row in rows], dtype=object)
                if rows and hasattr(rows[0], field_name):
                    return np.asarray([getattr(row, field_name) for row in rows], dtype=object)
            except (TypeError, ValueError):
                continue
        return None

    @staticmethod
    def _prediction_direction(value: object) -> float:
        if value is None:
            return np.nan
        if isinstance(value, str):
            normalized = value.strip().upper()
            if normalized in {"BUY", "LONG", "UP", "1"}:
                return 1.0
            if normalized in {"SELL", "SHORT", "DOWN", "-1"}:
                return -1.0
            if normalized in {"HOLD", "FLAT", "0"}:
                return 0.0
            return np.nan
        try:
            number = float(value)
        except (TypeError, ValueError):
            return np.nan
        if number > 0:
            return 1.0
        if number < 0:
            return -1.0
        return 0.0
