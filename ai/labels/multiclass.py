"""
ai/labels/multiclass.py - Multi-class direction labels

RESPONSIBILITY:
Create down/flat/up targets from future close-to-close returns.

VERSION: 1.0.0
"""

from __future__ import annotations

import numpy as np

from ai.labels.base import BaseLabeler, LabelResult


# ==============================================================================
# MULTI-CLASS DIRECTION
# ==============================================================================


class MultiClassDirectionLabeler(BaseLabeler):
    """Label rows as 0=down, 1=flat, 2=up using configured return thresholds."""

    method = "multiclass_direction"

    def label(self, candles: object, horizon: int | None = None, name: str | None = None) -> LabelResult:
        horizon = self._resolve_horizon(horizon)
        close = self._column(candles, "close")
        values, valid = self._empty_result_arrays(len(close))
        future = self._future_returns(close, horizon)

        thresholds = list(self.config.labels.multiclass_thresholds)
        if len(thresholds) != 2:
            raise ValueError("multiclass_thresholds must contain [down_threshold, up_threshold]")
        down_threshold, up_threshold = float(thresholds[0]), float(thresholds[1])
        if down_threshold > up_threshold:
            raise ValueError("multiclass down threshold must be <= up threshold")

        valid = self._base_valid(close, horizon) & np.isfinite(future)
        values[valid] = 1.0
        values[valid & (future < down_threshold)] = 0.0
        values[valid & (future > up_threshold)] = 2.0

        return self._result(
            values=values,
            name=name or f"{self.method}_{horizon}",
            method=self.method,
            horizon=horizon,
            valid_mask=valid,
            metadata={"classes": {"down": 0, "flat": 1, "up": 2}, "thresholds": thresholds},
        )
