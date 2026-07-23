"""
ai/labels/binary.py - Binary direction labels

RESPONSIBILITY:
Create upward/downward classification targets from future close-to-close returns.

VERSION: 1.0.0
"""

from __future__ import annotations

import numpy as np

from ai.labels.base import BaseLabeler, LabelResult


# ==============================================================================
# BINARY DIRECTION
# ==============================================================================


class BinaryDirectionLabeler(BaseLabeler):
    """Label rows as 1 when future return exceeds the configured threshold, else 0."""

    method = "binary_direction"

    def label(self, candles: object, horizon: int | None = None, name: str | None = None) -> LabelResult:
        horizon = self._resolve_horizon(horizon)
        close = self._column(candles, "close")
        values, valid = self._empty_result_arrays(len(close))
        future = self._future_returns(close, horizon)
        threshold = float(self.config.labels.binary_threshold)

        valid = self._base_valid(close, horizon) & np.isfinite(future)
        values[valid] = (future[valid] > threshold).astype(float)

        return self._result(
            values=values,
            name=name or f"{self.method}_{horizon}",
            method=self.method,
            horizon=horizon,
            valid_mask=valid,
            metadata={"threshold": threshold},
        )
