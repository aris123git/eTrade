"""
ai/labels/regression.py - Continuous future outcome labels

RESPONSIBILITY:
Create future return, excursion, and realized volatility regression targets.

VERSION: 1.0.0
"""

from __future__ import annotations

import numpy as np

from ai.labels.base import BaseLabeler, LabelResult
from ai.utils.math_ops import safe_div


# ==============================================================================
# BASE REGRESSION HELPERS
# ==============================================================================


class _FutureWindowLabeler(BaseLabeler):
    """Shared helpers for labels that inspect candles t+1 through t+h."""

    def _future_window_valid(self, close: np.ndarray, horizon: int) -> np.ndarray:
        valid = np.zeros(len(close), dtype=bool)
        if len(close) > horizon:
            valid[:-horizon] = np.isfinite(close[:-horizon])
        return valid


# ==============================================================================
# FUTURE RETURN
# ==============================================================================


class FutureReturn(_FutureWindowLabeler):
    """Close-to-close percentage return at the requested horizon."""

    method = "future_return"

    def label(self, candles: object, horizon: int | None = None, name: str | None = None) -> LabelResult:
        horizon = self._resolve_horizon(horizon)
        close = self._column(candles, "close")
        values = self._future_returns(close, horizon)
        valid = self._base_valid(close, horizon) & np.isfinite(values)
        return self._result(values, name or f"{self.method}_{horizon}", self.method, horizon, valid)


# ==============================================================================
# FUTURE HIGH / LOW EXCURSIONS
# ==============================================================================


class FutureHigh(_FutureWindowLabeler):
    """Maximum future high over the horizon, expressed as return from current close."""

    method = "future_high"

    def label(self, candles: object, horizon: int | None = None, name: str | None = None) -> LabelResult:
        horizon = self._resolve_horizon(horizon)
        _, high, _, close = self._ohlc(candles)
        values, valid = self._empty_result_arrays(len(close))

        for i in range(0, max(len(close) - horizon, 0)):
            window = high[i + 1 : i + horizon + 1]
            if np.isfinite(close[i]) and np.isfinite(window).all():
                values[i] = safe_div(np.max(window) - close[i], close[i], default=np.nan)
                valid[i] = np.isfinite(values[i])

        return self._result(values, name or f"{self.method}_{horizon}", self.method, horizon, valid)


class FutureLow(_FutureWindowLabeler):
    """Minimum future low over the horizon, expressed as return from current close."""

    method = "future_low"

    def label(self, candles: object, horizon: int | None = None, name: str | None = None) -> LabelResult:
        horizon = self._resolve_horizon(horizon)
        _, _, low, close = self._ohlc(candles)
        values, valid = self._empty_result_arrays(len(close))

        for i in range(0, max(len(close) - horizon, 0)):
            window = low[i + 1 : i + horizon + 1]
            if np.isfinite(close[i]) and np.isfinite(window).all():
                values[i] = safe_div(np.min(window) - close[i], close[i], default=np.nan)
                valid[i] = np.isfinite(values[i])

        return self._result(values, name or f"{self.method}_{horizon}", self.method, horizon, valid)


# ==============================================================================
# FUTURE VOLATILITY
# ==============================================================================


class FutureVolatility(_FutureWindowLabeler):
    """Realized standard deviation of one-step future returns inside the horizon."""

    method = "future_volatility"

    def label(self, candles: object, horizon: int | None = None, name: str | None = None) -> LabelResult:
        horizon = self._resolve_horizon(horizon)
        close = self._column(candles, "close")
        values, valid = self._empty_result_arrays(len(close))

        for i in range(0, max(len(close) - horizon, 0)):
            window = close[i : i + horizon + 1]
            if np.isfinite(window).all():
                step_returns = safe_div(window[1:] - window[:-1], window[:-1], default=np.nan)
                if np.isfinite(step_returns).all():
                    values[i] = float(np.std(step_returns, ddof=0))
                    valid[i] = True

        return self._result(values, name or f"{self.method}_{horizon}", self.method, horizon, valid)
