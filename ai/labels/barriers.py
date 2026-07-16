"""
ai/labels/barriers.py - Barrier and risk/reward labels

RESPONSIBILITY:
Create Lopez de Prado style triple-barrier outcomes and SL/TP distance targets.

VERSION: 1.0.0
"""

from __future__ import annotations

from typing import Dict

import numpy as np

from ai.labels.base import BaseLabeler, LabelResult
from ai.utils.math_ops import atr, safe_div


# ==============================================================================
# TRIPLE BARRIER METHOD
# ==============================================================================


class TripleBarrierMethod(BaseLabeler):
    """
    Label first barrier touch over t+1..t+h.

    Values are 1 for take-profit touch first, -1 for stop-loss touch first, and 0
    when only the vertical barrier is reached. If both horizontal barriers are
    inside the same candle, stop-loss wins because intrabar order is unknowable.
    """

    method = "triple_barrier"

    def label(self, candles: object, horizon: int | None = None, name: str | None = None) -> LabelResult:
        horizon = self._resolve_horizon(horizon)
        _, high, low, close = self._ohlc(candles)
        values, valid = self._empty_result_arrays(len(close))
        atr_values = atr(high, low, close, period=int(self.config.labels.atr_period))
        tp_mult = float(self.config.labels.take_profit_atr_mult)
        sl_mult = float(self.config.labels.stop_loss_atr_mult)

        for i in range(0, max(len(close) - horizon, 0)):
            if not (np.isfinite(close[i]) and np.isfinite(atr_values[i]) and atr_values[i] > 0):
                continue

            tp_level = close[i] + tp_mult * atr_values[i]
            sl_level = close[i] - sl_mult * atr_values[i]
            outcome = 0.0

            for j in range(i + 1, i + horizon + 1):
                hit_tp = np.isfinite(high[j]) and high[j] >= tp_level
                hit_sl = np.isfinite(low[j]) and low[j] <= sl_level
                if hit_sl:
                    outcome = -1.0
                    break
                if hit_tp:
                    outcome = 1.0
                    break

            if outcome == 0.0 and np.isfinite(close[i + horizon]):
                future_return = safe_div(close[i + horizon] - close[i], close[i], default=0.0)
                outcome = float(np.sign(future_return))

            values[i] = outcome
            valid[i] = True

        return self._result(
            values=values,
            name=name or f"{self.method}_{horizon}",
            method=self.method,
            horizon=horizon,
            valid_mask=valid,
            metadata={
                "take_profit_atr_mult": tp_mult,
                "stop_loss_atr_mult": sl_mult,
                "atr_period": int(self.config.labels.atr_period),
                "classes": {"stop_loss": -1, "vertical": 0, "take_profit": 1},
            },
        )


# ==============================================================================
# RISK / REWARD
# ==============================================================================


class RiskReward(BaseLabeler):
    """Future favorable excursion divided by adverse excursion."""

    method = "risk_reward"

    def label(self, candles: object, horizon: int | None = None, name: str | None = None) -> LabelResult:
        horizon = self._resolve_horizon(horizon)
        _, high, low, close = self._ohlc(candles)
        values, valid = self._empty_result_arrays(len(close))

        for i in range(0, max(len(close) - horizon, 0)):
            high_window = high[i + 1 : i + horizon + 1]
            low_window = low[i + 1 : i + horizon + 1]
            if not (np.isfinite(close[i]) and np.isfinite(high_window).all() and np.isfinite(low_window).all()):
                continue
            reward = max(float(np.max(high_window) - close[i]), 0.0)
            risk = max(float(close[i] - np.min(low_window)), 0.0)
            values[i] = safe_div(reward, risk, default=np.nan)
            valid[i] = np.isfinite(values[i])

        return self._result(values, name or f"{self.method}_{horizon}", self.method, horizon, valid)


# ==============================================================================
# SL / TP DISTANCE LABELS
# ==============================================================================


class StopLossDistance(BaseLabeler):
    """Maximum adverse move below entry over the future horizon."""

    method = "stop_loss_distance"

    def label(self, candles: object, horizon: int | None = None, name: str | None = None) -> LabelResult:
        return _distance_label(self, candles, horizon, name, direction="stop_loss")


class TakeProfitDistance(BaseLabeler):
    """Maximum favorable move above entry over the future horizon."""

    method = "take_profit_distance"

    def label(self, candles: object, horizon: int | None = None, name: str | None = None) -> LabelResult:
        return _distance_label(self, candles, horizon, name, direction="take_profit")


def _distance_label(
    labeler: BaseLabeler,
    candles: object,
    horizon: int | None,
    name: str | None,
    direction: str,
) -> LabelResult:
    resolved_horizon = labeler._resolve_horizon(horizon)
    _, high, low, close = labeler._ohlc(candles)
    values, valid = labeler._empty_result_arrays(len(close))

    for i in range(0, max(len(close) - resolved_horizon, 0)):
        window = low[i + 1 : i + resolved_horizon + 1] if direction == "stop_loss" else high[i + 1 : i + resolved_horizon + 1]
        if not (np.isfinite(close[i]) and np.isfinite(window).all()):
            continue
        raw_distance = close[i] - np.min(window) if direction == "stop_loss" else np.max(window) - close[i]
        values[i] = safe_div(max(float(raw_distance), 0.0), close[i], default=np.nan)
        valid[i] = np.isfinite(values[i])

    method = f"{direction}_distance"
    metadata: Dict[str, str] = {"unit": "fraction_of_entry_price"}
    return labeler._result(values, name or f"{method}_{resolved_horizon}", method, resolved_horizon, valid, metadata)
