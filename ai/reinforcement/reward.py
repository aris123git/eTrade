"""
ai/reinforcement/reward.py - Trading reward functions.

VERSION: 1.0.0
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class RewardBreakdown:
    """Serializable reward decomposition."""

    reward: float
    pnl: float
    sharpe_proxy: float
    drawdown_penalty: float
    position_penalty: float
    turnover_penalty: float


@dataclass
class RewardFunction:
    """Reward based on PnL, local Sharpe quality, and risk penalties."""

    pnl_weight: float = 1.0
    sharpe_weight: float = 0.1
    drawdown_weight: float = 0.5
    position_weight: float = 0.001
    turnover_weight: float = 0.001
    rolling_window: int = 20
    epsilon: float = 1e-12
    returns: list[float] = field(default_factory=list)
    equity_curve: list[float] = field(default_factory=list)

    def reset(self, initial_equity: float = 1.0) -> None:
        """Clear reward history for a new episode."""

        self.returns.clear()
        self.equity_curve[:] = [float(initial_equity)]

    def compute(
        self,
        *,
        pnl: float,
        equity: float,
        previous_equity: float,
        position: float = 0.0,
        previous_position: float = 0.0,
    ) -> RewardBreakdown:
        """Compute reward and update rolling return statistics."""

        previous = float(previous_equity)
        current = float(equity)
        step_return = float(pnl) / (abs(previous) + self.epsilon)
        self.returns.append(step_return)
        self.equity_curve.append(current)
        sharpe_value = self.sharpe_proxy(self.returns[-max(2, int(self.rolling_window)) :])
        drawdown_penalty = abs(min(0.0, self.current_drawdown()))
        position_penalty = abs(float(position))
        turnover_penalty = abs(float(position) - float(previous_position))
        reward = (
            self.pnl_weight * step_return
            + self.sharpe_weight * sharpe_value
            - self.drawdown_weight * drawdown_penalty
            - self.position_weight * position_penalty
            - self.turnover_weight * turnover_penalty
        )
        return RewardBreakdown(
            reward=float(reward),
            pnl=float(pnl),
            sharpe_proxy=float(sharpe_value),
            drawdown_penalty=float(drawdown_penalty),
            position_penalty=float(position_penalty),
            turnover_penalty=float(turnover_penalty),
        )

    def current_drawdown(self) -> float:
        """Return current drawdown as a negative decimal."""

        equity = np.asarray(self.equity_curve, dtype=float)
        if equity.size == 0:
            return 0.0
        peak = float(np.max(equity))
        if peak <= 0.0:
            return 0.0
        return float((equity[-1] - peak) / peak)

    @staticmethod
    def pnl_reward(pnl: float, previous_equity: float, epsilon: float = 1e-12) -> float:
        """Scale absolute PnL by previous equity."""

        return float(pnl) / (abs(float(previous_equity)) + float(epsilon))

    @staticmethod
    def sharpe_proxy(returns: Sequence[float] | NDArray[np.floating], epsilon: float = 1e-12) -> float:
        """Return a stable rolling Sharpe-like score."""

        arr = np.asarray(returns, dtype=float)
        arr = arr[np.isfinite(arr)]
        if arr.size < 2:
            return 0.0
        return float(np.mean(arr) / (np.std(arr, ddof=1) + epsilon))

    @staticmethod
    def drawdown_penalty(equity: Sequence[float] | NDArray[np.floating]) -> float:
        """Return maximum drawdown magnitude for an equity curve."""

        arr = np.asarray(equity, dtype=float)
        arr = arr[np.isfinite(arr)]
        if arr.size == 0:
            return 0.0
        peaks = np.maximum.accumulate(arr)
        mask = peaks > 0.0
        drawdowns = np.zeros_like(arr, dtype=float)
        drawdowns[mask] = (arr[mask] - peaks[mask]) / peaks[mask]
        return float(abs(np.min(drawdowns)))
