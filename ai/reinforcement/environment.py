"""
ai/reinforcement/environment.py - Trading environment for RL agents.

VERSION: 1.0.0
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Sequence

import numpy as np
from numpy.typing import NDArray

from ai.config.settings import AIConfig
from ai.reinforcement.action import ActionSpace, TradeAction
from ai.reinforcement.reward import RewardBreakdown, RewardFunction
from ai.reinforcement.state import StateBuilder


@dataclass(frozen=True)
class StepResult:
    """Result returned by TradingEnvironment.step."""

    state: NDArray[np.floating]
    reward: float
    done: bool
    info: Dict[str, Any]


@dataclass
class TradingEnvironment:
    """Deterministic single-asset trading environment with discrete actions."""

    features: NDArray[np.floating] | Sequence[Sequence[float]]
    prices: Sequence[float] | NDArray[np.floating]
    config: AIConfig = field(default_factory=AIConfig)
    initial_equity: float = 10_000.0
    quantity: float = 1.0
    action_space: ActionSpace = field(default_factory=ActionSpace)
    state_builder: StateBuilder = field(default_factory=StateBuilder)
    reward_function: RewardFunction = field(default_factory=RewardFunction)
    commission: float | None = None
    slippage: float | None = None

    def __post_init__(self) -> None:
        self.features = np.asarray(self.features, dtype=float)
        if self.features.ndim == 1:
            self.features = self.features.reshape(-1, 1)
        self.prices = np.asarray(self.prices, dtype=float).reshape(-1)
        if len(self.features) != len(self.prices):
            raise ValueError("features and prices must have the same number of rows")
        if len(self.prices) < 2:
            raise ValueError("TradingEnvironment requires at least two price rows")
        self.commission = (
            float(self.config.execution.commission_per_lot) * float(self.quantity)
            if self.commission is None
            else float(self.commission)
        )
        self.slippage = (
            float(self.config.execution.slippage_points) * 0.0001
            if self.slippage is None
            else float(self.slippage)
        )
        self.reset()

    def reset(self) -> NDArray[np.floating]:
        """Reset the episode and return the first state."""

        self.current_step = 0
        self.position = 0.0
        self.entry_price = 0.0
        self.cash = float(self.initial_equity)
        self.equity = float(self.initial_equity)
        self.realized_pnl = 0.0
        self.unrealized_pnl = 0.0
        self.closed_trades = 0
        self.reward_function.reset(self.initial_equity)
        return self._state()

    def step(self, action: TradeAction | str | int) -> StepResult:
        """Apply an action and advance one market row."""

        if self.current_step >= len(self.prices) - 1:
            return StepResult(self._state(), 0.0, True, self._info(None))

        trade_action = self.action_space.from_index(action) if isinstance(action, (int, np.integer)) else self.action_space.normalize(action)
        previous_equity = float(self.equity)
        previous_position = float(self.position)
        current_price = float(self.prices[self.current_step])
        next_price = float(self.prices[self.current_step + 1])

        realized = self._execute(trade_action, current_price)
        self.unrealized_pnl = self._mark_to_market(next_price)
        interval_pnl = self._position_pnl(previous_position if trade_action == TradeAction.HOLD else self.position, current_price, next_price)
        step_pnl = realized + interval_pnl
        self.equity = self.cash + self.unrealized_pnl
        self.current_step += 1
        done = self.current_step >= len(self.prices) - 1
        breakdown = self.reward_function.compute(
            pnl=step_pnl,
            equity=self.equity,
            previous_equity=previous_equity,
            position=self.position,
            previous_position=previous_position,
        )
        return StepResult(
            state=self._state(),
            reward=breakdown.reward,
            done=done,
            info=self._info(breakdown, action=trade_action.value, pnl=step_pnl),
        )

    def _execute(self, action: TradeAction, price: float) -> float:
        realized = 0.0
        if action == TradeAction.HOLD:
            return realized
        if action == TradeAction.CLOSE:
            return self._close(price)
        if action == TradeAction.BUY:
            if self.position < 0.0:
                realized += self._close(price)
            if self.position == 0.0:
                self._open(1.0, price)
            return realized
        if action == TradeAction.SELL:
            if self.position > 0.0:
                realized += self._close(price)
            if self.position == 0.0:
                self._open(-1.0, price)
            return realized
        return realized

    def _open(self, side: float, price: float) -> None:
        execution_price = price + float(side) * float(self.slippage or 0.0)
        self.position = float(side)
        self.entry_price = execution_price
        self.cash -= float(self.commission or 0.0)

    def _close(self, price: float) -> float:
        if self.position == 0.0:
            return 0.0
        execution_price = price - float(self.position) * float(self.slippage or 0.0)
        pnl = (execution_price - self.entry_price) * self.position * float(self.quantity)
        pnl -= float(self.commission or 0.0)
        self.cash += pnl
        self.realized_pnl += pnl
        self.position = 0.0
        self.entry_price = 0.0
        self.unrealized_pnl = 0.0
        self.closed_trades += 1
        return float(pnl)

    def _mark_to_market(self, price: float) -> float:
        if self.position == 0.0:
            return 0.0
        return float((price - self.entry_price) * self.position * float(self.quantity))

    def _position_pnl(self, position: float, start_price: float, end_price: float) -> float:
        return float((end_price - start_price) * float(position) * float(self.quantity))

    def _state(self) -> NDArray[np.floating]:
        idx = min(int(self.current_step), len(self.features) - 1)
        return self.state_builder.build(
            self.features[idx],
            position=self.position,
            unrealized_pnl=self.unrealized_pnl,
            equity=self.equity,
            initial_equity=self.initial_equity,
        )

    def _info(self, reward: RewardBreakdown | None, **extra: Any) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "step": int(self.current_step),
            "price": float(self.prices[min(self.current_step, len(self.prices) - 1)]),
            "position": float(self.position),
            "cash": float(self.cash),
            "equity": float(self.equity),
            "realized_pnl": float(self.realized_pnl),
            "unrealized_pnl": float(self.unrealized_pnl),
            "closed_trades": int(self.closed_trades),
        }
        if reward is not None:
            payload["reward_breakdown"] = reward.__dict__
        payload.update(extra)
        return payload
