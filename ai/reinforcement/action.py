"""
ai/reinforcement/action.py - Trading action space primitives.

VERSION: 1.0.0
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Iterable, Sequence

import numpy as np
from numpy.typing import NDArray


class TradeAction(str, Enum):
    """Discrete actions supported by the trading environment."""

    HOLD = "hold"
    BUY = "buy"
    SELL = "sell"
    CLOSE = "close"


@dataclass(frozen=True)
class ActionSpace:
    """Small gym-like action space for discrete trading actions."""

    actions: Sequence[TradeAction | str] = field(
        default_factory=lambda: (
            TradeAction.HOLD,
            TradeAction.BUY,
            TradeAction.SELL,
            TradeAction.CLOSE,
        )
    )

    def __post_init__(self) -> None:
        normalized = tuple(self.normalize(action) for action in self.actions)
        if not normalized:
            raise ValueError("ActionSpace requires at least one action")
        if len(set(normalized)) != len(normalized):
            raise ValueError("ActionSpace actions must be unique")
        object.__setattr__(self, "actions", normalized)

    @property
    def n(self) -> int:
        """Return the number of available actions."""

        return len(self.actions)

    def sample(self, rng: np.random.Generator | None = None) -> TradeAction:
        """Sample one action uniformly."""

        generator = rng or np.random.default_rng()
        return self.actions[int(generator.integers(0, self.n))]

    def contains(self, action: TradeAction | str | int) -> bool:
        """Return True when the action belongs to this space."""

        try:
            self.to_index(action)
        except (TypeError, ValueError):
            return False
        return True

    def to_index(self, action: TradeAction | str | int) -> int:
        """Convert an action value to its stable integer index."""

        if isinstance(action, (int, np.integer)):
            index = int(action)
            if 0 <= index < self.n:
                return index
            raise ValueError(f"Action index out of range: {action}")
        normalized = self.normalize(action)
        try:
            return list(self.actions).index(normalized)
        except ValueError as exc:
            raise ValueError(f"Unsupported action: {action!r}") from exc

    def from_index(self, index: int) -> TradeAction:
        """Return the action for an integer index."""

        return self.actions[self.to_index(index)]

    def one_hot(self, action: TradeAction | str | int) -> NDArray[np.floating]:
        """Encode an action as a one-hot vector."""

        out = np.zeros(self.n, dtype=float)
        out[self.to_index(action)] = 1.0
        return out

    def mask(self, allowed: Iterable[TradeAction | str | int]) -> NDArray[np.bool_]:
        """Build a boolean mask for allowed actions."""

        out = np.zeros(self.n, dtype=bool)
        for action in allowed:
            out[self.to_index(action)] = True
        return out

    @staticmethod
    def normalize(action: TradeAction | str) -> TradeAction:
        """Normalize strings and enum values into TradeAction."""

        if isinstance(action, TradeAction):
            return action
        value = str(action).strip().lower()
        aliases = {
            "0": TradeAction.HOLD,
            "hold": TradeAction.HOLD,
            "flat": TradeAction.HOLD,
            "1": TradeAction.BUY,
            "buy": TradeAction.BUY,
            "long": TradeAction.BUY,
            "2": TradeAction.SELL,
            "sell": TradeAction.SELL,
            "short": TradeAction.SELL,
            "3": TradeAction.CLOSE,
            "close": TradeAction.CLOSE,
            "exit": TradeAction.CLOSE,
        }
        if value not in aliases:
            raise ValueError(f"Unsupported trading action: {action!r}")
        return aliases[value]
