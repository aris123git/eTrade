"""
ai/reinforcement/policy.py - Action selection policies.

VERSION: 1.0.0
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
from numpy.typing import NDArray

from ai.reinforcement.action import ActionSpace, TradeAction


@dataclass
class Policy(ABC):
    """Base policy interface for discrete RL agents."""

    action_space: ActionSpace = field(default_factory=ActionSpace)
    random_seed: int | None = None

    def __post_init__(self) -> None:
        self.rng = np.random.default_rng(self.random_seed)

    @abstractmethod
    def select_action(
        self,
        state: NDArray[np.floating],
        q_values: Sequence[float] | NDArray[np.floating] | None = None,
    ) -> TradeAction:
        """Choose an action for a state."""

        raise RuntimeError("Subclasses must implement select_action")


@dataclass
class RandomPolicy(Policy):
    """Uniform random policy."""

    def select_action(
        self,
        state: NDArray[np.floating],
        q_values: Sequence[float] | NDArray[np.floating] | None = None,
    ) -> TradeAction:
        del state, q_values
        return self.action_space.sample(self.rng)


@dataclass
class EpsilonGreedy(Policy):
    """Epsilon-greedy policy over action values."""

    epsilon: float = 0.1
    min_epsilon: float = 0.01
    decay: float = 0.995

    def select_action(
        self,
        state: NDArray[np.floating],
        q_values: Sequence[float] | NDArray[np.floating] | None = None,
    ) -> TradeAction:
        del state
        if q_values is None or self.rng.random() < float(self.epsilon):
            return self.action_space.sample(self.rng)
        values = np.asarray(q_values, dtype=float).reshape(-1)
        if values.size != self.action_space.n:
            raise ValueError("q_values length must match action_space.n")
        best = np.flatnonzero(values == np.max(values))
        return self.action_space.from_index(int(self.rng.choice(best)))

    def update(self) -> float:
        """Apply epsilon decay and return the new epsilon."""

        self.epsilon = max(float(self.min_epsilon), float(self.epsilon) * float(self.decay))
        return float(self.epsilon)
