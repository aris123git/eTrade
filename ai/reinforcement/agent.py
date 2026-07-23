"""
ai/reinforcement/agent.py - Reinforcement learning trading agents.

VERSION: 1.0.0
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, Sequence

import numpy as np
from numpy.typing import NDArray

from ai.config.settings import AIConfig
from ai.reinforcement.action import ActionSpace, TradeAction
from ai.reinforcement.policy import EpsilonGreedy, Policy
from ai.reinforcement.state import StateBuilder


@dataclass
class Agent(ABC):
    """Base interface for trading RL agents."""

    config: AIConfig = field(default_factory=AIConfig)
    action_space: ActionSpace = field(default_factory=ActionSpace)

    @abstractmethod
    def act(self, state: NDArray[np.floating], training: bool = True) -> TradeAction:
        """Choose an action."""

        raise RuntimeError("Subclasses must implement act")

    @abstractmethod
    def learn(
        self,
        state: NDArray[np.floating],
        action: TradeAction | str | int,
        reward: float,
        next_state: NDArray[np.floating],
        done: bool,
    ) -> float:
        """Update the agent from one transition and return loss."""

        raise RuntimeError("Subclasses must implement learn")


@dataclass
class QLearningAgent(Agent):
    """Tabular Q-learning agent with numpy-backed state discretization."""

    learning_rate: float = 0.1
    discount: float = 0.95
    bins: int = 10
    policy: Policy | None = None
    state_builder: StateBuilder = field(default_factory=StateBuilder)

    def __post_init__(self) -> None:
        if self.policy is None:
            self.policy = EpsilonGreedy(
                action_space=self.action_space,
                random_seed=self.config.random_seed,
            )
        self.q_table: Dict[tuple[int, ...], NDArray[np.floating]] = {}
        self.training_steps = 0

    def act(self, state: NDArray[np.floating], training: bool = True) -> TradeAction:
        """Choose an epsilon-greedy action in training or greedy action in serving."""

        key = self.state_key(state)
        q_values = self.q_values(key)
        if training and self.policy is not None:
            return self.policy.select_action(state, q_values)
        best = np.flatnonzero(q_values == np.max(q_values))
        rng = getattr(self.policy, "rng", np.random.default_rng(self.config.random_seed))
        return self.action_space.from_index(int(rng.choice(best)))

    def learn(
        self,
        state: NDArray[np.floating],
        action: TradeAction | str | int,
        reward: float,
        next_state: NDArray[np.floating],
        done: bool,
    ) -> float:
        """Run one Q-learning update."""

        state_key = self.state_key(state)
        next_key = self.state_key(next_state)
        action_index = self.action_space.to_index(action)
        values = self.q_values(state_key)
        target = float(reward)
        if not done:
            target += float(self.discount) * float(np.max(self.q_values(next_key)))
        old_value = float(values[action_index])
        td_error = target - old_value
        values[action_index] = old_value + float(self.learning_rate) * td_error
        self.q_table[state_key] = values
        self.training_steps += 1
        if isinstance(self.policy, EpsilonGreedy):
            self.policy.update()
        return float(td_error * td_error)

    def fit_episode(self, environment: object, max_steps: int | None = None) -> Dict[str, float]:
        """Train over one environment episode."""

        state = environment.reset()
        total_reward = 0.0
        total_loss = 0.0
        steps = 0
        while True:
            action = self.act(state, training=True)
            result = environment.step(action)
            loss = self.learn(state, action, result.reward, result.state, result.done)
            total_reward += float(result.reward)
            total_loss += loss
            steps += 1
            state = result.state
            if result.done or (max_steps is not None and steps >= int(max_steps)):
                break
        return {
            "steps": float(steps),
            "reward": float(total_reward),
            "loss": float(total_loss / max(steps, 1)),
        }

    def state_key(self, state: Sequence[float] | NDArray[np.floating]) -> tuple[int, ...]:
        """Return the discretized state table key."""

        return self.state_builder.discretize(np.asarray(state, dtype=float), bins=int(self.bins))

    def q_values(self, key: tuple[int, ...]) -> NDArray[np.floating]:
        """Return mutable Q values for a state key."""

        if key not in self.q_table:
            self.q_table[key] = np.zeros(self.action_space.n, dtype=float)
        return self.q_table[key]
