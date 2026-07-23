"""
ai/reinforcement/replay.py - Numpy circular replay buffer.

VERSION: 1.0.0
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np
from numpy.typing import NDArray


@dataclass
class ReplayBuffer:
    """Fixed-capacity circular buffer for off-policy learning."""

    capacity: int
    state_dim: int
    random_seed: int | None = None

    def __post_init__(self) -> None:
        if self.capacity <= 0:
            raise ValueError("capacity must be > 0")
        if self.state_dim <= 0:
            raise ValueError("state_dim must be > 0")
        self.states = np.zeros((int(self.capacity), int(self.state_dim)), dtype=float)
        self.next_states = np.zeros((int(self.capacity), int(self.state_dim)), dtype=float)
        self.actions = np.zeros(int(self.capacity), dtype=int)
        self.rewards = np.zeros(int(self.capacity), dtype=float)
        self.dones = np.zeros(int(self.capacity), dtype=bool)
        self.position = 0
        self.size = 0
        self.rng = np.random.default_rng(self.random_seed)

    def add(
        self,
        state: NDArray[np.floating],
        action: int,
        reward: float,
        next_state: NDArray[np.floating],
        done: bool,
    ) -> None:
        """Store one transition."""

        idx = self.position
        self.states[idx] = np.asarray(state, dtype=float).reshape(self.state_dim)
        self.actions[idx] = int(action)
        self.rewards[idx] = float(reward)
        self.next_states[idx] = np.asarray(next_state, dtype=float).reshape(self.state_dim)
        self.dones[idx] = bool(done)
        self.position = (idx + 1) % int(self.capacity)
        self.size = min(self.size + 1, int(self.capacity))

    def sample(self, batch_size: int) -> Dict[str, NDArray[np.generic]]:
        """Return a random mini-batch."""

        if self.size == 0:
            raise ValueError("Cannot sample from an empty ReplayBuffer")
        count = min(max(1, int(batch_size)), self.size)
        indexes = self.rng.choice(self.size, size=count, replace=False)
        return {
            "states": self.states[indexes].copy(),
            "actions": self.actions[indexes].copy(),
            "rewards": self.rewards[indexes].copy(),
            "next_states": self.next_states[indexes].copy(),
            "dones": self.dones[indexes].copy(),
        }

    def clear(self) -> None:
        """Remove all transitions without reallocating arrays."""

        self.position = 0
        self.size = 0

    def __len__(self) -> int:
        return int(self.size)
