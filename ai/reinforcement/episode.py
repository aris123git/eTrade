"""
ai/reinforcement/episode.py - Episode execution utilities.

VERSION: 1.0.0
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List

import numpy as np


@dataclass(frozen=True)
class EpisodeResult:
    """Summary of an episode rollout."""

    total_reward: float
    steps: int
    final_equity: float
    actions: List[str]
    rewards: List[float]
    info: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EpisodeRunner:
    """Run training or evaluation rollouts for RL agents."""

    max_steps: int | None = None
    training: bool = True

    def run(self, environment: Any, agent: Any, training: bool | None = None) -> EpisodeResult:
        """Execute one episode and optionally update the agent."""

        active_training = self.training if training is None else bool(training)
        state = environment.reset()
        actions: List[str] = []
        rewards: List[float] = []
        losses: List[float] = []
        last_info: Dict[str, Any] = {}

        for step in range(self.max_steps or 10**9):
            action = agent.act(state, training=active_training)
            result = environment.step(action)
            if active_training:
                losses.append(float(agent.learn(state, action, result.reward, result.state, result.done)))
            actions.append(str(getattr(action, "value", action)))
            rewards.append(float(result.reward))
            state = result.state
            last_info = dict(result.info)
            if result.done:
                break
        final_equity = float(last_info.get("equity", getattr(environment, "equity", 0.0)))
        return EpisodeResult(
            total_reward=float(np.sum(rewards)) if rewards else 0.0,
            steps=len(rewards),
            final_equity=final_equity,
            actions=actions,
            rewards=rewards,
            info={
                "last": last_info,
                "mean_loss": float(np.mean(losses)) if losses else 0.0,
                "training": active_training,
                "unique_actions": sorted(set(actions)),
            },
        )
