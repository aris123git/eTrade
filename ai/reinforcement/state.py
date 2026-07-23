"""
ai/reinforcement/state.py - State construction for RL trading agents.

VERSION: 1.0.0
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Mapping, Sequence

import numpy as np
from numpy.typing import NDArray


@dataclass
class StateBuilder:
    """Build numeric states from market features and portfolio context."""

    include_position: bool = True
    include_unrealized_pnl: bool = True
    include_equity_ratio: bool = True
    clip_value: float | None = 10.0
    discretization_bins: int = 10
    feature_names: Sequence[str] = field(default_factory=tuple)

    def build(
        self,
        features: Sequence[float] | Mapping[str, float] | NDArray[np.floating],
        *,
        position: float = 0.0,
        unrealized_pnl: float = 0.0,
        equity: float = 1.0,
        initial_equity: float = 1.0,
    ) -> NDArray[np.floating]:
        """Return a one-dimensional state vector."""

        base = self._feature_vector(features)
        extra: list[float] = []
        if self.include_position:
            extra.append(float(np.clip(position, -1.0, 1.0)))
        if self.include_unrealized_pnl:
            denominator = abs(float(initial_equity)) or 1.0
            extra.append(float(unrealized_pnl) / denominator)
        if self.include_equity_ratio:
            denominator = abs(float(initial_equity)) or 1.0
            extra.append(float(equity) / denominator - 1.0)
        state = np.concatenate([base, np.asarray(extra, dtype=float)]) if extra else base
        state = np.nan_to_num(state.astype(float), nan=0.0, posinf=0.0, neginf=0.0)
        if self.clip_value is not None:
            limit = abs(float(self.clip_value))
            state = np.clip(state, -limit, limit)
        return state

    def build_batch(
        self,
        features: NDArray[np.floating],
        positions: Iterable[float] | None = None,
        unrealized_pnl: Iterable[float] | None = None,
        equity: Iterable[float] | None = None,
        initial_equity: float = 1.0,
    ) -> NDArray[np.floating]:
        """Build states for a matrix of feature rows."""

        matrix = np.asarray(features, dtype=float)
        if matrix.ndim == 1:
            matrix = matrix.reshape(1, -1)
        n_rows = matrix.shape[0]
        pos = list(positions) if positions is not None else [0.0] * n_rows
        pnl = list(unrealized_pnl) if unrealized_pnl is not None else [0.0] * n_rows
        eq = list(equity) if equity is not None else [initial_equity] * n_rows
        return np.vstack(
            [
                self.build(
                    row,
                    position=pos[idx],
                    unrealized_pnl=pnl[idx],
                    equity=eq[idx],
                    initial_equity=initial_equity,
                )
                for idx, row in enumerate(matrix)
            ]
        )

    def discretize(
        self,
        state: Sequence[float] | NDArray[np.floating],
        bins: int | Sequence[float] | None = None,
    ) -> tuple[int, ...]:
        """Convert a continuous state into a tabular key."""

        vector = np.asarray(state, dtype=float).reshape(-1)
        if bins is None:
            count = max(2, int(self.discretization_bins))
            edges = np.linspace(-1.0, 1.0, count + 1)[1:-1]
        else:
            edges = np.asarray(bins, dtype=float)
            if edges.ndim == 0:
                count = max(2, int(edges))
                edges = np.linspace(-1.0, 1.0, count + 1)[1:-1]
        clipped = np.tanh(np.nan_to_num(vector, nan=0.0, posinf=0.0, neginf=0.0))
        return tuple(int(value) for value in np.digitize(clipped, edges))

    def _feature_vector(
        self,
        features: Sequence[float] | Mapping[str, float] | NDArray[np.floating],
    ) -> NDArray[np.floating]:
        if isinstance(features, Mapping):
            names = list(self.feature_names) if self.feature_names else sorted(features)
            return np.asarray([features.get(name, 0.0) for name in names], dtype=float)
        arr = np.asarray(features, dtype=float)
        if arr.ndim == 0:
            return arr.reshape(1)
        return arr.reshape(-1)
