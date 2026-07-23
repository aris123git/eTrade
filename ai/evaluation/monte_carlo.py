"""
ai/evaluation/monte_carlo.py - Monte Carlo trade-return reshuffling.

VERSION: 1.0.0
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Sequence

import numpy as np
from numpy.typing import NDArray

from ai.evaluation.trading_metrics import equity_curve


@dataclass(frozen=True)
class MonteCarloResult:
    """Container for Monte Carlo equity-path simulations."""

    simulations: NDArray[np.floating]
    mean_curve: NDArray[np.floating]
    lower_band: NDArray[np.floating]
    upper_band: NDArray[np.floating]
    final_equity: NDArray[np.floating]
    confidence: float
    replacement: bool
    initial_equity: float

    def to_dict(self) -> Dict[str, Any]:
        """Serialize the result to JSON-compatible primitives."""
        return {
            "simulations": self.simulations.tolist(),
            "mean_curve": self.mean_curve.tolist(),
            "lower_band": self.lower_band.tolist(),
            "upper_band": self.upper_band.tolist(),
            "final_equity": self.final_equity.tolist(),
            "confidence": self.confidence,
            "replacement": self.replacement,
            "initial_equity": self.initial_equity,
        }


def monte_carlo_reshuffle(
    trade_returns: Sequence[float],
    n_simulations: int = 1000,
    confidence: float = 0.95,
    initial_equity: float = 1.0,
    compounded: bool = False,
    replacement: bool = False,
    random_seed: int | None = None,
) -> MonteCarloResult:
    """
    Reshuffle trade returns and return simulated equity paths plus confidence bands.

    With replacement disabled, every path is a permutation of the observed trades.
    With replacement enabled, paths are bootstrap samples of equal length.
    """
    if n_simulations <= 0:
        raise ValueError("n_simulations must be > 0")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be between 0 and 1")

    trades = np.asarray(trade_returns, dtype=float).reshape(-1)
    trades = trades[np.isfinite(trades)]
    rng = np.random.default_rng(random_seed)
    simulations = np.empty((n_simulations, len(trades) + 1), dtype=float)

    for idx in range(n_simulations):
        if len(trades) == 0:
            sampled = trades
        elif replacement:
            sampled = rng.choice(trades, size=len(trades), replace=True)
        else:
            sampled = rng.permutation(trades)
        simulations[idx] = equity_curve(sampled, initial_equity=initial_equity, compounded=compounded)

    tail = (1.0 - confidence) / 2.0
    lower = np.quantile(simulations, tail, axis=0)
    upper = np.quantile(simulations, 1.0 - tail, axis=0)
    return MonteCarloResult(
        simulations=simulations,
        mean_curve=np.mean(simulations, axis=0),
        lower_band=lower,
        upper_band=upper,
        final_equity=simulations[:, -1],
        confidence=float(confidence),
        replacement=bool(replacement),
        initial_equity=float(initial_equity),
    )


def confidence_bands(
    simulations: NDArray[np.floating],
    confidence: float = 0.95,
) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
    """Compute lower and upper confidence bands for existing simulations."""
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be between 0 and 1")
    paths = np.asarray(simulations, dtype=float)
    if paths.ndim != 2:
        raise ValueError("simulations must be a 2D array")
    tail = (1.0 - confidence) / 2.0
    return np.quantile(paths, tail, axis=0), np.quantile(paths, 1.0 - tail, axis=0)
