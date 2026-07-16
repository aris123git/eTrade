"""
ai/optimization/objective.py - Trading objective functions.

VERSION: 1.0.0
"""

from __future__ import annotations

from typing import Dict, Sequence

import numpy as np
from numpy.typing import NDArray

from ai.evaluation.trading_metrics import max_drawdown, profit_factor, sharpe, sortino, win_rate


def total_return(returns: Sequence[float] | NDArray[np.floating]) -> float:
    """Return compounded total return."""

    arr = _finite(returns)
    return float(np.prod(1.0 + arr) - 1.0) if arr.size else 0.0


def sharpe_objective(returns: Sequence[float] | NDArray[np.floating], periods: int = 252) -> float:
    """Objective to maximize annualized Sharpe."""

    return float(sharpe(_finite(returns), periods=periods))


def sortino_objective(returns: Sequence[float] | NDArray[np.floating], periods: int = 252) -> float:
    """Objective to maximize annualized Sortino."""

    return float(sortino(_finite(returns), periods=periods))


def calmar_objective(returns: Sequence[float] | NDArray[np.floating], periods: int = 252) -> float:
    """Objective to maximize annualized return over drawdown."""

    arr = _finite(returns)
    if arr.size == 0:
        return 0.0
    annualized = float(np.mean(arr) * periods)
    equity = np.cumprod(1.0 + arr)
    drawdown = abs(float(max_drawdown(equity)))
    return annualized / drawdown if drawdown > 0.0 else 0.0


def risk_adjusted_return(
    returns: Sequence[float] | NDArray[np.floating],
    drawdown_weight: float = 1.0,
    turnover: Sequence[float] | NDArray[np.floating] | None = None,
    turnover_weight: float = 0.0,
) -> float:
    """Return mean return penalized by drawdown and optional turnover."""

    arr = _finite(returns)
    if arr.size == 0:
        return 0.0
    equity = np.cumprod(1.0 + arr)
    penalty = float(drawdown_weight) * abs(float(max_drawdown(equity)))
    if turnover is not None:
        penalty += float(turnover_weight) * float(np.mean(np.abs(_finite(turnover))))
    return float(np.mean(arr) - penalty)


def trading_objectives(
    returns: Sequence[float] | NDArray[np.floating],
    periods: int = 252,
) -> Dict[str, float]:
    """Compute standard trading objectives in one pass."""

    arr = _finite(returns)
    equity = np.cumprod(1.0 + arr) if arr.size else np.asarray([], dtype=float)
    return {
        "total_return": total_return(arr),
        "mean_return": float(np.mean(arr)) if arr.size else 0.0,
        "sharpe": sharpe_objective(arr, periods=periods),
        "sortino": sortino_objective(arr, periods=periods),
        "calmar": calmar_objective(arr, periods=periods),
        "max_drawdown": float(max_drawdown(equity)) if equity.size else 0.0,
        "profit_factor": float(profit_factor(arr)),
        "win_rate": float(win_rate(arr)),
    }


def objective_score(
    returns: Sequence[float] | NDArray[np.floating],
    objective: str = "sharpe",
    periods: int = 252,
) -> float:
    """Dispatch an objective name to a scalar score."""

    name = str(objective).lower().strip()
    scores = trading_objectives(returns, periods=periods)
    aliases = {"return": "total_return", "mdd": "max_drawdown"}
    key = aliases.get(name, name)
    if key not in scores:
        raise ValueError(f"Unsupported objective: {objective!r}")
    return float(scores[key])


def _finite(values: Sequence[float] | NDArray[np.floating]) -> NDArray[np.floating]:
    arr = np.asarray(values, dtype=float).reshape(-1)
    return arr[np.isfinite(arr)]
