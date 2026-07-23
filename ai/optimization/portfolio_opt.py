"""
ai/optimization/portfolio_opt.py - Numpy portfolio allocation methods.

VERSION: 1.0.0
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Sequence

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class PortfolioAllocation:
    """Portfolio weights and diagnostics."""

    weights: NDArray[np.floating]
    expected_return: float
    volatility: float
    sharpe: float
    method: str

    def as_dict(self, assets: Sequence[str] | None = None) -> Dict[str, float | Dict[str, float]]:
        names = list(assets) if assets is not None else [f"asset_{idx}" for idx in range(len(self.weights))]
        return {
            "weights": {name: float(weight) for name, weight in zip(names, self.weights)},
            "expected_return": float(self.expected_return),
            "volatility": float(self.volatility),
            "sharpe": float(self.sharpe),
        }


def mean_variance_allocation(
    returns: NDArray[np.floating] | Sequence[Sequence[float]],
    risk_free_rate: float = 0.0,
    long_only: bool = True,
    ridge: float = 1e-6,
) -> PortfolioAllocation:
    """Compute tangency-style mean-variance weights."""

    matrix = _returns_matrix(returns)
    mu = np.nanmean(matrix, axis=0)
    cov = _covariance(matrix, ridge)
    excess = mu - float(risk_free_rate) / max(len(matrix), 1)
    try:
        raw = np.linalg.pinv(cov) @ excess
    except np.linalg.LinAlgError:
        raw = excess.copy()
    if long_only:
        raw = np.maximum(raw, 0.0)
    weights = _normalize_weights(raw)
    return _allocation(weights, mu, cov, "mean_variance", risk_free_rate)


def minimum_variance_allocation(
    returns: NDArray[np.floating] | Sequence[Sequence[float]],
    long_only: bool = True,
    ridge: float = 1e-6,
) -> PortfolioAllocation:
    """Compute global minimum variance weights."""

    matrix = _returns_matrix(returns)
    mu = np.nanmean(matrix, axis=0)
    cov = _covariance(matrix, ridge)
    ones = np.ones(cov.shape[0], dtype=float)
    raw = np.linalg.pinv(cov) @ ones
    if long_only:
        raw = np.maximum(raw, 0.0)
    weights = _normalize_weights(raw)
    return _allocation(weights, mu, cov, "minimum_variance", 0.0)


def risk_parity_allocation(
    returns: NDArray[np.floating] | Sequence[Sequence[float]],
    iterations: int = 1_000,
    learning_rate: float = 0.05,
    ridge: float = 1e-6,
) -> PortfolioAllocation:
    """Approximate equal risk contribution weights with projected gradients."""

    matrix = _returns_matrix(returns)
    mu = np.nanmean(matrix, axis=0)
    cov = _covariance(matrix, ridge)
    n_assets = cov.shape[0]
    weights = np.full(n_assets, 1.0 / n_assets, dtype=float)
    target = np.full(n_assets, 1.0 / n_assets, dtype=float)
    for _ in range(max(1, int(iterations))):
        portfolio_var = float(weights @ cov @ weights)
        if portfolio_var <= 0.0:
            break
        marginal = cov @ weights
        contribution = weights * marginal / portfolio_var
        gradient = contribution - target
        weights = np.maximum(weights - float(learning_rate) * gradient, 0.0)
        weights = _normalize_weights(weights)
    return _allocation(weights, mu, cov, "risk_parity", 0.0)


def inverse_volatility_allocation(
    returns: NDArray[np.floating] | Sequence[Sequence[float]],
) -> PortfolioAllocation:
    """Allocate inversely proportional to realized volatility."""

    matrix = _returns_matrix(returns)
    mu = np.nanmean(matrix, axis=0)
    cov = _covariance(matrix, 1e-6)
    vol = np.sqrt(np.maximum(np.diag(cov), 1e-12))
    weights = _normalize_weights(1.0 / vol)
    return _allocation(weights, mu, cov, "inverse_volatility", 0.0)


def _returns_matrix(values: NDArray[np.floating] | Sequence[Sequence[float]]) -> NDArray[np.floating]:
    matrix = np.asarray(values, dtype=float)
    if matrix.ndim != 2:
        raise ValueError("returns must be a 2D matrix")
    if matrix.shape[1] == 0:
        raise ValueError("returns must contain at least one asset")
    return np.nan_to_num(matrix, nan=0.0, posinf=0.0, neginf=0.0)


def _covariance(matrix: NDArray[np.floating], ridge: float) -> NDArray[np.floating]:
    if matrix.shape[0] < 2:
        cov = np.eye(matrix.shape[1], dtype=float) * float(ridge)
    else:
        cov = np.cov(matrix, rowvar=False)
        if cov.ndim == 0:
            cov = np.asarray([[float(cov)]], dtype=float)
    return np.asarray(cov, dtype=float) + np.eye(matrix.shape[1]) * float(ridge)


def _normalize_weights(weights: NDArray[np.floating]) -> NDArray[np.floating]:
    arr = np.nan_to_num(np.asarray(weights, dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
    total = float(np.sum(arr))
    if abs(total) <= 1e-12:
        return np.full(arr.size, 1.0 / max(arr.size, 1), dtype=float)
    return arr / total


def _allocation(
    weights: NDArray[np.floating],
    mu: NDArray[np.floating],
    cov: NDArray[np.floating],
    method: str,
    risk_free_rate: float,
) -> PortfolioAllocation:
    expected = float(weights @ mu)
    volatility = float(np.sqrt(max(weights @ cov @ weights, 0.0)))
    sharpe = (expected - float(risk_free_rate)) / volatility if volatility > 0.0 else 0.0
    return PortfolioAllocation(weights=weights, expected_return=expected, volatility=volatility, sharpe=float(sharpe), method=method)
