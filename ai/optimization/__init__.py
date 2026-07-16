"""Optimization utilities for model search and portfolio construction."""

from ai.optimization.hyperparams import (
    DEFAULT_SEARCH_SPACES,
    SearchSpace,
    get_search_space,
    grid_from_space,
    list_search_spaces,
    merge_search_space,
)
from ai.optimization.objective import objective_score, trading_objectives
from ai.optimization.portfolio_opt import (
    PortfolioAllocation,
    inverse_volatility_allocation,
    mean_variance_allocation,
    minimum_variance_allocation,
    risk_parity_allocation,
)
from ai.optimization.walk_forward_opt import WalkForwardFold, WalkForwardOptimizer, WalkForwardResult

__all__ = [
    "DEFAULT_SEARCH_SPACES",
    "SearchSpace",
    "get_search_space",
    "grid_from_space",
    "list_search_spaces",
    "merge_search_space",
    "objective_score",
    "trading_objectives",
    "PortfolioAllocation",
    "inverse_volatility_allocation",
    "mean_variance_allocation",
    "minimum_variance_allocation",
    "risk_parity_allocation",
    "WalkForwardFold",
    "WalkForwardOptimizer",
    "WalkForwardResult",
]
