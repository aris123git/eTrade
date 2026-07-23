"""
ai/training/hyperparam.py - Lightweight hyperparameter search.

RESPONSIBILITY:
Run grid and random search over BaseModel parameter spaces without requiring
external optimization services.

VERSION: 1.0.0
"""

from __future__ import annotations

from dataclasses import dataclass, field
from importlib import import_module
from itertools import product
from typing import Any, Callable, Dict, Iterable, List, Sequence
import copy

import numpy as np
from numpy.typing import NDArray

from ai.models.base import BaseModel, flatten_features, flatten_target
from ai.training.validation import cross_val, summarize_scores


# ==============================================================================
# SEARCH RESULTS
# ==============================================================================


ModelFactory = Callable[[Dict[str, Any]], BaseModel]


@dataclass
class SearchResult:
    """Hyperparameter search outcome."""

    best_params: Dict[str, Any]
    best_score: float
    results: List[Dict[str, Any]] = field(default_factory=list)
    best_model: BaseModel | None = None


def _param_grid(param_grid: Dict[str, Sequence[Any]]) -> List[Dict[str, Any]]:
    keys = list(param_grid)
    values = [list(param_grid[key]) for key in keys]
    return [dict(zip(keys, combo)) for combo in product(*values)]


def _sample_space(param_space: Dict[str, Sequence[Any]], trials: int, random_seed: int = 42) -> List[Dict[str, Any]]:
    rng = np.random.default_rng(random_seed)
    keys = list(param_space)
    samples: list[Dict[str, Any]] = []
    for _ in range(max(1, int(trials))):
        params = {}
        for key in keys:
            values = list(param_space[key])
            if not values:
                raise ValueError(f"Parameter space for '{key}' is empty")
            params[key] = values[int(rng.integers(0, len(values)))]
        samples.append(params)
    return samples


def _evaluate(
    model_factory: ModelFactory,
    params: Dict[str, Any],
    X: NDArray[np.floating],
    y: NDArray[np.floating],
    folds: int,
    score_metric: str,
) -> Dict[str, Any]:
    model = model_factory(dict(params))
    scores = cross_val(model, flatten_features(X), flatten_target(y), folds=folds)
    summary = summarize_scores(scores)
    return {"params": dict(params), "metrics": summary, "score": float(summary.get(score_metric, 0.0))}


def _run_parallel(
    fn: Callable[[Dict[str, Any]], Dict[str, Any]],
    candidates: Iterable[Dict[str, Any]],
    n_jobs: int,
) -> List[Dict[str, Any]]:
    if n_jobs == 1:
        return [fn(params) for params in candidates]
    try:
        joblib = import_module("joblib")
    except ModuleNotFoundError:
        return [fn(params) for params in candidates]
    delayed = getattr(joblib, "delayed")
    parallel = getattr(joblib, "Parallel")
    return list(parallel(n_jobs=n_jobs)(delayed(fn)(params) for params in candidates))


def grid_search(
    model_factory: ModelFactory,
    param_grid: Dict[str, Sequence[Any]],
    X: NDArray[np.floating],
    y: NDArray[np.floating],
    folds: int = 3,
    score_metric: str = "f1",
    minimize: bool = False,
    n_jobs: int = 1,
) -> SearchResult:
    """Run exhaustive grid search."""
    candidates = _param_grid(param_grid)
    return _search_candidates(
        model_factory=model_factory,
        candidates=candidates,
        X=X,
        y=y,
        folds=folds,
        score_metric=score_metric,
        minimize=minimize,
        n_jobs=n_jobs,
    )


def random_search(
    model_factory: ModelFactory,
    param_space: Dict[str, Sequence[Any]],
    X: NDArray[np.floating],
    y: NDArray[np.floating],
    trials: int = 20,
    folds: int = 3,
    score_metric: str = "f1",
    minimize: bool = False,
    random_seed: int = 42,
    n_jobs: int = 1,
) -> SearchResult:
    """Run random search over discrete parameter choices."""
    candidates = _sample_space(param_space, trials=trials, random_seed=random_seed)
    return _search_candidates(
        model_factory=model_factory,
        candidates=candidates,
        X=X,
        y=y,
        folds=folds,
        score_metric=score_metric,
        minimize=minimize,
        n_jobs=n_jobs,
    )


def _search_candidates(
    model_factory: ModelFactory,
    candidates: List[Dict[str, Any]],
    X: NDArray[np.floating],
    y: NDArray[np.floating],
    folds: int,
    score_metric: str,
    minimize: bool,
    n_jobs: int,
) -> SearchResult:
    if not candidates:
        raise ValueError("Hyperparameter search requires at least one candidate")
    evaluator = lambda params: _evaluate(model_factory, params, X, y, folds, score_metric)
    results = _run_parallel(evaluator, candidates, n_jobs=n_jobs)
    key = lambda item: float(item["score"])
    best = min(results, key=key) if minimize else max(results, key=key)
    best_model = model_factory(dict(best["params"]))
    best_model.fit(flatten_features(X), flatten_target(y))
    return SearchResult(
        best_params=dict(best["params"]),
        best_score=float(best["score"]),
        results=results,
        best_model=best_model,
    )


@dataclass
class HyperparameterSearch:
    """Object-oriented facade for grid and random search."""

    model: BaseModel
    folds: int = 3
    score_metric: str = "f1"
    minimize: bool = False
    n_jobs: int = 1

    def _factory(self, params: Dict[str, Any]) -> BaseModel:
        candidate = copy.deepcopy(self.model)
        candidate.set_params(**params)
        return candidate

    def grid(self, param_grid: Dict[str, Sequence[Any]], X: NDArray[np.floating], y: NDArray[np.floating]) -> SearchResult:
        """Run grid search for this model."""
        return grid_search(
            self._factory,
            param_grid,
            X,
            y,
            folds=self.folds,
            score_metric=self.score_metric,
            minimize=self.minimize,
            n_jobs=self.n_jobs,
        )

    def random(
        self,
        param_space: Dict[str, Sequence[Any]],
        X: NDArray[np.floating],
        y: NDArray[np.floating],
        trials: int = 20,
        random_seed: int = 42,
    ) -> SearchResult:
        """Run random search for this model."""
        return random_search(
            self._factory,
            param_space,
            X,
            y,
            trials=trials,
            folds=self.folds,
            score_metric=self.score_metric,
            minimize=self.minimize,
            random_seed=random_seed,
            n_jobs=self.n_jobs,
        )
