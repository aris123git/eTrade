"""
ai/optimization/walk_forward_opt.py - Walk-forward parameter optimization.

VERSION: 1.0.0
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Sequence

import numpy as np


ParameterSet = Dict[str, Any]
EvaluateFn = Callable[[ParameterSet, slice, slice], float | Dict[str, float]]


@dataclass(frozen=True)
class WalkForwardFold:
    """Single walk-forward train/test split."""

    train: slice
    test: slice


@dataclass(frozen=True)
class WalkForwardResult:
    """Walk-forward optimization result."""

    best_params: ParameterSet
    folds: List[Dict[str, Any]]
    objective: str
    score: float


@dataclass
class WalkForwardOptimizer:
    """Select parameters by rolling in-sample optimization and out-of-sample scoring."""

    train_size: int
    test_size: int
    step_size: int | None = None
    objective: str = "score"
    maximize: bool = True
    folds: List[WalkForwardFold] = field(default_factory=list)

    def split(self, n_samples: int) -> List[WalkForwardFold]:
        """Build rolling walk-forward folds."""

        if self.train_size <= 0 or self.test_size <= 0:
            raise ValueError("train_size and test_size must be > 0")
        step = int(self.step_size or self.test_size)
        folds: List[WalkForwardFold] = []
        start = 0
        while start + self.train_size + self.test_size <= int(n_samples):
            train = slice(start, start + self.train_size)
            test = slice(start + self.train_size, start + self.train_size + self.test_size)
            folds.append(WalkForwardFold(train=train, test=test))
            start += step
        self.folds = folds
        return folds

    def optimize(
        self,
        parameter_grid: Iterable[ParameterSet],
        n_samples: int,
        evaluate: EvaluateFn,
    ) -> WalkForwardResult:
        """Evaluate each parameter set across rolling folds."""

        folds = self.split(n_samples)
        if not folds:
            raise ValueError("No walk-forward folds can be built for the requested sizes")
        params_list = [dict(params) for params in parameter_grid]
        if not params_list:
            raise ValueError("parameter_grid must contain at least one parameter set")

        fold_summaries: List[Dict[str, Any]] = []
        aggregate: Dict[int, List[float]] = {idx: [] for idx in range(len(params_list))}
        for fold_idx, fold in enumerate(folds):
            fold_scores: List[float] = []
            for params_idx, params in enumerate(params_list):
                raw_score = evaluate(params, fold.train, fold.test)
                score = self._extract_score(raw_score)
                aggregate[params_idx].append(score)
                fold_scores.append(score)
            best_idx = self._best_index(fold_scores)
            fold_summaries.append(
                {
                    "fold": fold_idx,
                    "train": [fold.train.start, fold.train.stop],
                    "test": [fold.test.start, fold.test.stop],
                    "best_params": params_list[best_idx],
                    "best_score": float(fold_scores[best_idx]),
                    "scores": fold_scores,
                }
            )

        mean_scores = [float(np.mean(aggregate[idx])) for idx in range(len(params_list))]
        best_idx = self._best_index(mean_scores)
        return WalkForwardResult(
            best_params=params_list[best_idx],
            folds=fold_summaries,
            objective=self.objective,
            score=float(mean_scores[best_idx]),
        )

    def _extract_score(self, value: float | Dict[str, float]) -> float:
        if isinstance(value, dict):
            if self.objective not in value:
                raise ValueError(f"Evaluation result missing objective {self.objective!r}")
            return float(value[self.objective])
        return float(value)

    def _best_index(self, values: Sequence[float]) -> int:
        arr = np.asarray(values, dtype=float)
        return int(np.nanargmax(arr) if self.maximize else np.nanargmin(arr))
