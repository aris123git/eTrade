"""
ai/preprocessing/selector.py - Feature selection

RESPONSIBILITY:
Select stable, informative feature columns without leaking validation/test data.

VERSION: 1.0.0
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

from ai.config.settings import AIConfig
from ai.utils.validation import AIValidationError, ensure_1d, ensure_2d


# ==============================================================================
# SELECTOR
# ==============================================================================


@dataclass
class FeatureSelector:
    """Variance-threshold and top-k feature selector with optional mutual information."""

    config: AIConfig = field(default_factory=AIConfig)
    variance_threshold: float = 0.0
    method: str = "mutual_info"
    selected_indices_: NDArray[np.integer] | None = None
    scores_: NDArray[np.floating] | None = None
    variances_: NDArray[np.floating] | None = None
    n_features_in_: int | None = None

    @property
    def fitted(self) -> bool:
        return self.selected_indices_ is not None and self.n_features_in_ is not None

    def fit(
        self,
        features: NDArray[np.floating],
        target: NDArray[np.floating] | None = None,
        top_k: int | None = None,
    ) -> "FeatureSelector":
        """Fit selected columns from training features and optional training target."""
        x = ensure_2d(features)
        y = ensure_1d(target) if target is not None else None
        if y is not None and len(y) != x.shape[0]:
            raise AIValidationError(f"target length {len(y)} does not match feature rows {x.shape[0]}")

        self.n_features_in_ = int(x.shape[1])
        self.variances_ = np.nanvar(x, axis=0)
        candidate_indices = np.flatnonzero(self.variances_ > float(self.variance_threshold))
        if len(candidate_indices) == 0:
            candidate_indices = np.asarray([int(np.nanargmax(self.variances_))], dtype=int)

        scores = self._score_features(x[:, candidate_indices], y)
        self.scores_ = np.zeros(x.shape[1], dtype=float)
        self.scores_[candidate_indices] = scores

        configured_k = self.config.datasets.feature_selection_k if top_k is None else top_k
        if configured_k is None or configured_k <= 0 or configured_k >= len(candidate_indices):
            selected = candidate_indices
        else:
            order = np.argsort(scores)[::-1][: int(configured_k)]
            selected = np.sort(candidate_indices[order])

        self.selected_indices_ = np.asarray(selected, dtype=int)
        return self

    def transform(self, features: NDArray[np.floating]) -> NDArray[np.floating]:
        """Return selected feature columns in original column order."""
        self._require_fitted()
        x = ensure_2d(features)
        if x.shape[1] != self.n_features_in_:
            raise AIValidationError(f"Expected {self.n_features_in_} features, got {x.shape[1]}")
        return x[:, self.selected_indices_]  # type: ignore[index]

    def fit_transform(
        self,
        features: NDArray[np.floating],
        target: NDArray[np.floating] | None = None,
        top_k: int | None = None,
    ) -> NDArray[np.floating]:
        """Fit on training data and return selected training columns."""
        return self.fit(features, target=target, top_k=top_k).transform(features)

    def get_support(self) -> NDArray[np.bool_]:
        """Boolean mask of selected input columns."""
        self._require_fitted()
        mask = np.zeros(int(self.n_features_in_), dtype=bool)
        mask[self.selected_indices_] = True  # type: ignore[index]
        return mask

    def _score_features(self, features: NDArray[np.floating], target: NDArray[np.floating] | None) -> NDArray[np.floating]:
        if target is None or self.method == "variance":
            return np.nanvar(features, axis=0)
        if self.method == "mutual_info":
            sklearn_scores = self._mutual_information_scores(features, target)
            if sklearn_scores is not None:
                return sklearn_scores
        return self._correlation_scores(features, target)

    def _mutual_information_scores(
        self,
        features: NDArray[np.floating],
        target: NDArray[np.floating],
    ) -> NDArray[np.floating] | None:
        try:
            from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
        except ImportError:
            return None

        x = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
        y = np.nan_to_num(target, nan=0.0, posinf=0.0, neginf=0.0)
        if self._looks_classification_target(y):
            scores = mutual_info_classif(x, y.astype(int), random_state=self.config.random_seed)
        else:
            scores = mutual_info_regression(x, y, random_state=self.config.random_seed)
        return np.asarray(scores, dtype=float)

    @staticmethod
    def _correlation_scores(features: NDArray[np.floating], target: NDArray[np.floating]) -> NDArray[np.floating]:
        y = np.asarray(target, dtype=float)
        y_mask = np.isfinite(y)
        scores = np.zeros(features.shape[1], dtype=float)
        for col in range(features.shape[1]):
            x = features[:, col]
            mask = y_mask & np.isfinite(x)
            if int(mask.sum()) < 2:
                continue
            x_valid = x[mask]
            y_valid = y[mask]
            x_std = float(np.std(x_valid))
            y_std = float(np.std(y_valid))
            if x_std == 0.0 or y_std == 0.0:
                continue
            scores[col] = abs(float(np.corrcoef(x_valid, y_valid)[0, 1]))
        return scores

    @staticmethod
    def _looks_classification_target(target: NDArray[np.floating]) -> bool:
        finite = target[np.isfinite(target)]
        if len(finite) == 0:
            return False
        unique = np.unique(finite)
        return len(unique) <= max(20, int(np.sqrt(len(finite)))) and np.allclose(unique, unique.astype(int))

    def _require_fitted(self) -> None:
        if not self.fitted:
            raise AIValidationError("FeatureSelector must be fitted before use")


def create_feature_selector(
    config: AIConfig | None = None,
    variance_threshold: float = 0.0,
    method: str = "mutual_info",
) -> FeatureSelector:
    """Factory for feature selectors."""
    return FeatureSelector(config=config or AIConfig(), variance_threshold=variance_threshold, method=method)
