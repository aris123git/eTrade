"""
ai/models/ensemble.py - Ensembles over BaseModel instances.

RESPONSIBILITY:
Compose fitted or unfitted BaseModel estimators with voting, bagging, stacking,
and blending strategies.

VERSION: 1.0.0
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
import copy

import numpy as np
from numpy.typing import NDArray

from ai.models.base import BaseModel, ModelTask, flatten_features, flatten_target


# ==============================================================================
# HELPERS
# ==============================================================================


def _clone_model(model: BaseModel) -> BaseModel:
    return copy.deepcopy(model)


def _classification_vote(predictions: NDArray[np.generic]) -> NDArray[np.generic]:
    rows: list[Any] = []
    for row in predictions.T:
        labels, counts = np.unique(row, return_counts=True)
        rows.append(labels[int(np.argmax(counts))])
    return np.asarray(rows)


def _meta_features(models: List[BaseModel], X: NDArray[np.floating]) -> NDArray[np.floating]:
    features: list[NDArray[np.floating]] = []
    for model in models:
        probs = model.predict_proba(X)
        if probs is not None:
            features.append(np.asarray(probs, dtype=float))
        else:
            features.append(np.asarray(model.predict(X), dtype=float).reshape(-1, 1))
    if not features:
        raise RuntimeError("At least one base model is required for ensemble meta features")
    return np.hstack(features)


# ==============================================================================
# ENSEMBLES
# ==============================================================================


@dataclass
class VotingEnsemble(BaseModel):
    """Average probabilities or predictions from multiple base models."""

    estimators: List[BaseModel] = field(default_factory=list)

    def fit(
        self,
        X: NDArray[np.floating],
        y: NDArray[np.floating],
        X_val: NDArray[np.floating] | None = None,
        y_val: NDArray[np.floating] | None = None,
    ) -> BaseModel:
        if not self.estimators:
            raise RuntimeError("VotingEnsemble requires at least one estimator")
        self.estimators = [
            _clone_model(model).fit(X, y, X_val=X_val, y_val=y_val)
            for model in self.estimators
        ]
        return self

    def predict(self, X: NDArray[np.floating]) -> NDArray[np.floating]:
        if not self.estimators:
            raise RuntimeError("VotingEnsemble requires fitted estimators before prediction")
        if self.task == ModelTask.CLASSIFICATION:
            probs = self.predict_proba(X)
            if probs is not None:
                first = self.estimators[0]
                classes = getattr(getattr(first, "estimator_", None), "classes_", None)
                if classes is None:
                    classes = getattr(first, "classes_", None)
                if classes is not None and len(classes) == probs.shape[1]:
                    return np.asarray(classes)[np.argmax(probs, axis=1)]
            predictions = np.vstack([model.predict(X) for model in self.estimators])
            return _classification_vote(predictions)
        predictions = np.vstack([np.asarray(model.predict(X), dtype=float) for model in self.estimators])
        return np.mean(predictions, axis=0)

    def predict_proba(self, X: NDArray[np.floating]) -> Optional[NDArray[np.floating]]:
        if self.task != ModelTask.CLASSIFICATION:
            return None
        probabilities = [model.predict_proba(X) for model in self.estimators]
        clean = [np.asarray(proba, dtype=float) for proba in probabilities if proba is not None]
        if len(clean) != len(self.estimators) or not clean:
            return None
        return np.mean(np.stack(clean, axis=0), axis=0)


@dataclass
class BaggingEnsemble(BaseModel):
    """Bootstrap aggregation for any BaseModel estimator."""

    base_model: BaseModel | None = None
    estimators: List[BaseModel] = field(default_factory=list)

    def fit(
        self,
        X: NDArray[np.floating],
        y: NDArray[np.floating],
        X_val: NDArray[np.floating] | None = None,
        y_val: NDArray[np.floating] | None = None,
    ) -> BaseModel:
        x = flatten_features(X)
        target = flatten_target(y)
        if not self.estimators and self.base_model is None:
            raise RuntimeError("BaggingEnsemble requires a base_model or estimators")
        n_estimators = int(self.params.get("n_estimators", self.config.model.n_estimators))
        rng = np.random.default_rng(self.config.model.random_state)
        template = self.base_model or self.estimators[0]
        fitted: list[BaseModel] = []
        for _ in range(max(1, n_estimators)):
            sample_idx = rng.integers(0, len(x), size=len(x))
            model = _clone_model(template)
            model.fit(x[sample_idx], target[sample_idx], X_val=X_val, y_val=y_val)
            fitted.append(model)
        self.estimators = fitted
        return self

    def predict(self, X: NDArray[np.floating]) -> NDArray[np.floating]:
        return VotingEnsemble(config=self.config, task=self.task, estimators=self.estimators).predict(X)

    def predict_proba(self, X: NDArray[np.floating]) -> Optional[NDArray[np.floating]]:
        return VotingEnsemble(config=self.config, task=self.task, estimators=self.estimators).predict_proba(X)


@dataclass
class StackingEnsemble(BaseModel):
    """Fit base estimators and a meta learner on their predictions."""

    estimators: List[BaseModel] = field(default_factory=list)
    meta_model: BaseModel | None = None

    def fit(
        self,
        X: NDArray[np.floating],
        y: NDArray[np.floating],
        X_val: NDArray[np.floating] | None = None,
        y_val: NDArray[np.floating] | None = None,
    ) -> BaseModel:
        if not self.estimators:
            raise RuntimeError("StackingEnsemble requires at least one estimator")
        self.estimators = [
            _clone_model(model).fit(X, y, X_val=X_val, y_val=y_val)
            for model in self.estimators
        ]
        meta_x = _meta_features(self.estimators, X)
        self.meta_model = self.meta_model or self._default_meta_model()
        self.meta_model.fit(meta_x, y)
        return self

    def predict(self, X: NDArray[np.floating]) -> NDArray[np.floating]:
        if self.meta_model is None:
            raise RuntimeError("StackingEnsemble must be fitted before prediction")
        return self.meta_model.predict(_meta_features(self.estimators, X))

    def predict_proba(self, X: NDArray[np.floating]) -> Optional[NDArray[np.floating]]:
        if self.task != ModelTask.CLASSIFICATION or self.meta_model is None:
            return None
        return self.meta_model.predict_proba(_meta_features(self.estimators, X))

    def _default_meta_model(self) -> BaseModel:
        if self.task == ModelTask.CLASSIFICATION:
            from ai.models.classical import LogisticRegressionModel

            return LogisticRegressionModel(config=self.config, task=self.task)
        from ai.models.classical import LinearRegressionModel

        return LinearRegressionModel(config=self.config, task=self.task)


@dataclass
class BlendingEnsemble(BaseModel):
    """Train base models on a prefix split and meta learner on a holdout blend set."""

    estimators: List[BaseModel] = field(default_factory=list)
    meta_model: BaseModel | None = None
    holdout_fraction: float = 0.2

    def fit(
        self,
        X: NDArray[np.floating],
        y: NDArray[np.floating],
        X_val: NDArray[np.floating] | None = None,
        y_val: NDArray[np.floating] | None = None,
    ) -> BaseModel:
        if not self.estimators:
            raise RuntimeError("BlendingEnsemble requires at least one estimator")
        x = flatten_features(X)
        target = flatten_target(y)
        split = int(len(x) * (1.0 - float(np.clip(self.holdout_fraction, 0.05, 0.5))))
        split = max(1, min(split, len(x) - 1))
        train_x, blend_x = x[:split], x[split:]
        train_y, blend_y = target[:split], target[split:]
        self.estimators = [
            _clone_model(model).fit(train_x, train_y, X_val=X_val, y_val=y_val)
            for model in self.estimators
        ]
        self.meta_model = self.meta_model or StackingEnsemble(config=self.config, task=self.task)._default_meta_model()
        self.meta_model.fit(_meta_features(self.estimators, blend_x), blend_y)
        return self

    def predict(self, X: NDArray[np.floating]) -> NDArray[np.floating]:
        if self.meta_model is None:
            raise RuntimeError("BlendingEnsemble must be fitted before prediction")
        return self.meta_model.predict(_meta_features(self.estimators, X))

    def predict_proba(self, X: NDArray[np.floating]) -> Optional[NDArray[np.floating]]:
        if self.task != ModelTask.CLASSIFICATION or self.meta_model is None:
            return None
        return self.meta_model.predict_proba(_meta_features(self.estimators, X))


ENSEMBLE_MODELS: Dict[str, type[BaseModel]] = {
    "voting": VotingEnsemble,
    "voting_ensemble": VotingEnsemble,
    "bagging": BaggingEnsemble,
    "bagging_ensemble": BaggingEnsemble,
    "stacking": StackingEnsemble,
    "stacking_ensemble": StackingEnsemble,
    "blending": BlendingEnsemble,
    "blending_ensemble": BlendingEnsemble,
}
