"""
ai/models/anomaly.py - Anomaly detection models.

RESPONSIBILITY:
Expose outlier detectors through the shared BaseModel lifecycle.

VERSION: 1.0.0
"""

from __future__ import annotations

from importlib import import_module
from typing import Any, Dict, Optional

import numpy as np
from numpy.typing import NDArray

from ai.models.base import BaseModel, ModelTask, flatten_features


# ==============================================================================
# FALLBACK DETECTOR
# ==============================================================================


class _NumpyIsolationFallback:
    """Distance-based anomaly detector used when sklearn is unavailable."""

    def __init__(self, contamination: float = 0.05) -> None:
        self.contamination = float(np.clip(contamination, 0.001, 0.499))
        self.center_: NDArray[np.floating] | None = None
        self.scale_: NDArray[np.floating] | None = None
        self.threshold_: float | None = None

    def fit(self, X: NDArray[np.floating], y: NDArray[np.floating] | None = None) -> "_NumpyIsolationFallback":
        x = flatten_features(X)
        self.center_ = np.nanmedian(x, axis=0)
        scale = np.nanmedian(np.abs(x - self.center_), axis=0)
        scale[~np.isfinite(scale) | (scale == 0.0)] = 1.0
        self.scale_ = scale
        scores = self.score_samples(x)
        self.threshold_ = float(np.quantile(scores, self.contamination))
        return self

    def predict(self, X: NDArray[np.floating]) -> NDArray[np.integer]:
        scores = self.score_samples(X)
        if self.threshold_ is None:
            raise RuntimeError("Isolation fallback must be fitted before prediction")
        return np.where(scores < self.threshold_, -1, 1)

    def score_samples(self, X: NDArray[np.floating]) -> NDArray[np.floating]:
        if self.center_ is None or self.scale_ is None:
            raise RuntimeError("Isolation fallback must be fitted before scoring")
        z = np.abs((flatten_features(X) - self.center_) / self.scale_)
        return -np.mean(z, axis=1)


class IsolationForestModel(BaseModel):
    """IsolationForest wrapper with a deterministic numpy fallback."""

    estimator_: Any = None

    def __post_init__(self) -> None:
        super().__post_init__()
        self.task = ModelTask.ANOMALY

    def _make_estimator(self) -> Any:
        params: Dict[str, Any] = {
            "n_estimators": self.config.model.n_estimators,
            "contamination": self.params.get("contamination", "auto"),
            "random_state": self.config.model.random_state,
            "n_jobs": self.config.model.n_jobs,
        }
        params.update(self.params)
        try:
            cls = getattr(import_module("sklearn.ensemble"), "IsolationForest")
            return cls(**params)
        except ModuleNotFoundError as exc:
            if exc.name == "sklearn" or str(exc.name).startswith("sklearn."):
                contamination = params.get("contamination", 0.05)
                if contamination == "auto":
                    contamination = 0.05
                return _NumpyIsolationFallback(contamination=float(contamination))
            raise

    def fit(
        self,
        X: NDArray[np.floating],
        y: NDArray[np.floating],
        X_val: NDArray[np.floating] | None = None,
        y_val: NDArray[np.floating] | None = None,
    ) -> BaseModel:
        self.estimator_ = self._make_estimator()
        self.estimator_.fit(flatten_features(X))
        return self

    def predict(self, X: NDArray[np.floating]) -> NDArray[np.floating]:
        if self.estimator_ is None:
            raise RuntimeError("IsolationForestModel must be fitted before prediction")
        return np.asarray(self.estimator_.predict(flatten_features(X)))

    def predict_proba(self, X: NDArray[np.floating]) -> Optional[NDArray[np.floating]]:
        return None


ANOMALY_MODELS: Dict[str, type[BaseModel]] = {
    "isolation_forest": IsolationForestModel,
    "isolationforest": IsolationForestModel,
    "iforest": IsolationForestModel,
}
