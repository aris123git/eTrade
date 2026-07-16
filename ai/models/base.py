"""
ai/models/base.py - Shared model contracts.

RESPONSIBILITY:
Define the production interface implemented by every model wrapper.

VERSION: 1.0.0
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Optional
import pickle

import numpy as np
from numpy.typing import NDArray

from ai.config.settings import AIConfig


# ==============================================================================
# TASKS
# ==============================================================================


class ModelTask(str, Enum):
    """Supported supervised and unsupervised model tasks."""

    CLASSIFICATION = "classification"
    REGRESSION = "regression"
    ANOMALY = "anomaly"

    @classmethod
    def from_value(cls, value: str | "ModelTask" | None) -> "ModelTask":
        """Normalize user configuration into a task enum."""
        if isinstance(value, cls):
            return value
        normalized = str(value or cls.CLASSIFICATION.value).lower()
        aliases = {
            "classify": cls.CLASSIFICATION,
            "classifier": cls.CLASSIFICATION,
            "classification": cls.CLASSIFICATION,
            "regress": cls.REGRESSION,
            "regressor": cls.REGRESSION,
            "regression": cls.REGRESSION,
            "anomaly": cls.ANOMALY,
            "outlier": cls.ANOMALY,
        }
        if normalized not in aliases:
            raise ValueError(f"Unsupported model task: {value}")
        return aliases[normalized]


# ==============================================================================
# BASE MODEL
# ==============================================================================


@dataclass
class BaseModel(ABC):
    """
    Abstract model interface used by training, validation, and serving layers.

    Implementations may wrap external estimators or pure numpy fallbacks, but
    every model exposes the same typed surface for lifecycle operations.
    """

    config: AIConfig = field(default_factory=AIConfig)
    task: ModelTask | str | None = None
    params: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.task = ModelTask.from_value(self.task or self.config.model.task)
        merged = dict(getattr(self.config.model, "extra_params", {}) or {})
        merged.update(self.params)
        self.params = merged

    @abstractmethod
    def fit(
        self,
        X: NDArray[np.floating],
        y: NDArray[np.floating],
        X_val: NDArray[np.floating] | None = None,
        y_val: NDArray[np.floating] | None = None,
    ) -> "BaseModel":
        """Fit model state from training arrays."""
        raise RuntimeError("Subclasses must implement fit")

    @abstractmethod
    def predict(self, X: NDArray[np.floating]) -> NDArray[np.floating]:
        """Return model predictions for feature rows."""
        raise RuntimeError("Subclasses must implement predict")

    def predict_proba(self, X: NDArray[np.floating]) -> Optional[NDArray[np.floating]]:
        """Return class probabilities when the model supports them."""
        return None

    def save(self, path: Path | str) -> None:
        """Persist the model instance with pickle."""
        target = Path(path)
        if target.suffix == "":
            target = target / "model.pkl"
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("wb") as handle:
            pickle.dump(self, handle, protocol=pickle.HIGHEST_PROTOCOL)

    @classmethod
    def load(cls, path: Path | str) -> "BaseModel":
        """Load a model instance saved by BaseModel.save."""
        source = Path(path)
        if source.is_dir():
            source = source / "model.pkl"
        with source.open("rb") as handle:
            model = pickle.load(handle)
        if not isinstance(model, BaseModel):
            raise TypeError(f"Checkpoint did not contain a BaseModel: {type(model)!r}")
        return model

    def get_params(self) -> Dict[str, Any]:
        """Return a shallow copy of model hyperparameters."""
        return dict(self.params)

    def set_params(self, **kwargs: Any) -> "BaseModel":
        """Update model hyperparameters in place."""
        self.params.update(kwargs)
        return self

    @property
    def feature_importances_(self) -> Optional[NDArray[np.floating]]:
        """Return feature importances when the implementation exposes them."""
        return None


def flatten_features(X: NDArray[np.floating]) -> NDArray[np.floating]:
    """Convert tabular or sequence arrays into a 2D feature matrix."""
    arr = np.asarray(X, dtype=float)
    if arr.ndim == 1:
        return arr.reshape(-1, 1)
    if arr.ndim == 2:
        return arr
    if arr.ndim >= 3:
        return arr.reshape(arr.shape[0], -1)
    raise ValueError(f"Expected at least 1 dimension, got shape {arr.shape}")


def flatten_target(y: NDArray[np.floating]) -> NDArray[np.floating]:
    """Convert labels into a one-dimensional target vector when possible."""
    arr = np.asarray(y)
    if arr.ndim <= 1:
        return arr.reshape(-1)
    if arr.shape[1] == 1:
        return arr.reshape(-1)
    return arr
