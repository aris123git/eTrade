"""
ai/preprocessing/scaler.py - Feature scaling primitives

RESPONSIBILITY:
Fit, apply, invert, and persist deterministic feature scaling state.

VERSION: 1.0.0
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict
import json

import numpy as np
from numpy.typing import NDArray

from ai.config.settings import AIConfig
from ai.utils.validation import AIValidationError, ensure_2d


# ==============================================================================
# SCALER
# ==============================================================================


@dataclass
class FeatureScaler:
    """Stateful feature scaler supporting zscore, minmax, robust, and none."""

    config: AIConfig = field(default_factory=AIConfig)
    method: str | None = None
    center_: NDArray[np.floating] | None = None
    scale_: NDArray[np.floating] | None = None
    data_min_: NDArray[np.floating] | None = None
    data_max_: NDArray[np.floating] | None = None
    n_features_in_: int | None = None

    def __post_init__(self) -> None:
        self.method = (self.method or self.config.datasets.scaling_method).lower()
        if self.method not in {"zscore", "minmax", "robust", "none"}:
            raise ValueError(f"Unsupported scaling method: {self.method}")

    @property
    def fitted(self) -> bool:
        return self.center_ is not None and self.scale_ is not None and self.n_features_in_ is not None

    def fit(self, features: NDArray[np.floating]) -> "FeatureScaler":
        """Fit scaler state from training features only."""
        x = ensure_2d(features)
        self.n_features_in_ = int(x.shape[1])

        if self.method == "zscore":
            center = np.nanmean(x, axis=0)
            scale = np.nanstd(x, axis=0, ddof=0)
        elif self.method == "minmax":
            self.data_min_ = np.nanmin(x, axis=0)
            self.data_max_ = np.nanmax(x, axis=0)
            center = self.data_min_.copy()
            scale = self.data_max_ - self.data_min_
        elif self.method == "robust":
            q25 = np.nanpercentile(x, 25, axis=0)
            q75 = np.nanpercentile(x, 75, axis=0)
            center = np.nanmedian(x, axis=0)
            scale = q75 - q25
        else:
            center = np.zeros(x.shape[1], dtype=float)
            scale = np.ones(x.shape[1], dtype=float)

        self.center_ = np.asarray(center, dtype=float)
        self.scale_ = self._nonzero_scale(np.asarray(scale, dtype=float))
        if self.data_min_ is None:
            self.data_min_ = np.nanmin(x, axis=0)
        if self.data_max_ is None:
            self.data_max_ = np.nanmax(x, axis=0)
        return self

    def transform(self, features: NDArray[np.floating]) -> NDArray[np.floating]:
        """Scale features using previously fitted training state."""
        self._require_fitted()
        x = ensure_2d(features)
        self._validate_feature_count(x)
        if self.method == "none":
            return x.copy()
        return (x - self.center_) / self.scale_  # type: ignore[operator]

    def inverse_transform(self, features: NDArray[np.floating]) -> NDArray[np.floating]:
        """Invert scaled features back to the original feature space."""
        self._require_fitted()
        x = ensure_2d(features)
        self._validate_feature_count(x)
        if self.method == "none":
            return x.copy()
        return x * self.scale_ + self.center_  # type: ignore[operator]

    def fit_transform(self, features: NDArray[np.floating]) -> NDArray[np.floating]:
        """Fit on features and return their scaled representation."""
        return self.fit(features).transform(features)

    def save(self, path: Path | str) -> None:
        """Persist scaler metadata as JSON and arrays as NPZ."""
        self._require_fitted()
        json_path, npz_path = self._state_paths(path)
        json_path.parent.mkdir(parents=True, exist_ok=True)
        metadata: Dict[str, Any] = {
            "method": self.method,
            "n_features_in": self.n_features_in_,
            "npz": npz_path.name,
        }
        with json_path.open("w", encoding="utf-8") as handle:
            json.dump(metadata, handle, indent=2)
        np.savez(
            npz_path,
            center=self.center_,
            scale=self.scale_,
            data_min=self.data_min_,
            data_max=self.data_max_,
        )

    @classmethod
    def load(cls, path: Path | str, config: AIConfig | None = None) -> "FeatureScaler":
        """Load scaler state saved by FeatureScaler.save."""
        json_path, npz_path = cls._state_paths(path)
        with json_path.open("r", encoding="utf-8") as handle:
            metadata = json.load(handle)
        if "npz" in metadata:
            npz_path = json_path.parent / metadata["npz"]
        arrays = np.load(npz_path)
        scaler = cls(config=config or AIConfig(), method=str(metadata["method"]))
        scaler.n_features_in_ = int(metadata["n_features_in"])
        scaler.center_ = np.asarray(arrays["center"], dtype=float)
        scaler.scale_ = np.asarray(arrays["scale"], dtype=float)
        scaler.data_min_ = np.asarray(arrays["data_min"], dtype=float)
        scaler.data_max_ = np.asarray(arrays["data_max"], dtype=float)
        return scaler

    def _require_fitted(self) -> None:
        if not self.fitted:
            raise AIValidationError("FeatureScaler must be fitted before use")

    def _validate_feature_count(self, features: NDArray[np.floating]) -> None:
        if self.n_features_in_ is not None and features.shape[1] != self.n_features_in_:
            raise AIValidationError(
                f"Expected {self.n_features_in_} features, got {features.shape[1]}"
            )

    @staticmethod
    def _nonzero_scale(scale: NDArray[np.floating]) -> NDArray[np.floating]:
        clean = np.asarray(scale, dtype=float).copy()
        clean[~np.isfinite(clean) | (clean == 0.0)] = 1.0
        return clean

    @staticmethod
    def _state_paths(path: Path | str) -> tuple[Path, Path]:
        base = Path(path)
        if base.suffix:
            json_path = base if base.suffix == ".json" else base.with_suffix(".json")
            npz_path = base.with_suffix(".npz")
        else:
            json_path = base / "scaler.json"
            npz_path = base / "scaler.npz"
        return json_path, npz_path


def create_feature_scaler(config: AIConfig | None = None, method: str | None = None) -> FeatureScaler:
    """Factory for feature scalers."""
    return FeatureScaler(config=config or AIConfig(), method=method)
