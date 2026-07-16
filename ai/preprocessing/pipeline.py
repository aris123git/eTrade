"""
ai/preprocessing/pipeline.py - Train-fitted preprocessing pipeline

RESPONSIBILITY:
Compose feature selection and scaling while fitting state only on training data.

VERSION: 1.0.0
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Tuple

import numpy as np
from numpy.typing import NDArray

from ai.config.settings import AIConfig
from ai.preprocessing.scaler import FeatureScaler, create_feature_scaler
from ai.preprocessing.selector import FeatureSelector, create_feature_selector
from ai.utils.validation import AIValidationError, ensure_2d


# ==============================================================================
# PIPELINE
# ==============================================================================


@dataclass
class PreprocessPipeline:
    """Pipeline that selects columns then scales selected features."""

    config: AIConfig = field(default_factory=AIConfig)
    selector: FeatureSelector | None = None
    scaler: FeatureScaler | None = None

    def __post_init__(self) -> None:
        if self.selector is None:
            self.selector = create_feature_selector(self.config)
        if self.scaler is None:
            self.scaler = create_feature_scaler(self.config)

    @property
    def fitted(self) -> bool:
        return bool(self.selector and self.selector.fitted and self.scaler and self.scaler.fitted)

    def fit(
        self,
        train_features: NDArray[np.floating],
        train_target: NDArray[np.floating] | None = None,
    ) -> "PreprocessPipeline":
        """Fit selector and scaler using training data only."""
        x_train = ensure_2d(train_features)
        selected_train = self.selector.fit_transform(x_train, target=train_target)  # type: ignore[union-attr]
        self.scaler.fit(selected_train)  # type: ignore[union-attr]
        return self

    def transform(self, features: NDArray[np.floating]) -> NDArray[np.floating]:
        """Apply train-fitted selection and scaling to any split."""
        self._require_fitted()
        x = ensure_2d(features)
        selected = self.selector.transform(x)  # type: ignore[union-attr]
        return self.scaler.transform(selected)  # type: ignore[union-attr]

    def fit_transform(
        self,
        train_features: NDArray[np.floating],
        train_target: NDArray[np.floating] | None = None,
    ) -> NDArray[np.floating]:
        """Fit on training features and return transformed training features."""
        return self.fit(train_features, train_target=train_target).transform(train_features)

    def transform_splits(
        self,
        train_features: NDArray[np.floating],
        val_features: NDArray[np.floating],
        test_features: NDArray[np.floating],
        train_target: NDArray[np.floating] | None = None,
    ) -> Tuple[NDArray[np.floating], NDArray[np.floating], NDArray[np.floating]]:
        """Fit on train, then transform train/validation/test consistently."""
        self.fit(train_features, train_target=train_target)
        return (
            self.transform(train_features),
            self.transform(val_features),
            self.transform(test_features),
        )

    def inverse_transform(self, features: NDArray[np.floating]) -> NDArray[np.floating]:
        """Invert scaling for the selected feature space."""
        self._require_fitted()
        return self.scaler.inverse_transform(features)  # type: ignore[union-attr]

    def metadata(self) -> Dict[str, object]:
        """Return fitted preprocessing metadata for experiment tracking."""
        self._require_fitted()
        return {
            "scaling_method": self.scaler.method,  # type: ignore[union-attr]
            "selected_indices": self.selector.selected_indices_.astype(int).tolist(),  # type: ignore[union-attr]
            "n_features_in": self.selector.n_features_in_,  # type: ignore[union-attr]
            "n_features_out": int(len(self.selector.selected_indices_)),  # type: ignore[union-attr]
        }

    def _require_fitted(self) -> None:
        if not self.fitted:
            raise AIValidationError("PreprocessPipeline must be fitted before use")


def create_preprocess_pipeline(
    config: AIConfig | None = None,
    selector: FeatureSelector | None = None,
    scaler: FeatureScaler | None = None,
) -> PreprocessPipeline:
    """Factory for preprocessing pipelines."""
    active_config = config or AIConfig()
    return PreprocessPipeline(config=active_config, selector=selector, scaler=scaler)
