"""
ai/datasets/schema.py - Dataset container contracts.

VERSION: 1.0.0
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from datetime import datetime
from typing import Any, Dict, List, Optional

import numpy as np
from numpy.typing import NDArray


ArrayLike = NDArray[np.floating] | np.memmap
TimestampList = List[datetime]


@dataclass(frozen=True)
class DatasetBundle:
    """
    Chronologically split dataset arrays plus enough context to reproduce them.

    X arrays are 2D for tabular models and 3D for sequence models. y arrays may
    be 1D or 2D depending on the label generator output.
    """

    X_train: ArrayLike
    y_train: ArrayLike
    X_val: ArrayLike
    y_val: ArrayLike
    X_test: ArrayLike
    y_test: ArrayLike
    feature_names: List[str]
    timestamps: TimestampList
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def n_train(self) -> int:
        return int(len(self.X_train))

    @property
    def n_val(self) -> int:
        return int(len(self.X_val))

    @property
    def n_test(self) -> int:
        return int(len(self.X_test))

    @property
    def n_samples(self) -> int:
        return self.n_train + self.n_val + self.n_test

    @property
    def n_features(self) -> int:
        if self.X_train.ndim == 3:
            return int(self.X_train.shape[-1])
        if self.X_train.ndim == 2:
            return int(self.X_train.shape[1])
        return 0

    @property
    def is_sequence(self) -> bool:
        return bool(self.X_train.ndim == 3)

    @property
    def has_test(self) -> bool:
        return self.n_test > 0

    @property
    def train_timestamps(self) -> TimestampList:
        return list(self.metadata.get("train_timestamps", []))

    @property
    def val_timestamps(self) -> TimestampList:
        return list(self.metadata.get("val_timestamps", []))

    @property
    def test_timestamps(self) -> TimestampList:
        return list(self.metadata.get("test_timestamps", []))

    def with_metadata(self, **updates: Any) -> "DatasetBundle":
        metadata = dict(self.metadata)
        metadata.update(updates)
        return replace(self, metadata=metadata)

    def summary(self) -> Dict[str, Any]:
        return {
            "n_samples": self.n_samples,
            "n_train": self.n_train,
            "n_val": self.n_val,
            "n_test": self.n_test,
            "n_features": self.n_features,
            "is_sequence": self.is_sequence,
            "feature_count": len(self.feature_names),
            "start": self.timestamps[0].isoformat() if self.timestamps else None,
            "end": self.timestamps[-1].isoformat() if self.timestamps else None,
            "metadata": {
                key: value
                for key, value in self.metadata.items()
                if key not in {"train_timestamps", "val_timestamps", "test_timestamps"}
            },
        }


def empty_array(shape: tuple[int, ...], dtype: np.dtype[Any] | type = float) -> NDArray[np.floating]:
    return np.empty(shape, dtype=dtype)


def empty_bundle(feature_names: Optional[List[str]] = None) -> DatasetBundle:
    features = feature_names or []
    return DatasetBundle(
        X_train=empty_array((0, len(features))),
        y_train=empty_array((0,)),
        X_val=empty_array((0, len(features))),
        y_val=empty_array((0,)),
        X_test=empty_array((0, len(features))),
        y_test=empty_array((0,)),
        feature_names=features,
        timestamps=[],
        metadata={},
    )
