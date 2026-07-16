"""
ai/optimization/hyperparams.py - Model hyperparameter search spaces.

VERSION: 1.0.0
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List


@dataclass(frozen=True)
class SearchSpace:
    """Serializable search space entry."""

    kind: str
    values: List[Any] | None = None
    low: float | int | None = None
    high: float | int | None = None
    log: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "kind": self.kind,
            "values": self.values,
            "low": self.low,
            "high": self.high,
            "log": self.log,
        }


DEFAULT_SEARCH_SPACES: Dict[str, Dict[str, SearchSpace]] = {
    "random_forest": {
        "n_estimators": SearchSpace("int", low=100, high=800),
        "max_depth": SearchSpace("choice", values=[3, 5, 8, 12, None]),
        "min_samples_leaf": SearchSpace("int", low=1, high=20),
        "max_features": SearchSpace("choice", values=["sqrt", "log2", None]),
    },
    "extra_trees": {
        "n_estimators": SearchSpace("int", low=100, high=800),
        "max_depth": SearchSpace("choice", values=[3, 5, 8, 12, None]),
        "min_samples_leaf": SearchSpace("int", low=1, high=20),
    },
    "lightgbm": {
        "n_estimators": SearchSpace("int", low=100, high=1200),
        "learning_rate": SearchSpace("float", low=0.005, high=0.2, log=True),
        "max_depth": SearchSpace("choice", values=[-1, 3, 5, 8, 12]),
        "num_leaves": SearchSpace("int", low=15, high=255),
        "subsample": SearchSpace("float", low=0.5, high=1.0),
        "colsample_bytree": SearchSpace("float", low=0.5, high=1.0),
    },
    "xgboost": {
        "n_estimators": SearchSpace("int", low=100, high=1200),
        "learning_rate": SearchSpace("float", low=0.005, high=0.2, log=True),
        "max_depth": SearchSpace("int", low=2, high=10),
        "subsample": SearchSpace("float", low=0.5, high=1.0),
        "colsample_bytree": SearchSpace("float", low=0.5, high=1.0),
        "reg_lambda": SearchSpace("float", low=1e-3, high=20.0, log=True),
    },
    "logistic_regression": {
        "C": SearchSpace("float", low=1e-3, high=100.0, log=True),
        "penalty": SearchSpace("choice", values=["l2"]),
    },
    "mlp": {
        "hidden_units": SearchSpace("choice", values=[32, 64, 128, 256]),
        "dropout": SearchSpace("float", low=0.0, high=0.5),
        "learning_rate": SearchSpace("float", low=1e-5, high=1e-2, log=True),
    },
    "lstm": {
        "lstm_units": SearchSpace("choice", values=[32, 64, 128]),
        "lstm_layers": SearchSpace("int", low=1, high=4),
        "dropout": SearchSpace("float", low=0.0, high=0.5),
        "learning_rate": SearchSpace("float", low=1e-5, high=1e-2, log=True),
    },
}


def get_search_space(model_type: str, include_aliases: bool = True) -> Dict[str, Dict[str, Any]]:
    """Return a serializable search space for a model type."""

    name = str(model_type).lower().strip()
    aliases = {
        "rf": "random_forest",
        "forest": "random_forest",
        "et": "extra_trees",
        "lgbm": "lightgbm",
        "xgb": "xgboost",
        "logistic": "logistic_regression",
    }
    if include_aliases:
        name = aliases.get(name, name)
    if name not in DEFAULT_SEARCH_SPACES:
        raise ValueError(f"No search space registered for model_type={model_type!r}")
    return {key: value.to_dict() for key, value in DEFAULT_SEARCH_SPACES[name].items()}


def list_search_spaces() -> List[str]:
    """List registered model types."""

    return sorted(DEFAULT_SEARCH_SPACES)


def merge_search_space(
    model_type: str,
    overrides: Dict[str, SearchSpace | Dict[str, Any]],
) -> Dict[str, Dict[str, Any]]:
    """Return a model search space with caller overrides applied."""

    merged = get_search_space(model_type)
    for key, value in overrides.items():
        if isinstance(value, SearchSpace):
            merged[key] = value.to_dict()
        else:
            merged[key] = dict(value)
    return merged


def grid_from_space(space: Dict[str, Dict[str, Any]], limit: int = 256) -> List[Dict[str, Any]]:
    """Build a bounded grid from choice-style search space entries."""

    grids: List[Dict[str, Any]] = [{}]
    for name, spec in space.items():
        values: Iterable[Any]
        if spec.get("kind") == "choice":
            values = spec.get("values") or []
        elif spec.get("kind") == "int":
            low, high = int(spec["low"]), int(spec["high"])
            values = sorted({low, (low + high) // 2, high})
        else:
            low, high = float(spec["low"]), float(spec["high"])
            values = sorted({low, (low + high) / 2.0, high})
        grids = [dict(existing, **{name: value}) for existing in grids for value in values]
        if len(grids) >= limit:
            return grids[:limit]
    return grids
