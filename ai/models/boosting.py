"""
ai/models/boosting.py - Optional gradient boosting integrations.

RESPONSIBILITY:
Wrap XGBoost, LightGBM, and CatBoost behind the shared BaseModel interface.

VERSION: 1.0.0
"""

from __future__ import annotations

from importlib import import_module
from typing import Any, Dict, Optional

import numpy as np
from numpy.typing import NDArray

from ai.models.base import BaseModel, ModelTask, flatten_features, flatten_target


# ==============================================================================
# OPTIONAL BOOSTERS
# ==============================================================================


class _OptionalBoostingModel(BaseModel):
    """Base class for optional third-party boosting libraries."""

    package_name: str = ""
    install_name: str = ""
    classifier_name: str = ""
    regressor_name: str = ""
    estimator_: Any = None

    def _import_package(self) -> Any:
        try:
            return import_module(self.package_name)
        except ModuleNotFoundError as exc:
            if exc.name == self.package_name:
                raise ImportError(
                    f"{self.__class__.__name__} requires {self.install_name}. "
                    f"Install {self.install_name} to use model_type='{self.package_name}'."
                ) from exc
            raise

    def _estimator_params(self) -> Dict[str, Any]:
        params: Dict[str, Any] = {
            "n_estimators": self.config.model.n_estimators,
            "max_depth": self.config.model.max_depth,
            "learning_rate": self.config.model.learning_rate,
            "random_state": self.config.model.random_state,
        }
        params.update(self.params)
        return params

    def _make_estimator(self) -> Any:
        package = self._import_package()
        class_name = self.classifier_name if self.task == ModelTask.CLASSIFICATION else self.regressor_name
        estimator_cls = getattr(package, class_name)
        return estimator_cls(**self._estimator_params())

    def fit(
        self,
        X: NDArray[np.floating],
        y: NDArray[np.floating],
        X_val: NDArray[np.floating] | None = None,
        y_val: NDArray[np.floating] | None = None,
    ) -> BaseModel:
        estimator = self._make_estimator()
        fit_kwargs: Dict[str, Any] = {}
        if X_val is not None and y_val is not None and self.package_name != "catboost":
            fit_kwargs["eval_set"] = [(flatten_features(X_val), flatten_target(y_val))]
        if self.package_name == "catboost":
            fit_kwargs["verbose"] = bool(self.params.get("verbose", False))
            if X_val is not None and y_val is not None:
                fit_kwargs["eval_set"] = (flatten_features(X_val), flatten_target(y_val))
        estimator.fit(flatten_features(X), flatten_target(y), **fit_kwargs)
        self.estimator_ = estimator
        return self

    def predict(self, X: NDArray[np.floating]) -> NDArray[np.floating]:
        if self.estimator_ is None:
            raise RuntimeError(f"{self.__class__.__name__} must be fitted before prediction")
        return np.asarray(self.estimator_.predict(flatten_features(X)))

    def predict_proba(self, X: NDArray[np.floating]) -> Optional[NDArray[np.floating]]:
        if self.task != ModelTask.CLASSIFICATION:
            return None
        if self.estimator_ is None:
            raise RuntimeError(f"{self.__class__.__name__} must be fitted before probability prediction")
        if hasattr(self.estimator_, "predict_proba"):
            return np.asarray(self.estimator_.predict_proba(flatten_features(X)), dtype=float)
        return None

    @property
    def feature_importances_(self) -> Optional[NDArray[np.floating]]:
        if self.estimator_ is None:
            return None
        values = getattr(self.estimator_, "feature_importances_", None)
        return None if values is None else np.asarray(values, dtype=float)


class XGBoostModel(_OptionalBoostingModel):
    """XGBoost classifier/regressor wrapper."""

    package_name = "xgboost"
    install_name = "xgboost"
    classifier_name = "XGBClassifier"
    regressor_name = "XGBRegressor"

    def _estimator_params(self) -> Dict[str, Any]:
        params = super()._estimator_params()
        params.setdefault("subsample", self.config.model.subsample)
        params.setdefault("colsample_bytree", self.config.model.colsample_bytree)
        params.setdefault("reg_alpha", self.config.model.reg_alpha)
        params.setdefault("reg_lambda", self.config.model.reg_lambda)
        params.setdefault("n_jobs", self.config.model.n_jobs)
        params.setdefault("eval_metric", "logloss" if self.task == ModelTask.CLASSIFICATION else "rmse")
        return params


class LightGBMModel(_OptionalBoostingModel):
    """LightGBM classifier/regressor wrapper."""

    package_name = "lightgbm"
    install_name = "lightgbm"
    classifier_name = "LGBMClassifier"
    regressor_name = "LGBMRegressor"

    def _estimator_params(self) -> Dict[str, Any]:
        params = super()._estimator_params()
        params.setdefault("subsample", self.config.model.subsample)
        params.setdefault("colsample_bytree", self.config.model.colsample_bytree)
        params.setdefault("reg_alpha", self.config.model.reg_alpha)
        params.setdefault("reg_lambda", self.config.model.reg_lambda)
        params.setdefault("n_jobs", self.config.model.n_jobs)
        return params


class CatBoostModel(_OptionalBoostingModel):
    """CatBoost classifier/regressor wrapper."""

    package_name = "catboost"
    install_name = "catboost"
    classifier_name = "CatBoostClassifier"
    regressor_name = "CatBoostRegressor"

    def _estimator_params(self) -> Dict[str, Any]:
        params = super()._estimator_params()
        params.setdefault("depth", params.pop("max_depth", self.config.model.max_depth))
        params.setdefault("l2_leaf_reg", self.config.model.reg_lambda)
        params.setdefault("verbose", False)
        return params


BOOSTING_MODELS: Dict[str, type[BaseModel]] = {
    "xgboost": XGBoostModel,
    "xgb": XGBoostModel,
    "lightgbm": LightGBMModel,
    "lgbm": LightGBMModel,
    "catboost": CatBoostModel,
    "cat": CatBoostModel,
}
