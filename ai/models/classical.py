"""
ai/models/classical.py - Classical machine-learning model wrappers.

RESPONSIBILITY:
Expose sklearn-compatible estimators behind the BaseModel contract, with numpy
fallbacks for core production models when sklearn is not installed.

VERSION: 1.0.0
"""

from __future__ import annotations

from dataclasses import dataclass
from importlib import import_module
from typing import Any, Callable, Dict, Optional

import numpy as np
from numpy.typing import NDArray

from ai.config.settings import AIConfig
from ai.models.base import BaseModel, ModelTask, flatten_features, flatten_target


# ==============================================================================
# DEPENDENCY HELPERS
# ==============================================================================


def _sklearn_class(module: str, name: str) -> type:
    """Resolve a sklearn estimator or raise a targeted runtime error."""
    try:
        return getattr(import_module(module), name)
    except ModuleNotFoundError as exc:
        if exc.name == "sklearn" or str(exc.name).startswith("sklearn."):
            raise RuntimeError(
                f"{name} requires scikit-learn. Install scikit-learn to use this model."
            ) from exc
        raise


def _base_tree_params(config: AIConfig) -> Dict[str, Any]:
    return {
        "n_estimators": config.model.n_estimators,
        "max_depth": config.model.max_depth,
        "random_state": config.model.random_state,
        "n_jobs": config.model.n_jobs,
    }


def _boosting_params(config: AIConfig) -> Dict[str, Any]:
    return {
        "n_estimators": config.model.n_estimators,
        "max_depth": config.model.max_depth,
        "learning_rate": config.model.learning_rate,
        "random_state": config.model.random_state,
    }


# ==============================================================================
# NUMPY DECISION TREE FALLBACK
# ==============================================================================


@dataclass
class _TreeNode:
    feature: int = -1
    threshold: float = 0.0
    value: NDArray[np.floating] | float | int | None = None
    left: Optional["_TreeNode"] = None
    right: Optional["_TreeNode"] = None

    @property
    def is_leaf(self) -> bool:
        return self.value is not None


class _NumpyDecisionTree:
    """Small CART-style tree used by numpy fallbacks."""

    def __init__(
        self,
        task: ModelTask,
        max_depth: int = 6,
        min_samples_split: int = 4,
        max_features: int | None = None,
        random_state: int | None = None,
        classes: NDArray[np.generic] | None = None,
    ) -> None:
        self.task = task
        self.max_depth = max(1, int(max_depth))
        self.min_samples_split = max(2, int(min_samples_split))
        self.max_features = max_features
        self.rng = np.random.default_rng(random_state)
        self.root: _TreeNode | None = None
        self.classes_: NDArray[np.generic] | None = classes
        self.importances_: NDArray[np.floating] | None = None

    def fit(self, X: NDArray[np.floating], y: NDArray[np.floating]) -> "_NumpyDecisionTree":
        x = flatten_features(X)
        target = flatten_target(y)
        self.importances_ = np.zeros(x.shape[1], dtype=float)
        if self.task == ModelTask.CLASSIFICATION and self.classes_ is None:
            self.classes_ = np.unique(target)
        self.root = self._build(x, target, depth=0)
        total = float(np.sum(self.importances_))
        if total > 0.0:
            self.importances_ = self.importances_ / total
        return self

    def predict(self, X: NDArray[np.floating]) -> NDArray[np.floating]:
        if self.root is None:
            raise RuntimeError("Numpy decision tree must be fitted before prediction")
        x = flatten_features(X)
        values = [self._predict_row(row, self.root) for row in x]
        if self.task == ModelTask.CLASSIFICATION:
            if self.classes_ is None:
                raise RuntimeError("Numpy decision tree has no fitted classes")
            probs = np.vstack(values).astype(float)
            return self.classes_[np.argmax(probs, axis=1)]
        return np.asarray(values)

    def predict_proba(self, X: NDArray[np.floating]) -> NDArray[np.floating]:
        if self.task != ModelTask.CLASSIFICATION:
            raise RuntimeError("Probabilities are only available for classification trees")
        if self.root is None or self.classes_ is None:
            raise RuntimeError("Numpy decision tree must be fitted before probability prediction")
        x = flatten_features(X)
        values = [self._predict_row(row, self.root) for row in x]
        return np.vstack(values).astype(float)

    def _build(self, X: NDArray[np.floating], y: NDArray[np.floating], depth: int) -> _TreeNode:
        if (
            depth >= self.max_depth
            or len(X) < self.min_samples_split
            or len(np.unique(y)) <= 1
        ):
            return _TreeNode(value=self._leaf_value(y))

        split = self._best_split(X, y)
        if split is None:
            return _TreeNode(value=self._leaf_value(y))

        feature, threshold, gain = split
        mask = X[:, feature] <= threshold
        if self.importances_ is not None:
            self.importances_[feature] += max(float(gain), 0.0)
        return _TreeNode(
            feature=feature,
            threshold=threshold,
            left=self._build(X[mask], y[mask], depth + 1),
            right=self._build(X[~mask], y[~mask], depth + 1),
        )

    def _best_split(self, X: NDArray[np.floating], y: NDArray[np.floating]) -> tuple[int, float, float] | None:
        n_features = X.shape[1]
        feature_count = self.max_features or n_features
        feature_count = max(1, min(feature_count, n_features))
        features = self.rng.choice(n_features, size=feature_count, replace=False)
        parent_score = self._impurity(y)
        best_gain = 0.0
        best: tuple[int, float, float] | None = None

        for feature in features:
            values = X[:, int(feature)]
            finite = values[np.isfinite(values)]
            if finite.size <= 1:
                continue
            quantiles = np.linspace(10.0, 90.0, num=9)
            thresholds = np.unique(np.percentile(finite, quantiles))
            for raw_threshold in thresholds:
                threshold = float(raw_threshold)
                left_mask = values <= threshold
                right_mask = ~left_mask
                if not left_mask.any() or not right_mask.any():
                    continue
                left_score = self._impurity(y[left_mask])
                right_score = self._impurity(y[right_mask])
                weight_left = float(np.mean(left_mask))
                weighted = weight_left * left_score + (1.0 - weight_left) * right_score
                gain = parent_score - weighted
                if gain > best_gain:
                    best_gain = float(gain)
                    best = (int(feature), threshold, best_gain)
        return best

    def _impurity(self, y: NDArray[np.floating]) -> float:
        if len(y) == 0:
            return 0.0
        if self.task == ModelTask.CLASSIFICATION:
            _, counts = np.unique(y, return_counts=True)
            probs = counts.astype(float) / float(len(y))
            return float(1.0 - np.sum(probs * probs))
        mean = float(np.mean(y))
        return float(np.mean((y.astype(float) - mean) ** 2))

    def _leaf_value(self, y: NDArray[np.floating]) -> NDArray[np.floating] | float:
        if self.task == ModelTask.CLASSIFICATION:
            if self.classes_ is None:
                self.classes_ = np.unique(y)
            counts = np.array([np.sum(y == label) for label in self.classes_], dtype=float)
            total = float(np.sum(counts))
            if total == 0.0:
                counts[:] = 1.0 / float(len(counts))
            else:
                counts = counts / total
            return counts
        return float(np.mean(y.astype(float))) if len(y) else 0.0

    def _predict_row(self, row: NDArray[np.floating], node: _TreeNode) -> Any:
        if node.is_leaf:
            if self.task == ModelTask.CLASSIFICATION:
                return np.asarray(node.value, dtype=float)
            return float(node.value)
        child = node.left if row[node.feature] <= node.threshold else node.right
        if child is None:
            return node.value
        return self._predict_row(row, child)


# ==============================================================================
# NUMPY LINEAR FALLBACKS
# ==============================================================================


class _NumpyLinearRegressor:
    """Closed-form ridge-style linear regressor."""

    def __init__(self, reg_lambda: float = 1e-8) -> None:
        self.reg_lambda = float(reg_lambda)
        self.coef_: NDArray[np.floating] | None = None

    def fit(self, X: NDArray[np.floating], y: NDArray[np.floating]) -> "_NumpyLinearRegressor":
        x = flatten_features(X)
        target = flatten_target(y).astype(float)
        design = np.c_[np.ones(len(x)), x]
        reg = np.eye(design.shape[1]) * self.reg_lambda
        reg[0, 0] = 0.0
        self.coef_ = np.linalg.pinv(design.T @ design + reg) @ design.T @ target
        return self

    def predict(self, X: NDArray[np.floating]) -> NDArray[np.floating]:
        if self.coef_ is None:
            raise RuntimeError("Numpy linear regressor must be fitted before prediction")
        x = flatten_features(X)
        design = np.c_[np.ones(len(x)), x]
        return np.asarray(design @ self.coef_, dtype=float)


class _NumpyLogisticRegression:
    """Softmax logistic regression trained with gradient descent."""

    def __init__(
        self,
        learning_rate: float = 0.05,
        max_iter: int = 300,
        reg_lambda: float = 1e-4,
        random_state: int = 42,
    ) -> None:
        self.learning_rate = float(learning_rate)
        self.max_iter = int(max_iter)
        self.reg_lambda = float(reg_lambda)
        self.rng = np.random.default_rng(random_state)
        self.classes_: NDArray[np.generic] | None = None
        self.weights_: NDArray[np.floating] | None = None

    def fit(self, X: NDArray[np.floating], y: NDArray[np.floating]) -> "_NumpyLogisticRegression":
        x = flatten_features(X)
        target = flatten_target(y)
        self.classes_, encoded = np.unique(target, return_inverse=True)
        class_count = len(self.classes_)
        design = np.c_[np.ones(len(x)), x]
        self.weights_ = self.rng.normal(0.0, 0.01, size=(design.shape[1], class_count))
        y_one_hot = np.eye(class_count)[encoded]
        for _ in range(self.max_iter):
            probs = self._softmax(design @ self.weights_)
            grad = design.T @ (probs - y_one_hot) / float(len(design))
            grad[1:] += self.reg_lambda * self.weights_[1:]
            self.weights_ -= self.learning_rate * grad
        return self

    def predict(self, X: NDArray[np.floating]) -> NDArray[np.floating]:
        probs = self.predict_proba(X)
        if self.classes_ is None:
            raise RuntimeError("Numpy logistic regression has no fitted classes")
        return self.classes_[np.argmax(probs, axis=1)]

    def predict_proba(self, X: NDArray[np.floating]) -> NDArray[np.floating]:
        if self.weights_ is None:
            raise RuntimeError("Numpy logistic regression must be fitted before prediction")
        x = flatten_features(X)
        design = np.c_[np.ones(len(x)), x]
        return self._softmax(design @ self.weights_)

    @staticmethod
    def _softmax(scores: NDArray[np.floating]) -> NDArray[np.floating]:
        stable = scores - np.max(scores, axis=1, keepdims=True)
        exp = np.exp(stable)
        denom = np.sum(exp, axis=1, keepdims=True)
        denom[denom == 0.0] = 1.0
        return exp / denom


class _NumpyRandomForest:
    """Bootstrap ensemble of compact numpy decision trees."""

    def __init__(
        self,
        task: ModelTask,
        n_estimators: int = 50,
        max_depth: int = 6,
        random_state: int = 42,
    ) -> None:
        self.task = task
        self.n_estimators = max(1, int(n_estimators))
        self.max_depth = max(1, int(max_depth))
        self.rng = np.random.default_rng(random_state)
        self.trees: list[_NumpyDecisionTree] = []
        self.classes_: NDArray[np.generic] | None = None
        self.importances_: NDArray[np.floating] | None = None

    def fit(self, X: NDArray[np.floating], y: NDArray[np.floating]) -> "_NumpyRandomForest":
        x = flatten_features(X)
        target = flatten_target(y)
        if self.task == ModelTask.CLASSIFICATION:
            self.classes_ = np.unique(target)
        max_features = max(1, int(np.sqrt(x.shape[1])))
        self.trees = []
        importances = np.zeros(x.shape[1], dtype=float)
        for idx in range(self.n_estimators):
            sample_idx = self.rng.integers(0, len(x), size=len(x))
            tree = _NumpyDecisionTree(
                task=self.task,
                max_depth=self.max_depth,
                min_samples_split=4,
                max_features=max_features,
                random_state=int(self.rng.integers(0, 2**31 - 1)),
                classes=self.classes_,
            )
            tree.fit(x[sample_idx], target[sample_idx])
            self.trees.append(tree)
            if tree.importances_ is not None:
                importances += tree.importances_
        total = float(np.sum(importances))
        self.importances_ = importances / total if total > 0.0 else importances
        return self

    def predict(self, X: NDArray[np.floating]) -> NDArray[np.floating]:
        if not self.trees:
            raise RuntimeError("Numpy random forest must be fitted before prediction")
        if self.task == ModelTask.CLASSIFICATION:
            probs = self.predict_proba(X)
            if self.classes_ is None:
                raise RuntimeError("Numpy random forest has no fitted classes")
            return self.classes_[np.argmax(probs, axis=1)]
        preds = np.vstack([tree.predict(X).astype(float) for tree in self.trees])
        return np.mean(preds, axis=0)

    def predict_proba(self, X: NDArray[np.floating]) -> NDArray[np.floating]:
        if self.task != ModelTask.CLASSIFICATION:
            raise RuntimeError("Probabilities are only available for classification forests")
        if not self.trees:
            raise RuntimeError("Numpy random forest must be fitted before probability prediction")
        probs = [tree.predict_proba(X) for tree in self.trees]
        return np.mean(np.stack(probs, axis=0), axis=0)


# ==============================================================================
# WRAPPERS
# ==============================================================================


class _EstimatorBackedModel(BaseModel):
    """Base wrapper for sklearn estimators and numpy substitutes."""

    estimator_: Any = None

    def _make_estimator(self) -> Any:
        raise RuntimeError("Concrete model must provide an estimator factory")

    def fit(
        self,
        X: NDArray[np.floating],
        y: NDArray[np.floating],
        X_val: NDArray[np.floating] | None = None,
        y_val: NDArray[np.floating] | None = None,
    ) -> BaseModel:
        estimator = self._make_estimator()
        estimator.fit(flatten_features(X), flatten_target(y))
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
        if values is None:
            values = getattr(self.estimator_, "importances_", None)
        return None if values is None else np.asarray(values, dtype=float)


class RandomForestModel(_EstimatorBackedModel):
    """Random forest wrapper with a numpy fallback."""

    def _make_estimator(self) -> Any:
        params = _base_tree_params(self.config)
        params.update(self.params)
        try:
            if self.task == ModelTask.CLASSIFICATION:
                cls = _sklearn_class("sklearn.ensemble", "RandomForestClassifier")
            else:
                cls = _sklearn_class("sklearn.ensemble", "RandomForestRegressor")
            return cls(**params)
        except RuntimeError:
            return _NumpyRandomForest(
                task=ModelTask.from_value(self.task),
                n_estimators=int(params.get("n_estimators", 50)),
                max_depth=int(params.get("max_depth", 6) or 6),
                random_state=int(params.get("random_state", 42)),
            )


class ExtraTreesModel(_EstimatorBackedModel):
    """Extremely randomized trees wrapper."""

    def _make_estimator(self) -> Any:
        params = _base_tree_params(self.config)
        params.update(self.params)
        cls_name = "ExtraTreesClassifier" if self.task == ModelTask.CLASSIFICATION else "ExtraTreesRegressor"
        return _sklearn_class("sklearn.ensemble", cls_name)(**params)


class GradientBoostingModel(_EstimatorBackedModel):
    """Gradient boosting wrapper."""

    def _make_estimator(self) -> Any:
        params = _boosting_params(self.config)
        params.update(self.params)
        params.pop("n_jobs", None)
        cls_name = (
            "GradientBoostingClassifier"
            if self.task == ModelTask.CLASSIFICATION
            else "GradientBoostingRegressor"
        )
        return _sklearn_class("sklearn.ensemble", cls_name)(**params)


class LogisticRegressionModel(_EstimatorBackedModel):
    """Logistic regression wrapper with a numpy fallback."""

    def _make_estimator(self) -> Any:
        params = {
            "max_iter": int(self.params.get("max_iter", 1000)),
            "random_state": self.config.model.random_state,
            "n_jobs": self.config.model.n_jobs,
        }
        params.update(self.params)
        try:
            return _sklearn_class("sklearn.linear_model", "LogisticRegression")(**params)
        except RuntimeError:
            return _NumpyLogisticRegression(
                learning_rate=float(params.get("learning_rate", self.config.model.learning_rate)),
                max_iter=int(params.get("max_iter", 300)),
                reg_lambda=float(params.get("reg_lambda", self.config.model.reg_lambda)),
                random_state=int(params.get("random_state", self.config.model.random_state)),
            )


class LinearRegressionModel(_EstimatorBackedModel):
    """Linear regression wrapper with a numpy fallback."""

    def _make_estimator(self) -> Any:
        params = dict(self.params)
        try:
            return _sklearn_class("sklearn.linear_model", "LinearRegression")(**params)
        except RuntimeError:
            return _NumpyLinearRegressor(reg_lambda=float(params.get("reg_lambda", 1e-8)))


class ElasticNetModel(_EstimatorBackedModel):
    """ElasticNet regression wrapper."""

    def _make_estimator(self) -> Any:
        params = {
            "alpha": float(self.params.get("alpha", self.config.model.reg_lambda)),
            "l1_ratio": float(self.params.get("l1_ratio", 0.5)),
            "random_state": self.config.model.random_state,
        }
        params.update(self.params)
        return _sklearn_class("sklearn.linear_model", "ElasticNet")(**params)


class SVMModel(_EstimatorBackedModel):
    """Support-vector machine wrapper."""

    def _make_estimator(self) -> Any:
        params = {"probability": self.task == ModelTask.CLASSIFICATION}
        params.update(self.params)
        cls_name = "SVC" if self.task == ModelTask.CLASSIFICATION else "SVR"
        return _sklearn_class("sklearn.svm", cls_name)(**params)


class KNNModel(_EstimatorBackedModel):
    """K-nearest-neighbors wrapper."""

    def _make_estimator(self) -> Any:
        params = {"n_neighbors": int(self.params.get("n_neighbors", 5))}
        params.update(self.params)
        cls_name = "KNeighborsClassifier" if self.task == ModelTask.CLASSIFICATION else "KNeighborsRegressor"
        return _sklearn_class("sklearn.neighbors", cls_name)(**params)


class NaiveBayesModel(_EstimatorBackedModel):
    """Gaussian naive Bayes classifier wrapper."""

    def _make_estimator(self) -> Any:
        if self.task != ModelTask.CLASSIFICATION:
            raise RuntimeError("NaiveBayesModel supports classification only")
        return _sklearn_class("sklearn.naive_bayes", "GaussianNB")(**self.params)


class DecisionTreeModel(_EstimatorBackedModel):
    """Decision tree wrapper with a compact numpy fallback."""

    def _make_estimator(self) -> Any:
        params = {
            "max_depth": self.config.model.max_depth,
            "random_state": self.config.model.random_state,
        }
        params.update(self.params)
        try:
            cls_name = "DecisionTreeClassifier" if self.task == ModelTask.CLASSIFICATION else "DecisionTreeRegressor"
            return _sklearn_class("sklearn.tree", cls_name)(**params)
        except RuntimeError:
            return _NumpyDecisionTree(
                task=ModelTask.from_value(self.task),
                max_depth=int(params.get("max_depth", 6) or 6),
                random_state=int(params.get("random_state", 42)),
            )


CLASSICAL_MODELS: Dict[str, type[BaseModel]] = {
    "random_forest": RandomForestModel,
    "rf": RandomForestModel,
    "extra_trees": ExtraTreesModel,
    "extratrees": ExtraTreesModel,
    "gradient_boosting": GradientBoostingModel,
    "gbm": GradientBoostingModel,
    "logistic_regression": LogisticRegressionModel,
    "logreg": LogisticRegressionModel,
    "linear_regression": LinearRegressionModel,
    "linear": LinearRegressionModel,
    "elastic_net": ElasticNetModel,
    "elasticnet": ElasticNetModel,
    "svm": SVMModel,
    "svc": SVMModel,
    "svr": SVMModel,
    "knn": KNNModel,
    "naive_bayes": NaiveBayesModel,
    "nb": NaiveBayesModel,
    "decision_tree": DecisionTreeModel,
    "tree": DecisionTreeModel,
}
