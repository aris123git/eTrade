"""
ai/models - Production model registry and factories.

RESPONSIBILITY:
Expose every model wrapper through a stable registry and create_model factory.

VERSION: 1.0.0
"""

from __future__ import annotations

from typing import Dict, List

from ai.config.settings import AIConfig
from ai.models.anomaly import ANOMALY_MODELS, IsolationForestModel
from ai.models.base import BaseModel, ModelTask
from ai.models.boosting import BOOSTING_MODELS, CatBoostModel, LightGBMModel, XGBoostModel
from ai.models.classical import (
    CLASSICAL_MODELS,
    DecisionTreeModel,
    ElasticNetModel,
    ExtraTreesModel,
    GradientBoostingModel,
    KNNModel,
    LinearRegressionModel,
    LogisticRegressionModel,
    NaiveBayesModel,
    RandomForestModel,
    SVMModel,
)
from ai.models.ensemble import (
    ENSEMBLE_MODELS,
    BaggingEnsemble,
    BlendingEnsemble,
    StackingEnsemble,
    VotingEnsemble,
)
from ai.models.neural import (
    NEURAL_MODELS,
    AutoEncoderModel,
    CNNLSTMModel,
    GRUModel,
    LSTMModel,
    MLPModel,
    TCNModel,
    TransformerModel,
)


# ==============================================================================
# REGISTRY
# ==============================================================================


MODEL_REGISTRY: Dict[str, type[BaseModel]] = {}
MODEL_REGISTRY.update(CLASSICAL_MODELS)
MODEL_REGISTRY.update(BOOSTING_MODELS)
MODEL_REGISTRY.update(NEURAL_MODELS)
MODEL_REGISTRY.update(ANOMALY_MODELS)
MODEL_REGISTRY.update(ENSEMBLE_MODELS)
MODEL_REGISTRY["ensemble"] = VotingEnsemble


def create_model(model_type: str, config: AIConfig) -> BaseModel:
    """Create a model instance by registry name."""
    name = str(model_type or config.model.model_type).lower().strip()
    if name == "ensemble":
        name = str(config.model.ensemble_method or "voting").lower().strip()
    if name not in MODEL_REGISTRY:
        available = ", ".join(sorted(MODEL_REGISTRY))
        raise ValueError(f"Unknown model_type '{model_type}'. Available models: {available}")

    cls = MODEL_REGISTRY[name]
    task = ModelTask.from_value(config.model.task)
    if cls in {VotingEnsemble, StackingEnsemble, BlendingEnsemble}:
        estimators = _create_ensemble_estimators(config, current_name=name)
        return cls(config=config, task=task, estimators=estimators)
    if cls is BaggingEnsemble:
        estimators = _create_ensemble_estimators(config, current_name=name)
        base_model = estimators[0] if estimators else RandomForestModel(config=config, task=task)
        return BaggingEnsemble(config=config, task=task, base_model=base_model)
    return cls(config=config, task=task)


def _create_ensemble_estimators(config: AIConfig, current_name: str) -> List[BaseModel]:
    names = [str(name).lower().strip() for name in config.model.ensemble_models]
    estimators: list[BaseModel] = []
    for name in names:
        if name in {"ensemble", current_name} or name in ENSEMBLE_MODELS:
            continue
        estimators.append(create_model(name, config))
    if estimators:
        return estimators
    return [
        RandomForestModel(config=config, task=config.model.task),
        LogisticRegressionModel(config=config, task=config.model.task),
    ]


__all__ = [
    "BaseModel",
    "ModelTask",
    "MODEL_REGISTRY",
    "create_model",
    "RandomForestModel",
    "ExtraTreesModel",
    "GradientBoostingModel",
    "LogisticRegressionModel",
    "LinearRegressionModel",
    "ElasticNetModel",
    "SVMModel",
    "KNNModel",
    "NaiveBayesModel",
    "DecisionTreeModel",
    "XGBoostModel",
    "LightGBMModel",
    "CatBoostModel",
    "MLPModel",
    "LSTMModel",
    "GRUModel",
    "TransformerModel",
    "TCNModel",
    "CNNLSTMModel",
    "AutoEncoderModel",
    "IsolationForestModel",
    "VotingEnsemble",
    "BaggingEnsemble",
    "StackingEnsemble",
    "BlendingEnsemble",
]
