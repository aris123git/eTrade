"""
ai/services/pipeline.py - End-to-end AI pipeline orchestration.

VERSION: 1.0.0
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Sequence

import numpy as np
from numpy.typing import NDArray

from ai.config.settings import AIConfig
from ai.datasets.schema import DatasetBundle
from ai.features.engine import FeatureEngine, FeatureFrame, normalize_timestamp
from ai.labels.base import LabelResult
from ai.labels.generator import LabelGenerator
from ai.models import create_model
from ai.models.base import BaseModel, flatten_target
from ai.monitoring.tracker import PerformanceTracker
from ai.storage.registry import ModelRegistry, RegisteredModel
from ai.training.trainer import TrainResult, Trainer
from ai.training.validation import default_metrics
from ai.utils.types import CandleDict, PredictionResult, SignalType


@dataclass(frozen=True)
class PipelineDataset:
    """Feature, label, and split bundle produced by the pipeline."""

    features: FeatureFrame
    label: LabelResult
    bundle: DatasetBundle


@dataclass(frozen=True)
class PipelineRunResult:
    """Full training pipeline output."""

    dataset: PipelineDataset
    train: TrainResult
    evaluation: Dict[str, float]
    registration: RegisteredModel | None


@dataclass
class AIPipeline:
    """Main integration surface for data, training, serving, signals, and risk."""

    config: AIConfig = field(default_factory=AIConfig)
    candle_source: Any = None
    feature_engine: FeatureEngine | None = None
    label_generator: LabelGenerator | None = None
    trainer: Trainer | None = None
    registry: ModelRegistry | None = None
    tracker: PerformanceTracker | None = None
    model: BaseModel | None = None
    model_version: str | None = None

    def __post_init__(self) -> None:
        self.feature_engine = self.feature_engine or FeatureEngine(config=self.config)
        self.label_generator = self.label_generator or LabelGenerator(config=self.config)
        self.trainer = self.trainer or Trainer(config=self.config)
        self.registry = self.registry or ModelRegistry(config=self.config)
        self.tracker = self.tracker or PerformanceTracker(config=self.config)

    def load_candles(
        self,
        symbol: str | None = None,
        timeframe: str | None = None,
        start: datetime | None = None,
        end: datetime | None = None,
        limit: int = 500,
    ) -> List[CandleDict]:
        """Load candles from the injected source."""

        if self.candle_source is None:
            raise RuntimeError("A candle_source is required to load candles")
        active_symbol = symbol or (self.config.symbols[0] if self.config.symbols else "")
        active_timeframe = timeframe or self.config.primary_timeframe
        if start is not None and end is not None and hasattr(self.candle_source, "stream_candles"):
            return list(
                self.candle_source.stream_candles(
                    active_symbol,
                    active_timeframe,
                    start,
                    end,
                    batch_size=limit,
                    order="ASC",
                )
            )
        if hasattr(self.candle_source, "get_last_n"):
            return list(self.candle_source.get_last_n(active_symbol, active_timeframe, n=limit))
        raise RuntimeError("candle_source must expose get_last_n or stream_candles")

    def build_features(self, candles: Sequence[CandleDict]) -> FeatureFrame:
        """Transform raw candles into model-ready features."""

        assert self.feature_engine is not None
        return self.feature_engine.transform(candles, self.config)

    def build_labels(self, candles: Sequence[CandleDict]) -> Dict[str, LabelResult]:
        """Generate configured labels from base candles."""

        assert self.label_generator is not None
        return self.label_generator.generate(self._base_candles(candles), self.config)

    def build_dataset(self, candles: Sequence[CandleDict], label_name: str | None = None) -> PipelineDataset:
        """Build an aligned DatasetBundle for supervised training."""

        base_candles = self._base_candles(candles)
        features = self.build_features(base_candles)
        labels = self.build_labels(base_candles)
        label = self._select_label(labels, label_name)
        X, y, timestamps = self._align_features_labels(features, label, base_candles)
        if len(X) == 0:
            raise ValueError("No valid aligned feature/label rows were produced")
        train_end, val_end = self._split_points(len(X))
        metadata = {
            "label": label.name,
            "label_method": label.method,
            "label_horizon": int(label.horizon),
            "train_timestamps": timestamps[:train_end],
            "val_timestamps": timestamps[train_end:val_end],
            "test_timestamps": timestamps[val_end:],
        }
        bundle = DatasetBundle(
            X_train=X[:train_end],
            y_train=y[:train_end],
            X_val=X[train_end:val_end],
            y_val=y[train_end:val_end],
            X_test=X[val_end:],
            y_test=y[val_end:],
            feature_names=list(features.feature_names),
            timestamps=timestamps,
            metadata=metadata,
        )
        return PipelineDataset(features=features, label=label, bundle=bundle)

    def train(
        self,
        dataset: DatasetBundle,
        model: BaseModel | None = None,
    ) -> TrainResult:
        """Train a configured model on a dataset bundle."""

        active_model = model or create_model(self.config.model.model_type, self.config)
        assert self.trainer is not None
        result = self.trainer.fit(active_model, dataset, self.config)
        self.model = result.model
        return result

    def evaluate(
        self,
        model: BaseModel | None,
        dataset: DatasetBundle,
    ) -> Dict[str, float]:
        """Evaluate a model on validation/test data."""

        active_model = model or self.model
        if active_model is None:
            raise RuntimeError("No model is available for evaluation")
        metrics: Dict[str, float] = {}
        if dataset.n_val:
            metrics.update({f"val_{key}": value for key, value in self._metrics(active_model, dataset.X_val, dataset.y_val).items()})
        if dataset.n_test:
            metrics.update({f"test_{key}": value for key, value in self._metrics(active_model, dataset.X_test, dataset.y_test).items()})
        return metrics

    def register(
        self,
        model: BaseModel | None = None,
        metrics: Dict[str, float] | None = None,
        name: str | None = None,
        features: Sequence[str] | None = None,
    ) -> RegisteredModel:
        """Register a trained model artifact."""

        active_model = model or self.model
        if active_model is None:
            raise RuntimeError("No model is available for registration")
        assert self.registry is not None
        model_name = name or self.config.model.model_type
        registered = self.registry.register(
            name=model_name,
            model=active_model,
            features=features,
            metrics=metrics or {},
            params=active_model.get_params(),
            metadata={"project": self.config.project_name, "registered_at": datetime.now(timezone.utc).isoformat()},
        )
        self.model_version = registered.version
        return registered

    def predict(
        self,
        candles: Sequence[CandleDict] | None = None,
        features: NDArray[np.floating] | Sequence[Sequence[float]] | None = None,
        model: BaseModel | None = None,
        symbol: str | None = None,
        timeframe: str | None = None,
    ) -> PredictionResult:
        """Run model prediction and return a standard prediction payload."""

        active_model = model or self.model
        if active_model is None:
            active_model = self._load_latest_model()
        if features is None:
            if candles is None:
                candles = self.load_candles(symbol=symbol, timeframe=timeframe)
            frame = self.build_features(candles)
            matrix = frame.matrix[-1:].astype(float)
            timestamp = frame.timestamps[-1] if frame.timestamps else datetime.now(timezone.utc)
            feature_names = frame.feature_names
        else:
            matrix = np.asarray(features, dtype=float)
            if matrix.ndim == 1:
                matrix = matrix.reshape(1, -1)
            timestamp = datetime.now(timezone.utc)
            feature_names = []
        prediction_values = np.asarray(active_model.predict(matrix)).reshape(-1)
        prediction = prediction_values[0].item() if hasattr(prediction_values[0], "item") else prediction_values[0]
        probabilities = self._probabilities(active_model, matrix)
        confidence = max(probabilities.values()) if probabilities else min(1.0, abs(float(prediction)))
        expected_return = float(prediction) if self.config.model.task == "regression" else None
        result = PredictionResult(
            symbol=symbol or (self.config.symbols[0] if self.config.symbols else ""),
            timeframe=timeframe or self.config.primary_timeframe,
            timestamp=timestamp,
            prediction=prediction,
            probabilities=probabilities,
            confidence=float(confidence),
            expected_return=expected_return,
            model_version=self.model_version,
            metadata={"feature_count": len(feature_names)},
        )
        assert self.tracker is not None
        self.tracker.record_prediction(float(prediction) if _is_number(prediction) else 0.0)
        return result

    def create_signal(self, prediction: PredictionResult) -> Dict[str, Any]:
        """Convert a prediction into a trading signal dictionary."""

        confidence_ok = float(prediction.confidence) >= float(self.config.risk.min_confidence)
        value = float(prediction.prediction) if _is_number(prediction.prediction) else 0.0
        if not confidence_ok:
            signal = SignalType.HOLD.value
        elif self.config.model.task == "classification":
            signal = SignalType.BUY.value if value > 0.0 else SignalType.SELL.value if value < 0.0 else SignalType.HOLD.value
        else:
            threshold = float(self.config.labels.binary_threshold)
            signal = SignalType.BUY.value if value > threshold else SignalType.SELL.value if value < -threshold else SignalType.HOLD.value
        return {
            "symbol": prediction.symbol,
            "timeframe": prediction.timeframe,
            "timestamp": prediction.timestamp,
            "signal": signal,
            "confidence": prediction.confidence,
            "expected_return": prediction.expected_return,
            "metadata": {"model_version": prediction.model_version},
        }

    def apply_risk(self, signal: Dict[str, Any], equity: float | None = None) -> Dict[str, Any]:
        """Attach risk sizing and execution guards to a signal."""

        account_equity = float(equity or 10_000.0)
        risk_amount = account_equity * float(self.config.risk.risk_per_trade)
        lot_size = min(float(self.config.risk.max_lot_size), max(0.0, risk_amount / max(account_equity, 1.0)))
        enriched = dict(signal)
        enriched["risk"] = {
            "account_equity": account_equity,
            "risk_amount": risk_amount,
            "lot_size": max(lot_size, float(self.config.risk.default_lot_size)),
            "max_open_trades": int(self.config.risk.max_open_trades),
            "daily_loss_limit": float(self.config.risk.daily_loss_limit),
            "max_drawdown": float(self.config.risk.max_drawdown),
        }
        return enriched

    def run_training(
        self,
        candles: Sequence[CandleDict] | None = None,
        *,
        register: bool = True,
        label_name: str | None = None,
        model_name: str | None = None,
    ) -> PipelineRunResult:
        """Execute data -> features -> labels -> train -> evaluate -> register."""

        active_candles = list(candles) if candles is not None else self.load_candles()
        dataset = self.build_dataset(active_candles, label_name=label_name)
        train_result = self.train(dataset.bundle)
        evaluation = self.evaluate(train_result.model, dataset.bundle)
        registration = self.register(
            train_result.model,
            metrics={**train_result.metrics, **evaluation},
            name=model_name,
            features=dataset.bundle.feature_names,
        ) if register else None
        return PipelineRunResult(dataset=dataset, train=train_result, evaluation=evaluation, registration=registration)

    def run_prediction(
        self,
        candles: Sequence[CandleDict] | None = None,
        *,
        equity: float | None = None,
    ) -> Dict[str, Any]:
        """Execute predict -> signal -> risk."""

        prediction = self.predict(candles=candles)
        signal = self.create_signal(prediction)
        return {"prediction": prediction.to_dict(), "signal": self.apply_risk(signal, equity=equity)}

    def _metrics(self, model: BaseModel, X: NDArray[np.floating], y: NDArray[np.floating]) -> Dict[str, float]:
        return default_metrics(flatten_target(y), np.asarray(model.predict(X)), task=model.task)

    def _load_latest_model(self) -> BaseModel:
        assert self.registry is not None
        artifacts = self.registry.load(self.config.model.model_type)
        model = artifacts.get("model")
        if not isinstance(model, BaseModel):
            raise TypeError("Registered artifact did not contain a BaseModel")
        self.model = model
        self.model_version = str(artifacts.get("version"))
        return model

    def _probabilities(self, model: BaseModel, X: NDArray[np.floating]) -> Dict[str, float] | None:
        proba = model.predict_proba(X)
        if proba is None:
            return None
        arr = np.asarray(proba, dtype=float).reshape(1, -1)
        return {str(idx): float(value) for idx, value in enumerate(arr[0])}

    def _select_label(self, labels: Dict[str, LabelResult], label_name: str | None) -> LabelResult:
        if not labels:
            raise ValueError("No labels were generated")
        if label_name is not None:
            if label_name not in labels:
                raise ValueError(f"Unknown label_name {label_name!r}")
            return labels[label_name]
        return next(iter(labels.values()))

    def _align_features_labels(
        self,
        features: FeatureFrame,
        label: LabelResult,
        candles: Sequence[CandleDict],
    ) -> tuple[NDArray[np.floating], NDArray[np.floating], List[datetime]]:
        label_by_timestamp = {
            normalize_timestamp(candle.get("timestamp")): idx for idx, candle in enumerate(self._base_candles(candles))
        }
        rows: List[NDArray[np.floating]] = []
        targets: List[float] = []
        timestamps: List[datetime] = []
        for row, timestamp in zip(features.matrix, features.timestamps):
            idx = label_by_timestamp.get(timestamp)
            if idx is None or idx >= len(label.values) or not bool(label.valid_mask[idx]):
                continue
            value = float(label.values[idx])
            if not np.isfinite(value) or not np.all(np.isfinite(row)):
                continue
            rows.append(np.asarray(row, dtype=float))
            targets.append(value)
            timestamps.append(timestamp)
        if not rows:
            return np.empty((0, len(features.feature_names)), dtype=float), np.empty((0,), dtype=float), []
        return np.vstack(rows), np.asarray(targets, dtype=float), timestamps

    def _split_points(self, n_samples: int) -> tuple[int, int]:
        train_ratio = float(self.config.datasets.train_ratio)
        val_ratio = float(self.config.datasets.val_ratio)
        train_end = max(1, min(n_samples, int(n_samples * train_ratio)))
        val_end = max(train_end, min(n_samples, train_end + int(n_samples * val_ratio)))
        if train_end == n_samples and n_samples > 1:
            train_end = n_samples - 1
        return train_end, val_end

    def _base_candles(self, candles: Sequence[CandleDict]) -> List[CandleDict]:
        rows = list(candles)
        if not rows:
            return []
        primary_tf = self.config.primary_timeframe.upper()
        symbol = str(rows[0].get("symbol", self.config.symbols[0] if self.config.symbols else "")).upper()
        filtered = [
            row
            for row in rows
            if str(row.get("symbol", symbol)).upper() == symbol
            and str(row.get("timeframe", primary_tf)).upper() == primary_tf
        ]
        return sorted(filtered or rows, key=lambda row: normalize_timestamp(row.get("timestamp")))


def create_ai_pipeline(config: AIConfig | None = None, candle_source: Any = None) -> AIPipeline:
    """Factory for the production AI pipeline."""

    return AIPipeline(config=config or AIConfig(), candle_source=candle_source)


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float, np.integer, np.floating))
