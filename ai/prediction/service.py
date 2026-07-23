"""
ai/prediction/service.py - Live model serving for trading predictions.

RESPONSIBILITY:
Load registered model artifacts, build live feature matrices, and return typed
PredictionResult values for single, batch, and streaming candle inputs.

VERSION: 1.0.0
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, Iterator, List, Mapping, Sequence

import numpy as np
from numpy.typing import NDArray

from ai.config.settings import AIConfig
from ai.features.engine import FeatureEngine, FeatureFrame, create_feature_engine
from ai.models import create_model
from ai.storage import ModelRegistry, create_model_registry
from ai.utils.types import CandleDict, PredictionResult, SignalType


ArrayLike = NDArray[np.floating]


@dataclass
class PredictionService:
    """Production prediction facade backed by ModelRegistry artifacts."""

    config: AIConfig = field(default_factory=AIConfig)
    model_name: str | None = None
    model_version: str | None = None
    registry: ModelRegistry | None = None
    feature_engine: FeatureEngine | None = None
    model: Any = None
    scaler: Any = None
    feature_names: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    autoload: bool = True

    def __post_init__(self) -> None:
        self.registry = self.registry or create_model_registry(self.config)
        self.feature_engine = self.feature_engine or create_feature_engine(self.config)
        if self.model is None and self.autoload:
            self._load_or_prepare_model()

    def load(self, name: str | None = None, version: str | None = None) -> "PredictionService":
        """Load model, scaler metadata, and feature list from the registry."""

        active_name = name or self.model_name or self.config.model.model_type
        active_version = version or self.model_version
        artifacts = self.registry.load(active_name, active_version)  # type: ignore[union-attr]
        self.model = artifacts["model"]
        self.scaler = artifacts.get("scaler")
        self.feature_names = list(artifacts.get("features") or [])
        self.model_name = str(artifacts.get("name", active_name))
        self.model_version = str(artifacts.get("version", active_version or ""))
        self.metadata = {
            "artifact_directory": str(artifacts.get("directory", "")),
            "artifact_meta": artifacts.get("meta", {}),
            "artifact_metrics": artifacts.get("metrics", {}),
            "source": "registry",
        }
        return self

    def predict(
        self,
        candles: Sequence[CandleDict] | FeatureFrame | ArrayLike,
        *,
        symbol: str | None = None,
        timeframe: str | None = None,
    ) -> PredictionResult:
        """Return a typed prediction for the latest feature row."""

        matrix, context = self._prepare_features(candles)
        latest = self._latest_row(matrix)
        raw_prediction = self._predict_matrix(latest)[0]
        probabilities = self._predict_probabilities(latest)
        confidence = _confidence(probabilities, raw_prediction)
        prediction_value = _json_scalar(raw_prediction)
        return PredictionResult(
            symbol=symbol or context["symbol"],
            timeframe=timeframe or context["timeframe"],
            timestamp=context["timestamp"],
            prediction=prediction_value,
            probabilities=probabilities,
            confidence=confidence,
            expected_return=_expected_return(prediction_value, probabilities),
            feature_contributions=self._feature_contributions(latest),
            model_version=self.model_version,
            metadata={
                "model_name": self.model_name,
                "feature_count": int(latest.shape[1]),
                "service": "PredictionService",
                **context["metadata"],
            },
        )

    def predict_proba(
        self,
        candles: Sequence[CandleDict] | FeatureFrame | ArrayLike,
        *,
        symbol: str | None = None,
        timeframe: str | None = None,
    ) -> PredictionResult:
        """Return a PredictionResult focused on class probabilities."""

        result = self.predict(candles, symbol=symbol, timeframe=timeframe)
        if result.probabilities is None:
            result.probabilities = _probabilities_from_prediction(result.prediction)
            result.confidence = _confidence(result.probabilities, result.prediction)
        return result

    def predict_signal(self, candles: Sequence[CandleDict] | FeatureFrame | ArrayLike) -> SignalType:
        """Map the latest prediction into a coarse signal side."""

        result = self.predict_proba(candles)
        return prediction_to_signal(result)

    def predict_batch(
        self,
        batches: Iterable[Sequence[CandleDict] | FeatureFrame | ArrayLike],
    ) -> List[PredictionResult]:
        """Predict one result for each supplied candle or feature batch."""

        return [self.predict(batch) for batch in batches]

    def predict_stream(
        self,
        candles: Iterable[CandleDict],
        *,
        window_size: int | None = None,
        min_candles: int = 1,
    ) -> Iterator[PredictionResult]:
        """Yield predictions as incoming candles extend the live window."""

        history: List[CandleDict] = []
        for candle in candles:
            history.append(candle)
            if window_size is not None and window_size > 0 and len(history) > window_size:
                history = history[-window_size:]
            if len(history) >= min_candles:
                try:
                    yield self.predict(history)
                except ValueError as exc:
                    if "No usable feature rows" not in str(exc):
                        raise

    def _load_or_prepare_model(self) -> None:
        active_name = self.model_name or self.config.model.model_type
        try:
            self.load(active_name, self.model_version)
        except FileNotFoundError:
            self.model = create_model(self.config.model.model_type, self.config)
            self.model_name = active_name
            self.model_version = None
            self.metadata = {"source": "factory", "ready": False}

    def _prepare_features(
        self,
        source: Sequence[CandleDict] | FeatureFrame | ArrayLike,
    ) -> tuple[ArrayLike, Dict[str, Any]]:
        if isinstance(source, FeatureFrame):
            matrix = self._align_features(source.matrix, source.feature_names)
            context = _context_from_feature_frame(source, self.config)
        elif _is_array_like(source):
            matrix = np.asarray(source, dtype=float)
            context = _default_context(self.config)
        else:
            candles = list(source)
            if not candles:
                raise ValueError("Prediction requires at least one candle")
            frame = self.feature_engine.transform(candles, self.config)  # type: ignore[union-attr]
            matrix = self._align_features(frame.matrix, frame.feature_names)
            context = _context_from_candles(candles, frame, self.config)

        if matrix.ndim == 1:
            matrix = matrix.reshape(1, -1)
        if matrix.shape[0] == 0:
            raise ValueError("No usable feature rows after feature engineering")
        scaled = self._scale_matrix(matrix)
        return scaled, context

    def _align_features(self, matrix: ArrayLike, names: Sequence[str]) -> ArrayLike:
        arr = np.asarray(matrix, dtype=float)
        if not self.feature_names:
            return arr
        index = {name: pos for pos, name in enumerate(names)}
        aligned = np.zeros((arr.shape[0], len(self.feature_names)), dtype=float)
        missing: List[str] = []
        for target_pos, name in enumerate(self.feature_names):
            source_pos = index.get(name)
            if source_pos is None:
                missing.append(name)
            else:
                aligned[:, target_pos] = arr[:, source_pos]
        if missing:
            self.metadata["missing_live_features"] = missing
        return aligned

    def _scale_matrix(self, matrix: ArrayLike) -> ArrayLike:
        if self.scaler is None or self.scaler == {}:
            return np.asarray(matrix, dtype=float)
        if hasattr(self.scaler, "transform") and callable(self.scaler.transform):
            return np.asarray(self.scaler.transform(matrix), dtype=float)
        if isinstance(self.scaler, Mapping):
            return _scale_from_mapping(matrix, self.scaler)
        return np.asarray(matrix, dtype=float)

    def _predict_matrix(self, matrix: ArrayLike) -> NDArray[Any]:
        model = self._require_ready_model()
        if not hasattr(model, "predict") or not callable(model.predict):
            raise TypeError(f"Loaded model does not expose predict(): {type(model)!r}")
        return np.asarray(model.predict(matrix))

    def _predict_probabilities(self, matrix: ArrayLike) -> Dict[str, float] | None:
        model = self._require_ready_model()
        if not hasattr(model, "predict_proba") or not callable(model.predict_proba):
            return None
        raw = model.predict_proba(matrix)
        if raw is None:
            return None
        arr = np.asarray(raw, dtype=float)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        if arr.size == 0:
            return None
        return _probability_dict(arr[0], getattr(model, "classes_", None))

    def _feature_contributions(self, matrix: ArrayLike) -> Dict[str, float] | None:
        model = self.model
        importances = getattr(model, "feature_importances_", None)
        if importances is None:
            return None
        values = np.asarray(importances, dtype=float).reshape(-1)
        if values.size == 0:
            return None
        names = self.feature_names or [f"feature_{idx}" for idx in range(matrix.shape[1])]
        total = float(np.sum(np.abs(values))) or 1.0
        return {
            name: float(value / total)
            for name, value in zip(names, values)
        }

    def _latest_row(self, matrix: ArrayLike) -> ArrayLike:
        return np.asarray(matrix[-1:], dtype=float)

    def _require_ready_model(self) -> Any:
        if self.model is None:
            self.load()
        if self.metadata.get("source") == "factory" and self.metadata.get("ready") is False:
            raise FileNotFoundError(
                f"No registered model artifact found for {self.model_name!r}; train and register a model before live prediction"
            )
        return self.model


def prediction_to_signal(result: PredictionResult) -> SignalType:
    """Convert a PredictionResult into BUY, SELL, or HOLD."""

    if result.probabilities:
        buy = _prob_lookup(result.probabilities, SignalType.BUY.value)
        sell = _prob_lookup(result.probabilities, SignalType.SELL.value)
        hold = _prob_lookup(result.probabilities, SignalType.HOLD.value)
        if buy > max(sell, hold):
            return SignalType.BUY
        if sell > max(buy, hold):
            return SignalType.SELL
        return SignalType.HOLD

    value = result.prediction
    if isinstance(value, str):
        normalized = value.upper()
        if normalized in SignalType.__members__:
            return SignalType[normalized]
        for signal in SignalType:
            if normalized == signal.value:
                return signal
    numeric = _to_float(value, 0.0)
    if numeric > 0:
        return SignalType.BUY
    if numeric < 0:
        return SignalType.SELL
    return SignalType.HOLD


def create_prediction_service(
    config: AIConfig | None = None,
    *,
    model_name: str | None = None,
    model_version: str | None = None,
    registry: ModelRegistry | None = None,
    feature_engine: FeatureEngine | None = None,
    autoload: bool = True,
) -> PredictionService:
    """Factory for a configured PredictionService."""

    active_config = config or AIConfig()
    return PredictionService(
        config=active_config,
        model_name=model_name,
        model_version=model_version,
        registry=registry,
        feature_engine=feature_engine,
        autoload=autoload,
    )


def _scale_from_mapping(matrix: ArrayLike, payload: Mapping[str, Any]) -> ArrayLike:
    arr = np.asarray(matrix, dtype=float)
    mean = _mapping_array(payload, ("mean_", "mean", "center"))
    scale = _mapping_array(payload, ("scale_", "scale", "std"))
    min_values = _mapping_array(payload, ("min_", "min"))
    data_min = _mapping_array(payload, ("data_min_", "data_min"))
    data_max = _mapping_array(payload, ("data_max_", "data_max"))
    if mean is not None and scale is not None:
        return (arr - _fit_length(mean, arr.shape[1])) / _safe_denominator(_fit_length(scale, arr.shape[1]))
    if data_min is not None and data_max is not None:
        low = _fit_length(data_min, arr.shape[1])
        high = _fit_length(data_max, arr.shape[1])
        return (arr - low) / _safe_denominator(high - low)
    if min_values is not None and scale is not None:
        return arr * _fit_length(scale, arr.shape[1]) + _fit_length(min_values, arr.shape[1])
    return arr


def _mapping_array(payload: Mapping[str, Any], keys: Sequence[str]) -> NDArray[np.floating] | None:
    for key in keys:
        value = payload.get(key)
        if value is not None:
            return np.asarray(value, dtype=float).reshape(-1)
    value_payload = payload.get("value")
    if isinstance(value_payload, Mapping):
        return _mapping_array(value_payload, keys)
    return None


def _fit_length(values: NDArray[np.floating], width: int) -> NDArray[np.floating]:
    if values.size == width:
        return values
    if values.size == 1:
        return np.repeat(values, width)
    if values.size > width:
        return values[:width]
    padded = np.ones(width, dtype=float)
    padded[: values.size] = values
    return padded


def _safe_denominator(values: NDArray[np.floating]) -> NDArray[np.floating]:
    out = np.asarray(values, dtype=float)
    out[np.isclose(out, 0.0)] = 1.0
    return out


def _context_from_candles(candles: Sequence[CandleDict], frame: FeatureFrame, config: AIConfig) -> Dict[str, Any]:
    latest_candle = candles[-1]
    timestamp = frame.timestamps[-1] if frame.timestamps else _timestamp(latest_candle.get("timestamp"))
    return {
        "symbol": str(latest_candle.get("symbol", config.symbols[0] if config.symbols else "")),
        "timeframe": str(latest_candle.get("timeframe", config.primary_timeframe)),
        "timestamp": timestamp,
        "metadata": {
            "feature_names": frame.feature_names,
            "feature_metadata": frame.metadata,
        },
    }


def _context_from_feature_frame(frame: FeatureFrame, config: AIConfig) -> Dict[str, Any]:
    timestamp = frame.timestamps[-1] if frame.timestamps else datetime.now(timezone.utc)
    return {
        "symbol": config.symbols[0] if config.symbols else "",
        "timeframe": config.primary_timeframe,
        "timestamp": timestamp,
        "metadata": {
            "feature_names": frame.feature_names,
            "feature_metadata": frame.metadata,
        },
    }


def _default_context(config: AIConfig) -> Dict[str, Any]:
    return {
        "symbol": config.symbols[0] if config.symbols else "",
        "timeframe": config.primary_timeframe,
        "timestamp": datetime.now(timezone.utc),
        "metadata": {"feature_names": [], "feature_metadata": {}},
    }


def _probability_dict(values: NDArray[np.floating], classes: Any) -> Dict[str, float]:
    labels = _class_labels(values.size, classes)
    clipped = np.clip(values.astype(float), 0.0, 1.0)
    total = float(np.sum(clipped))
    normalized = clipped / total if total > 0 else clipped
    return {labels[idx]: float(normalized[idx]) for idx in range(len(labels))}


def _class_labels(count: int, classes: Any) -> List[str]:
    if classes is not None:
        labels = [str(item).upper() for item in np.asarray(classes).reshape(-1).tolist()]
        if len(labels) == count:
            return [_normalize_label(label) for label in labels]
    if count == 2:
        return [SignalType.SELL.value, SignalType.BUY.value]
    if count == 3:
        return [SignalType.SELL.value, SignalType.HOLD.value, SignalType.BUY.value]
    return [f"class_{idx}" for idx in range(count)]


def _normalize_label(value: str) -> str:
    aliases = {
        "-1": SignalType.SELL.value,
        "0": SignalType.HOLD.value,
        "1": SignalType.BUY.value,
        "LONG": SignalType.BUY.value,
        "SHORT": SignalType.SELL.value,
        "REDUCE": SignalType.REDUCE.value,
    }
    return aliases.get(value.upper(), value.upper())


def _confidence(probabilities: Dict[str, float] | None, prediction: Any) -> float:
    if probabilities:
        return float(max(probabilities.values()))
    numeric = abs(_to_float(prediction, 0.0))
    return float(min(max(numeric, 0.0), 1.0))


def _expected_return(prediction: Any, probabilities: Dict[str, float] | None) -> float | None:
    numeric = _to_float(prediction, float("nan"))
    if np.isfinite(numeric):
        return float(numeric)
    if probabilities:
        return _prob_lookup(probabilities, SignalType.BUY.value) - _prob_lookup(probabilities, SignalType.SELL.value)
    return None


def _probabilities_from_prediction(prediction: Any) -> Dict[str, float]:
    signal = prediction_to_signal(
        PredictionResult(
            symbol="",
            timeframe="",
            timestamp=datetime.now(timezone.utc),
            prediction=prediction,
        )
    )
    if signal == SignalType.BUY:
        return {SignalType.BUY.value: 1.0, SignalType.SELL.value: 0.0, SignalType.HOLD.value: 0.0}
    if signal == SignalType.SELL:
        return {SignalType.BUY.value: 0.0, SignalType.SELL.value: 1.0, SignalType.HOLD.value: 0.0}
    return {SignalType.BUY.value: 0.0, SignalType.SELL.value: 0.0, SignalType.HOLD.value: 1.0}


def _prob_lookup(probabilities: Mapping[str, float], label: str) -> float:
    aliases = {label, label.upper(), label.lower(), _normalize_label(label)}
    for key, value in probabilities.items():
        if str(key) in aliases or _normalize_label(str(key)) in aliases:
            return float(value)
    return 0.0


def _timestamp(value: Any) -> datetime:
    if isinstance(value, datetime):
        return value if value.tzinfo is not None else value.replace(tzinfo=timezone.utc)
    if isinstance(value, (int, float)):
        return datetime.fromtimestamp(float(value), tz=timezone.utc)
    if isinstance(value, str):
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        return parsed if parsed.tzinfo is not None else parsed.replace(tzinfo=timezone.utc)
    return datetime.now(timezone.utc)


def _is_array_like(value: Any) -> bool:
    return isinstance(value, np.ndarray)


def _json_scalar(value: Any) -> float | int | str:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (float, int, str)):
        return value
    return str(value)


def _to_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default
