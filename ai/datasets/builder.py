"""
ai/datasets/builder.py - Streaming dataset construction.

VERSION: 1.0.0
"""

from __future__ import annotations

import os
import tempfile
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Mapping, Optional, Protocol, Sequence, Tuple, runtime_checkable

import numpy as np
from numpy.typing import NDArray

from ai.config.settings import AIConfig, DatasetConfig
from ai.data.normalizers import normalize_candle_dict
from ai.data.protocols import CandleSource
from ai.datasets.schema import DatasetBundle
from ai.datasets.windows import apply_stride, generate_sequences
from ai.utils.types import CandleDict
from ai.utils.validation import AIValidationError, validate_finite


@runtime_checkable
class FeatureEngineLike(Protocol):
    def build_features(self, candles: Sequence[CandleDict], **kwargs: Any) -> Any:
        ...


@runtime_checkable
class LabelGeneratorLike(Protocol):
    def generate_labels(self, candles: Sequence[CandleDict], **kwargs: Any) -> Any:
        ...


class _ArrayAccumulator:
    """
    Chunk accumulator that switches to a growable numpy memmap above a row cap.
    """

    def __init__(self, max_memory_rows: int, prefix: str, temp_dir: Optional[Path] = None) -> None:
        self.max_memory_rows = max(1, int(max_memory_rows))
        self.prefix = prefix
        self.temp_dir = temp_dir
        self._chunks: List[NDArray[np.floating]] = []
        self._memmap: Optional[np.memmap] = None
        self._path: Optional[Path] = None
        self._count = 0
        self._capacity = 0
        self._row_shape: Optional[tuple[int, ...]] = None
        self._dtype = np.dtype(float)

    @property
    def count(self) -> int:
        return self._count

    @property
    def path(self) -> Optional[str]:
        return str(self._path) if self._path is not None else None

    def append(self, rows: NDArray[np.floating]) -> None:
        arr = np.asarray(rows, dtype=float)
        if arr.ndim == 0:
            arr = arr.reshape(1)
        if len(arr) == 0:
            return
        row_shape = tuple(arr.shape[1:])
        if self._row_shape is None:
            self._row_shape = row_shape
            self._dtype = arr.dtype
        elif row_shape != self._row_shape:
            raise AIValidationError(f"{self.prefix} row shape changed: {self._row_shape} -> {row_shape}")

        required = self._count + len(arr)
        if self._memmap is None and required <= self.max_memory_rows:
            self._chunks.append(arr.copy())
            self._count = required
            return

        if self._memmap is None:
            self._activate_memmap(max(required, self.max_memory_rows * 2))
        self._ensure_capacity(required)
        assert self._memmap is not None
        self._memmap[self._count : required] = arr
        self._memmap.flush()
        self._count = required

    def to_array(self) -> NDArray[np.floating] | np.memmap:
        if self._memmap is not None:
            return self._memmap[: self._count]
        if not self._chunks:
            shape = (0, *(self._row_shape or ()))
            return np.empty(shape, dtype=float)
        if len(self._chunks) == 1:
            return self._chunks[0]
        return np.concatenate(self._chunks, axis=0)

    def _activate_memmap(self, capacity: int) -> None:
        row_shape = self._row_shape or ()
        fd, raw_path = tempfile.mkstemp(prefix=f"{self.prefix}_", suffix=".npy", dir=self.temp_dir)
        os.close(fd)
        self._path = Path(raw_path)
        self._capacity = int(capacity)
        self._memmap = np.lib.format.open_memmap(
            self._path,
            mode="w+",
            dtype=self._dtype,
            shape=(self._capacity, *row_shape),
        )
        offset = 0
        for chunk in self._chunks:
            end = offset + len(chunk)
            self._memmap[offset:end] = chunk
            offset = end
        self._chunks.clear()

    def _ensure_capacity(self, required: int) -> None:
        if required <= self._capacity:
            return
        assert self._path is not None
        assert self._memmap is not None
        row_shape = self._row_shape or ()
        new_capacity = max(required, self._capacity * 2)
        fd, raw_path = tempfile.mkstemp(prefix=f"{self.prefix}_", suffix=".npy", dir=self.temp_dir)
        os.close(fd)
        new_path = Path(raw_path)
        new_map = np.lib.format.open_memmap(
            new_path,
            mode="w+",
            dtype=self._dtype,
            shape=(new_capacity, *row_shape),
        )
        new_map[: self._count] = self._memmap[: self._count]
        new_map.flush()
        old_path = self._path
        self._memmap = new_map
        self._path = new_path
        self._capacity = new_capacity
        try:
            old_path.unlink(missing_ok=True)
        except OSError:
            pass


class DatasetBuilder:
    """Build model-ready datasets from streaming candle sources."""

    def __init__(self, config: AIConfig | DatasetConfig | None = None) -> None:
        self.config = config or AIConfig()

    def build(
        self,
        source: CandleSource,
        symbol: str,
        timeframe: str,
        start: datetime,
        end: datetime,
        feature_engine: Optional[FeatureEngineLike],
        label_generator: Optional[LabelGeneratorLike],
        config: AIConfig | DatasetConfig | None = None,
        *,
        use_sequences: Optional[bool] = None,
    ) -> DatasetBundle:
        run_config = config or self.config
        dataset_config = self._dataset_config(run_config)
        if end < start:
            raise AIValidationError("end must be >= start")

        max_memory_rows = max(1, int(dataset_config.max_memory_rows))
        stream_batch_size = max(
            1,
            min(max_memory_rows, max(int(dataset_config.batch_size), self._carry_rows(run_config) + 1)),
        )
        feature_acc = _ArrayAccumulator(max_memory_rows, "dataset_X")
        label_acc = _ArrayAccumulator(max_memory_rows, "dataset_y")
        timestamps: List[datetime] = []
        feature_names: List[str] = []
        carry: List[CandleDict] = []
        last_emitted_ts: Optional[datetime] = None
        total_candles = 0
        emitted_rows = 0
        carry_rows = self._carry_rows(run_config)
        forecast_horizon = self._label_horizon(run_config)

        for chunk in self._stream_candle_chunks(source, symbol, timeframe, start, end, stream_batch_size):
            total_candles += len(chunk)
            buffer = carry + chunk
            X_chunk, y_chunk, ts_chunk, names = self._build_aligned_chunk(
                buffer=buffer,
                symbol=symbol,
                timeframe=timeframe,
                feature_engine=feature_engine,
                label_generator=label_generator,
                config=run_config,
                safe_horizon=forecast_horizon,
                last_emitted_ts=last_emitted_ts,
            )
            if len(ts_chunk):
                if not feature_names:
                    feature_names = names
                elif names != feature_names:
                    raise AIValidationError("Feature names changed while streaming")
                feature_acc.append(X_chunk)
                label_acc.append(y_chunk)
                timestamps.extend(ts_chunk)
                last_emitted_ts = ts_chunk[-1]
                emitted_rows += len(ts_chunk)
            carry = buffer[-carry_rows:] if carry_rows else []

        X = feature_acc.to_array()
        y = label_acc.to_array()
        if emitted_rows == 0:
            raise AIValidationError("No aligned feature/label rows were produced")

        sequence_enabled = self._use_sequences(run_config, use_sequences)
        if sequence_enabled and int(dataset_config.sequence_length) > 1:
            sequence_data = generate_sequences(
                np.asarray(X, dtype=float),
                np.asarray(y, dtype=float),
                timestamps,
                sequence_length=int(dataset_config.sequence_length),
                stride=int(dataset_config.stride),
                drop_incomplete=bool(dataset_config.drop_incomplete),
            )
            X = sequence_data.X
            y = sequence_data.y if sequence_data.y is not None else np.empty((0,), dtype=float)
            timestamps = sequence_data.timestamps
        else:
            X, y, timestamps = apply_stride(
                np.asarray(X, dtype=float),
                np.asarray(y, dtype=float),
                timestamps,
                stride=int(dataset_config.stride),
            )

        if len(X) == 0:
            raise AIValidationError("No samples remain after windowing/stride")
        validate_finite(np.asarray(X, dtype=float), "features")
        validate_finite(np.asarray(y, dtype=float), "labels")

        train_end, val_end = self._split_points(len(X), dataset_config)
        train_ts = timestamps[:train_end]
        val_ts = timestamps[train_end:val_end]
        test_ts = timestamps[val_end:]
        metadata = {
            "symbol": symbol.upper(),
            "timeframe": timeframe.upper(),
            "start": start.isoformat(),
            "end": end.isoformat(),
            "total_candles": total_candles,
            "samples": int(len(X)),
            "sequence_length": int(dataset_config.sequence_length) if sequence_enabled else 1,
            "use_sequences": sequence_enabled,
            "stride": int(dataset_config.stride),
            "forecast_horizon": forecast_horizon,
            "max_memory_rows": max_memory_rows,
            "feature_memmap_path": feature_acc.path,
            "label_memmap_path": label_acc.path,
            "train_timestamps": train_ts,
            "val_timestamps": val_ts,
            "test_timestamps": test_ts,
            "config": self._config_snapshot(run_config),
        }
        return DatasetBundle(
            X_train=X[:train_end],
            y_train=y[:train_end],
            X_val=X[train_end:val_end],
            y_val=y[train_end:val_end],
            X_test=X[val_end:],
            y_test=y[val_end:],
            feature_names=feature_names,
            timestamps=timestamps,
            metadata=metadata,
        )

    def _stream_candle_chunks(
        self,
        source: CandleSource,
        symbol: str,
        timeframe: str,
        start: datetime,
        end: datetime,
        batch_size: int,
    ) -> Iterator[List[CandleDict]]:
        chunk: List[CandleDict] = []
        for candle in source.stream_candles(
            symbol=symbol,
            timeframe=timeframe,
            start_time=start,
            end_time=end,
            batch_size=batch_size,
            order="ASC",
        ):
            normalized = normalize_candle_dict(candle)
            normalized["timestamp"] = self._coerce_timestamp(normalized["timestamp"])
            chunk.append(normalized)
            if len(chunk) >= batch_size:
                yield chunk
                chunk = []
        if chunk:
            yield chunk

    def _build_aligned_chunk(
        self,
        buffer: Sequence[CandleDict],
        symbol: str,
        timeframe: str,
        feature_engine: Optional[FeatureEngineLike],
        label_generator: Optional[LabelGeneratorLike],
        config: AIConfig | DatasetConfig,
        safe_horizon: int,
        last_emitted_ts: Optional[datetime],
    ) -> tuple[NDArray[np.floating], NDArray[np.floating], List[datetime], List[str]]:
        if not buffer:
            return np.empty((0, 0), dtype=float), np.empty((0,), dtype=float), [], []

        feature_matrix, feature_names, feature_ts = self._build_features(
            feature_engine,
            buffer,
            symbol=symbol,
            timeframe=timeframe,
            config=config,
        )
        label_array, label_ts = self._build_labels(
            label_generator,
            buffer,
            symbol=symbol,
            timeframe=timeframe,
            config=config,
        )
        if len(feature_matrix) != len(feature_ts):
            raise AIValidationError("feature matrix/timestamp length mismatch")
        if len(label_array) != len(label_ts):
            raise AIValidationError("label array/timestamp length mismatch")

        safe_cutoff = buffer[-safe_horizon - 1]["timestamp"] if safe_horizon and len(buffer) > safe_horizon else buffer[-1]["timestamp"]
        label_by_ts = {ts: label_array[idx] for idx, ts in enumerate(label_ts)}
        rows: List[NDArray[np.floating]] = []
        labels: List[NDArray[np.floating] | float] = []
        timestamps: List[datetime] = []
        for idx, ts in enumerate(feature_ts):
            if last_emitted_ts is not None and ts <= last_emitted_ts:
                continue
            if ts > safe_cutoff:
                continue
            if ts not in label_by_ts:
                continue
            feature_row = np.asarray(feature_matrix[idx], dtype=float)
            label_row = np.asarray(label_by_ts[ts], dtype=float)
            if not np.isfinite(feature_row).all() or not np.isfinite(label_row).all():
                continue
            rows.append(feature_row)
            labels.append(label_row)
            timestamps.append(ts)

        if not rows:
            y_shape = label_array.shape[1:] if label_array.ndim > 1 else ()
            return np.empty((0, feature_matrix.shape[1]), dtype=float), np.empty((0, *y_shape), dtype=float), [], feature_names
        return (
            np.vstack(rows).astype(float, copy=False),
            np.asarray(labels, dtype=float),
            timestamps,
            feature_names,
        )

    def _build_features(
        self,
        feature_engine: Optional[FeatureEngineLike],
        candles: Sequence[CandleDict],
        **context: Any,
    ) -> tuple[NDArray[np.floating], List[str], List[datetime]]:
        if feature_engine is None:
            return self._default_features(candles)
        result = self._invoke_engine(
            feature_engine,
            ("build_features", "transform", "fit_transform", "generate_features", "build"),
            candles,
            **context,
        )
        return self._normalize_feature_result(result, candles)

    def _build_labels(
        self,
        label_generator: Optional[LabelGeneratorLike],
        candles: Sequence[CandleDict],
        **context: Any,
    ) -> tuple[NDArray[np.floating], List[datetime]]:
        if label_generator is None:
            config = context.get("config")
            return self._default_labels(candles, config)
        result = self._invoke_engine(
            label_generator,
            ("generate_labels", "build_labels", "transform", "generate", "build"),
            candles,
            **context,
        )
        return self._normalize_label_result(result, candles, context.get("config"))

    def _invoke_engine(self, engine: Any, method_names: Sequence[str], candles: Sequence[CandleDict], **context: Any) -> Any:
        last_error: Optional[Exception] = None
        for method_name in method_names:
            method = getattr(engine, method_name, None)
            if method is None:
                continue
            for kwargs in self._engine_call_variants(context):
                try:
                    return method(candles, **kwargs)
                except TypeError as exc:
                    last_error = exc
        if callable(engine):
            for kwargs in self._engine_call_variants(context):
                try:
                    return engine(candles, **kwargs)
                except TypeError as exc:
                    last_error = exc
        if last_error is not None:
            raise last_error
        raise TypeError(f"{engine!r} does not expose a supported dataset engine method")

    def _normalize_feature_result(
        self,
        result: Any,
        candles: Sequence[CandleDict],
    ) -> tuple[NDArray[np.floating], List[str], List[datetime]]:
        timestamps = self._extract_timestamps(result)
        names = self._extract_feature_names(result)
        payload = self._extract_payload(result, ("features", "X", "data", "matrix", "values"))

        if isinstance(payload, Mapping) and payload and all(self._is_numeric_sequence(v) for v in payload.values()):
            names = list(payload.keys())
            matrix = np.column_stack([np.asarray(payload[name], dtype=float) for name in names])
        elif self._is_sequence_of_mappings(payload):
            rows = [dict(row) for row in payload]
            names = names or list(rows[0].keys())
            matrix = np.asarray([[float(row.get(name, np.nan)) for name in names] for row in rows], dtype=float)
        else:
            matrix = np.asarray(payload, dtype=float)
            if matrix.ndim == 1:
                matrix = matrix.reshape(-1, 1)
            if matrix.ndim != 2:
                raise AIValidationError(f"feature result must be 2D-compatible, got shape {matrix.shape}")
            names = names or [f"feature_{idx}" for idx in range(matrix.shape[1])]

        if len(names) != matrix.shape[1]:
            raise AIValidationError(f"feature name count mismatch: {len(names)} != {matrix.shape[1]}")
        if not timestamps:
            timestamps = self._align_result_timestamps(candles, len(matrix), align="tail")
        return matrix.astype(float, copy=False), [str(name) for name in names], timestamps

    def _normalize_label_result(
        self,
        result: Any,
        candles: Sequence[CandleDict],
        config: AIConfig | DatasetConfig | None,
    ) -> tuple[NDArray[np.floating], List[datetime]]:
        timestamps = self._extract_timestamps(result)
        payload = self._extract_payload(result, ("labels", "y", "targets", "target", "data", "values"))

        if isinstance(payload, Mapping) and payload and all(hasattr(v, "values") for v in payload.values()):
            columns: List[NDArray[np.floating]] = []
            valid_masks: List[NDArray[np.bool_]] = []
            for value in payload.values():
                column = np.asarray(getattr(value, "values"), dtype=float).reshape(-1)
                valid_mask = getattr(value, "valid_mask", np.isfinite(column))
                valid_masks.append(np.asarray(valid_mask, dtype=bool).reshape(-1))
                columns.append(column)
            arr = np.column_stack(columns).astype(float, copy=False)
            combined_valid = np.logical_and.reduce(valid_masks)
            arr[~combined_valid] = np.nan
            if arr.shape[1] == 1:
                arr = arr.reshape(-1)
        elif isinstance(payload, Mapping) and payload and all(self._is_numeric_sequence(v) for v in payload.values()):
            keys = list(payload.keys())
            arr = np.column_stack([np.asarray(payload[key], dtype=float) for key in keys])
            if arr.shape[1] == 1:
                arr = arr.reshape(-1)
        elif self._is_sequence_of_mappings(payload):
            rows = [dict(row) for row in payload]
            keys = list(rows[0].keys())
            arr = np.asarray([[float(row.get(key, np.nan)) for key in keys] for row in rows], dtype=float)
            if arr.shape[1] == 1:
                arr = arr.reshape(-1)
        else:
            arr = np.asarray(payload, dtype=float)
            if arr.ndim > 2:
                raise AIValidationError(f"label result must be 1D/2D-compatible, got shape {arr.shape}")

        if not timestamps:
            horizon = int(getattr(self._dataset_config(config), "forecast_horizon", 0)) if config is not None else 0
            align = "head" if horizon > 0 else "tail"
            timestamps = self._align_result_timestamps(candles, len(arr), align=align)
        return arr.astype(float, copy=False), timestamps

    def _default_features(self, candles: Sequence[CandleDict]) -> tuple[NDArray[np.floating], List[str], List[datetime]]:
        close = np.asarray([c["close"] for c in candles], dtype=float)
        open_ = np.asarray([c["open"] for c in candles], dtype=float)
        high = np.asarray([c["high"] for c in candles], dtype=float)
        low = np.asarray([c["low"] for c in candles], dtype=float)
        volume = np.asarray([c.get("volume", 0.0) for c in candles], dtype=float)
        prev_close = np.roll(close, 1)
        prev_close[0] = close[0]
        denom = np.where(prev_close == 0.0, np.nan, prev_close)
        returns = (close - prev_close) / denom
        ranges = high - low
        body = close - open_
        matrix = np.column_stack([open_, high, low, close, volume, returns, ranges, body])
        names = ["open", "high", "low", "close", "volume", "return_1", "range", "body"]
        timestamps = [c["timestamp"] for c in candles]
        return matrix, names, timestamps

    def _default_labels(
        self,
        candles: Sequence[CandleDict],
        config: AIConfig | DatasetConfig | None,
    ) -> tuple[NDArray[np.floating], List[datetime]]:
        dataset_config = self._dataset_config(config)
        horizon = max(1, int(getattr(dataset_config, "forecast_horizon", 1)))
        if len(candles) <= horizon:
            return np.empty((0,), dtype=float), []
        closes = np.asarray([c["close"] for c in candles], dtype=float)
        future = closes[horizon:]
        current = closes[:-horizon]
        returns = np.divide(future - current, current, out=np.zeros_like(current), where=current != 0.0)
        method = "binary_direction"
        threshold = 0.0
        if isinstance(config, AIConfig):
            methods = getattr(config.labels, "methods", [])
            method = str(methods[0]) if methods else method
            threshold = float(getattr(config.labels, "binary_threshold", 0.0))
        labels = returns if method == "future_return" else (returns > threshold).astype(float)
        return labels.astype(float, copy=False), [c["timestamp"] for c in candles[:-horizon]]

    def _extract_payload(self, result: Any, keys: Sequence[str]) -> Any:
        if isinstance(result, tuple) and result:
            return result[0]
        if isinstance(result, Mapping):
            for key in keys:
                if key in result:
                    return result[key]
            return result
        for key in keys:
            if hasattr(result, key):
                return getattr(result, key)
        return result

    def _extract_feature_names(self, result: Any) -> List[str]:
        if isinstance(result, tuple) and len(result) > 1 and self._is_string_sequence(result[1]):
            return [str(name) for name in result[1]]
        if isinstance(result, Mapping):
            for key in ("feature_names", "columns", "names"):
                value = result.get(key)
                if self._is_string_sequence(value):
                    return [str(name) for name in value]
        for key in ("feature_names", "columns", "names"):
            value = getattr(result, key, None)
            if self._is_string_sequence(value):
                return [str(name) for name in value]
        return []

    def _extract_timestamps(self, result: Any) -> List[datetime]:
        value = None
        if isinstance(result, Mapping):
            value = result.get("timestamps", result.get("timestamp"))
        if value is None:
            value = getattr(result, "timestamps", None)
        if value is None:
            return []
        if isinstance(value, np.ndarray):
            value = value.tolist()
        if not isinstance(value, Iterable) or isinstance(value, (str, bytes)):
            return []
        return [self._coerce_timestamp(ts) for ts in value]

    def _align_result_timestamps(self, candles: Sequence[CandleDict], n_rows: int, align: str) -> List[datetime]:
        if n_rows > len(candles):
            raise AIValidationError(f"result length {n_rows} exceeds candle length {len(candles)}")
        if n_rows == len(candles):
            selected = candles
        elif align == "head":
            selected = candles[:n_rows]
        elif align == "tail":
            selected = candles[-n_rows:]
        else:
            raise AIValidationError(f"unsupported timestamp alignment: {align}")
        return [self._coerce_timestamp(c["timestamp"]) for c in selected]

    def _engine_call_variants(self, context: Mapping[str, Any]) -> tuple[Dict[str, Any], ...]:
        variants: List[Dict[str, Any]] = [dict(context)]
        if "config" in context:
            variants.append({"config": context["config"]})
        variants.append({})
        return tuple(variants)

    def _coerce_timestamp(self, value: Any) -> datetime:
        if isinstance(value, datetime):
            ts = value
        else:
            ts = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        if ts.tzinfo is not None:
            ts = ts.astimezone(timezone.utc).replace(tzinfo=None)
        return ts

    def _split_points(self, n_samples: int, config: DatasetConfig) -> tuple[int, int]:
        train_ratio = float(config.train_ratio)
        val_ratio = float(config.val_ratio)
        test_ratio = float(config.test_ratio)
        if min(train_ratio, val_ratio, test_ratio) < 0:
            raise AIValidationError("split ratios must be >= 0")
        ratio_sum = train_ratio + val_ratio + test_ratio
        if ratio_sum <= 0:
            raise AIValidationError("at least one split ratio must be > 0")
        train_ratio /= ratio_sum
        val_ratio /= ratio_sum
        train_end = int(n_samples * train_ratio)
        val_end = train_end + int(n_samples * val_ratio)
        if n_samples >= 3:
            train_end = max(1, min(train_end, n_samples - 2))
            val_end = max(train_end + 1, min(val_end, n_samples - 1))
        else:
            train_end = max(1, min(train_end, n_samples))
            val_end = max(train_end, min(val_end, n_samples))
        return train_end, val_end

    def _dataset_config(self, config: AIConfig | DatasetConfig | None) -> DatasetConfig:
        if isinstance(config, AIConfig):
            return config.datasets
        if isinstance(config, DatasetConfig):
            return config
        return AIConfig().datasets

    def _use_sequences(self, config: AIConfig | DatasetConfig, explicit: Optional[bool]) -> bool:
        if explicit is not None:
            return bool(explicit)
        dataset_config = self._dataset_config(config)
        dataset_flag = getattr(dataset_config, "use_sequences", None)
        if dataset_flag is not None:
            return bool(dataset_flag)
        if isinstance(config, AIConfig):
            model_type = str(getattr(config.model, "model_type", "")).lower()
            return model_type in {"lstm", "gru", "rnn", "transformer", "temporal_cnn", "tcn"}
        return False

    def _carry_rows(self, config: AIConfig | DatasetConfig) -> int:
        dataset_config = self._dataset_config(config)
        candidates = [
            int(getattr(dataset_config, "sequence_length", 1)),
            self._label_horizon(config) + 1,
        ]
        if isinstance(config, AIConfig):
            features = config.features
            for name in (
                "sma_periods",
                "ema_periods",
                "rolling_windows",
                "multi_timeframes",
                "horizons",
            ):
                value = getattr(features, name, None)
                if isinstance(value, Sequence) and value and not isinstance(value, (str, bytes)):
                    numeric = [int(v) for v in value if isinstance(v, (int, float))]
                    if numeric:
                        candidates.append(max(numeric))
            for name in (
                "rsi_period",
                "atr_period",
                "adx_period",
                "bollinger_period",
                "stochastic_k",
                "williams_period",
                "cci_period",
                "donchian_period",
                "keltner_period",
                "mfi_period",
                "cmf_period",
                "swing_lookback",
                "support_resistance_lookback",
            ):
                value = getattr(features, name, None)
                if isinstance(value, (int, float)):
                    candidates.append(int(value))
        return max(candidates) + 2

    def _label_horizon(self, config: AIConfig | DatasetConfig) -> int:
        dataset_horizon = max(0, int(getattr(self._dataset_config(config), "forecast_horizon", 0)))
        if not isinstance(config, AIConfig):
            return dataset_horizon
        label_candidates = [dataset_horizon, int(getattr(config.labels, "horizon", 0))]
        label_candidates.extend(int(item) for item in getattr(config.labels, "horizons", []) if int(item) > 0)
        return max(label_candidates)

    def _config_snapshot(self, config: AIConfig | DatasetConfig) -> Dict[str, Any]:
        if isinstance(config, AIConfig):
            return config.to_dict()
        if is_dataclass(config):
            return asdict(config)
        return {}

    def _is_sequence_of_mappings(self, value: Any) -> bool:
        return isinstance(value, Sequence) and not isinstance(value, (str, bytes, np.ndarray)) and bool(value) and isinstance(value[0], Mapping)

    def _is_string_sequence(self, value: Any) -> bool:
        return isinstance(value, Sequence) and not isinstance(value, (str, bytes)) and all(isinstance(item, str) for item in value)

    def _is_numeric_sequence(self, value: Any) -> bool:
        if isinstance(value, (str, bytes, Mapping)):
            return False
        try:
            arr = np.asarray(value, dtype=float)
        except (TypeError, ValueError):
            return False
        return arr.ndim == 1


def create_dataset_builder(config: AIConfig | DatasetConfig | None = None) -> DatasetBuilder:
    return DatasetBuilder(config=config)
