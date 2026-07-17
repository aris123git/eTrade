"""
ai/features/engine.py - Feature orchestration for model-ready matrices

RESPONSIBILITY:
Coordinate feature groups, normalize candle inputs, handle warm-up rows, and
return a typed FeatureFrame for downstream AI pipelines.

VERSION: 1.0.0
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Sequence, Tuple

import numpy as np
from numpy.typing import NDArray

from ai.config.settings import AIConfig
from ai.utils.types import CandleDict


# ==============================================================================
# PUBLIC CONTRACTS
# ==============================================================================


FeatureMap = Dict[str, NDArray[np.floating]]


class FeatureGroup(str, Enum):
    """Named feature groups accepted by FeatureConfig.enabled_groups."""

    PRICE = "price"
    RETURNS = "returns"
    MOVING_AVERAGES = "moving_averages"
    MOMENTUM = "momentum"
    VOLATILITY = "volatility"
    CHANNELS = "channels"
    VOLUME = "volume"
    CANDLE_STRUCTURE = "candle_structure"
    PATTERNS = "patterns"
    STRUCTURE = "structure"
    SESSION = "session"
    REGIME = "regime"
    MICROSTRUCTURE = "microstructure"
    MULTI_TIMEFRAME = "multi_timeframe"
    CORRELATION = "correlation"


@dataclass(frozen=True)
class FeatureFrame:
    """Dense feature matrix with column names and row metadata."""

    matrix: NDArray[np.floating]
    feature_names: List[str]
    timestamps: List[datetime]
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def shape(self) -> Tuple[int, int]:
        """Return matrix shape as (rows, columns)."""

        return int(self.matrix.shape[0]), int(self.matrix.shape[1])

    def to_pandas(self) -> Any:
        """Export to pandas.DataFrame when pandas is installed."""

        try:
            import pandas as pd
        except ImportError as exc:
            raise RuntimeError("pandas is required for FeatureFrame.to_pandas()") from exc
        return pd.DataFrame(self.matrix, index=self.timestamps, columns=self.feature_names)


@dataclass(frozen=True)
class CandleArrays:
    """Vectorized OHLCV representation extracted from CandleDict inputs."""

    timestamps: List[datetime]
    open: NDArray[np.floating]
    high: NDArray[np.floating]
    low: NDArray[np.floating]
    close: NDArray[np.floating]
    volume: NDArray[np.floating]
    spread: NDArray[np.floating]


@dataclass
class FeatureEngine:
    """Production feature engine driven by AIConfig.FeatureConfig."""

    config: AIConfig | None = None

    def transform(self, candles: Sequence[CandleDict], config: AIConfig | None = None) -> FeatureFrame:
        """Transform candle dictionaries into a model-ready FeatureFrame."""

        active_config = config or self.config or AIConfig()
        base_candles, higher_timeframes, peer_symbols = self._select_base_scope(candles, active_config)
        arrays = candles_to_arrays(base_candles)
        feature_config = active_config.features
        enabled = {group.lower() for group in feature_config.enabled_groups}

        feature_maps: List[Tuple[str, FeatureMap]] = []
        if self._enabled(enabled, FeatureGroup.PRICE, FeatureGroup.RETURNS):
            from ai.features.price import compute_price_features

            feature_maps.append(
                (
                    "price",
                    compute_price_features(
                        arrays.open,
                        arrays.high,
                        arrays.low,
                        arrays.close,
                        arrays.volume,
                        rolling_windows=feature_config.rolling_windows,
                        include_price=FeatureGroup.PRICE.value in enabled,
                        include_returns=FeatureGroup.RETURNS.value in enabled,
                    ),
                )
            )

        if self._enabled(enabled, FeatureGroup.MOVING_AVERAGES):
            from ai.features.moving_averages import compute_moving_average_features

            feature_maps.append(
                (
                    "moving_averages",
                    compute_moving_average_features(
                        arrays.high,
                        arrays.low,
                        arrays.close,
                        arrays.volume,
                        sma_periods=feature_config.sma_periods,
                        ema_periods=feature_config.ema_periods,
                        vwap_windows=feature_config.rolling_windows,
                    ),
                )
            )

        if self._enabled(enabled, FeatureGroup.MOMENTUM):
            from ai.features.momentum import compute_momentum_features

            feature_maps.append(
                (
                    "momentum",
                    compute_momentum_features(
                        arrays.high,
                        arrays.low,
                        arrays.close,
                        rsi_period=feature_config.rsi_period,
                        adx_period=feature_config.adx_period,
                        macd_fast=feature_config.macd_fast,
                        macd_slow=feature_config.macd_slow,
                        macd_signal=feature_config.macd_signal,
                        stochastic_k=feature_config.stochastic_k,
                        stochastic_d=feature_config.stochastic_d,
                        williams_period=feature_config.williams_period,
                        cci_period=feature_config.cci_period,
                        roc_period=feature_config.roc_period,
                        momentum_period=feature_config.momentum_period,
                    ),
                )
            )

        if self._enabled(enabled, FeatureGroup.VOLATILITY, FeatureGroup.CHANNELS):
            from ai.features.volatility import compute_volatility_features

            feature_maps.append(
                (
                    "volatility",
                    compute_volatility_features(
                        arrays.high,
                        arrays.low,
                        arrays.close,
                        rolling_windows=feature_config.rolling_windows,
                        atr_period=feature_config.atr_period,
                        bollinger_period=feature_config.bollinger_period,
                        bollinger_std=feature_config.bollinger_std,
                        donchian_period=feature_config.donchian_period,
                        keltner_period=feature_config.keltner_period,
                        keltner_atr_mult=feature_config.keltner_atr_mult,
                        supertrend_period=feature_config.supertrend_period,
                        supertrend_mult=feature_config.supertrend_mult,
                        include_volatility=FeatureGroup.VOLATILITY.value in enabled,
                        include_channels=(
                            FeatureGroup.CHANNELS.value in enabled
                            or FeatureGroup.VOLATILITY.value in enabled
                        ),
                    ),
                )
            )

        if self._enabled(enabled, FeatureGroup.VOLUME):
            from ai.features.volume import compute_volume_features

            feature_maps.append(
                (
                    "volume",
                    compute_volume_features(
                        arrays.open,
                        arrays.high,
                        arrays.low,
                        arrays.close,
                        arrays.volume,
                        rolling_windows=feature_config.rolling_windows,
                        mfi_period=feature_config.mfi_period,
                        cmf_period=feature_config.cmf_period,
                    ),
                )
            )

        if self._enabled(enabled, FeatureGroup.CANDLE_STRUCTURE):
            from ai.features.candle_structure import compute_candle_structure_features

            feature_maps.append(
                (
                    "candle_structure",
                    compute_candle_structure_features(
                        arrays.open,
                        arrays.high,
                        arrays.low,
                        arrays.close,
                        arrays.spread,
                    ),
                )
            )

        if self._enabled(enabled, FeatureGroup.PATTERNS):
            from ai.features.patterns import compute_pattern_features

            feature_maps.append(
                (
                    "patterns",
                    compute_pattern_features(
                        arrays.open,
                        arrays.high,
                        arrays.low,
                        arrays.close,
                        fractal_window=feature_config.fractal_window,
                    ),
                )
            )

        if self._enabled(enabled, FeatureGroup.STRUCTURE):
            from ai.features.structure import compute_structure_features

            feature_maps.append(
                (
                    "structure",
                    compute_structure_features(
                        arrays.high,
                        arrays.low,
                        arrays.close,
                        swing_lookback=feature_config.swing_lookback,
                        support_resistance_lookback=feature_config.support_resistance_lookback,
                    ),
                )
            )

        if self._enabled(enabled, FeatureGroup.SESSION):
            from ai.features.session import compute_session_features

            feature_maps.append(
                (
                    "session",
                    compute_session_features(
                        arrays.timestamps,
                        open_=arrays.open,
                        high=arrays.high,
                        low=arrays.low,
                        close=arrays.close,
                    ),
                )
            )

        if self._enabled(enabled, FeatureGroup.REGIME):
            from ai.features.regime import compute_regime_features

            feature_maps.append(
                (
                    "regime",
                    compute_regime_features(
                        arrays.high,
                        arrays.low,
                        arrays.close,
                        rolling_windows=feature_config.rolling_windows,
                        fx_vix_window=int(getattr(feature_config, "fx_vix_window", 20)),
                        fx_vix_lookback=int(getattr(feature_config, "fx_vix_lookback", 252)),
                    ),
                )
            )

        if self._enabled(enabled, FeatureGroup.MICROSTRUCTURE):
            from ai.features.microstructure import compute_microstructure_features

            bid_vol, ask_vol = _extract_book_volumes(base_candles)
            feature_maps.append(
                (
                    "microstructure",
                    compute_microstructure_features(
                        arrays.open,
                        arrays.high,
                        arrays.low,
                        arrays.close,
                        arrays.volume,
                        arrays.spread,
                        bid_volume=bid_vol,
                        ask_volume=ask_vol,
                        rolling_windows=feature_config.rolling_windows,
                    ),
                )
            )

        if self._enabled(enabled, FeatureGroup.MULTI_TIMEFRAME) and higher_timeframes:
            from ai.features.multi_timeframe import compute_multi_timeframe_features

            feature_maps.append(
                (
                    "multi_timeframe",
                    compute_multi_timeframe_features(
                        arrays.timestamps,
                        higher_timeframes,
                        ema_periods=feature_config.ema_periods,
                        atr_period=feature_config.atr_period,
                    ),
                )
            )

        if self._enabled(enabled, FeatureGroup.CORRELATION) and peer_symbols:
            from ai.features.correlation import compute_correlation_features

            feature_maps.append(
                (
                    "correlation",
                    compute_correlation_features(
                        base_candles,
                        peer_symbols,
                        window=feature_config.correlation_window,
                    ),
                )
            )

        matrix, feature_names = assemble_feature_matrix(feature_maps, row_count=len(arrays.close))
        matrix, timestamps, row_metadata = apply_nan_policy(
            matrix,
            arrays.timestamps,
            dropna=feature_config.dropna,
            fill_method=feature_config.fill_method,
        )
        metadata: Dict[str, Any] = {
            "original_rows": len(arrays.timestamps),
            "output_rows": len(timestamps),
            "feature_count": len(feature_names),
            "enabled_groups": sorted(enabled),
            "generated_groups": [name for name, fmap in feature_maps if fmap],
            "dropna": feature_config.dropna,
            "fill_method": feature_config.fill_method,
            "periods": {
                "sma": list(feature_config.sma_periods),
                "ema": list(feature_config.ema_periods),
                "rolling_windows": list(feature_config.rolling_windows),
                "rsi": feature_config.rsi_period,
                "atr": feature_config.atr_period,
                "adx": feature_config.adx_period,
                "bollinger": feature_config.bollinger_period,
            },
        }
        metadata.update(row_metadata)
        return FeatureFrame(matrix=matrix, feature_names=feature_names, timestamps=timestamps, metadata=metadata)

    @staticmethod
    def _enabled(enabled: set[str], *groups: FeatureGroup) -> bool:
        return any(group.value in enabled for group in groups)

    @staticmethod
    def _select_base_scope(
        candles: Sequence[CandleDict],
        config: AIConfig,
    ) -> Tuple[List[CandleDict], Dict[str, List[CandleDict]], Dict[str, List[CandleDict]]]:
        if not candles:
            return [], {}, {}

        by_timeframe: MutableMapping[str, List[CandleDict]] = {}
        by_symbol: MutableMapping[str, List[CandleDict]] = {}
        for candle in candles:
            timeframe = str(candle.get("timeframe", config.primary_timeframe)).upper()
            symbol = str(candle.get("symbol", config.symbols[0] if config.symbols else "")).upper()
            by_timeframe.setdefault(timeframe, []).append(candle)
            by_symbol.setdefault(symbol, []).append(candle)

        primary_tf = config.primary_timeframe.upper()
        if primary_tf in by_timeframe and len(by_timeframe) > 1:
            base_candidates = list(by_timeframe[primary_tf])
        else:
            base_candidates = list(candles)

        base_symbol = str(base_candidates[0].get("symbol", config.symbols[0] if config.symbols else "")).upper()
        base_timeframe = str(base_candidates[0].get("timeframe", primary_tf)).upper()
        base_candles = [
            c
            for c in base_candidates
            if str(c.get("symbol", base_symbol)).upper() == base_symbol
            and str(c.get("timeframe", base_timeframe)).upper() == base_timeframe
        ]
        base_candles = sorted(base_candles, key=lambda c: normalize_timestamp(c.get("timestamp")))

        higher_timeframes = {
            tf: sorted(rows, key=lambda c: normalize_timestamp(c.get("timestamp")))
            for tf, rows in by_timeframe.items()
            if tf != base_timeframe
        }
        configured_peers = {symbol.upper() for symbol in config.features.correlation_symbols}
        # When correlation is enabled with an empty peer list, accept any other
        # symbols present in the candle batch (caller-supplied cross-asset pack).
        peer_symbols = {
            symbol: sorted(rows, key=lambda c: normalize_timestamp(c.get("timestamp")))
            for symbol, rows in by_symbol.items()
            if symbol != base_symbol and (not configured_peers or symbol in configured_peers)
        }
        return base_candles, higher_timeframes, peer_symbols


# ==============================================================================
# CANDLE AND MATRIX HELPERS
# ==============================================================================


def normalize_timestamp(value: Any) -> datetime:
    """Normalize timestamps to naive UTC datetimes for repository alignment."""

    if isinstance(value, datetime):
        return value.replace(tzinfo=None) if value.tzinfo is not None else value
    if isinstance(value, np.datetime64):
        seconds = value.astype("datetime64[s]").astype(int)
        return datetime.utcfromtimestamp(float(seconds))
    if isinstance(value, (int, float)):
        return datetime.utcfromtimestamp(float(value))
    if isinstance(value, str):
        cleaned = value.replace("Z", "+00:00")
        parsed = datetime.fromisoformat(cleaned)
        return parsed.replace(tzinfo=None) if parsed.tzinfo is not None else parsed
    raise ValueError("Each candle must include a valid timestamp")


def candles_to_arrays(candles: Sequence[CandleDict]) -> CandleArrays:
    """Extract sorted OHLCV arrays from CandleDict records."""

    ordered = sorted(candles, key=lambda c: normalize_timestamp(c.get("timestamp")))
    timestamps = [normalize_timestamp(c.get("timestamp")) for c in ordered]
    open_arr = _extract_required_float(ordered, "open")
    high_arr = _extract_required_float(ordered, "high")
    low_arr = _extract_required_float(ordered, "low")
    close_arr = _extract_required_float(ordered, "close")
    volume_arr = np.array([_first_float(c, ("volume", "tick_volume", "real_volume"), 0.0) for c in ordered], dtype=float)
    spread_arr = np.array([_first_float(c, ("spread",), 0.0) for c in ordered], dtype=float)
    return CandleArrays(
        timestamps=timestamps,
        open=open_arr,
        high=high_arr,
        low=low_arr,
        close=close_arr,
        volume=volume_arr,
        spread=spread_arr,
    )


def assemble_feature_matrix(feature_maps: Iterable[Tuple[str, FeatureMap]], row_count: int) -> Tuple[NDArray[np.floating], List[str]]:
    """Assemble ordered feature maps into a dense matrix."""

    columns: List[NDArray[np.floating]] = []
    names: List[str] = []
    for _group_name, fmap in feature_maps:
        for name, values in fmap.items():
            arr = np.asarray(values, dtype=float)
            if len(arr) != row_count:
                raise ValueError(f"Feature {name!r} length {len(arr)} does not match row count {row_count}")
            columns.append(arr)
            names.append(name)
    if not columns:
        return np.empty((row_count, 0), dtype=float), []
    matrix = np.column_stack(columns).astype(float, copy=False)
    matrix[~np.isfinite(matrix)] = np.nan
    return matrix, names


def apply_nan_policy(
    matrix: NDArray[np.floating],
    timestamps: Sequence[datetime],
    *,
    dropna: bool,
    fill_method: str,
) -> Tuple[NDArray[np.floating], List[datetime], Dict[str, Any]]:
    """Drop or fill warm-up NaNs according to FeatureConfig."""

    if matrix.shape[1] == 0:
        return matrix, list(timestamps), {"dropped_rows": 0, "nan_policy": "empty"}

    clean = np.array(matrix, dtype=float, copy=True)
    clean[~np.isfinite(clean)] = np.nan
    if dropna:
        valid = ~np.isnan(clean).any(axis=1)
        return clean[valid], [ts for ts, keep in zip(timestamps, valid) if bool(keep)], {
            "dropped_rows": int(len(timestamps) - np.count_nonzero(valid)),
            "nan_policy": "dropna",
        }

    filled = fill_nan_matrix(clean, method=fill_method)
    return filled, list(timestamps), {"dropped_rows": 0, "nan_policy": f"fill:{fill_method}"}


def fill_nan_matrix(matrix: NDArray[np.floating], method: str = "ffill") -> NDArray[np.floating]:
    """Fill NaNs without changing row count."""

    method_l = method.lower()
    out = np.array(matrix, dtype=float, copy=True)
    if method_l in {"none", "nan"}:
        return out
    if method_l == "zero":
        out[np.isnan(out)] = 0.0
        return out
    if method_l == "mean":
        means = np.nanmean(out, axis=0)
        means[np.isnan(means)] = 0.0
        rows, cols = np.where(np.isnan(out))
        out[rows, cols] = means[cols]
        return out
    if method_l == "bfill":
        out = _backward_fill(out)
        out = _forward_fill(out)
        out[np.isnan(out)] = 0.0
        return out
    out = _forward_fill(out)
    out = _backward_fill(out)
    out[np.isnan(out)] = 0.0
    return out


def create_feature_engine(config: AIConfig | None = None) -> FeatureEngine:
    """Factory for a configured FeatureEngine."""

    return FeatureEngine(config=config)


def _extract_required_float(candles: Sequence[CandleDict], key: str) -> NDArray[np.floating]:
    values: List[float] = []
    for candle in candles:
        if key not in candle:
            raise ValueError(f"Each candle must include {key!r}")
        values.append(float(candle[key]))  # type: ignore[index]
    return np.array(values, dtype=float)


def _first_float(candle: Mapping[str, Any], keys: Sequence[str], default: float) -> float:
    for key in keys:
        value = candle.get(key)
        if value is not None:
            return float(value)
    return default


def _extract_book_volumes(
    candles: Sequence[CandleDict],
) -> tuple[NDArray[np.floating] | None, NDArray[np.floating] | None]:
    """Pull optional bid/ask volumes from candle fields or metadata."""

    if not candles:
        return None, None
    bids: list[float] = []
    asks: list[float] = []
    any_book = False
    for candle in candles:
        meta = candle.get("metadata") if isinstance(candle, Mapping) else None
        meta = meta if isinstance(meta, Mapping) else {}
        bid = _first_float(candle, ("bid_volume", "bid_vol", "buy_volume"), np.nan)
        ask = _first_float(candle, ("ask_volume", "ask_vol", "sell_volume"), np.nan)
        if not np.isfinite(bid):
            bid = _first_float(meta, ("bid_volume", "bid_vol", "buy_volume"), np.nan)
        if not np.isfinite(ask):
            ask = _first_float(meta, ("ask_volume", "ask_vol", "sell_volume"), np.nan)
        if np.isfinite(bid) or np.isfinite(ask):
            any_book = True
        bids.append(float(bid) if np.isfinite(bid) else np.nan)
        asks.append(float(ask) if np.isfinite(ask) else np.nan)
    if not any_book:
        return None, None
    return np.asarray(bids, dtype=float), np.asarray(asks, dtype=float)


def _forward_fill(matrix: NDArray[np.floating]) -> NDArray[np.floating]:
    out = np.array(matrix, dtype=float, copy=True)
    for col in range(out.shape[1]):
        last = np.nan
        for row in range(out.shape[0]):
            if np.isnan(out[row, col]):
                out[row, col] = last
            else:
                last = out[row, col]
    return out


def _backward_fill(matrix: NDArray[np.floating]) -> NDArray[np.floating]:
    out = np.array(matrix, dtype=float, copy=True)
    for col in range(out.shape[1]):
        next_value = np.nan
        for row in range(out.shape[0] - 1, -1, -1):
            if np.isnan(out[row, col]):
                out[row, col] = next_value
            else:
                next_value = out[row, col]
    return out
