"""
ai/features/__init__.py - Public feature engineering API

RESPONSIBILITY:
Expose the production feature engine, FeatureFrame contract, factories, and
indicator group entry points.

VERSION: 1.0.0
"""

from ai.features.candle_structure import CandleDirection, compute_candle_structure_features
from ai.features.correlation import (
    ASSET_CLASS_BY_SYMBOL,
    DEFAULT_CROSS_ASSET_SYMBOLS,
    AssetClass,
    CorrelationMeasure,
    CorrelationSpec,
    asset_class_for,
    compute_correlation_features,
)
from ai.features.engine import (
    CandleArrays,
    FeatureFrame,
    FeatureGroup,
    FeatureMap,
    FeatureEngine,
    apply_nan_policy,
    assemble_feature_matrix,
    candles_to_arrays,
    create_feature_engine,
)
from ai.features.microstructure import MicrostructureSignal, compute_microstructure_features
from ai.features.momentum import MomentumIndicator, compute_momentum_features
from ai.features.moving_averages import MovingAverageKind, compute_moving_average_features
from ai.features.multi_timeframe import AlignmentMode, TimeframeFeatureSpec, compute_multi_timeframe_features
from ai.features.patterns import PatternKind, compute_pattern_features
from ai.features.price import PriceBasis, compute_price_features
from ai.features.regime import MarketRegime, compute_regime_features
from ai.features.session import TradingSession, compute_session_features
from ai.features.structure import TrendDirection, compute_structure_features
from ai.features.volatility import VolatilityIndicator, compute_volatility_features
from ai.features.volume import VolumeIndicator, compute_volume_features

__all__ = [
    "ASSET_CLASS_BY_SYMBOL",
    "AlignmentMode",
    "AssetClass",
    "CandleArrays",
    "CandleDirection",
    "CorrelationMeasure",
    "CorrelationSpec",
    "DEFAULT_CROSS_ASSET_SYMBOLS",
    "FeatureEngine",
    "FeatureFrame",
    "FeatureGroup",
    "FeatureMap",
    "MarketRegime",
    "MicrostructureSignal",
    "MomentumIndicator",
    "MovingAverageKind",
    "PatternKind",
    "PriceBasis",
    "TimeframeFeatureSpec",
    "TradingSession",
    "TrendDirection",
    "VolatilityIndicator",
    "VolumeIndicator",
    "apply_nan_policy",
    "assemble_feature_matrix",
    "asset_class_for",
    "candles_to_arrays",
    "compute_candle_structure_features",
    "compute_correlation_features",
    "compute_microstructure_features",
    "compute_momentum_features",
    "compute_moving_average_features",
    "compute_multi_timeframe_features",
    "compute_pattern_features",
    "compute_price_features",
    "compute_regime_features",
    "compute_session_features",
    "compute_structure_features",
    "compute_volatility_features",
    "compute_volume_features",
    "create_feature_engine",
]
