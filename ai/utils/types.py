"""
ai/utils/types.py - Shared typed contracts for the AI engine

VERSION: 1.0.0
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, TypedDict, Union
import numpy as np
from numpy.typing import NDArray


FeatureMatrix = NDArray[np.floating]
LabelArray = NDArray[np.floating]


class CandleDict(TypedDict, total=False):
    """Canonical candle dictionary used across the AI pipeline."""

    symbol: str
    timeframe: str
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float
    tick_volume: float
    real_volume: float
    spread: float
    market_id: int
    broker_id: int


class SignalType(str, Enum):
    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"
    CLOSE = "CLOSE"
    REDUCE = "REDUCE_POSITION"


class OrderSide(str, Enum):
    BUY = "BUY"
    SELL = "SELL"


class OrderType(str, Enum):
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"


class PositionSide(str, Enum):
    LONG = "LONG"
    SHORT = "SHORT"
    FLAT = "FLAT"


@dataclass
class PredictionResult:
    """Standard prediction payload returned by live and batch services."""

    symbol: str
    timeframe: str
    timestamp: datetime
    prediction: Union[float, int]
    probabilities: Optional[Dict[str, float]] = None
    confidence: float = 0.0
    expected_return: Optional[float] = None
    feature_contributions: Optional[Dict[str, float]] = None
    model_version: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "symbol": self.symbol,
            "timeframe": self.timeframe,
            "timestamp": self.timestamp.isoformat() if isinstance(self.timestamp, datetime) else self.timestamp,
            "prediction": self.prediction,
            "probabilities": self.probabilities,
            "confidence": self.confidence,
            "expected_return": self.expected_return,
            "feature_contributions": self.feature_contributions,
            "model_version": self.model_version,
            "metadata": self.metadata,
        }
