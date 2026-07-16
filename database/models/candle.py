"""database.models.candle - Candle domain model."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, Optional


class CandleStatus(str, Enum):
    ACTIVE = "active"
    ARCHIVED = "archived"
    INVALID = "invalid"


@dataclass
class Candle:
    candle_id: Optional[int]
    candle_uuid: str
    symbol: str
    timeframe: str
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float
    market_id: Optional[int] = None
    broker_id: Optional[int] = None
    spread: Optional[float] = None
    tick_volume: Optional[int] = None
    status: Optional[CandleStatus] = CandleStatus.ACTIVE
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
