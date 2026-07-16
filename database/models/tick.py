"""database.models.tick - Tick domain model."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, Optional


class TickStatus(str, Enum):
    ACTIVE = "active"
    ARCHIVED = "archived"
    INVALID = "invalid"


@dataclass
class Tick:
    tick_id: Optional[int]
    tick_uuid: str
    symbol: str
    timestamp: datetime
    bid: float
    ask: float
    last: float = 0.0
    volume: float = 0.0
    flags: int = 0
    market_id: Optional[int] = None
    broker_id: Optional[int] = None
    status: Optional[TickStatus] = TickStatus.ACTIVE
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
