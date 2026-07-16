"""database.models.timeframe - Timeframe domain model."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, Optional


class TimeframeCategory(str, Enum):
    MINUTE = "minute"
    HOURLY = "hourly"
    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"
    CUSTOM = "custom"


class TimeframeStatus(str, Enum):
    ACTIVE = "active"
    INACTIVE = "inactive"


@dataclass
class Timeframe:
    timeframe_id: Optional[int]
    timeframe_uuid: str
    name: str
    seconds: int
    sort_order: int = 0
    description: Optional[str] = None
    category: Optional[TimeframeCategory] = None
    status: Optional[TimeframeStatus] = TimeframeStatus.ACTIVE
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
