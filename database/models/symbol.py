"""database.models.symbol - Symbol domain model."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, Optional


class SymbolStatus(str, Enum):
    ACTIVE = "active"
    INACTIVE = "inactive"


@dataclass
class Symbol:
    """Canonical trading symbol metadata (often mirrored from markets)."""

    symbol_id: Optional[int]
    name: str
    category: Optional[str] = None
    description: Optional[str] = None
    digits: Optional[int] = None
    point: Optional[float] = None
    spread: Optional[float] = None
    trade_mode: Optional[int] = None
    currency_base: Optional[str] = None
    currency_profit: Optional[str] = None
    currency_margin: Optional[str] = None
    broker_id: Optional[int] = None
    market_id: Optional[int] = None
    status: Optional[SymbolStatus] = SymbolStatus.ACTIVE
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
