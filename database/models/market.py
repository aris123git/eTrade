"""database.models.market - Market/symbol domain model."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, Optional


class MarketType(str, Enum):
    FOREX = "FOREX"
    METAL = "METAL"
    ENERGY = "ENERGY"
    CRYPTO = "CRYPTO"
    INDEX = "INDEX"
    STOCK = "STOCK"
    CFD = "CFD"
    COMMODITY = "COMMODITY"
    UNKNOWN = "UNKNOWN"


class MarketStatus(str, Enum):
    ACTIVE = "active"
    INACTIVE = "inactive"
    DELISTED = "delisted"


@dataclass
class Market:
    market_id: Optional[int]
    broker_id: Optional[int]
    symbol: str
    market_type: Optional[MarketType]
    status: Optional[MarketStatus]
    description: Optional[str] = None
    base_currency: Optional[str] = None
    quote_currency: Optional[str] = None
    pip_size: Optional[float] = None
    point: Optional[float] = None
    digits: Optional[int] = None
    contract_size: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
