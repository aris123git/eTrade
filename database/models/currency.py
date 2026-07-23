"""database.models.currency - Currency domain model."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, Optional


class CurrencyType(str, Enum):
    FIAT = "fiat"
    CRYPTO = "crypto"
    METAL = "metal"
    COMMODITY = "commodity"
    OTHER = "other"
    UNKNOWN = "unknown"


class CurrencyStatus(str, Enum):
    ACTIVE = "active"
    INACTIVE = "inactive"


@dataclass
class Currency:
    currency_id: Optional[int]
    currency_uuid: str
    code: str
    name: str
    currency_type: Optional[CurrencyType] = CurrencyType.FIAT
    symbol: Optional[str] = None
    iso_number: Optional[int] = None
    decimals: int = 2
    description: Optional[str] = None
    status: Optional[CurrencyStatus] = CurrencyStatus.ACTIVE
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
