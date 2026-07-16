"""database.models.broker - Broker domain model."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, Optional


class BrokerType(str, Enum):
    CFD = "cfd"
    FOREX = "forex"
    FUTURES = "futures"
    CRYPTO = "crypto"
    MULTI = "multi"
    OTHER = "other"


class BrokerStatus(str, Enum):
    ACTIVE = "active"
    INACTIVE = "inactive"
    MAINTENANCE = "maintenance"
    ARCHIVED = "archived"


@dataclass
class Broker:
    broker_id: Optional[int]
    broker_uuid: str
    name: str
    broker_type: Optional[BrokerType] = BrokerType.CFD
    server: Optional[str] = None
    host: Optional[str] = None
    port: Optional[int] = None
    description: Optional[str] = None
    login: Optional[str] = None
    password_encrypted: Optional[str] = None
    status: Optional[BrokerStatus] = BrokerStatus.ACTIVE
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
