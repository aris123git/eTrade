"""
database/repositories/symbol_repository.py - Symbol Repository

VERSION: 1.0.0
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional
from uuid import uuid4

from database.core.connection import DatabaseManager
from database.models.symbol import Symbol, SymbolStatus
from database.repositories.base_repository import BaseRepository

logger = logging.getLogger(__name__)


class SymbolRepository(BaseRepository[Symbol]):
    """Repository for trading symbol metadata."""

    TABLE = "symbols"
    MODEL = Symbol

    def __init__(self, db_manager: DatabaseManager):
        super().__init__(db_manager)
        self.logger = logging.getLogger(__name__)

    def create(
        self,
        name: str,
        category: Optional[str] = None,
        description: Optional[str] = None,
        digits: Optional[int] = None,
        point: Optional[float] = None,
        spread: Optional[float] = None,
        trade_mode: Optional[int] = None,
        currency_base: Optional[str] = None,
        currency_profit: Optional[str] = None,
        currency_margin: Optional[str] = None,
        broker_id: Optional[int] = None,
        market_id: Optional[int] = None,
        status: SymbolStatus = SymbolStatus.ACTIVE,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Symbol:
        now = datetime.utcnow()
        symbol = Symbol(
            symbol_id=None,
            name=name.upper(),
            category=category,
            description=description,
            digits=digits,
            point=point,
            spread=spread,
            trade_mode=trade_mode,
            currency_base=currency_base,
            currency_profit=currency_profit,
            currency_margin=currency_margin,
            broker_id=broker_id,
            market_id=market_id,
            status=status,
            metadata=metadata or {},
            created_at=now,
            updated_at=now,
        )
        symbol_id = self.upsert(self._entity_to_dict(symbol), ["name"])
        symbol.symbol_id = int(symbol_id) if symbol_id is not None else None
        return symbol

    def find_by_name(self, name: str) -> Optional[Symbol]:
        return self.find_one(name=name.upper())

    def find_active(self) -> List[Symbol]:
        return self.find_all_where("status = ?", (SymbolStatus.ACTIVE.value,))

    def _entity_to_dict(self, symbol: Symbol) -> Dict[str, Any]:
        return {
            "symbol_id": symbol.symbol_id,
            "name": symbol.name,
            "category": symbol.category,
            "description": symbol.description,
            "digits": symbol.digits,
            "point": symbol.point,
            "spread": symbol.spread,
            "trade_mode": symbol.trade_mode,
            "currency_base": symbol.currency_base,
            "currency_profit": symbol.currency_profit,
            "currency_margin": symbol.currency_margin,
            "broker_id": symbol.broker_id,
            "market_id": symbol.market_id,
            "status": symbol.status.value if symbol.status else None,
            "metadata": json.dumps(symbol.metadata) if symbol.metadata else "{}",
            "created_at": symbol.created_at.isoformat(timespec="seconds") if isinstance(symbol.created_at, datetime) else symbol.created_at,
            "updated_at": symbol.updated_at.isoformat(timespec="seconds") if isinstance(symbol.updated_at, datetime) else symbol.updated_at,
        }

    def _row_to_entity(self, row: Dict[str, Any]) -> Symbol:
        created = row.get("created_at")
        updated = row.get("updated_at")
        if isinstance(created, str):
            created = datetime.fromisoformat(created)
        if isinstance(updated, str):
            updated = datetime.fromisoformat(updated)
        return Symbol(
            symbol_id=row["symbol_id"],
            name=row["name"],
            category=row.get("category"),
            description=row.get("description"),
            digits=row.get("digits"),
            point=row.get("point"),
            spread=row.get("spread"),
            trade_mode=row.get("trade_mode"),
            currency_base=row.get("currency_base"),
            currency_profit=row.get("currency_profit"),
            currency_margin=row.get("currency_margin"),
            broker_id=row.get("broker_id"),
            market_id=row.get("market_id"),
            status=SymbolStatus(row["status"]) if row.get("status") else SymbolStatus.ACTIVE,
            metadata=json.loads(row["metadata"]) if row.get("metadata") else {},
            created_at=created,
            updated_at=updated,
        )

    def _get_id(self, symbol: Symbol) -> int:
        return symbol.symbol_id
