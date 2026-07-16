"""
database.models.market_model - Legacy MarketModel used by SymbolManager.

Preserves the public API: MarketModel(database).add_market(...).
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from typing import Any, Optional
from uuid import uuid4

logger = logging.getLogger(__name__)


class MarketModel:
    """Compatibility facade writing market rows for the collector."""

    def __init__(self, database: Any):
        self.db = database

    def _execute(self, sql: str, params: tuple = ()) -> Any:
        if hasattr(self.db, "get_adapter"):
            return self.db.get_adapter().execute(sql, params)
        if hasattr(self.db, "execute"):
            return self.db.execute(sql, params)
        if hasattr(self.db, "connection"):
            return self.db.connection.execute(sql, params)
        raise TypeError(f"Unsupported database object: {type(self.db)!r}")

    def _commit(self) -> None:
        if hasattr(self.db, "commit"):
            self.db.commit()
        elif hasattr(self.db, "get_adapter"):
            self.db.get_adapter().commit()
        elif hasattr(self.db, "connection"):
            self.db.connection.commit()

    def add_market(
        self,
        symbol: str,
        category: str = "UNKNOWN",
        description: Optional[str] = None,
        digits: Optional[int] = None,
        spread: Optional[float] = None,
        point: Optional[float] = None,
        trade_mode: Optional[int] = None,
        currency_base: Optional[str] = None,
        currency_profit: Optional[str] = None,
        currency_margin: Optional[str] = None,
        broker_id: Optional[int] = None,
        **kwargs: Any,
    ) -> None:
        """Insert or update a market/symbol row."""
        now = datetime.utcnow().isoformat(timespec="seconds")
        metadata = {
            "spread": spread,
            "trade_mode": trade_mode,
            "currency_margin": currency_margin,
            "currency_profit": currency_profit,
            **{k: v for k, v in kwargs.items() if v is not None},
        }
        self._execute(
            """
            INSERT INTO markets (
                broker_id, symbol, market_type, status, description,
                base_currency, quote_currency, pip_size, point, digits,
                contract_size, metadata, created_at, updated_at, active, name, category
            ) VALUES (?, ?, ?, 'active', ?, ?, ?, ?, ?, ?, NULL, ?, ?, ?, 1, ?, ?)
            ON CONFLICT(symbol) DO UPDATE SET
                market_type=excluded.market_type,
                description=excluded.description,
                base_currency=excluded.base_currency,
                quote_currency=excluded.quote_currency,
                point=excluded.point,
                digits=excluded.digits,
                metadata=excluded.metadata,
                updated_at=excluded.updated_at,
                active=1,
                name=excluded.name,
                category=excluded.category,
                status='active'
            """,
            (
                broker_id,
                symbol.upper(),
                str(category or "UNKNOWN").upper(),
                description,
                currency_base,
                currency_profit or currency_base,
                point,
                point,
                digits,
                json.dumps(metadata),
                now,
                now,
                symbol.upper(),
                str(category or "UNKNOWN").upper(),
            ),
        )
        self._commit()
        logger.debug("Market upserted: %s", symbol)
