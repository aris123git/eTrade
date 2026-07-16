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
        canonical_symbol: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        """Insert or update a market/symbol row for a specific broker."""
        from core.symbol_identity import canonicalize

        now = datetime.utcnow().isoformat(timespec="seconds")
        ident = canonicalize(symbol)
        canonical = canonical_symbol or ident.canonical_symbol
        base = currency_base or ident.base_currency
        quote = currency_profit or ident.quote_currency
        category_value = str(category or ident.asset_class or "UNKNOWN").upper()
        metadata = {
            "spread": spread,
            "trade_mode": trade_mode,
            "currency_margin": currency_margin,
            "currency_profit": currency_profit,
            "broker_symbol": symbol,
            "canonical_symbol": canonical,
            **{k: v for k, v in kwargs.items() if v is not None},
        }
        # Upsert on (broker_id, symbol). When broker_id is NULL, fall back to symbol match.
        self._execute(
            """
            INSERT INTO markets (
                broker_id, symbol, canonical_symbol, market_type, status, description,
                base_currency, quote_currency, pip_size, point, digits,
                contract_size, metadata, created_at, updated_at, active, name, category
            ) VALUES (?, ?, ?, ?, 'active', ?, ?, ?, ?, ?, ?, NULL, ?, ?, ?, 1, ?, ?)
            ON CONFLICT(broker_id, symbol) DO UPDATE SET
                canonical_symbol=excluded.canonical_symbol,
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
                symbol,
                canonical,
                category_value,
                description,
                base,
                quote,
                point,
                point,
                digits,
                json.dumps(metadata),
                now,
                now,
                symbol,
                category_value,
            ),
        )
        self._commit()
        logger.debug("Market upserted: %s [%s] broker=%s", symbol, canonical, broker_id)
