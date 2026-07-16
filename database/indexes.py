"""
database/indexes.py - Index creation for MarketAI tables.
"""

from __future__ import annotations

from typing import Any


INDEX_SQL = [
    "CREATE INDEX IF NOT EXISTS idx_brokers_status ON brokers(status)",
    "CREATE INDEX IF NOT EXISTS idx_currencies_status ON currencies(status)",
    "CREATE INDEX IF NOT EXISTS idx_timeframes_status ON timeframes(status)",
    "CREATE INDEX IF NOT EXISTS idx_markets_symbol ON markets(symbol)",
    "CREATE INDEX IF NOT EXISTS idx_markets_status ON markets(status)",
    "CREATE INDEX IF NOT EXISTS idx_markets_active ON markets(active)",
    "CREATE INDEX IF NOT EXISTS idx_markets_broker ON markets(broker_id)",
    "CREATE INDEX IF NOT EXISTS idx_symbols_name ON symbols(name)",
    "CREATE INDEX IF NOT EXISTS idx_candles_primary ON candles(symbol, timeframe, timestamp)",
    "CREATE INDEX IF NOT EXISTS idx_candles_market ON candles(market_id)",
    "CREATE INDEX IF NOT EXISTS idx_candles_broker ON candles(broker_id)",
    "CREATE INDEX IF NOT EXISTS idx_candles_active ON candles(symbol, timeframe, timestamp) WHERE status = 'active'",
    "CREATE INDEX IF NOT EXISTS idx_ticks_primary ON ticks(symbol, timestamp)",
    "CREATE INDEX IF NOT EXISTS idx_ticks_active ON ticks(symbol, timestamp) WHERE status = 'active'",
    "CREATE INDEX IF NOT EXISTS idx_sync_status_market ON sync_status(market_id)",
]


def _execute(db: Any, sql: str) -> None:
    if hasattr(db, "get_adapter"):
        db.get_adapter().execute(sql)
        return
    if hasattr(db, "execute"):
        db.execute(sql)
        return
    if hasattr(db, "connection"):
        db.connection.execute(sql)
        return
    raise TypeError(f"Unsupported db object: {type(db)!r}")


def _commit(db: Any) -> None:
    if hasattr(db, "commit"):
        db.commit()
    elif hasattr(db, "get_adapter"):
        db.get_adapter().commit()
    elif hasattr(db, "connection"):
        db.connection.commit()


def create_indexes(db: Any) -> None:
    """Create all performance indexes."""
    for statement in INDEX_SQL:
        _execute(db, statement)
    _commit(db)
