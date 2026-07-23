"""
database/schema.py - Canonical schema for MarketAI repositories + collector.

Creates tables matching repository models while retaining collector-compatible
columns (active/name on markets, sync_status).
"""

from __future__ import annotations

from typing import Any


SCHEMA_SQL = [
    """
    CREATE TABLE IF NOT EXISTS brokers (
        broker_id INTEGER PRIMARY KEY AUTOINCREMENT,
        broker_uuid TEXT UNIQUE,
        name TEXT UNIQUE NOT NULL,
        broker_type TEXT DEFAULT 'cfd',
        server TEXT,
        host TEXT,
        port INTEGER,
        description TEXT,
        login TEXT,
        password_encrypted TEXT,
        status TEXT DEFAULT 'active',
        metadata TEXT DEFAULT '{}',
        created_at TEXT,
        updated_at TEXT
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS currencies (
        currency_id INTEGER PRIMARY KEY AUTOINCREMENT,
        currency_uuid TEXT UNIQUE,
        code TEXT UNIQUE NOT NULL,
        name TEXT NOT NULL,
        currency_type TEXT DEFAULT 'fiat',
        symbol TEXT,
        iso_number INTEGER,
        decimals INTEGER DEFAULT 2,
        description TEXT,
        status TEXT DEFAULT 'active',
        metadata TEXT DEFAULT '{}',
        created_at TEXT,
        updated_at TEXT
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS timeframes (
        timeframe_id INTEGER PRIMARY KEY AUTOINCREMENT,
        timeframe_uuid TEXT UNIQUE,
        name TEXT UNIQUE NOT NULL,
        seconds INTEGER NOT NULL,
        sort_order INTEGER DEFAULT 0,
        description TEXT,
        category TEXT,
        status TEXT DEFAULT 'active',
        metadata TEXT DEFAULT '{}',
        created_at TEXT,
        updated_at TEXT
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS markets (
        market_id INTEGER PRIMARY KEY AUTOINCREMENT,
        broker_id INTEGER,
        symbol TEXT NOT NULL,
        canonical_symbol TEXT,
        market_type TEXT,
        status TEXT DEFAULT 'active',
        description TEXT,
        base_currency TEXT,
        quote_currency TEXT,
        pip_size REAL,
        point REAL,
        digits INTEGER,
        contract_size REAL,
        metadata TEXT DEFAULT '{}',
        created_at TEXT,
        updated_at TEXT,
        -- collector compatibility
        name TEXT,
        category TEXT,
        active INTEGER DEFAULT 1,
        spread REAL,
        trade_mode INTEGER,
        currency_base TEXT,
        currency_profit TEXT,
        currency_margin TEXT,
        FOREIGN KEY(broker_id) REFERENCES brokers(broker_id),
        -- Same broker symbol may exist once per broker; cross-broker join uses canonical_symbol
        UNIQUE(broker_id, symbol)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS symbol_aliases (
        alias_id INTEGER PRIMARY KEY AUTOINCREMENT,
        alias TEXT NOT NULL,
        canonical_symbol TEXT NOT NULL,
        asset_class TEXT,
        description TEXT,
        metadata TEXT DEFAULT '{}',
        created_at TEXT,
        updated_at TEXT,
        UNIQUE(alias)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS symbols (
        symbol_id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT UNIQUE NOT NULL,
        category TEXT,
        description TEXT,
        digits INTEGER,
        point REAL,
        spread REAL,
        trade_mode INTEGER,
        currency_base TEXT,
        currency_profit TEXT,
        currency_margin TEXT,
        broker_id INTEGER,
        market_id INTEGER,
        status TEXT DEFAULT 'active',
        metadata TEXT DEFAULT '{}',
        created_at TEXT,
        updated_at TEXT
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS candles (
        candle_id INTEGER PRIMARY KEY AUTOINCREMENT,
        candle_uuid TEXT,
        symbol TEXT NOT NULL,
        timeframe TEXT NOT NULL,
        timestamp TEXT NOT NULL,
        open REAL NOT NULL,
        high REAL NOT NULL,
        low REAL NOT NULL,
        close REAL NOT NULL,
        volume REAL DEFAULT 0,
        market_id INTEGER,
        broker_id INTEGER,
        spread REAL,
        tick_volume INTEGER,
        status TEXT DEFAULT 'active',
        metadata TEXT DEFAULT '{}',
        created_at TEXT,
        updated_at TEXT,
        -- Broker-scoped uniqueness: same symbol+time can exist for different brokers
        UNIQUE(broker_id, symbol, timeframe, timestamp)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS ticks (
        tick_id INTEGER PRIMARY KEY AUTOINCREMENT,
        tick_uuid TEXT,
        symbol TEXT NOT NULL,
        timestamp TEXT NOT NULL,
        bid REAL NOT NULL,
        ask REAL NOT NULL,
        last REAL DEFAULT 0,
        volume REAL DEFAULT 0,
        flags INTEGER DEFAULT 0,
        market_id INTEGER,
        broker_id INTEGER,
        status TEXT DEFAULT 'active',
        metadata TEXT DEFAULT '{}',
        created_at TEXT,
        updated_at TEXT,
        UNIQUE(broker_id, symbol, timestamp, bid, ask)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS sync_status (
        market_id INTEGER NOT NULL,
        timeframe TEXT NOT NULL,
        status TEXT NOT NULL,
        last_synced TEXT,
        last_candle_time TEXT,
        candles_count INTEGER DEFAULT 0,
        error_message TEXT,
        PRIMARY KEY (market_id, timeframe)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS schema_migrations (
        version TEXT PRIMARY KEY,
        applied_at TEXT NOT NULL
    )
    """,
]


def _execute(db: Any, sql: str, params: tuple = ()) -> None:
    if hasattr(db, "get_adapter"):
        db.get_adapter().execute(sql, params)
        return
    if hasattr(db, "execute"):
        db.execute(sql, params)
        return
    if hasattr(db, "connection"):
        db.connection.execute(sql, params)
        return
    raise TypeError(f"Unsupported db object: {type(db)!r}")


def _commit(db: Any) -> None:
    if hasattr(db, "commit"):
        db.commit()
    elif hasattr(db, "get_adapter"):
        db.get_adapter().commit()
    elif hasattr(db, "connection"):
        db.connection.commit()


def create_schema(db: Any) -> None:
    """Create the full MarketAI schema."""
    if hasattr(db, "create_schema") and db.__class__.__name__ == "Database":
        # Prefer canonical statements below for Database too
        pass
    for statement in SCHEMA_SQL:
        _execute(db, statement)
    _commit(db)


# Keep Database.create_schema in sync by also exposing statements
def apply_schema_to_connection(connection: Any) -> None:
    for statement in SCHEMA_SQL:
        connection.execute(statement)
    connection.commit()
