"""
database/database.py - SQLite compatibility layer for MarketAI.

The legacy collector imports ``Database`` directly, while newer modules expect
a manager/adapter shape. This module keeps both surfaces available without
pulling in the incomplete repository-layer dependencies.
"""

from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterable, Iterator, Optional, Sequence

from core.config import DATABASE_PATH


class Database:
    """Small SQLite wrapper used by main.py and legacy collectors."""

    def __init__(self, path: Optional[Any] = None):
        self.path = Path(path or DATABASE_PATH)
        self.path.parent.mkdir(parents=True, exist_ok=True)

        self.connection = sqlite3.connect(str(self.path), check_same_thread=False)
        self.connection.row_factory = sqlite3.Row
        self.cursor = self.connection.cursor()
        self.enable_performance()

    def enable_performance(self) -> None:
        """Enable SQLite pragmas suitable for collector workloads."""
        self.cursor.execute("PRAGMA journal_mode=WAL;")
        self.cursor.execute("PRAGMA synchronous=NORMAL;")
        self.cursor.execute("PRAGMA temp_store=MEMORY;")
        self.cursor.execute("PRAGMA foreign_keys=ON;")
        self.connection.commit()

    @contextmanager
    def transaction(self) -> Iterator[sqlite3.Connection]:
        """Run statements inside a transaction."""
        try:
            yield self.connection
            self.connection.commit()
        except Exception:
            self.connection.rollback()
            raise

    @contextmanager
    def connect(self) -> Iterator[sqlite3.Connection]:
        """Return the managed connection as a context manager."""
        yield self.connection

    def get_connection(self) -> sqlite3.Connection:
        """Return the underlying sqlite3 connection."""
        return self.connection

    def get_adapter(self) -> "Database":
        """Expose this object as a lightweight database adapter."""
        return self

    def execute(self, query: str, values: Sequence[Any] = ()) -> sqlite3.Cursor:
        """Execute a SQL statement and return the cursor."""
        self.cursor.execute(query, tuple(values))
        return self.cursor

    def executemany(self, query: str, values: Iterable[Sequence[Any]]) -> sqlite3.Cursor:
        """Execute a SQL statement for many parameter sets."""
        self.cursor.executemany(query, list(values))
        return self.cursor

    def fetchone(self) -> Optional[sqlite3.Row]:
        """Fetch one row from the last cursor operation."""
        return self.cursor.fetchone()

    def fetchall(self) -> list[sqlite3.Row]:
        """Fetch all rows from the last cursor operation."""
        return self.cursor.fetchall()

    def fetch_one(self, query: str, values: Sequence[Any] = ()) -> Optional[sqlite3.Row]:
        """Adapter-style fetch-one helper."""
        return self.execute(query, values).fetchone()

    def fetch_all(self, query: str, values: Sequence[Any] = ()) -> list[sqlite3.Row]:
        """Adapter-style fetch-all helper."""
        return self.execute(query, values).fetchall()

    def commit(self) -> None:
        """Commit the current transaction."""
        self.connection.commit()

    def rollback(self) -> None:
        """Roll back the current transaction."""
        self.connection.rollback()

    def close(self) -> None:
        """Close the database connection."""
        self.connection.close()

    def create_schema(self) -> None:
        """Create the minimal schema required by main.py and seed.py."""
        statements = [
            """
            CREATE TABLE IF NOT EXISTS timeframes (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT UNIQUE NOT NULL,
                seconds INTEGER NOT NULL,
                sort_order INTEGER,
                description TEXT,
                category TEXT,
                status TEXT DEFAULT 'active',
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                updated_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS currencies (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                code TEXT UNIQUE NOT NULL,
                name TEXT NOT NULL,
                currency_type TEXT DEFAULT 'fiat',
                symbol TEXT,
                iso_number INTEGER,
                decimals INTEGER DEFAULT 2,
                status TEXT DEFAULT 'active',
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                updated_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS brokers (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT UNIQUE NOT NULL,
                broker_type TEXT DEFAULT 'cfd',
                server TEXT,
                description TEXT,
                status TEXT DEFAULT 'active',
                metadata TEXT DEFAULT '{}',
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                updated_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS markets (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT UNIQUE,
                symbol TEXT UNIQUE,
                category TEXT,
                description TEXT,
                digits INTEGER,
                spread REAL,
                point REAL,
                trade_mode INTEGER,
                currency_base TEXT,
                currency_profit TEXT,
                currency_margin TEXT,
                active INTEGER DEFAULT 1,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                updated_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS candles (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                market_id INTEGER NOT NULL,
                timeframe_id INTEGER NOT NULL,
                time INTEGER NOT NULL,
                open REAL NOT NULL,
                high REAL NOT NULL,
                low REAL NOT NULL,
                close REAL NOT NULL,
                tick_volume INTEGER,
                spread INTEGER,
                real_volume INTEGER,
                flags INTEGER DEFAULT 0,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(market_id, timeframe_id, time)
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS sync_status (
                market_id INTEGER NOT NULL,
                timeframe_id INTEGER NOT NULL,
                status TEXT NOT NULL,
                last_synced TEXT,
                last_candle_time INTEGER,
                candles_count INTEGER DEFAULT 0,
                error_message TEXT,
                PRIMARY KEY (market_id, timeframe_id)
            )
            """,
        ]
        for statement in statements:
            self.execute(statement)
        self.commit()

    def create_indexes(self) -> None:
        """Create indexes for the compatibility schema."""
        statements = [
            "CREATE INDEX IF NOT EXISTS idx_timeframes_name ON timeframes(name)",
            "CREATE INDEX IF NOT EXISTS idx_currencies_code ON currencies(code)",
            "CREATE INDEX IF NOT EXISTS idx_brokers_name ON brokers(name)",
            "CREATE INDEX IF NOT EXISTS idx_markets_active ON markets(active)",
            "CREATE INDEX IF NOT EXISTS idx_markets_symbol ON markets(symbol)",
            "CREATE INDEX IF NOT EXISTS idx_candles_lookup ON candles(market_id, timeframe_id, time)",
        ]
        for statement in statements:
            self.execute(statement)
        self.commit()


class DatabaseManager:
    """Thin manager wrapper that satisfies collector DatabaseManager imports."""

    def __init__(self, path: Optional[Any] = None):
        self._database = Database(path)

    def get_connection(self) -> sqlite3.Connection:
        return self._database.get_connection()

    def get_adapter(self) -> Database:
        return self._database

    @contextmanager
    def connect(self) -> Iterator[sqlite3.Connection]:
        yield self._database.get_connection()

    def create_schema(self) -> None:
        self._database.create_schema()

    def create_indexes(self) -> None:
        self._database.create_indexes()

    def close(self) -> None:
        self._database.close()


def get_connection(path: Optional[Any] = None) -> sqlite3.Connection:
    """Return a new sqlite3 connection using the compatibility Database."""
    return Database(path).get_connection()


def get_database_manager() -> DatabaseManager:
    """Return a database manager, preferring the full implementation if present."""
    try:
        from database.core.connection import DatabaseManager as CoreDatabaseManager

        return CoreDatabaseManager()
    except ImportError:
        return DatabaseManager()
