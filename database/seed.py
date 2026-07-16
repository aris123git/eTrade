"""
database/seed.py - Production seed data for MarketAI.

This module intentionally keeps seeding small, deterministic, and idempotent.
It supports both the legacy ``Database`` wrapper used by main.py and newer
database managers that expose ``get_adapter()``.
"""

from __future__ import annotations

import logging
import re
from contextlib import nullcontext
from datetime import datetime
from typing import Any, Iterable, List, Optional, Sequence, Tuple

logger = logging.getLogger(__name__)


DEFAULT_TIMEFRAMES: List[Tuple[str, int, int, str, str]] = [
    ("M1", 60, 1, "1 Minute", "minute"),
    ("M5", 300, 2, "5 Minutes", "minute"),
    ("M15", 900, 3, "15 Minutes", "minute"),
    ("M30", 1800, 4, "30 Minutes", "minute"),
    ("H1", 3600, 5, "1 Hour", "hourly"),
    ("H4", 14400, 6, "4 Hours", "hourly"),
    ("D1", 86400, 7, "1 Day", "daily"),
    ("W1", 604800, 8, "1 Week", "weekly"),
    ("MN1", 2592000, 9, "1 Month", "monthly"),
]

MAJOR_CURRENCIES: List[Tuple[str, str, str, Optional[int], int]] = [
    ("USD", "US Dollar", "$", 840, 2),
    ("EUR", "Euro", "EUR", 978, 2),
    ("GBP", "British Pound", "GBP", 826, 2),
    ("JPY", "Japanese Yen", "JPY", 392, 0),
    ("CHF", "Swiss Franc", "CHF", 756, 2),
    ("CAD", "Canadian Dollar", "CAD", 124, 2),
    ("AUD", "Australian Dollar", "AUD", 36, 2),
    ("NZD", "New Zealand Dollar", "NZD", 554, 2),
]

DEFAULT_BROKER = ("Default", "cfd", "localhost", "Default broker")


class _Executor:
    """Small adapter around sqlite connections, cursors, and project adapters."""

    def __init__(self, target: Any):
        self.target = target

    def execute(self, sql: str, params: Sequence[Any] = ()) -> Any:
        if hasattr(self.target, "execute"):
            return self.target.execute(sql, tuple(params))
        raise TypeError(f"Object does not support execute(): {type(self.target)!r}")

    def executemany(self, sql: str, params: Iterable[Sequence[Any]]) -> Any:
        if hasattr(self.target, "executemany"):
            return self.target.executemany(sql, list(params))
        for item in params:
            self.execute(sql, item)
        return None


class DatabaseSeeder:
    """Seed required baseline rows into the MarketAI database."""

    def seed(self, db: Any) -> None:
        """
        Seed baseline timeframes, currencies, and a default broker.

        Args:
            db: Either a legacy Database-like object with ``connection`` and
                ``execute`` or a manager exposing ``get_adapter()``.
        """
        target = self._resolve_target(db)
        executor = _Executor(target)

        with self._transaction(target):
            self._ensure_schema(executor)
            self._seed_timeframes(executor)
            self._seed_currencies(executor)
            self._seed_default_broker(executor)

        self._commit(target)
        logger.info("Database seed complete")

    def _resolve_target(self, db: Any) -> Any:
        """Return an executable target from supported database abstractions."""
        if db is None:
            raise ValueError("db is required")

        if hasattr(db, "get_adapter"):
            return db.get_adapter()
        if hasattr(db, "connection"):
            return db.connection
        return db

    def _transaction(self, target: Any):
        """Use an existing transaction context if available."""
        if hasattr(target, "transaction"):
            return target.transaction()
        return nullcontext()

    def _commit(self, target: Any) -> None:
        """Commit when the target exposes an explicit commit method."""
        commit = getattr(target, "commit", None)
        if callable(commit):
            commit()

    def _ensure_schema(self, db: _Executor) -> None:
        """Create minimal seed tables when a caller has not created schema yet."""
        db.execute(
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
            """
        )
        db.execute(
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
            """
        )
        db.execute(
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
            """
        )

    def _seed_timeframes(self, db: _Executor) -> None:
        """Seed default trading timeframes."""
        now = datetime.utcnow().isoformat(timespec="seconds")
        rows = [
            (name, seconds, sort_order, description, category, "active", now, now)
            for name, seconds, sort_order, description, category in DEFAULT_TIMEFRAMES
        ]
        db.executemany(
            """
            INSERT OR IGNORE INTO timeframes (
                name, seconds, sort_order, description, category, status, created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            rows,
        )

    def _seed_currencies(self, db: _Executor) -> None:
        """Seed major fiat currencies."""
        now = datetime.utcnow().isoformat(timespec="seconds")
        rows = []
        for code, name, symbol, iso_number, decimals in MAJOR_CURRENCIES:
            if not re.fullmatch(r"[A-Z]{3}", code):
                raise ValueError(f"Invalid ISO currency code: {code}")
            rows.append((code, name, "fiat", symbol, iso_number, decimals, "active", now, now))

        db.executemany(
            """
            INSERT OR IGNORE INTO currencies (
                code, name, currency_type, symbol, iso_number, decimals, status, created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            rows,
        )

    def _seed_default_broker(self, db: _Executor) -> None:
        """Seed the default broker row."""
        now = datetime.utcnow().isoformat(timespec="seconds")
        name, broker_type, server, description = DEFAULT_BROKER
        db.execute(
            """
            INSERT OR IGNORE INTO brokers (
                name, broker_type, server, description, status, metadata, created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (name, broker_type, server, description, "active", '{"is_default": true}', now, now),
        )


def seed(db: Any) -> None:
    """Seed the database using the default DatabaseSeeder."""
    DatabaseSeeder().seed(db)
