"""
database/seed.py - Production seed data for MarketAI.

Supports legacy Database and DatabaseManager. Idempotent INSERT OR IGNORE.
"""

from __future__ import annotations

import logging
from contextlib import nullcontext
from datetime import datetime
from typing import Any, Iterable, List, Optional, Sequence, Tuple
from uuid import uuid4

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
    ("EUR", "Euro", "€", 978, 2),
    ("GBP", "British Pound", "£", 826, 2),
    ("JPY", "Japanese Yen", "¥", 392, 0),
    ("CHF", "Swiss Franc", "CHF", 756, 2),
    ("CAD", "Canadian Dollar", "CAD", 124, 2),
    ("AUD", "Australian Dollar", "AUD", 36, 2),
    ("NZD", "New Zealand Dollar", "NZD", 554, 2),
]


class _Executor:
    def __init__(self, target: Any):
        self.target = target

    def execute(self, sql: str, params: Sequence[Any] = ()) -> Any:
        if hasattr(self.target, "execute"):
            return self.target.execute(sql, tuple(params))
        raise TypeError(f"Object does not support execute(): {type(self.target)!r}")

    def executemany(self, sql: str, params: Iterable[Sequence[Any]]) -> Any:
        if hasattr(self.target, "execute_many"):
            return self.target.execute_many(sql, list(params))
        if hasattr(self.target, "executemany"):
            return self.target.executemany(sql, list(params))
        for item in params:
            self.execute(sql, item)
        return None


class DatabaseSeeder:
    """Seed required baseline rows into the MarketAI database."""

    def seed(self, db: Any) -> bool:
        target = self._resolve_target(db)
        executor = _Executor(target)
        with self._transaction(target):
            from database.schema import create_schema

            create_schema(db)
            self._seed_timeframes(executor)
            self._seed_currencies(executor)
            self._seed_default_broker(executor)
        self._commit(target)
        logger.info("Database seed complete")
        return True

    def _resolve_target(self, db: Any) -> Any:
        if db is None:
            raise ValueError("db is required")
        if hasattr(db, "get_adapter"):
            return db.get_adapter()
        if hasattr(db, "connection"):
            return db
        return db

    def _transaction(self, target: Any):
        if hasattr(target, "transaction"):
            return target.transaction()
        return nullcontext()

    def _commit(self, target: Any) -> None:
        commit = getattr(target, "commit", None)
        if callable(commit):
            commit()

    def _seed_timeframes(self, db: _Executor) -> None:
        now = datetime.utcnow().isoformat(timespec="seconds")
        rows = [
            (str(uuid4()), name, seconds, sort_order, description, category, "active", "{}", now, now)
            for name, seconds, sort_order, description, category in DEFAULT_TIMEFRAMES
        ]
        db.executemany(
            """
            INSERT OR IGNORE INTO timeframes (
                timeframe_uuid, name, seconds, sort_order, description,
                category, status, metadata, created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            rows,
        )

    def _seed_currencies(self, db: _Executor) -> None:
        now = datetime.utcnow().isoformat(timespec="seconds")
        rows = [
            (str(uuid4()), code, name, "fiat", symbol, iso, decimals, name, "active", "{}", now, now)
            for code, name, symbol, iso, decimals in MAJOR_CURRENCIES
        ]
        db.executemany(
            """
            INSERT OR IGNORE INTO currencies (
                currency_uuid, code, name, currency_type, symbol, iso_number,
                decimals, description, status, metadata, created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            rows,
        )

    def _seed_default_broker(self, db: _Executor) -> None:
        now = datetime.utcnow().isoformat(timespec="seconds")
        db.execute(
            """
            INSERT OR IGNORE INTO brokers (
                broker_uuid, name, broker_type, server, description,
                status, metadata, created_at, updated_at
            ) VALUES (?, 'Default', 'cfd', 'localhost', 'Default broker',
                      'active', '{}', ?, ?)
            """,
            (str(uuid4()), now, now),
        )


def seed(db: Any) -> bool:
    """Module-level entrypoint expected by main.py."""
    return DatabaseSeeder().seed(db)
