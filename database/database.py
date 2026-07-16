"""
database/database.py - SQLite compatibility layer for MarketAI.

``Database`` remains the simple object used by main.py.
``DatabaseManager`` is re-exported from database.core for collectors.
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
        self.cursor.execute("PRAGMA journal_mode=WAL;")
        self.cursor.execute("PRAGMA synchronous=NORMAL;")
        self.cursor.execute("PRAGMA temp_store=MEMORY;")
        self.cursor.execute("PRAGMA foreign_keys=ON;")
        self.connection.commit()

    @contextmanager
    def transaction(self) -> Iterator[sqlite3.Connection]:
        try:
            yield self.connection
            self.connection.commit()
        except Exception:
            self.connection.rollback()
            raise

    @contextmanager
    def connect(self) -> Iterator[sqlite3.Connection]:
        yield self.connection

    def get_connection(self) -> sqlite3.Connection:
        return self.connection

    def get_adapter(self) -> "Database":
        return self

    def execute(self, query: str, values: Sequence[Any] = ()) -> sqlite3.Cursor:
        self.cursor.execute(query, tuple(values))
        return self.cursor

    def executemany(self, query: str, values: Iterable[Sequence[Any]]) -> sqlite3.Cursor:
        self.cursor.executemany(query, list(values))
        return self.cursor

    def execute_many(self, query: str, values: Iterable[Sequence[Any]]) -> int:
        self.executemany(query, values)
        return self.cursor.rowcount

    def fetchone(self) -> Optional[sqlite3.Row]:
        return self.cursor.fetchone()

    def fetchall(self) -> list:
        return self.cursor.fetchall()

    def fetch_one(self, query: str, values: Sequence[Any] = ()) -> Optional[dict]:
        row = self.execute(query, values).fetchone()
        return dict(row) if row is not None else None

    def fetch_all(self, query: str, values: Sequence[Any] = ()) -> list:
        rows = self.execute(query, values).fetchall()
        return [dict(r) for r in rows]

    def fetch_count(self, query: str, values: Sequence[Any] = ()) -> int:
        row = self.fetch_one(query, values)
        if not row:
            return 0
        return int(next(iter(row.values())) or 0)

    def get_last_insert_id(self) -> int:
        row = self.execute("SELECT last_insert_rowid()").fetchone()
        return int(row[0]) if row else 0

    def vacuum(self) -> None:
        self.execute("VACUUM")

    def commit(self) -> None:
        self.connection.commit()

    def rollback(self) -> None:
        self.connection.rollback()

    def close(self) -> None:
        self.connection.close()

    def create_schema(self) -> None:
        from database.schema import create_schema

        create_schema(self)

    def create_indexes(self) -> None:
        from database.indexes import create_indexes

        create_indexes(self)


def get_connection(path: Optional[Any] = None) -> sqlite3.Connection:
    return Database(path).get_connection()


def get_database_manager(path: Optional[Any] = None):
    from database.core.connection import DatabaseManager

    return DatabaseManager(db_path=path)


# Prefer the full DatabaseManager implementation for collector imports
from database.core.connection import DatabaseManager  # noqa: E402
