"""
SQLite database adapter.

The adapter keeps a single WAL-enabled SQLite connection for repository usage
and exposes transaction-aware helper methods that return rows as dictionaries.
"""

from __future__ import annotations

import sqlite3
import threading
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterable, Iterator, List, Optional, Sequence, Union

from database.adapters.base_adapter import DatabaseAdapter, Params


class SQLiteAdapter(DatabaseAdapter):
    """Production SQLite implementation of :class:`DatabaseAdapter`."""

    def __init__(self, db_path: Union[str, Path], timeout: float = 30.0):
        self.db_path = Path(db_path)
        if self.db_path != Path(":memory:"):
            self.db_path.parent.mkdir(parents=True, exist_ok=True)

        self._lock = threading.RLock()
        self._transaction_depth = 0
        self._closed = False
        self._last_insert_id = 0
        self._connection = sqlite3.connect(
            str(self.db_path),
            timeout=timeout,
            detect_types=sqlite3.PARSE_DECLTYPES | sqlite3.PARSE_COLNAMES,
            check_same_thread=False,
            isolation_level=None,
        )
        self._connection.row_factory = sqlite3.Row
        self._configure_connection()

    def _configure_connection(self) -> None:
        with self._lock:
            self._connection.execute("PRAGMA journal_mode=WAL;")
            self._connection.execute("PRAGMA synchronous=NORMAL;")
            self._connection.execute("PRAGMA temp_store=MEMORY;")
            self._connection.execute("PRAGMA foreign_keys=ON;")
            self._connection.execute("PRAGMA busy_timeout=30000;")

    def _ensure_open(self) -> None:
        if self._closed:
            raise sqlite3.ProgrammingError("SQLiteAdapter connection is closed")

    @staticmethod
    def _row_to_dict(row: Optional[sqlite3.Row]) -> Optional[dict]:
        return dict(row) if row is not None else None

    def execute(self, sql: str, params: Params = ()) -> sqlite3.Cursor:
        """Execute SQL and commit automatically outside explicit transactions."""
        with self._lock:
            self._ensure_open()
            cursor = self._connection.execute(sql, tuple(params))
            self._last_insert_id = cursor.lastrowid or self._last_insert_id
            return cursor

    def execute_many(self, sql: str, params_list: Iterable[Params]) -> int:
        """Execute many statements and return SQLite's affected row count."""
        params = [tuple(item) for item in params_list]
        if not params:
            return 0

        with self._lock:
            self._ensure_open()
            cursor = self._connection.executemany(sql, params)
            self._last_insert_id = cursor.lastrowid or self._last_insert_id
            return cursor.rowcount if cursor.rowcount != -1 else len(params)

    def fetch_one(self, sql: str, params: Params = ()) -> Optional[dict]:
        with self._lock:
            self._ensure_open()
            cursor = self._connection.execute(sql, tuple(params))
            return self._row_to_dict(cursor.fetchone())

    def fetch_all(self, sql: str, params: Params = ()) -> List[dict]:
        with self._lock:
            self._ensure_open()
            cursor = self._connection.execute(sql, tuple(params))
            return [dict(row) for row in cursor.fetchall()]

    def fetch_count(self, sql: str, params: Params = ()) -> int:
        row = self.fetch_one(sql, params)
        if not row:
            return 0
        value = next(iter(row.values()), 0)
        return int(value or 0)

    def get_last_insert_id(self) -> int:
        with self._lock:
            self._ensure_open()
            row = self._connection.execute("SELECT last_insert_rowid()").fetchone()
            return int(row[0]) if row else int(self._last_insert_id or 0)

    @contextmanager
    def transaction(self) -> Iterator["SQLiteAdapter"]:
        """Run enclosed statements in an atomic transaction.

        Nested transactions use SQLite savepoints so repository-level bulk
        operations can be safely composed.
        """
        with self._lock:
            self._ensure_open()
            savepoint_name = None
            if self._transaction_depth == 0:
                self._connection.execute("BEGIN IMMEDIATE")
            else:
                savepoint_name = f"sp_{self._transaction_depth}"
                self._connection.execute(f"SAVEPOINT {savepoint_name}")
            self._transaction_depth += 1

        try:
            yield self
        except Exception:
            with self._lock:
                self._transaction_depth -= 1
                if savepoint_name:
                    self._connection.execute(f"ROLLBACK TO SAVEPOINT {savepoint_name}")
                    self._connection.execute(f"RELEASE SAVEPOINT {savepoint_name}")
                else:
                    self._connection.rollback()
            raise
        else:
            with self._lock:
                self._transaction_depth -= 1
                if savepoint_name:
                    self._connection.execute(f"RELEASE SAVEPOINT {savepoint_name}")
                else:
                    self._connection.commit()

    @contextmanager
    def get_connection(self) -> Iterator[sqlite3.Connection]:
        """Yield the raw SQLite connection with commit/rollback handling."""
        with self._lock:
            self._ensure_open()
            nested = self._transaction_depth > 0
        try:
            yield self._connection
        except Exception:
            if not nested:
                with self._lock:
                    self._connection.rollback()
            raise
        else:
            if not nested:
                with self._lock:
                    self._connection.commit()

    def vacuum(self) -> None:
        with self._lock:
            self._ensure_open()
            if self._transaction_depth > 0:
                raise sqlite3.OperationalError("VACUUM cannot run inside a transaction")
            self._connection.execute("VACUUM")

    def commit(self) -> None:
        with self._lock:
            self._ensure_open()
            self._connection.commit()

    def rollback(self) -> None:
        with self._lock:
            self._ensure_open()
            self._connection.rollback()

    def close(self) -> None:
        with self._lock:
            if not self._closed:
                self._connection.close()
                self._closed = True

    @property
    def connection(self) -> sqlite3.Connection:
        """Expose the raw connection for low-level compatibility code."""
        self._ensure_open()
        return self._connection
