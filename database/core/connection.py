"""
database/core/connection.py - DatabaseManager

RESPONSIBILITY:
Central connection/adapter lifecycle for all repositories.

VERSION: 2.0.0
"""

from __future__ import annotations

import logging
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator, Optional, Union

from core.config import DATABASE_PATH, Config
from database.adapters.base_adapter import DatabaseAdapter
from database.adapters.sqlite_adapter import SQLiteAdapter

logger = logging.getLogger(__name__)


class DatabaseManager:
    """
    Production database manager used by every repository.

    USAGE:
        db = DatabaseManager()
        adapter = db.get_adapter()
        with db.transaction():
            adapter.execute("INSERT ...")
    """

    def __init__(
        self,
        db_path: Optional[Union[str, Path]] = None,
        adapter: Optional[DatabaseAdapter] = None,
        config: Optional[Config] = None,
    ):
        self.config = config or Config()
        if adapter is not None:
            self._adapter = adapter
            self.db_path = Path(getattr(adapter, "db_path", db_path or self.config.database_path))
        else:
            path = Path(db_path or self.config.database_path or DATABASE_PATH)
            path.parent.mkdir(parents=True, exist_ok=True)
            self.db_path = path
            self._adapter = SQLiteAdapter(path)
        logger.info("DatabaseManager ready at %s", self.db_path)

    def get_adapter(self) -> DatabaseAdapter:
        return self._adapter

    @contextmanager
    def get_connection(self) -> Iterator[Any]:
        with self._adapter.get_connection() as conn:
            yield conn

    @contextmanager
    def transaction(self) -> Iterator[DatabaseAdapter]:
        with self._adapter.transaction() as adapter:
            yield adapter

    def execute(self, sql: str, params: tuple = ()) -> Any:
        return self._adapter.execute(sql, params)

    def fetch_one(self, sql: str, params: tuple = ()) -> Optional[dict]:
        return self._adapter.fetch_one(sql, params)

    def fetch_all(self, sql: str, params: tuple = ()) -> list:
        return self._adapter.fetch_all(sql, params)

    def commit(self) -> None:
        self._adapter.commit()

    def rollback(self) -> None:
        self._adapter.rollback()

    def close(self) -> None:
        self._adapter.close()
        logger.info("DatabaseManager closed")

    def __enter__(self) -> "DatabaseManager":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()


def create_database_manager(
    db_path: Optional[Union[str, Path]] = None,
    config: Optional[Config] = None,
) -> DatabaseManager:
    """Factory for DatabaseManager."""
    return DatabaseManager(db_path=db_path, config=config)
