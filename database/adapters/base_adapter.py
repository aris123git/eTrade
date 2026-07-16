"""
Database adapter abstraction.

Adapters own the mechanics of talking to a concrete database engine while
repositories and services work against this stable interface.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from contextlib import AbstractContextManager
from typing import Any, Iterable, List, Mapping, Optional, Sequence, Tuple

Params = Sequence[Any]
RowDict = Mapping[str, Any]


class DatabaseAdapter(ABC):
    """Abstract interface implemented by concrete database adapters."""

    @abstractmethod
    def execute(self, sql: str, params: Params = ()) -> Any:
        """Execute a single SQL statement and return the database cursor."""
        raise NotImplementedError

    @abstractmethod
    def execute_many(self, sql: str, params_list: Iterable[Params]) -> int:
        """Execute a statement for many parameter sets and return row count."""
        raise NotImplementedError

    @abstractmethod
    def fetch_one(self, sql: str, params: Params = ()) -> Optional[dict]:
        """Fetch a single row as a dictionary, or None if no row exists."""
        raise NotImplementedError

    @abstractmethod
    def fetch_all(self, sql: str, params: Params = ()) -> List[dict]:
        """Fetch all rows as dictionaries."""
        raise NotImplementedError

    @abstractmethod
    def fetch_count(self, sql: str, params: Params = ()) -> int:
        """Execute a count query and return the integer result."""
        raise NotImplementedError

    @abstractmethod
    def get_last_insert_id(self) -> int:
        """Return the last inserted row id for this connection."""
        raise NotImplementedError

    @abstractmethod
    def transaction(self) -> AbstractContextManager["DatabaseAdapter"]:
        """Return a context manager wrapping operations in a transaction."""
        raise NotImplementedError

    @abstractmethod
    def vacuum(self) -> None:
        """Compact/optimize the underlying database when supported."""
        raise NotImplementedError

    @abstractmethod
    def commit(self) -> None:
        """Commit the current transaction."""
        raise NotImplementedError

    @abstractmethod
    def rollback(self) -> None:
        """Rollback the current transaction."""
        raise NotImplementedError

    @abstractmethod
    def close(self) -> None:
        """Close any open database resources."""
        raise NotImplementedError

    @abstractmethod
    def get_connection(self) -> AbstractContextManager[Any]:
        """Return a context manager yielding the raw connection."""
        raise NotImplementedError
