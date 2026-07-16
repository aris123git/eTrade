"""Compatibility index entry point for main.py."""

from __future__ import annotations

from typing import Any


def create_indexes(db: Any) -> None:
    """Create database indexes using a Database-compatible object."""
    if hasattr(db, "create_indexes"):
        db.create_indexes()
        return
    raise TypeError(f"Object does not expose create_indexes(): {type(db)!r}")

