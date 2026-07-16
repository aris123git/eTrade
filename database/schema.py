"""Compatibility schema entry point for main.py."""

from __future__ import annotations

from typing import Any


def create_schema(db: Any) -> None:
    """Create database schema using a Database-compatible object."""
    if hasattr(db, "create_schema"):
        db.create_schema()
        return
    raise TypeError(f"Object does not expose create_schema(): {type(db)!r}")

