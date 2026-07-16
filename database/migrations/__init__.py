"""
database/migrations/__init__.py - Lightweight migration runner
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, List, Tuple

from database.indexes import create_indexes
from database.schema import create_schema

MIGRATIONS: List[Tuple[str, str]] = [
    ("001_initial_schema", "create_schema"),
    ("002_indexes", "create_indexes"),
    ("003_multi_broker_identity", "multi_broker"),
]


def _already_applied(db: Any, version: str) -> bool:
    row = None
    if hasattr(db, "fetch_one"):
        row = db.fetch_one("SELECT version FROM schema_migrations WHERE version = ?", (version,))
    elif hasattr(db, "get_adapter"):
        row = db.get_adapter().fetch_one(
            "SELECT version FROM schema_migrations WHERE version = ?",
            (version,),
        )
    return bool(row)


def _record(db: Any, version: str) -> None:
    now = datetime.utcnow().isoformat(timespec="seconds")
    sql = "INSERT OR IGNORE INTO schema_migrations (version, applied_at) VALUES (?, ?)"
    if hasattr(db, "get_adapter"):
        db.get_adapter().execute(sql, (version, now))
        db.get_adapter().commit()
    else:
        db.execute(sql, (version, now))
        db.commit()


def apply_migrations(db: Any) -> List[str]:
    """Apply pending schema migrations and return applied versions."""
    create_schema(db)

    # Idempotent multi-broker upgrade must run before indexes that need new columns.
    # Still recorded as migration 003 so status stays visible.
    from database.migrations.multi_broker import apply_multi_broker_migration

    apply_multi_broker_migration(db)

    applied: List[str] = []
    for version, action in MIGRATIONS:
        if _already_applied(db, version):
            continue
        if action == "create_indexes" or version == "002_indexes":
            create_indexes(db)
        elif action == "multi_broker":
            # Already applied above; create indexes again for new columns
            create_indexes(db)
        _record(db, version)
        applied.append(version)
    return applied
