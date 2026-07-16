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
]


def apply_migrations(db: Any) -> List[str]:
    """Apply pending schema migrations and return applied versions."""
    create_schema(db)
    applied: List[str] = []
    for version, _ in MIGRATIONS:
        row = None
        if hasattr(db, "fetch_one"):
            row = db.fetch_one("SELECT version FROM schema_migrations WHERE version = ?", (version,))
        elif hasattr(db, "get_adapter"):
            row = db.get_adapter().fetch_one(
                "SELECT version FROM schema_migrations WHERE version = ?",
                (version,),
            )
        if row:
            continue
        if version.endswith("indexes") or version == "002_indexes":
            create_indexes(db)
        now = datetime.utcnow().isoformat(timespec="seconds")
        sql = "INSERT OR IGNORE INTO schema_migrations (version, applied_at) VALUES (?, ?)"
        if hasattr(db, "get_adapter"):
            db.get_adapter().execute(sql, (version, now))
            db.get_adapter().commit()
        else:
            db.execute(sql, (version, now))
            db.commit()
        applied.append(version)
    return applied
