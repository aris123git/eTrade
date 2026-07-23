"""
database/migrations/multi_broker.py

Migration 003: multi-broker identity support.
- markets.canonical_symbol
- symbol_aliases table
- uniqueness (broker_id, symbol) for markets
- uniqueness (market_id, timeframe, timestamp) for candles
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, List, Optional


def _execute(db: Any, sql: str, params: tuple = ()) -> Any:
    if hasattr(db, "get_adapter"):
        return db.get_adapter().execute(sql, params)
    if hasattr(db, "execute"):
        return db.execute(sql, params)
    if hasattr(db, "connection"):
        return db.connection.execute(sql, params)
    raise TypeError(f"Unsupported db: {type(db)!r}")


def _fetch_all(db: Any, sql: str, params: tuple = ()) -> list:
    if hasattr(db, "fetch_all"):
        return db.fetch_all(sql, params)
    cur = _execute(db, sql, params)
    rows = cur.fetchall() if cur is not None else []
    out = []
    for row in rows:
        out.append(dict(row) if hasattr(row, "keys") else row)
    return out


def _fetch_one(db: Any, sql: str, params: tuple = ()) -> Optional[dict]:
    if hasattr(db, "fetch_one"):
        return db.fetch_one(sql, params)
    rows = _fetch_all(db, sql, params)
    if not rows:
        return None
    row = rows[0]
    return row if isinstance(row, dict) else {"sql": row[0]} if row else None


def _commit(db: Any) -> None:
    if hasattr(db, "commit"):
        db.commit()
    elif hasattr(db, "get_adapter"):
        db.get_adapter().commit()
    elif hasattr(db, "connection"):
        db.connection.commit()


def _table_sql(db: Any, table: str) -> str:
    row = _fetch_one(
        db,
        "SELECT sql FROM sqlite_master WHERE type='table' AND name=?",
        (table,),
    )
    if not row:
        return ""
    if isinstance(row, dict):
        return str(row.get("sql") or "")
    return str(row[0])


def _table_columns(db: Any, table: str) -> List[str]:
    rows = _fetch_all(db, f"PRAGMA table_info({table})")
    cols: List[str] = []
    for row in rows:
        if isinstance(row, dict):
            cols.append(str(row.get("name") or ""))
        else:
            cols.append(str(row[1]))
    return [c for c in cols if c]


def _norm_sql(sql: str) -> str:
    return "".join(sql.split())


def ensure_symbol_aliases_table(db: Any) -> None:
    _execute(
        db,
        """
        CREATE TABLE IF NOT EXISTS symbol_aliases (
            alias_id INTEGER PRIMARY KEY AUTOINCREMENT,
            alias TEXT NOT NULL,
            canonical_symbol TEXT NOT NULL,
            asset_class TEXT,
            description TEXT,
            metadata TEXT DEFAULT '{}',
            created_at TEXT,
            updated_at TEXT,
            UNIQUE(alias)
        )
        """,
    )


def seed_builtin_aliases(db: Any) -> int:
    from core.symbol_identity import classify_token, expand_alias_rows

    now = datetime.utcnow().isoformat(timespec="seconds")
    count = 0
    for alias, canonical in expand_alias_rows():
        _execute(
            db,
            """
            INSERT OR IGNORE INTO symbol_aliases
            (alias, canonical_symbol, asset_class, description, metadata, created_at, updated_at)
            VALUES (?, ?, ?, ?, '{}', ?, ?)
            """,
            (
                alias,
                canonical,
                classify_token(canonical),
                f"Built-in alias {alias} -> {canonical}",
                now,
                now,
            ),
        )
        count += 1
    return count


def backfill_canonical_symbols(db: Any) -> int:
    from core.symbol_identity import canonicalize

    cols = _table_columns(db, "markets")
    if "canonical_symbol" not in cols:
        _execute(db, "ALTER TABLE markets ADD COLUMN canonical_symbol TEXT")

    rows = _fetch_all(db, "SELECT market_id, symbol, canonical_symbol FROM markets")
    updated = 0
    for row in rows:
        if isinstance(row, dict):
            market_id = row["market_id"]
            symbol = row["symbol"]
            current = row.get("canonical_symbol")
        else:
            market_id, symbol, current = row[0], row[1], row[2]
        if current:
            continue
        ident = canonicalize(symbol)
        _execute(
            db,
            "UPDATE markets SET canonical_symbol = ?, "
            "base_currency = COALESCE(base_currency, ?), "
            "quote_currency = COALESCE(quote_currency, ?) "
            "WHERE market_id = ?",
            (ident.canonical_symbol, ident.base_currency, ident.quote_currency, market_id),
        )
        updated += 1
    return updated


def rebuild_markets_unique_if_needed(db: Any) -> bool:
    sql = _table_sql(db, "markets")
    if not sql:
        return False
    norm = _norm_sql(sql)
    if "UNIQUE(broker_id,symbol)" in norm:
        # Ensure canonical column exists on modern schemas
        if "canonical_symbol" not in _table_columns(db, "markets"):
            _execute(db, "ALTER TABLE markets ADD COLUMN canonical_symbol TEXT")
        return False
    if "symbolTEXTUNIQUE" not in norm and "symbolTEXTUNIQUENOTNULL" not in norm:
        if "canonical_symbol" not in _table_columns(db, "markets"):
            _execute(db, "ALTER TABLE markets ADD COLUMN canonical_symbol TEXT")
        return False

    legacy_cols = _table_columns(db, "markets")
    _execute(db, "ALTER TABLE markets RENAME TO markets_legacy_mb")
    _execute(
        db,
        """
        CREATE TABLE markets (
            market_id INTEGER PRIMARY KEY AUTOINCREMENT,
            broker_id INTEGER,
            symbol TEXT NOT NULL,
            canonical_symbol TEXT,
            market_type TEXT,
            status TEXT DEFAULT 'active',
            description TEXT,
            base_currency TEXT,
            quote_currency TEXT,
            pip_size REAL,
            point REAL,
            digits INTEGER,
            contract_size REAL,
            metadata TEXT DEFAULT '{}',
            created_at TEXT,
            updated_at TEXT,
            name TEXT,
            category TEXT,
            active INTEGER DEFAULT 1,
            spread REAL,
            trade_mode INTEGER,
            currency_base TEXT,
            currency_profit TEXT,
            currency_margin TEXT,
            FOREIGN KEY(broker_id) REFERENCES brokers(broker_id),
            UNIQUE(broker_id, symbol)
        )
        """,
    )
    new_cols = set(_table_columns(db, "markets"))
    copy_cols = [c for c in legacy_cols if c in new_cols]
    col_csv = ", ".join(copy_cols)
    _execute(db, f"INSERT INTO markets ({col_csv}) SELECT {col_csv} FROM markets_legacy_mb")
    _execute(db, "DROP TABLE markets_legacy_mb")
    return True


def rebuild_candles_unique_if_needed(db: Any) -> bool:
    sql = _table_sql(db, "candles")
    if not sql:
        return False
    norm = _norm_sql(sql)
    if "UNIQUE(broker_id,symbol,timeframe,timestamp)" in norm:
        return False
    if "UNIQUE(symbol,timeframe,timestamp)" not in norm and "UNIQUE(market_id,timeframe,timestamp)" not in norm:
        return False

    legacy_cols = _table_columns(db, "candles")
    _execute(db, "ALTER TABLE candles RENAME TO candles_legacy_mb")
    _execute(
        db,
        """
        CREATE TABLE candles (
            candle_id INTEGER PRIMARY KEY AUTOINCREMENT,
            candle_uuid TEXT,
            symbol TEXT NOT NULL,
            timeframe TEXT NOT NULL,
            timestamp TEXT NOT NULL,
            open REAL NOT NULL,
            high REAL NOT NULL,
            low REAL NOT NULL,
            close REAL NOT NULL,
            volume REAL DEFAULT 0,
            market_id INTEGER,
            broker_id INTEGER,
            spread REAL,
            tick_volume INTEGER,
            status TEXT DEFAULT 'active',
            metadata TEXT DEFAULT '{}',
            created_at TEXT,
            updated_at TEXT,
            UNIQUE(broker_id, symbol, timeframe, timestamp)
        )
        """,
    )
    new_cols = set(_table_columns(db, "candles"))
    copy_cols = [c for c in legacy_cols if c in new_cols]
    col_csv = ", ".join(copy_cols)
    _execute(db, f"INSERT INTO candles ({col_csv}) SELECT {col_csv} FROM candles_legacy_mb")
    _execute(db, "DROP TABLE candles_legacy_mb")
    return True


def rebuild_ticks_unique_if_needed(db: Any) -> bool:
    sql = _table_sql(db, "ticks")
    if not sql:
        return False
    norm = _norm_sql(sql)
    if "UNIQUE(broker_id,symbol,timestamp,bid,ask)" in norm:
        return False
    if "UNIQUE(symbol,timestamp,bid,ask)" not in norm and "UNIQUE(market_id,timestamp,bid,ask)" not in norm:
        return False

    legacy_cols = _table_columns(db, "ticks")
    _execute(db, "ALTER TABLE ticks RENAME TO ticks_legacy_mb")
    _execute(
        db,
        """
        CREATE TABLE ticks (
            tick_id INTEGER PRIMARY KEY AUTOINCREMENT,
            tick_uuid TEXT,
            symbol TEXT NOT NULL,
            timestamp TEXT NOT NULL,
            bid REAL NOT NULL,
            ask REAL NOT NULL,
            last REAL DEFAULT 0,
            volume REAL DEFAULT 0,
            flags INTEGER DEFAULT 0,
            market_id INTEGER,
            broker_id INTEGER,
            status TEXT DEFAULT 'active',
            metadata TEXT DEFAULT '{}',
            created_at TEXT,
            updated_at TEXT,
            UNIQUE(broker_id, symbol, timestamp, bid, ask)
        )
        """,
    )
    new_cols = set(_table_columns(db, "ticks"))
    copy_cols = [c for c in legacy_cols if c in new_cols]
    col_csv = ", ".join(copy_cols)
    _execute(db, f"INSERT INTO ticks ({col_csv}) SELECT {col_csv} FROM ticks_legacy_mb")
    _execute(db, "DROP TABLE ticks_legacy_mb")
    return True


def apply_multi_broker_migration(db: Any) -> List[str]:
    """Apply multi-broker identity changes. Returns human-readable steps done."""
    steps: List[str] = []
    ensure_symbol_aliases_table(db)
    steps.append("symbol_aliases_ready")
    n_alias = seed_builtin_aliases(db)
    steps.append(f"aliases_seeded:{n_alias}")
    if rebuild_markets_unique_if_needed(db):
        steps.append("markets_rebuilt_unique_broker_symbol")
    n = backfill_canonical_symbols(db)
    steps.append(f"canonical_backfill:{n}")
    if rebuild_candles_unique_if_needed(db):
        steps.append("candles_rebuilt_unique_broker")
    if rebuild_ticks_unique_if_needed(db):
        steps.append("ticks_rebuilt_unique_broker")
    _commit(db)
    return steps
