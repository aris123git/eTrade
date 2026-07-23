"""
collector/tick_history.py - Expand tick archive from broker sources (MT5).

Resumable: starts after last stored tick. bulk_upsert uses unique constraints
so duplicates are never stored twice.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Sequence
from uuid import uuid4

from collector.broker_sources.registry import BrokerSourceRegistry, build_default_registry
from core.symbol_identity import canonicalize
from database.models.market_model import MarketModel

logger = logging.getLogger(__name__)


@dataclass
class TickSeriesResult:
    broker: str
    symbol: str
    canonical_symbol: str
    ticks_downloaded: int = 0
    ticks_inserted: int = 0
    resumed: bool = False
    status: str = "ok"
    error: Optional[str] = None


@dataclass
class TickDownloadReport:
    series: List[TickSeriesResult] = field(default_factory=list)

    @property
    def ticks_inserted(self) -> int:
        return sum(s.ticks_inserted for s in self.series)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "ticks_inserted": self.ticks_inserted,
            "series": [
                {
                    "broker": s.broker,
                    "symbol": s.symbol,
                    "canonical_symbol": s.canonical_symbol,
                    "ticks_downloaded": s.ticks_downloaded,
                    "ticks_inserted": s.ticks_inserted,
                    "resumed": s.resumed,
                    "status": s.status,
                    "error": s.error,
                }
                for s in self.series
            ],
        }


class TickHistoryEngine:
    """Download and archive tick history into the ticks table."""

    def __init__(
        self,
        db: Any,
        *,
        registry: Optional[BrokerSourceRegistry] = None,
        include_mt5: bool = True,
        csv_brokers: Optional[Dict[str, str]] = None,
        lookback_days: int = 7,
    ):
        self.db = db
        self.registry = registry or build_default_registry(
            include_mt5=include_mt5,
            csv_brokers=csv_brokers or {},
        )
        self.lookback_days = lookback_days
        self.markets = MarketModel(db)

    def download_ticks(
        self,
        *,
        symbols: Sequence[str],
        start: Optional[datetime] = None,
        end: Optional[datetime] = None,
        resume: bool = True,
    ) -> TickDownloadReport:
        report = TickDownloadReport()
        end = end or datetime.utcnow()
        for source in self.registry.all():
            if not source.connect():
                logger.warning("Tick source unavailable: %s", source.name)
                continue
            try:
                discovered = {
                    canonicalize(s.broker_symbol).canonical_symbol: s
                    for s in source.discover_symbols()
                }
                for symbol in symbols:
                    canon = canonicalize(symbol).canonical_symbol
                    item = discovered.get(canon)
                    if item is None:
                        # try exact broker symbol
                        item = next(
                            (
                                s
                                for s in discovered.values()
                                if s.broker_symbol.upper() == symbol.upper()
                                or s.canonical_symbol == canon
                            ),
                            None,
                        )
                    if item is None:
                        report.series.append(
                            TickSeriesResult(
                                broker=source.name,
                                symbol=symbol,
                                canonical_symbol=canon,
                                status="symbol_not_found",
                            )
                        )
                        continue
                    result = self._download_symbol(
                        source,
                        item.broker_symbol,
                        canon,
                        start=start,
                        end=end,
                        resume=resume,
                    )
                    report.series.append(result)
            finally:
                try:
                    source.disconnect()
                except Exception:
                    pass
        return report

    def _download_symbol(
        self,
        source,
        broker_symbol: str,
        canonical_symbol: str,
        *,
        start: Optional[datetime],
        end: datetime,
        resume: bool,
    ) -> TickSeriesResult:
        broker_row = self._ensure_broker(source.name)
        broker_id = int(broker_row["broker_id"])
        self.markets.add_market(
            symbol=broker_symbol,
            category=canonicalize(broker_symbol).asset_class,
            broker_id=broker_id,
            canonical_symbol=canonical_symbol,
        )
        market_id = self._market_id(broker_id, broker_symbol)
        if market_id is None:
            raise RuntimeError(f"Failed to resolve market_id for {broker_symbol}")

        resumed = False
        effective_start = start or (end - timedelta(days=self.lookback_days))
        if resume:
            last = self._last_tick_time(market_id, broker_symbol)
            if last is not None and last >= effective_start:
                effective_start = last
                resumed = True

        try:
            ticks = source.download_ticks(
                broker_symbol,
                start=effective_start,
                end=end,
            )
            if resume and ticks:
                # Drop exact overlap on the resume boundary
                last = self._last_tick_time(market_id, broker_symbol)
                if last is not None:
                    ticks = [t for t in ticks if _as_dt(t["timestamp"]) > last]
            inserted = self._insert_ticks(broker_id, market_id, broker_symbol, ticks)
            self._update_sync(market_id, end if not ticks else _as_dt(ticks[-1]["timestamp"]), inserted)
            return TickSeriesResult(
                broker=source.name,
                symbol=broker_symbol,
                canonical_symbol=canonical_symbol,
                ticks_downloaded=len(ticks),
                ticks_inserted=inserted,
                resumed=resumed,
                status="ok" if ticks or resumed else "empty",
            )
        except Exception as exc:
            logger.exception("Tick download failed %s %s", source.name, broker_symbol)
            return TickSeriesResult(
                broker=source.name,
                symbol=broker_symbol,
                canonical_symbol=canonical_symbol,
                resumed=resumed,
                status="error",
                error=f"{exc.__class__.__name__}: {exc}",
            )

    def _insert_ticks(
        self,
        broker_id: int,
        market_id: int,
        symbol: str,
        ticks: Sequence[Dict[str, Any]],
    ) -> int:
        if not ticks:
            return 0
        before = self._count(market_id, symbol)
        now = datetime.utcnow().isoformat(timespec="seconds")
        for tick in ticks:
            ts = _as_dt(tick["timestamp"]).isoformat(timespec="seconds")
            self._execute(
                """
                INSERT OR IGNORE INTO ticks (
                    tick_uuid, symbol, timestamp, bid, ask, last, volume, flags,
                    market_id, broker_id, status, metadata, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'active', '{}', ?, ?)
                """,
                (
                    str(uuid4()),
                    symbol.upper(),
                    ts,
                    float(tick["bid"]),
                    float(tick["ask"]),
                    float(tick.get("last") or 0.0),
                    float(tick.get("volume") or 0.0),
                    int(tick.get("flags") or 0),
                    market_id,
                    broker_id,
                    now,
                    now,
                ),
            )
        self._commit()
        return max(0, self._count(market_id, symbol) - before)

    def _last_tick_time(self, market_id: int, symbol: str) -> Optional[datetime]:
        row = self._fetch_one(
            """
            SELECT MAX(timestamp) AS ts FROM ticks
            WHERE market_id=? AND symbol=? AND COALESCE(status,'active')='active'
            """,
            (market_id, symbol.upper()),
        )
        if not row:
            return None
        ts = row["ts"] if isinstance(row, dict) else row[0]
        return _as_dt(ts) if ts else None

    def _count(self, market_id: int, symbol: str) -> int:
        row = self._fetch_one(
            "SELECT COUNT(*) AS c FROM ticks WHERE market_id=? AND symbol=?",
            (market_id, symbol.upper()),
        )
        if not row:
            return 0
        return int(row["c"] if isinstance(row, dict) else row[0])

    def _update_sync(self, market_id: int, last_tick: datetime, count_delta: int) -> None:
        now = datetime.utcnow().isoformat(timespec="seconds")
        existing = self._fetch_one(
            "SELECT ticks_count FROM tick_sync_status WHERE market_id=?",
            (market_id,),
        )
        prev = int(existing["ticks_count"] if existing and isinstance(existing, dict) else (existing[0] if existing else 0) or 0)
        self._execute(
            """
            INSERT INTO tick_sync_status (market_id, status, last_synced, last_tick_time, ticks_count, error_message)
            VALUES (?, 'completed', ?, ?, ?, NULL)
            ON CONFLICT(market_id) DO UPDATE SET
                status=excluded.status,
                last_synced=excluded.last_synced,
                last_tick_time=excluded.last_tick_time,
                ticks_count=excluded.ticks_count
            """,
            (market_id, now, last_tick.isoformat(timespec="seconds"), prev + count_delta),
        )
        self._commit()

    def _market_id(self, broker_id: int, symbol: str) -> Optional[int]:
        row = self._fetch_one(
            "SELECT market_id FROM markets WHERE broker_id=? AND symbol=?",
            (broker_id, symbol),
        )
        if not row:
            return None
        return int(row["market_id"] if isinstance(row, dict) else row[0])

    def _ensure_broker(self, name: str) -> Dict[str, Any]:
        row = self._fetch_one("SELECT * FROM brokers WHERE name=?", (name,))
        if row:
            return dict(row) if hasattr(row, "keys") else {"broker_id": row[0], "name": name}
        now = datetime.utcnow().isoformat(timespec="seconds")
        self._execute(
            """
            INSERT INTO brokers (broker_uuid, name, broker_type, status, metadata, created_at, updated_at)
            VALUES (?, ?, 'cfd', 'active', '{}', ?, ?)
            """,
            (str(uuid4()), name, now, now),
        )
        self._commit()
        row = self._fetch_one("SELECT * FROM brokers WHERE name=?", (name,))
        return dict(row) if row and hasattr(row, "keys") else {"broker_id": 0, "name": name}

    def _execute(self, sql: str, params: tuple = ()):
        if hasattr(self.db, "get_adapter"):
            return self.db.get_adapter().execute(sql, params)
        return self.db.execute(sql, params)

    def _commit(self) -> None:
        if hasattr(self.db, "get_adapter"):
            self.db.get_adapter().commit()
        elif hasattr(self.db, "commit"):
            self.db.commit()

    def _fetch_one(self, sql: str, params: tuple = ()):
        if hasattr(self.db, "fetch_one"):
            return self.db.fetch_one(sql, params)
        cur = self._execute(sql, params)
        row = cur.fetchone()
        return dict(row) if row is not None and hasattr(row, "keys") else row


def _as_dt(value: Any) -> datetime:
    if isinstance(value, datetime):
        return value.replace(tzinfo=None)
    return datetime.fromisoformat(str(value).replace("Z", ""))
