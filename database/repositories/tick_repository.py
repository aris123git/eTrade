"""
database/repositories/tick_repository.py - Tick Repository

VERSION: 1.0.0
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timedelta
from typing import Any, Dict, Iterator, List, Optional
from uuid import uuid4

from database.core.connection import DatabaseManager
from database.models.tick import Tick, TickStatus
from database.repositories.base_repository import BaseRepository

logger = logging.getLogger(__name__)


class TickRepository(BaseRepository[Tick]):
    @staticmethod
    def _sql_ts(value):
        if isinstance(value, datetime):
            return value.isoformat(timespec="seconds")
        return value

    """Repository for tick-level market data with streaming support."""

    TABLE = "ticks"
    MODEL = Tick
    STREAM_BATCH_SIZE = 20000

    def __init__(self, db_manager: DatabaseManager):
        super().__init__(db_manager)
        self.logger = logging.getLogger(__name__)

    def create(
        self,
        symbol: str,
        timestamp: datetime,
        bid: float,
        ask: float,
        last: float = 0.0,
        volume: float = 0.0,
        flags: int = 0,
        market_id: Optional[int] = None,
        broker_id: Optional[int] = None,
        status: TickStatus = TickStatus.ACTIVE,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Tick:
        now = datetime.utcnow()
        tick = Tick(
            tick_id=None,
            tick_uuid=str(uuid4()),
            symbol=symbol.upper(),
            timestamp=timestamp,
            bid=float(bid),
            ask=float(ask),
            last=float(last or ((bid + ask) / 2.0)),
            volume=float(volume or 0.0),
            flags=int(flags or 0),
            market_id=market_id,
            broker_id=broker_id,
            status=status,
            metadata=metadata or {},
            created_at=now,
            updated_at=now,
        )
        data = self._entity_to_dict(tick)
        if data.get("broker_id") is None:
            data["broker_id"] = 0
        tick_id = self.upsert(data, ["broker_id", "symbol", "timestamp", "bid", "ask"])
        tick.tick_id = int(tick_id) if tick_id is not None else None
        return tick

    def bulk_upsert(self, ticks: List[Dict[str, Any]], batch_size: int = 5000) -> int:
        total = 0
        now = datetime.utcnow().isoformat(timespec="seconds")
        for i in range(0, len(ticks), batch_size):
            chunk = ticks[i : i + batch_size]
            rows = []
            for raw in chunk:
                ts = raw.get("timestamp") or raw.get("time")
                if isinstance(ts, datetime):
                    ts = ts.isoformat(timespec="seconds")
                rows.append(
                    (
                        raw.get("tick_uuid") or str(uuid4()),
                        str(raw["symbol"]).upper(),
                        ts,
                        float(raw["bid"]),
                        float(raw["ask"]),
                        float(raw.get("last", (float(raw["bid"]) + float(raw["ask"])) / 2.0)),
                        float(raw.get("volume", 0.0) or 0.0),
                        int(raw.get("flags", 0) or 0),
                        raw.get("market_id"),
                        raw.get("broker_id") if raw.get("broker_id") is not None else 0,
                        raw.get("status", TickStatus.ACTIVE.value),
                        json.dumps(raw.get("metadata") or {}),
                        now,
                        now,
                    )
                )
            sql = """
                INSERT INTO ticks (
                    tick_uuid, symbol, timestamp, bid, ask, last, volume, flags,
                    market_id, broker_id, status, metadata, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(broker_id, symbol, timestamp, bid, ask) DO UPDATE SET
                    last=excluded.last,
                    volume=excluded.volume,
                    flags=excluded.flags,
                    status=excluded.status,
                    metadata=excluded.metadata,
                    updated_at=excluded.updated_at
            """
            total += self.adapter.execute_many(sql, rows)
        return total


    def stream_ticks(
        self,
        symbol: str,
        start_time: datetime,
        end_time: datetime,
        batch_size: Optional[int] = None,
        order: str = "ASC",
    ) -> Iterator[Tick]:
        batch_size = batch_size or self.STREAM_BATCH_SIZE
        order = order.upper() if order.upper() in {"ASC", "DESC"} else "ASC"
        last_ts = start_time if order == "ASC" else end_time
        first_page = True
        while True:
            if order == "ASC":
                op = ">=" if first_page else ">"
                sql = f"""
                    SELECT * FROM {self.TABLE}
                    WHERE symbol = ? AND status = ?
                      AND timestamp {op} ? AND timestamp <= ?
                    ORDER BY timestamp ASC
                    LIMIT ?
                """
                params = (symbol.upper(), TickStatus.ACTIVE.value, self._sql_ts(last_ts), self._sql_ts(end_time), batch_size)
            else:
                op = "<=" if first_page else "<"
                sql = f"""
                    SELECT * FROM {self.TABLE}
                    WHERE symbol = ? AND status = ?
                      AND timestamp <= ? AND timestamp >= ?
                    ORDER BY timestamp DESC
                    LIMIT ?
                """
                # keep DESC first-page inclusive; subsequent pages use last_ts upper bound via reassignment below
                if not first_page:
                    sql = f"""
                        SELECT * FROM {self.TABLE}
                        WHERE symbol = ? AND status = ?
                          AND timestamp < ? AND timestamp >= ?
                        ORDER BY timestamp DESC
                        LIMIT ?
                    """
                params = (symbol.upper(), TickStatus.ACTIVE.value, self._sql_ts(last_ts), self._sql_ts(start_time), batch_size)
            rows = self.adapter.fetch_all(sql, params)
            if not rows:
                break
            for row in rows:
                tick = self._row_to_entity(row)
                yield tick
                last_ts = tick.timestamp
            first_page = False
            if len(rows) < batch_size:
                break


    def _entity_to_dict(self, tick: Tick) -> Dict[str, Any]:
        ts = tick.timestamp
        if isinstance(ts, datetime):
            ts = ts.isoformat(timespec="seconds")
        return {
            "tick_id": tick.tick_id,
            "tick_uuid": tick.tick_uuid,
            "symbol": tick.symbol,
            "timestamp": ts,
            "bid": tick.bid,
            "ask": tick.ask,
            "last": tick.last,
            "volume": tick.volume,
            "flags": tick.flags,
            "market_id": tick.market_id,
            "broker_id": tick.broker_id,
            "status": tick.status.value if tick.status else None,
            "metadata": json.dumps(tick.metadata) if tick.metadata else "{}",
            "created_at": tick.created_at.isoformat(timespec="seconds") if isinstance(tick.created_at, datetime) else tick.created_at,
            "updated_at": tick.updated_at.isoformat(timespec="seconds") if isinstance(tick.updated_at, datetime) else tick.updated_at,
        }

    def _row_to_entity(self, row: Dict[str, Any]) -> Tick:
        ts = row["timestamp"]
        if isinstance(ts, str):
            ts = datetime.fromisoformat(ts)
        created = row.get("created_at")
        updated = row.get("updated_at")
        if isinstance(created, str):
            created = datetime.fromisoformat(created)
        if isinstance(updated, str):
            updated = datetime.fromisoformat(updated)
        return Tick(
            tick_id=row["tick_id"],
            tick_uuid=row["tick_uuid"],
            symbol=row["symbol"],
            timestamp=ts,
            bid=row["bid"],
            ask=row["ask"],
            last=row.get("last") or 0.0,
            volume=row.get("volume") or 0.0,
            flags=row.get("flags") or 0,
            market_id=row.get("market_id"),
            broker_id=row.get("broker_id"),
            status=TickStatus(row["status"]) if row.get("status") else TickStatus.ACTIVE,
            metadata=json.loads(row["metadata"]) if row.get("metadata") else {},
            created_at=created,
            updated_at=updated,
        )

    def _get_id(self, tick: Tick) -> int:
        return tick.tick_id
