"""
ai/data/tick_adapter.py - Tick streaming adapters

VERSION: 1.0.0
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence


def normalize_tick(raw: Dict[str, Any]) -> Dict[str, Any]:
    ts = raw.get("timestamp", raw.get("time"))
    if isinstance(ts, (int, float)):
        value = float(ts)
        if value > 1e12:
            value /= 1000.0
        ts = datetime.utcfromtimestamp(value)
    elif isinstance(ts, str):
        ts = datetime.fromisoformat(ts.replace("Z", "+00:00")).replace(tzinfo=None)
    if not isinstance(ts, datetime):
        raise ValueError("Tick missing timestamp")
    bid = float(raw.get("bid", raw.get("price", 0.0)) or 0.0)
    ask = float(raw.get("ask", bid) or bid)
    return {
        "symbol": str(raw.get("symbol", "")).upper(),
        "timestamp": ts,
        "bid": bid,
        "ask": ask,
        "last": float(raw.get("last", (bid + ask) / 2.0)),
        "volume": float(raw.get("volume", 0.0) or 0.0),
        "flags": int(raw.get("flags", 0) or 0),
    }


class TickRepositoryAdapter:
    """Adapter around a future/present TickRepository."""

    def __init__(self, repository: Any):
        if not hasattr(repository, "stream_ticks"):
            raise TypeError("repository must expose stream_ticks()")
        self._repo = repository

    def stream_ticks(
        self,
        symbol: str,
        start_time: datetime,
        end_time: datetime,
        batch_size: Optional[int] = None,
        order: str = "ASC",
    ) -> Iterator[dict]:
        kwargs = {
            "symbol": symbol,
            "start_time": start_time,
            "end_time": end_time,
            "order": order,
        }
        if batch_size is not None:
            kwargs["batch_size"] = batch_size
        for tick in self._repo.stream_ticks(**kwargs):
            if hasattr(tick, "__dict__") and not isinstance(tick, dict):
                tick = {
                    "symbol": getattr(tick, "symbol", symbol),
                    "timestamp": getattr(tick, "timestamp", getattr(tick, "time", None)),
                    "bid": getattr(tick, "bid", None),
                    "ask": getattr(tick, "ask", None),
                    "last": getattr(tick, "last", None),
                    "volume": getattr(tick, "volume", 0.0),
                    "flags": getattr(tick, "flags", 0),
                }
            yield normalize_tick(tick)


class InMemoryTickSource:
    """In-memory tick source for tests and synthetic pipelines."""

    def __init__(self, ticks: Sequence[dict] | None = None):
        self._ticks: List[dict] = [normalize_tick(t) for t in (ticks or [])]

    def add(self, ticks: Iterable[dict]) -> None:
        self._ticks.extend(normalize_tick(t) for t in ticks)
        self._ticks.sort(key=lambda t: (t["symbol"], t["timestamp"]))

    def stream_ticks(
        self,
        symbol: str,
        start_time: datetime,
        end_time: datetime,
        batch_size: Optional[int] = None,
        order: str = "ASC",
    ) -> Iterator[dict]:
        symbol_u = symbol.upper()
        filtered = [
            t
            for t in self._ticks
            if t["symbol"] == symbol_u and start_time <= t["timestamp"] <= end_time
        ]
        filtered.sort(key=lambda t: t["timestamp"], reverse=(order.upper() == "DESC"))
        if batch_size is None or batch_size <= 0:
            yield from filtered
            return
        for i in range(0, len(filtered), batch_size):
            for tick in filtered[i : i + batch_size]:
                yield tick
