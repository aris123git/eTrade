"""
ai/data/candle_adapter.py - Adapters over CandleRepository and in-memory sources

VERSION: 1.0.0
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Iterable, Iterator, List, Optional, Sequence

from ai.data.normalizers import candle_entity_to_dict
from ai.utils.types import CandleDict


class CandleRepositoryAdapter:
    """
    Thin adapter around CandleRepository.

    Reuses repository streaming — never reimplements persistence.
    """

    def __init__(self, repository: Any):
        if not hasattr(repository, "stream_candles"):
            raise TypeError("repository must expose stream_candles()")
        self._repo = repository

    def stream_candles(
        self,
        symbol: str,
        timeframe: str,
        start_time: datetime,
        end_time: datetime,
        batch_size: Optional[int] = None,
        order: str = "ASC",
    ) -> Iterator[CandleDict]:
        kwargs = {
            "symbol": symbol,
            "timeframe": timeframe,
            "start_time": start_time,
            "end_time": end_time,
            "order": order,
        }
        if batch_size is not None:
            kwargs["batch_size"] = batch_size
        for entity in self._repo.stream_candles(**kwargs):
            yield candle_entity_to_dict(entity)

    def get_last_n(self, symbol: str, timeframe: str, n: int = 500) -> List[CandleDict]:
        if hasattr(self._repo, "get_last_n"):
            entities = self._repo.get_last_n(symbol, timeframe, n)
        elif hasattr(self._repo, "find_latest"):
            entities = self._repo.find_latest(symbol, timeframe, limit=n)
            entities = list(reversed(list(entities)))
        else:
            raise AttributeError("repository lacks get_last_n/find_latest")
        return [candle_entity_to_dict(e) for e in entities]


class InMemoryCandleSource:
    """In-memory candle source for tests and offline pipelines."""

    def __init__(self, candles: Sequence[CandleDict] | None = None):
        self._candles: List[CandleDict] = [
            candle_entity_to_dict(c) for c in (candles or [])
        ]

    def add(self, candles: Iterable[CandleDict]) -> None:
        self._candles.extend(candle_entity_to_dict(c) for c in candles)
        self._candles.sort(key=lambda c: (c["symbol"], c["timeframe"], c["timestamp"]))

    def stream_candles(
        self,
        symbol: str,
        timeframe: str,
        start_time: datetime,
        end_time: datetime,
        batch_size: Optional[int] = None,
        order: str = "ASC",
    ) -> Iterator[CandleDict]:
        symbol_u = symbol.upper()
        timeframe_u = timeframe.upper()
        filtered = [
            c
            for c in self._candles
            if c["symbol"] == symbol_u
            and c["timeframe"] == timeframe_u
            and start_time <= c["timestamp"] <= end_time
        ]
        filtered.sort(key=lambda c: c["timestamp"], reverse=(order.upper() == "DESC"))
        if batch_size is None or batch_size <= 0:
            yield from filtered
            return
        for i in range(0, len(filtered), batch_size):
            for candle in filtered[i : i + batch_size]:
                yield candle

    def get_last_n(self, symbol: str, timeframe: str, n: int = 500) -> List[CandleDict]:
        symbol_u = symbol.upper()
        timeframe_u = timeframe.upper()
        filtered = [
            c
            for c in self._candles
            if c["symbol"] == symbol_u and c["timeframe"] == timeframe_u
        ]
        filtered.sort(key=lambda c: c["timestamp"])
        return filtered[-n:]
