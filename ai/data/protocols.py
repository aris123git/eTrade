"""
ai/data/protocols.py - Repository-agnostic streaming contracts

VERSION: 1.0.0
"""

from __future__ import annotations

from datetime import datetime
from typing import Iterator, List, Optional, Protocol, runtime_checkable

from ai.utils.types import CandleDict


@runtime_checkable
class CandleSource(Protocol):
    """Protocol matching CandleRepository streaming capabilities."""

    def stream_candles(
        self,
        symbol: str,
        timeframe: str,
        start_time: datetime,
        end_time: datetime,
        batch_size: Optional[int] = None,
        order: str = "ASC",
    ) -> Iterator[CandleDict]:
        ...

    def get_last_n(
        self,
        symbol: str,
        timeframe: str,
        n: int = 500,
    ) -> List[CandleDict]:
        ...


@runtime_checkable
class TickSource(Protocol):
    """Protocol for tick-level streaming."""

    def stream_ticks(
        self,
        symbol: str,
        start_time: datetime,
        end_time: datetime,
        batch_size: Optional[int] = None,
        order: str = "ASC",
    ) -> Iterator[dict]:
        ...


@runtime_checkable
class MarketDataGateway(Protocol):
    """Combined gateway used by higher-level services."""

    @property
    def candles(self) -> CandleSource:
        ...

    @property
    def ticks(self) -> Optional[TickSource]:
        ...
