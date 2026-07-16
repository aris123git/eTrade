"""
collector.broker_sources.base - Broker data-source protocol
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, Iterable, List, Optional, Sequence


@dataclass
class DiscoveredSymbol:
    """A symbol as exposed by a specific broker/source."""

    broker_symbol: str
    canonical_symbol: str
    asset_class: str = "UNKNOWN"
    description: Optional[str] = None
    base_currency: Optional[str] = None
    quote_currency: Optional[str] = None
    digits: Optional[int] = None
    point: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class OHLCVBar:
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float = 0.0
    spread: Optional[float] = None
    tick_volume: Optional[int] = None


class BrokerSource(ABC):
    """
    Abstract market-data source.

    Implementations may talk to MetaTrader 5, import broker CSV dumps,
    or wrap REST APIs. All symbols are expected to be joinable via
    ``canonical_symbol``.
    """

    name: str
    source_type: str = "generic"

    @abstractmethod
    def connect(self) -> bool:
        """Open the source. Return True when ready."""

    @abstractmethod
    def disconnect(self) -> None:
        """Release resources."""

    @abstractmethod
    def discover_symbols(
        self,
        *,
        currency_pairs_only: bool = False,
    ) -> List[DiscoveredSymbol]:
        """List tradable symbols from this broker/source."""

    @abstractmethod
    def download_bars(
        self,
        broker_symbol: str,
        timeframe: str,
        *,
        start: Optional[datetime] = None,
        end: Optional[datetime] = None,
        count: Optional[int] = None,
    ) -> List[OHLCVBar]:
        """Download OHLCV bars for one broker symbol + timeframe."""

    def broker_metadata(self) -> Dict[str, Any]:
        """Optional metadata stored on the brokers row."""
        return {"source_type": self.source_type, "source_name": self.name}
