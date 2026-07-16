"""
ai/data - Data access adapters over repository streaming APIs

VERSION: 1.0.0
"""

from ai.data.protocols import CandleSource, TickSource, MarketDataGateway
from ai.data.candle_adapter import CandleRepositoryAdapter, InMemoryCandleSource
from ai.data.tick_adapter import TickRepositoryAdapter, InMemoryTickSource
from ai.data.normalizers import candle_entity_to_dict, normalize_candle_dict

__all__ = [
    "CandleSource",
    "TickSource",
    "MarketDataGateway",
    "CandleRepositoryAdapter",
    "InMemoryCandleSource",
    "TickRepositoryAdapter",
    "InMemoryTickSource",
    "candle_entity_to_dict",
    "normalize_candle_dict",
]
