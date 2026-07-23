"""
database/repositories/factory.py - Repository factories

VERSION: 1.0.0
"""

from __future__ import annotations

from typing import Any, Optional, Type

from database.core.connection import DatabaseManager
from database.repositories.base_repository import BaseRepository
from database.repositories.repository_manager import RepositoryManager


_REPO_MAP = None


def _repo_map():
    global _REPO_MAP
    if _REPO_MAP is not None:
        return _REPO_MAP
    from database.repositories.broker_repository import BrokerRepository
    from database.repositories.candle_repository import CandleRepository
    from database.repositories.currency_repository import CurrencyRepository
    from database.repositories.market_repository import MarketRepository
    from database.repositories.symbol_repository import SymbolRepository
    from database.repositories.research_repository import ResearchRepository
    from database.repositories.tick_repository import TickRepository
    from database.repositories.timeframe_repository import TimeframeRepository

    _REPO_MAP = {
        "brokers": BrokerRepository,
        "candles": CandleRepository,
        "currencies": CurrencyRepository,
        "markets": MarketRepository,
        "research": ResearchRepository,
        "symbols": SymbolRepository,
        "ticks": TickRepository,
        "timeframes": TimeframeRepository,
    }
    return _REPO_MAP


def create_repository(name: str, db_manager: DatabaseManager) -> BaseRepository:
    """Create a single repository by registry name."""
    repo_cls: Optional[Type[BaseRepository]] = _repo_map().get(name)
    if repo_cls is None:
        raise KeyError(f"Unknown repository: {name}")
    return repo_cls(db_manager)


def create_repository_manager(db_manager: Optional[DatabaseManager] = None, **kwargs: Any) -> RepositoryManager:
    """Create a RepositoryManager with all standard repositories registered."""
    manager_db = db_manager or DatabaseManager(**kwargs)
    return RepositoryManager(manager_db)
