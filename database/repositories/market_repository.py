"""
database/repositories/market_repository.py - Market Repository

RESPONSIBILITY:
Manage market/symbol data in the database.

ARCHITECTURAL PRINCIPLES:
1. Single Responsibility - Only handles market data
2. Repository Pattern - Mediates between domain and data mapping
3. Type Safety - Uses Market model with validation
4. Business Logic - Market-specific queries and operations

SCALABILITY VISION:
This repository will handle millions of market records across
multiple brokers and timeframes.

VERSION: 1.0.0
"""

import json
import logging
from datetime import datetime
from typing import List, Optional, Dict, Any, Set, Tuple
from uuid import uuid4

from database.core.connection import DatabaseManager
from database.repositories.base_repository import BaseRepository
from database.models.market import Market, MarketType, MarketStatus


logger = logging.getLogger(__name__)


class MarketRepository(BaseRepository[Market]):
    """
    Repository for market/symbol data.
    
    Provides CRUD operations and market-specific queries.
    
    USAGE:
        repo = MarketRepository(db_manager)
        
        # Create a market
        market = repo.create(
            symbol="EURUSD",
            market_type=MarketType.FOREX,
            description="Euro/US Dollar",
            broker_id=1,
        )
        
        # Find active markets
        active = repo.find_active()
        
        # Find by symbol
        eurusd = repo.find_by_symbol("EURUSD")
    """
    
    TABLE = "markets"
    MODEL = Market
    
    def __init__(self, db_manager: DatabaseManager):
        """Initialize the market repository."""
        super().__init__(db_manager)
        self.logger = logging.getLogger(__name__)
    
    # ==========================================================================
    # CREATE OPERATIONS
    # ==========================================================================
    
    def create(
        self,
        symbol: str,
        market_type: MarketType,
        description: Optional[str] = None,
        broker_id: Optional[int] = None,
        base_currency: Optional[str] = None,
        quote_currency: Optional[str] = None,
        pip_size: Optional[float] = None,
        point: Optional[float] = None,
        digits: Optional[int] = None,
        contract_size: float = 1.0,
        metadata: Optional[Dict[str, Any]] = None,
        status: MarketStatus = MarketStatus.ACTIVE,
        canonical_symbol: Optional[str] = None,
    ) -> Market:
        """
        Create a new market.
        
        Args:
            symbol: Market symbol (e.g., "EURUSD")
            market_type: Type of market
            description: Market description
            broker_id: Broker ID
            base_currency: Base currency code
            quote_currency: Quote currency code
            pip_size: Pip size
            point: Point value
            digits: Number of decimal places
            contract_size: Contract size
            metadata: Additional metadata
            status: Market status
            canonical_symbol: Cross-broker instrument key
            
        Returns:
            Created Market object
        """
        from core.symbol_identity import canonicalize

        ident = canonicalize(symbol)
        market = Market(
            market_id=None,
            broker_id=broker_id,
            symbol=symbol,
            market_type=market_type,
            status=status,
            description=description,
            base_currency=base_currency or ident.base_currency,
            quote_currency=quote_currency or ident.quote_currency,
            pip_size=pip_size,
            point=point,
            digits=digits,
            contract_size=contract_size,
            canonical_symbol=canonical_symbol or ident.canonical_symbol,
            metadata=metadata or {},
            created_at=datetime.now(),
            updated_at=datetime.now(),
        )
        
        # Insert into database
        data = self._entity_to_dict(market)
        market_id = self.insert_dict(data)
        market.market_id = market_id
        
        self.logger.info(f"✅ Market created: {symbol} (ID: {market_id})")
        return market
    
    def create_or_update(
        self,
        symbol: str,
        data: Dict[str, Any],
    ) -> Market:
        """
        Create or update a market by symbol.
        
        Args:
            symbol: Market symbol
            data: Market data
            
        Returns:
            Market object
        """
        existing = self.find_by_symbol(symbol)
        
        if existing:
            # Update existing
            self.update(existing.market_id, data)
            return self.find_by_symbol(symbol)
        else:
            # Create new
            return self.create(symbol=symbol, **data)
    
    # ==========================================================================
    # READ OPERATIONS
    # ==========================================================================
    
    def find_by_symbol(self, symbol: str) -> Optional[Market]:
        """
        Find a market by broker symbol (first match).
        
        Args:
            symbol: Market symbol
            
        Returns:
            Market object or None
        """
        row = self.adapter.fetch_one(
            f"SELECT * FROM {self.TABLE} WHERE symbol = ?",
            (symbol,)
        )
        return self._row_to_entity(row) if row else None

    def find_by_broker_symbol(self, broker_id: int, symbol: str) -> Optional[Market]:
        """Find a market by broker + local symbol name."""
        row = self.adapter.fetch_one(
            f"SELECT * FROM {self.TABLE} WHERE broker_id = ? AND symbol = ?",
            (broker_id, symbol),
        )
        return self._row_to_entity(row) if row else None

    def find_by_canonical(self, canonical_symbol: str) -> List[Market]:
        """
        Find all broker markets that map to the same canonical instrument.
        """
        from core.symbol_identity import canonicalize

        canon = canonicalize(canonical_symbol).canonical_symbol
        rows = self.adapter.fetch_all(
            f"SELECT * FROM {self.TABLE} WHERE canonical_symbol = ?",
            (canon,),
        )
        return [self._row_to_entity(row) for row in rows]
    
    def find_by_broker(self, broker_id: int) -> List[Market]:
        """
        Find all markets for a broker.
        
        Args:
            broker_id: Broker ID
            
        Returns:
            List of Market objects
        """
        rows = self.adapter.fetch_all(
            f"SELECT * FROM {self.TABLE} WHERE broker_id = ?",
            (broker_id,)
        )
        return [self._row_to_entity(row) for row in rows]
    
    def find_by_type(self, market_type: MarketType) -> List[Market]:
        """
        Find all markets of a specific type.
        
        Args:
            market_type: Market type
            
        Returns:
            List of Market objects
        """
        rows = self.adapter.fetch_all(
            f"SELECT * FROM {self.TABLE} WHERE market_type = ?",
            (market_type.value,)
        )
        return [self._row_to_entity(row) for row in rows]
    
    def find_active(self) -> List[Market]:
        """
        Find all active markets.
        
        Returns:
            List of active Market objects
        """
        rows = self.adapter.fetch_all(
            f"SELECT * FROM {self.TABLE} WHERE status = ?",
            (MarketStatus.ACTIVE.value,)
        )
        return [self._row_to_entity(row) for row in rows]
    
    def find_by_status(self, status: MarketStatus) -> List[Market]:
        """
        Find all markets with a specific status.
        
        Args:
            status: Market status
            
        Returns:
            List of Market objects
        """
        rows = self.adapter.fetch_all(
            f"SELECT * FROM {self.TABLE} WHERE status = ?",
            (status.value,)
        )
        return [self._row_to_entity(row) for row in rows]
    
    def find_forex(self) -> List[Market]:
        """Find all Forex markets."""
        return self.find_by_type(MarketType.FOREX)
    
    def find_crypto(self) -> List[Market]:
        """Find all Cryptocurrency markets."""
        return self.find_by_type(MarketType.CRYPTO)
    
    def find_indices(self) -> List[Market]:
        """Find all Index markets."""
        return self.find_by_type(MarketType.INDEX)
    
    def find_commodities(self) -> List[Market]:
        """Find all Commodity markets."""
        return self.find_by_type(MarketType.COMMODITY)
    
    def find_by_base_currency(self, currency: str) -> List[Market]:
        """
        Find markets by base currency.
        
        Args:
            currency: Currency code (e.g., "USD")
            
        Returns:
            List of Market objects
        """
        rows = self.adapter.fetch_all(
            f"SELECT * FROM {self.TABLE} WHERE base_currency = ?",
            (currency,)
        )
        return [self._row_to_entity(row) for row in rows]
    
    def find_by_quote_currency(self, currency: str) -> List[Market]:
        """
        Find markets by quote currency.
        
        Args:
            currency: Currency code (e.g., "USD")
            
        Returns:
            List of Market objects
        """
        rows = self.adapter.fetch_all(
            f"SELECT * FROM {self.TABLE} WHERE quote_currency = ?",
            (currency,)
        )
        return [self._row_to_entity(row) for row in rows]
    
    def find_by_currencies(self, base: str, quote: str) -> Optional[Market]:
        """
        Find a market by base and quote currencies.
        
        Args:
            base: Base currency code
            quote: Quote currency code
            
        Returns:
            Market object or None
        """
        row = self.adapter.fetch_one(
            f"SELECT * FROM {self.TABLE} WHERE base_currency = ? AND quote_currency = ?",
            (base, quote)
        )
        return self._row_to_entity(row) if row else None
    
    def find_all_symbols(self) -> List[str]:
        """
        Get all market symbols.
        
        Returns:
            List of symbols
        """
        rows = self.adapter.fetch_all(
            f"SELECT symbol FROM {self.TABLE} WHERE status = ?",
            (MarketStatus.ACTIVE.value,)
        )
        return [row['symbol'] for row in rows]
    
    def find_tradable(self) -> List[Market]:
        """
        Find all tradable markets.
        
        Markets with status ACTIVE and have required fields.
        
        Returns:
            List of tradable Market objects
        """
        rows = self.adapter.fetch_all(f"""
            SELECT * FROM {self.TABLE} 
            WHERE status = ? 
            AND pip_size IS NOT NULL 
            AND pip_size > 0
            AND contract_size > 0
        """, (MarketStatus.ACTIVE.value,))
        return [self._row_to_entity(row) for row in rows]
    
    def get_type_counts(self) -> Dict[str, int]:
        """
        Get count of markets by type.
        
        Returns:
            Dictionary mapping type to count
        """
        rows = self.adapter.fetch_all(f"""
            SELECT market_type, COUNT(*) as count 
            FROM {self.TABLE} 
            WHERE status = ?
            GROUP BY market_type
        """, (MarketStatus.ACTIVE.value,))
        
        return {row['market_type']: row['count'] for row in rows}
    
    def get_broker_markets(self, broker_id: int) -> Dict[str, Any]:
        """
        Get all markets for a broker with statistics.
        
        Args:
            broker_id: Broker ID
            
        Returns:
            Dictionary with markets and statistics
        """
        markets = self.find_by_broker(broker_id)
        return {
            'broker_id': broker_id,
            'total_markets': len(markets),
            'markets': markets,
            'types': self._get_type_counts_for_broker(broker_id),
        }
    
    # ==========================================================================
    # UPDATE OPERATIONS
    # ==========================================================================
    
    def update_status(self, market_id: int, status: MarketStatus) -> bool:
        """
        Update market status.
        
        Args:
            market_id: Market ID
            status: New status
            
        Returns:
            True if updated, False otherwise
        """
        return self.update(market_id, {'status': status.value})
    
    def activate(self, market_id: int) -> bool:
        """Activate a market."""
        return self.update_status(market_id, MarketStatus.ACTIVE)
    
    def deactivate(self, market_id: int) -> bool:
        """Deactivate a market."""
        return self.update_status(market_id, MarketStatus.INACTIVE)
    
    def update_metadata(self, market_id: int, metadata: Dict[str, Any]) -> bool:
        """
        Update market metadata.
        
        Args:
            market_id: Market ID
            metadata: New metadata
            
        Returns:
            True if updated, False otherwise
        """
        existing = self.get_by_id(market_id)
        if not existing:
            return False
        
        new_metadata = {**existing.metadata, **metadata}
        return self.update(market_id, {'metadata': new_metadata})
    
    # ==========================================================================
    # DELETE OPERATIONS
    # ==========================================================================
    
    def delete_by_symbol(self, symbol: str) -> bool:
        """
        Delete a market by symbol.
        
        Args:
            symbol: Market symbol
            
        Returns:
            True if deleted, False otherwise
        """
        market = self.find_by_symbol(symbol)
        if not market:
            return False
        return self.delete(market.market_id)
    
    # ==========================================================================
    # UTILITY METHODS
    # ==========================================================================
    
    def get_symbols_by_type(self, market_type: MarketType) -> List[str]:
        """
        Get symbols by market type.
        
        Args:
            market_type: Market type
            
        Returns:
            List of symbols
        """
        markets = self.find_by_type(market_type)
        return [m.symbol for m in markets]
    
    def get_currency_pairs(self) -> List[Tuple[str, str]]:
        """
        Get all currency pairs (base, quote).
        
        Returns:
            List of (base_currency, quote_currency) tuples
        """
        rows = self.adapter.fetch_all(f"""
            SELECT base_currency, quote_currency 
            FROM {self.TABLE} 
            WHERE status = ? 
            AND base_currency IS NOT NULL 
            AND quote_currency IS NOT NULL
        """, (MarketStatus.ACTIVE.value,))
        return [(row['base_currency'], row['quote_currency']) for row in rows]
    
    def market_exists(self, symbol: str) -> bool:
        """
        Check if a market exists.
        
        Args:
            symbol: Market symbol
            
        Returns:
            True if exists, False otherwise
        """
        return self.count(f"symbol = ?", (symbol,)) > 0
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get repository statistics.
        
        Returns:
            Dictionary with statistics
        """
        base_stats = super().get_statistics()
        
        active_count = self.count("status = ?", (MarketStatus.ACTIVE.value,))
        inactive_count = self.count("status = ?", (MarketStatus.INACTIVE.value,))
        type_counts = self.get_type_counts()
        
        return {
            **base_stats,
            'active_markets': active_count,
            'inactive_markets': inactive_count,
            'market_types': type_counts,
        }
    
    def get_market_summary(self) -> Dict[str, Any]:
        """
        Get a summary of all markets.
        
        Returns:
            Dictionary with market summary
        """
        active = self.find_active()
        
        return {
            'total_active': len(active),
            'by_type': self.get_type_counts(),
            'brokers': self._get_broker_counts(),
            'symbols': [m.symbol for m in active],
        }
    
    # ==========================================================================
    # PRIVATE METHODS
    # ==========================================================================
    
    def _get_type_counts_for_broker(self, broker_id: int) -> Dict[str, int]:
        """Get type counts for a specific broker."""
        rows = self.adapter.fetch_all(f"""
            SELECT market_type, COUNT(*) as count 
            FROM {self.TABLE} 
            WHERE broker_id = ? AND status = ?
            GROUP BY market_type
        """, (broker_id, MarketStatus.ACTIVE.value))
        
        return {row['market_type']: row['count'] for row in rows}
    
    def _get_broker_counts(self) -> Dict[int, int]:
        """Get count of markets by broker."""
        rows = self.adapter.fetch_all(f"""
            SELECT broker_id, COUNT(*) as count 
            FROM {self.TABLE} 
            WHERE status = ?
            GROUP BY broker_id
        """, (MarketStatus.ACTIVE.value,))
        
        return {row['broker_id']: row['count'] for row in rows}
    
    # ==========================================================================
    # CONVERSION METHODS
    # ==========================================================================
    
    def _entity_to_dict(self, market: Market) -> Dict[str, Any]:
        """Convert Market entity to dictionary."""
        return {
            'market_id': market.market_id,
            'broker_id': market.broker_id,
            'symbol': market.symbol,
            'canonical_symbol': market.canonical_symbol,
            'market_type': market.market_type.value if market.market_type else None,
            'status': market.status.value if market.status else None,
            'description': market.description,
            'base_currency': market.base_currency,
            'quote_currency': market.quote_currency,
            'pip_size': market.pip_size,
            'point': market.point,
            'digits': market.digits,
            'contract_size': market.contract_size,
            'metadata': json.dumps(market.metadata) if market.metadata else '{}',
            'created_at': market.created_at.isoformat() if market.created_at else None,
            'updated_at': market.updated_at.isoformat() if market.updated_at else None,
        }
    
    def _row_to_entity(self, row: Dict[str, Any]) -> Market:
        """Convert database row to Market entity."""
        return Market(
            market_id=row['market_id'],
            broker_id=row['broker_id'],
            symbol=row['symbol'],
            market_type=MarketType(row['market_type']) if row['market_type'] else None,
            status=MarketStatus(row['status']) if row['status'] else None,
            description=row['description'],
            base_currency=row['base_currency'],
            quote_currency=row['quote_currency'],
            pip_size=row['pip_size'],
            point=row['point'],
            digits=row['digits'],
            contract_size=row['contract_size'],
            canonical_symbol=row.get('canonical_symbol'),
            metadata=json.loads(row['metadata']) if row['metadata'] else {},
            created_at=datetime.fromisoformat(row['created_at']) if row['created_at'] else None,
            updated_at=datetime.fromisoformat(row['updated_at']) if row['updated_at'] else None,
        )
    
    def _get_id(self, market: Market) -> int:
        """Get ID from Market entity."""
        return market.market_id


# ==============================================================================
# REGISTER IN REPOSITORY MANAGER
# ==============================================================================

# To be added to REPOSITORIES list in repository_manager.py:
# ('markets', MarketRepository, None)