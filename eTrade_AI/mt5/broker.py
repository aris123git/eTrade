"""
database/broker.py - Broker Repository

RESPONSIBILITY:
Manage broker data with CRUD operations and statistics.

ARCHITECTURAL DECISIONS:
1. Extends BaseRepository - Inherits common database operations
2. UUID support - For distributed systems and cross-database synchronization
3. Active/Inactive tracking - Soft delete for data retention
4. Statistics aggregation - Market count, candle count, sync time
5. Thread-safe - All operations use the repository's connection
6. Proper error handling - All exceptions logged and raised as DatabaseError

USAGE:
    from database.broker import BrokerRepository, Broker
    
    repo = BrokerRepository(db_manager)
    broker = repo.add("ICMarkets", "forex", "icmarkets.com")
    
    # Get broker
    broker = repo.get_by_id(1)
    
    # Get statistics
    stats = repo.get_statistics(1)

VERSION: 1.0.0
"""

import logging
import sqlite3
from datetime import datetime
from typing import Optional, List, Dict, Any, Tuple
from uuid import uuid4

from database.repositories import BaseRepository
from core.exceptions import DatabaseError, DatabaseQueryError
from core.utils import to_datetime, format_datetime

logger = logging.getLogger(__name__)


# ==============================================================================
# DATA CLASS
# ==============================================================================

class Broker:
    """
    Broker data class representing a brokerage.
    
    Attributes:
        broker_id: Primary key (integer)
        broker_uuid: UUID for distributed systems
        name: Broker name (e.g., "ICMarkets")
        broker_type: Type of broker (e.g., "forex", "crypto")
        host: Server host (e.g., "icmarkets.com")
        active: Whether the broker is active
        created_at: Creation timestamp
        updated_at: Last update timestamp
    """
    
    def __init__(
        self,
        broker_id: Optional[int] = None,
        broker_uuid: Optional[str] = None,
        name: Optional[str] = None,
        broker_type: Optional[str] = None,
        host: Optional[str] = None,
        active: bool = True,
        created_at: Optional[datetime] = None,
        updated_at: Optional[datetime] = None,
    ):
        self.broker_id = broker_id
        self.broker_uuid = broker_uuid or str(uuid4())
        self.name = name
        self.broker_type = broker_type
        self.host = host
        self.active = active
        self.created_at = created_at or datetime.now()
        self.updated_at = updated_at or datetime.now()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'broker_id': self.broker_id,
            'broker_uuid': self.broker_uuid,
            'name': self.name,
            'broker_type': self.broker_type,
            'host': self.host,
            'active': self.active,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'updated_at': self.updated_at.isoformat() if self.updated_at else None,
        }
    
    @classmethod
    def from_row(cls, row: sqlite3.Row) -> 'Broker':
        """
        Create Broker from database row.
        
        Args:
            row: SQLite row object
        
        Returns:
            Broker instance
        """
        return cls(
            broker_id=row['broker_id'],
            broker_uuid=row['broker_uuid'],
            name=row['name'],
            broker_type=row['broker_type'],
            host=row['host'],
            active=bool(row['active']),
            created_at=to_datetime(row['created_at']),
            updated_at=to_datetime(row['updated_at']),
        )
    
    def __repr__(self) -> str:
        return f"Broker(id={self.broker_id}, name='{self.name}', type='{self.broker_type}')"


# ==============================================================================
# BROKER REPOSITORY
# ==============================================================================

class BrokerRepository(BaseRepository):
    """
    Repository for broker data.
    
    Provides CRUD operations and statistics aggregation.
    """
    
    # Table name
    TABLE = "brokers"
    
    # Schema definition
    SCHEMA = """
        CREATE TABLE IF NOT EXISTS brokers (
            broker_id INTEGER PRIMARY KEY AUTOINCREMENT,
            broker_uuid TEXT UNIQUE NOT NULL,
            name TEXT NOT NULL,
            broker_type TEXT,
            host TEXT,
            active BOOLEAN DEFAULT 1,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """
    
    # Indexes
    INDEXES = [
        "CREATE INDEX IF NOT EXISTS idx_brokers_name ON brokers(name)",
        "CREATE INDEX IF NOT EXISTS idx_brokers_type ON brokers(broker_type)",
        "CREATE INDEX IF NOT EXISTS idx_brokers_active ON brokers(active)",
        "CREATE INDEX IF NOT EXISTS idx_brokers_uuid ON brokers(broker_uuid)",
    ]
    
    def __init__(self, db_manager):
        """Initialize repository."""
        super().__init__(db_manager, self.TABLE, self.SCHEMA, self.INDEXES)
        self._table = self.TABLE
    
    # ==========================================================================
    # CREATE OPERATIONS
    # ==========================================================================
    
    def add(
        self,
        name: str,
        broker_type: str,
        host: Optional[str] = None,
    ) -> int:
        """
        Add a new broker.
        
        Args:
            name: Broker name
            broker_type: Type of broker (e.g., "forex")
            host: Server host (optional)
        
        Returns:
            broker_id of the new broker
        
        Raises:
            DatabaseError: If insert fails
        """
        self._ensure_initialized()
        
        try:
            broker_uuid = str(uuid4())
            
            with self._db_manager.connect() as conn:
                cursor = conn.execute("""
                    INSERT INTO brokers (
                        broker_uuid, name, broker_type, host, created_at, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?)
                """, (
                    broker_uuid,
                    name,
                    broker_type,
                    host,
                    datetime.now().isoformat(),
                    datetime.now().isoformat(),
                ))
                
                broker_id = cursor.lastrowid
                
                logger.info(f"✅ Broker added: {name} (ID: {broker_id})")
                return broker_id
                
        except sqlite3.IntegrityError as e:
            raise DatabaseError(f"Broker '{name}' already exists: {e}")
        except Exception as e:
            raise DatabaseError(f"Failed to add broker '{name}': {e}")
    
    # ==========================================================================
    # READ OPERATIONS
    # ==========================================================================
    
    def get_by_id(self, broker_id: int) -> Optional[Broker]:
        """
        Get broker by ID.
        
        Args:
            broker_id: Broker ID
        
        Returns:
            Broker instance or None
        """
        self._ensure_initialized()
        
        try:
            with self._db_manager.connect() as conn:
                cursor = conn.execute("""
                    SELECT * FROM brokers WHERE broker_id = ?
                """, (broker_id,))
                row = cursor.fetchone()
                
                if row:
                    return Broker.from_row(row)
                return None
                
        except Exception as e:
            logger.error(f"Error getting broker {broker_id}: {e}")
            return None
    
    def get_by_uuid(self, broker_uuid: str) -> Optional[Broker]:
        """
        Get broker by UUID.
        
        Args:
            broker_uuid: Broker UUID
        
        Returns:
            Broker instance or None
        """
        self._ensure_initialized()
        
        try:
            with self._db_manager.connect() as conn:
                cursor = conn.execute("""
                    SELECT * FROM brokers WHERE broker_uuid = ?
                """, (broker_uuid,))
                row = cursor.fetchone()
                
                if row:
                    return Broker.from_row(row)
                return None
                
        except Exception as e:
            logger.error(f"Error getting broker by UUID {broker_uuid}: {e}")
            return None
    
    def get_by_name(self, name: str) -> Optional[Broker]:
        """
        Get broker by name.
        
        Args:
            name: Broker name
        
        Returns:
            Broker instance or None
        """
        self._ensure_initialized()
        
        try:
            with self._db_manager.connect() as conn:
                cursor = conn.execute("""
                    SELECT * FROM brokers WHERE name = ?
                """, (name,))
                row = cursor.fetchone()
                
                if row:
                    return Broker.from_row(row)
                return None
                
        except Exception as e:
            logger.error(f"Error getting broker by name '{name}': {e}")
            return None
    
    def get_all(self, active_only: bool = True) -> List[Broker]:
        """
        Get all brokers.
        
        Args:
            active_only: If True, only return active brokers
        
        Returns:
            List of Broker instances
        """
        self._ensure_initialized()
        
        try:
            with self._db_manager.connect() as conn:
                if active_only:
                    cursor = conn.execute("""
                        SELECT * FROM brokers WHERE active = 1
                        ORDER BY name
                    """)
                else:
                    cursor = conn.execute("""
                        SELECT * FROM brokers ORDER BY name
                    """)
                
                return [Broker.from_row(row) for row in cursor.fetchall()]
                
        except Exception as e:
            logger.error(f"Error getting brokers: {e}")
            return []
    
    def get_by_type(self, broker_type: str, active_only: bool = True) -> List[Broker]:
        """
        Get brokers by type.
        
        Args:
            broker_type: Type of broker
            active_only: If True, only return active brokers
        
        Returns:
            List of Broker instances
        """
        self._ensure_initialized()
        
        try:
            with self._db_manager.connect() as conn:
                if active_only:
                    cursor = conn.execute("""
                        SELECT * FROM brokers 
                        WHERE broker_type = ? AND active = 1
                        ORDER BY name
                    """, (broker_type,))
                else:
                    cursor = conn.execute("""
                        SELECT * FROM brokers 
                        WHERE broker_type = ?
                        ORDER BY name
                    """, (broker_type,))
                
                return [Broker.from_row(row) for row in cursor.fetchall()]
                
        except Exception as e:
            logger.error(f"Error getting brokers by type '{broker_type}': {e}")
            return []
    
    # ==========================================================================
    # UPDATE OPERATIONS
    # ==========================================================================
    
    def update(self, broker_id: int, **kwargs) -> bool:
        """
        Update broker fields.
        
        Args:
            broker_id: Broker ID
            **kwargs: Fields to update (name, broker_type, host, active)
        
        Returns:
            True if updated, False otherwise
        """
        self._ensure_initialized()
        
        # Valid fields
        valid_fields = {'name', 'broker_type', 'host', 'active'}
        update_fields = {k: v for k, v in kwargs.items() if k in valid_fields}
        
        if not update_fields:
            logger.warning(f"No valid fields to update for broker {broker_id}")
            return False
        
        try:
            # Build SET clause
            set_clause = ", ".join([f"{k} = ?" for k in update_fields.keys()])
            values = list(update_fields.values())
            values.append(datetime.now().isoformat())
            values.append(broker_id)
            
            with self._db_manager.connect() as conn:
                cursor = conn.execute(f"""
                    UPDATE brokers 
                    SET {set_clause}, updated_at = ?
                    WHERE broker_id = ?
                """, values)
                
                updated = cursor.rowcount > 0
                
                if updated:
                    logger.info(f"✅ Broker {broker_id} updated: {update_fields}")
                else:
                    logger.warning(f"Broker {broker_id} not found for update")
                
                return updated
                
        except Exception as e:
            logger.error(f"Error updating broker {broker_id}: {e}")
            return False
    
    def set_active(self, broker_id: int, active: bool) -> bool:
        """
        Set broker active status.
        
        Args:
            broker_id: Broker ID
            active: Active status
        
        Returns:
            True if updated, False otherwise
        """
        return self.update(broker_id, active=active)
    
    # ==========================================================================
    # DELETE OPERATIONS
    # ==========================================================================
    
    def delete(self, broker_id: int, soft: bool = True) -> bool:
        """
        Delete broker.
        
        Args:
            broker_id: Broker ID
            soft: If True, soft delete (set active=False)
        
        Returns:
            True if deleted, False otherwise
        """
        self._ensure_initialized()
        
        if soft:
            return self.set_active(broker_id, False)
        
        # Hard delete
        try:
            with self._db_manager.connect() as conn:
                cursor = conn.execute("""
                    DELETE FROM brokers WHERE broker_id = ?
                """, (broker_id,))
                
                deleted = cursor.rowcount > 0
                
                if deleted:
                    logger.info(f"🗑️ Broker {broker_id} hard deleted")
                else:
                    logger.warning(f"Broker {broker_id} not found for deletion")
                
                return deleted
                
        except Exception as e:
            logger.error(f"Error deleting broker {broker_id}: {e}")
            return False
    
    # ==========================================================================
    # UTILITY OPERATIONS
    # ==========================================================================
    
    def exists(self, name: str) -> bool:
        """
        Check if broker exists by name.
        
        Args:
            name: Broker name
        
        Returns:
            True if exists, False otherwise
        """
        self._ensure_initialized()
        
        try:
            with self._db_manager.connect() as conn:
                cursor = conn.execute("""
                    SELECT COUNT(*) as count FROM brokers WHERE name = ?
                """, (name,))
                row = cursor.fetchone()
                return row['count'] > 0
                
        except Exception as e:
            logger.error(f"Error checking broker existence '{name}': {e}")
            return False
    
    def exists_by_uuid(self, broker_uuid: str) -> bool:
        """
        Check if broker exists by UUID.
        
        Args:
            broker_uuid: Broker UUID
        
        Returns:
            True if exists, False otherwise
        """
        self._ensure_initialized()
        
        try:
            with self._db_manager.connect() as conn:
                cursor = conn.execute("""
                    SELECT COUNT(*) as count FROM brokers WHERE broker_uuid = ?
                """, (broker_uuid,))
                row = cursor.fetchone()
                return row['count'] > 0
                
        except Exception as e:
            logger.error(f"Error checking broker existence by UUID: {e}")
            return False
    
    def count(self, active_only: bool = True) -> int:
        """
        Count brokers.
        
        Args:
            active_only: If True, only count active brokers
        
        Returns:
            Number of brokers
        """
        self._ensure_initialized()
        
        try:
            with self._db_manager.connect() as conn:
                if active_only:
                    cursor = conn.execute("""
                        SELECT COUNT(*) as count FROM brokers WHERE active = 1
                    """)
                else:
                    cursor = conn.execute("""
                        SELECT COUNT(*) as count FROM brokers
                    """)
                row = cursor.fetchone()
                return row['count']
                
        except Exception as e:
            logger.error(f"Error counting brokers: {e}")
            return 0
    
    # ==========================================================================
    # STATISTICS OPERATIONS
    # ==========================================================================
    
    def get_statistics(self, broker_id: int) -> Dict[str, Any]:
        """
        Get statistics for a broker.
        
        Args:
            broker_id: Broker ID
        
        Returns:
            Dictionary with statistics:
            - market_count: Number of markets
            - candle_count: Total candles
            - last_sync: Last sync timestamp
            - symbols: List of symbols
        """
        self._ensure_initialized()
        
        stats = {
            'broker_id': broker_id,
            'market_count': 0,
            'candle_count': 0,
            'last_sync': None,
            'symbols': [],
        }
        
        try:
            with self._db_manager.connect() as conn:
                # Get market count
                cursor = conn.execute("""
                    SELECT COUNT(*) as count 
                    FROM markets 
                    WHERE broker_id = ? AND active = 1
                """, (broker_id,))
                row = cursor.fetchone()
                stats['market_count'] = row['count'] if row else 0
                
                # Get symbols
                cursor = conn.execute("""
                    SELECT symbol 
                    FROM markets 
                    WHERE broker_id = ? AND active = 1
                    ORDER BY symbol
                """, (broker_id,))
                stats['symbols'] = [row['symbol'] for row in cursor.fetchall()]
                
                # Get candle count (approximate)
                cursor = conn.execute("""
                    SELECT COUNT(*) as count 
                    FROM candles c
                    JOIN markets m ON c.market_id = m.market_id
                    WHERE m.broker_id = ?
                """, (broker_id,))
                row = cursor.fetchone()
                stats['candle_count'] = row['count'] if row else 0
                
                # Get last sync time
                cursor = conn.execute("""
                    SELECT MAX(c.time) as last_sync
                    FROM candles c
                    JOIN markets m ON c.market_id = m.market_id
                    WHERE m.broker_id = ?
                """, (broker_id,))
                row = cursor.fetchone()
                if row and row['last_sync']:
                    stats['last_sync'] = to_datetime(row['last_sync']).isoformat()
                
                return stats
                
        except Exception as e:
            logger.error(f"Error getting statistics for broker {broker_id}: {e}")
            return stats
    
    def get_broker_summary(self) -> List[Dict[str, Any]]:
        """
        Get summary of all brokers with statistics.
        
        Returns:
            List of broker summaries
        """
        self._ensure_initialized()
        
        brokers = self.get_all(active_only=True)
        summaries = []
        
        for broker in brokers:
            stats = self.get_statistics(broker.broker_id)
            summaries.append({
                'broker_id': broker.broker_id,
                'name': broker.name,
                'type': broker.broker_type,
                'host': broker.host,
                'market_count': stats['market_count'],
                'candle_count': stats['candle_count'],
                'last_sync': stats['last_sync'],
            })
        
        return summaries
    
    # ==========================================================================
    # BULK OPERATIONS
    # ==========================================================================
    
    def get_or_create(self, name: str, broker_type: str, host: Optional[str] = None) -> Tuple[Broker, bool]:
        """
        Get broker by name or create if not exists.
        
        Args:
            name: Broker name
            broker_type: Type of broker
            host: Server host
        
        Returns:
            Tuple of (Broker, created) where created is True if new
        """
        broker = self.get_by_name(name)
        if broker:
            return broker, False
        
        broker_id = self.add(name, broker_type, host)
        broker = self.get_by_id(broker_id)
        return broker, True
    
    # ==========================================================================
    # INTERNAL METHODS
    # ==========================================================================
    
    def _ensure_initialized(self) -> None:
        """Ensure repository is initialized."""
        if not self._initialized:
            self.init()