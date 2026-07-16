"""
database/repositories/broker_repository.py - Broker Repository

RESPONSIBILITY:
Manage broker data in the database.

ARCHITECTURAL PRINCIPLES:
1. Single Responsibility - Only handles broker data
2. Repository Pattern - Mediates between domain and data mapping
3. Type Safety - Uses Broker model with validation
4. Business Logic - Broker-specific queries and operations

SCALABILITY VISION:
This repository will handle broker configurations, connections,
and multi-broker support for the AI trading platform.

VERSION: 1.0.0
"""

import json
import logging
from datetime import datetime
from typing import List, Optional, Dict, Any, Set
from uuid import uuid4

from database.core.connection import DatabaseManager
from database.repositories.base_repository import BaseRepository
from database.models.broker import Broker, BrokerStatus, BrokerType


logger = logging.getLogger(__name__)


class BrokerRepository(BaseRepository[Broker]):
    """
    Repository for broker data.
    
    Provides CRUD operations and broker-specific queries.
    
    USAGE:
        repo = BrokerRepository(db_manager)
        
        # Create a broker
        broker = repo.create(
            name="ICMarkets",
            broker_type=BrokerType.FOREX,
            server="icmarkets.com",
            description="IC Markets Forex Broker",
        )
        
        # Find active brokers
        active = repo.find_active()
        
        # Find by name
        icmarkets = repo.find_by_name("ICMarkets")
    """
    
    TABLE = "brokers"
    MODEL = Broker
    
    def __init__(self, db_manager: DatabaseManager):
        """Initialize the broker repository."""
        super().__init__(db_manager)
        self.logger = logging.getLogger(__name__)
    
    # ==========================================================================
    # CREATE OPERATIONS
    # ==========================================================================
    
    def create(
        self,
        name: str,
        broker_type: BrokerType,
        server: Optional[str] = None,
        host: Optional[str] = None,
        port: Optional[int] = None,
        description: Optional[str] = None,
        login: Optional[str] = None,
        password_encrypted: Optional[str] = None,
        status: BrokerStatus = BrokerStatus.ACTIVE,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Broker:
        """
        Create a new broker.
        
        Args:
            name: Broker name
            broker_type: Type of broker
            server: Server name
            host: Host address
            port: Port number
            description: Broker description
            login: Login credentials (encrypted)
            password_encrypted: Password (encrypted)
            status: Broker status
            metadata: Additional metadata
            
        Returns:
            Created Broker object
        """
        broker = Broker(
            broker_id=None,
            broker_uuid=str(uuid4()),
            name=name,
            broker_type=broker_type,
            server=server,
            host=host,
            port=port,
            description=description,
            login=login,
            password_encrypted=password_encrypted,
            status=status,
            metadata=metadata or {},
            created_at=datetime.now(),
            updated_at=datetime.now(),
        )
        
        # Insert into database
        data = self._entity_to_dict(broker)
        broker_id = self.insert_dict(data)
        broker.broker_id = broker_id
        
        self.logger.info(f"✅ Broker created: {name} (ID: {broker_id})")
        return broker
    
    def create_or_update(
        self,
        name: str,
        data: Dict[str, Any],
    ) -> Broker:
        """
        Create or update a broker by name.
        
        Args:
            name: Broker name
            data: Broker data
            
        Returns:
            Broker object
        """
        existing = self.find_by_name(name)
        
        if existing:
            # Update existing
            self.update(existing.broker_id, data)
            return self.find_by_name(name)
        else:
            # Create new
            return self.create(name=name, **data)
    
    def create_default_broker(self) -> Broker:
        """
        Create the default broker configuration.
        
        Returns:
            Broker object
        """
        return self.create(
            name="Default Broker",
            broker_type=BrokerType.CFD,
            server="localhost",
            description="Default broker configuration for MarketAI",
            metadata={
                'is_default': True,
                'version': '1.0.0',
            },
        )
    
    # ==========================================================================
    # READ OPERATIONS
    # ==========================================================================
    
    def find_by_name(self, name: str) -> Optional[Broker]:
        """
        Find a broker by name.
        
        Args:
            name: Broker name
            
        Returns:
            Broker object or None
        """
        row = self.adapter.fetch_one(
            f"SELECT * FROM {self.TABLE} WHERE name = ?",
            (name,)
        )
        return self._row_to_entity(row) if row else None
    
    def find_by_uuid(self, broker_uuid: str) -> Optional[Broker]:
        """
        Find a broker by UUID.
        
        Args:
            broker_uuid: Broker UUID
            
        Returns:
            Broker object or None
        """
        row = self.adapter.fetch_one(
            f"SELECT * FROM {self.TABLE} WHERE broker_uuid = ?",
            (broker_uuid,)
        )
        return self._row_to_entity(row) if row else None
    
    def find_by_type(self, broker_type: BrokerType) -> List[Broker]:
        """
        Find all brokers of a specific type.
        
        Args:
            broker_type: Broker type
            
        Returns:
            List of Broker objects
        """
        rows = self.adapter.fetch_all(
            f"SELECT * FROM {self.TABLE} WHERE broker_type = ?",
            (broker_type.value,)
        )
        return [self._row_to_entity(row) for row in rows]
    
    def find_active(self) -> List[Broker]:
        """
        Find all active brokers.
        
        Returns:
            List of active Broker objects
        """
        rows = self.adapter.fetch_all(
            f"SELECT * FROM {self.TABLE} WHERE status = ?",
            (BrokerStatus.ACTIVE.value,)
        )
        return [self._row_to_entity(row) for row in rows]
    
    def find_by_status(self, status: BrokerStatus) -> List[Broker]:
        """
        Find all brokers with a specific status.
        
        Args:
            status: Broker status
            
        Returns:
            List of Broker objects
        """
        rows = self.adapter.fetch_all(
            f"SELECT * FROM {self.TABLE} WHERE status = ?",
            (status.value,)
        )
        return [self._row_to_entity(row) for row in rows]
    
    def find_connected(self) -> List[Broker]:
        """
        Find all connected brokers.
        
        Returns:
            List of connected Broker objects
        """
        rows = self.adapter.fetch_all(
            f"SELECT * FROM {self.TABLE} WHERE status = ? AND server IS NOT NULL",
            (BrokerStatus.ACTIVE.value,)
        )
        return [self._row_to_entity(row) for row in rows]
    
    def find_default(self) -> Optional[Broker]:
        """
        Find the default broker.
        
        Returns:
            Default Broker object or None
        """
        rows = self.adapter.fetch_all(f"""
            SELECT * FROM {self.TABLE} 
            WHERE metadata LIKE '%is_default%true%' 
            AND status = ?
            LIMIT 1
        """, (BrokerStatus.ACTIVE.value,))
        
        if rows:
            return self._row_to_entity(rows[0])
        return None
    
    def find_all_names(self) -> List[str]:
        """
        Get all broker names.
        
        Returns:
            List of broker names
        """
        rows = self.adapter.fetch_all(
            f"SELECT name FROM {self.TABLE} WHERE status = ?",
            (BrokerStatus.ACTIVE.value,)
        )
        return [row['name'] for row in rows]
    
    def find_by_server(self, server: str) -> List[Broker]:
        """
        Find brokers by server.
        
        Args:
            server: Server name
            
        Returns:
            List of Broker objects
        """
        rows = self.adapter.fetch_all(
            f"SELECT * FROM {self.TABLE} WHERE server = ?",
            (server,)
        )
        return [self._row_to_entity(row) for row in rows]
    
    def find_by_host(self, host: str) -> List[Broker]:
        """
        Find brokers by host.
        
        Args:
            host: Host address
            
        Returns:
            List of Broker objects
        """
        rows = self.adapter.fetch_all(
            f"SELECT * FROM {self.TABLE} WHERE host = ?",
            (host,)
        )
        return [self._row_to_entity(row) for row in rows]
    
    def get_type_counts(self) -> Dict[str, int]:
        """
        Get count of brokers by type.
        
        Returns:
            Dictionary mapping type to count
        """
        rows = self.adapter.fetch_all(f"""
            SELECT broker_type, COUNT(*) as count 
            FROM {self.TABLE} 
            WHERE status = ?
            GROUP BY broker_type
        """, (BrokerStatus.ACTIVE.value,))
        
        return {row['broker_type']: row['count'] for row in rows}
    
    def get_broker_summary(self, broker_id: int) -> Dict[str, Any]:
        """
        Get detailed summary of a broker.
        
        Args:
            broker_id: Broker ID
            
        Returns:
            Dictionary with broker details
        """
        broker = self.get_by_id(broker_id)
        if not broker:
            return {'error': f'Broker {broker_id} not found'}
        
        return {
            'broker_id': broker.broker_id,
            'name': broker.name,
            'type': broker.broker_type.value,
            'server': broker.server,
            'host': broker.host,
            'status': broker.status.value,
            'created_at': broker.created_at,
            'updated_at': broker.updated_at,
            'metadata': broker.metadata,
            'has_credentials': bool(broker.login and broker.password_encrypted),
        }
    
    # ==========================================================================
    # UPDATE OPERATIONS
    # ==========================================================================
    
    def update_status(self, broker_id: int, status: BrokerStatus) -> bool:
        """
        Update broker status.
        
        Args:
            broker_id: Broker ID
            status: New status
            
        Returns:
            True if updated, False otherwise
        """
        return self.update(broker_id, {'status': status.value})
    
    def activate(self, broker_id: int) -> bool:
        """Activate a broker."""
        return self.update_status(broker_id, BrokerStatus.ACTIVE)
    
    def deactivate(self, broker_id: int) -> bool:
        """Deactivate a broker."""
        return self.update_status(broker_id, BrokerStatus.INACTIVE)
    
    def update_credentials(
        self,
        broker_id: int,
        login: str,
        password_encrypted: str,
    ) -> bool:
        """
        Update broker credentials.
        
        Args:
            broker_id: Broker ID
            login: Login (encrypted)
            password_encrypted: Password (encrypted)
            
        Returns:
            True if updated, False otherwise
        """
        return self.update(broker_id, {
            'login': login,
            'password_encrypted': password_encrypted,
        })
    
    def update_connection(
        self,
        broker_id: int,
        server: Optional[str] = None,
        host: Optional[str] = None,
        port: Optional[int] = None,
    ) -> bool:
        """
        Update broker connection settings.
        
        Args:
            broker_id: Broker ID
            server: Server name
            host: Host address
            port: Port number
            
        Returns:
            True if updated, False otherwise
        """
        data = {}
        if server is not None:
            data['server'] = server
        if host is not None:
            data['host'] = host
        if port is not None:
            data['port'] = port
        
        if not data:
            return False
        
        return self.update(broker_id, data)
    
    def update_metadata(self, broker_id: int, metadata: Dict[str, Any]) -> bool:
        """
        Update broker metadata.
        
        Args:
            broker_id: Broker ID
            metadata: New metadata
            
        Returns:
            True if updated, False otherwise
        """
        existing = self.get_by_id(broker_id)
        if not existing:
            return False
        
        new_metadata = {**existing.metadata, **metadata}
        return self.update(broker_id, {'metadata': new_metadata})
    
    # ==========================================================================
    # DELETE OPERATIONS
    # ==========================================================================
    
    def delete_by_name(self, name: str) -> bool:
        """
        Delete a broker by name.
        
        Args:
            name: Broker name
            
        Returns:
            True if deleted, False otherwise
        """
        broker = self.find_by_name(name)
        if not broker:
            return False
        return self.delete(broker.broker_id)
    
    def archive(self, broker_id: int) -> bool:
        """
        Archive a broker (soft delete).
        
        Args:
            broker_id: Broker ID
            
        Returns:
            True if archived, False otherwise
        """
        return self.update_status(broker_id, BrokerStatus.ARCHIVED)
    
    # ==========================================================================
    # UTILITY METHODS
    # ==========================================================================
    
    def broker_exists(self, name: str) -> bool:
        """
        Check if a broker exists.
        
        Args:
            name: Broker name
            
        Returns:
            True if exists, False otherwise
        """
        return self.count(f"name = ?", (name,)) > 0
    
    def get_broker_health(self, broker_id: int) -> Dict[str, Any]:
        """
        Get health status of a broker.
        
        Args:
            broker_id: Broker ID
            
        Returns:
            Dictionary with health information
        """
        broker = self.get_by_id(broker_id)
        if not broker:
            return {'status': 'not_found', 'broker_id': broker_id}
        
        return {
            'broker_id': broker.broker_id,
            'name': broker.name,
            'status': broker.status.value,
            'has_credentials': bool(broker.login and broker.password_encrypted),
            'has_server': bool(broker.server),
            'has_host': bool(broker.host),
            'health': 'ok' if broker.status == BrokerStatus.ACTIVE else 'inactive',
        }
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get repository statistics.
        
        Returns:
            Dictionary with statistics
        """
        base_stats = super().get_statistics()
        
        active_count = self.count("status = ?", (BrokerStatus.ACTIVE.value,))
        inactive_count = self.count("status = ?", (BrokerStatus.INACTIVE.value,))
        archived_count = self.count("status = ?", (BrokerStatus.ARCHIVED.value,))
        type_counts = self.get_type_counts()
        
        return {
            **base_stats,
            'active_brokers': active_count,
            'inactive_brokers': inactive_count,
            'archived_brokers': archived_count,
            'broker_types': type_counts,
            'default_broker': self.find_default(),
        }
    
    # ==========================================================================
    # PRIVATE METHODS
    # ==========================================================================
    
    def _entity_to_dict(self, broker: Broker) -> Dict[str, Any]:
        """Convert Broker entity to dictionary."""
        return {
            'broker_id': broker.broker_id,
            'broker_uuid': broker.broker_uuid,
            'name': broker.name,
            'broker_type': broker.broker_type.value if broker.broker_type else None,
            'server': broker.server,
            'host': broker.host,
            'port': broker.port,
            'description': broker.description,
            'login': broker.login,
            'password_encrypted': broker.password_encrypted,
            'status': broker.status.value if broker.status else None,
            'metadata': json.dumps(broker.metadata) if broker.metadata else '{}',
            'created_at': broker.created_at.isoformat() if broker.created_at else None,
            'updated_at': broker.updated_at.isoformat() if broker.updated_at else None,
        }
    
    def _row_to_entity(self, row: Dict[str, Any]) -> Broker:
        """Convert database row to Broker entity."""
        return Broker(
            broker_id=row['broker_id'],
            broker_uuid=row['broker_uuid'],
            name=row['name'],
            broker_type=BrokerType(row['broker_type']) if row['broker_type'] else None,
            server=row['server'],
            host=row['host'],
            port=row['port'],
            description=row['description'],
            login=row['login'],
            password_encrypted=row['password_encrypted'],
            status=BrokerStatus(row['status']) if row['status'] else None,
            metadata=json.loads(row['metadata']) if row['metadata'] else {},
            created_at=datetime.fromisoformat(row['created_at']) if row['created_at'] else None,
            updated_at=datetime.fromisoformat(row['updated_at']) if row['updated_at'] else None,
        )
    
    def _get_id(self, broker: Broker) -> int:
        """Get ID from Broker entity."""
        return broker.broker_id


# ==============================================================================
# REGISTER IN REPOSITORY MANAGER
# ==============================================================================

# To be added to REPOSITORIES list in repository_manager.py:
# ('brokers', BrokerRepository, None)