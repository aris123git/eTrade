"""
database/repositories/base_repository.py - Base Repository Pattern

RESPONSIBILITY:
Provide a generic repository base class for all entities.

ARCHITECTURAL PRINCIPLES:
1. Repository Pattern - Mediates between domain and data mapping layers
2. CRUD Operations - Create, Read, Update, Delete
3. Single Responsibility - Only data access, no business logic
4. Type Safety - Generic type hints
5. Database Agnostic - Works with any adapter

SCALABILITY VISION:
This is the foundation for all repositories in MarketAI.
It will scale from 20,000 to 1,000,000+ lines.

DIFFERENCE FROM ADAPTER:
- Adapter: How to talk to the database (connections, SQL)
- Repository: What data to store/retrieve (entities, business logic)

VERSION: 1.0.0
"""

import json
import logging
from abc import ABC, abstractmethod
from typing import (
    TypeVar, Generic, Optional, List, Dict, Any,
    Tuple, Union, Callable, Iterator, Set, Type
)
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

from database.core.connection import DatabaseManager
from database.adapters.base_adapter import DatabaseAdapter


# ==============================================================================
# TYPES
# ==============================================================================

T = TypeVar('T')  # Entity type
ID = Union[int, str]  # ID type


# ==============================================================================
# ENUMS
# ==============================================================================

class SortOrder(Enum):
    """Sort order for queries."""
    ASC = "ASC"
    DESC = "DESC"


class Operator(Enum):
    """Comparison operators for queries."""
    EQ = "="
    NE = "!="
    GT = ">"
    GTE = ">="
    LT = "<"
    LTE = "<="
    LIKE = "LIKE"
    IN = "IN"
    NOT_IN = "NOT IN"
    BETWEEN = "BETWEEN"
    IS_NULL = "IS NULL"
    IS_NOT_NULL = "IS NOT NULL"


# ==============================================================================
# QUERY BUILDER
# ==============================================================================

@dataclass
class QueryCondition:
    """A condition for query building."""
    field: str
    operator: Operator
    value: Any
    is_or: bool = False


@dataclass
class QueryOrder:
    """Order by clause for query building."""
    field: str
    order: SortOrder


@dataclass
class QueryLimit:
    """Limit clause for query building."""
    limit: int
    offset: int = 0


class QueryBuilder:
    """
    SQL query builder for repositories.
    
    Builds SQL queries programmatically with type safety.
    Supports complex queries with proper AND/OR precedence.
    
    USAGE:
        builder = QueryBuilder("markets")
        builder.where("symbol", Operator.EQ, "EURUSD")
        builder.or_where("symbol", Operator.EQ, "GBPUSD")
        sql, params = builder.build()
    """
    
    def __init__(self, table: str):
        """
        Initialize query builder.
        
        Args:
            table: Table name
        """
        self._table = table
        self._select_fields: List[str] = []
        self._conditions: List[QueryCondition] = []
        self._orders: List[QueryOrder] = []
        self._limit: Optional[QueryLimit] = None
        self._distinct: bool = False
        self._join: Optional[Tuple[str, str, str]] = None  # table, on_field, on_target
        self._group_by: List[str] = []
        self._having: List[QueryCondition] = []
    
    def select(self, *fields: str) -> 'QueryBuilder':
        """Set select fields."""
        self._select_fields = list(fields)
        return self
    
    def where(self, field: str, operator: Union[Operator, str], value: Any) -> 'QueryBuilder':
        """Add a WHERE condition (AND)."""
        if isinstance(operator, str):
            operator = Operator(operator)
        self._conditions.append(QueryCondition(field, operator, value, is_or=False))
        return self
    
    def or_where(self, field: str, operator: Union[Operator, str], value: Any) -> 'QueryBuilder':
        """Add an OR WHERE condition."""
        if isinstance(operator, str):
            operator = Operator(operator)
        self._conditions.append(QueryCondition(field, operator, value, is_or=True))
        return self
    
    def order_by(self, field: str, order: Union[SortOrder, str] = SortOrder.ASC) -> 'QueryBuilder':
        """Add ORDER BY clause."""
        if isinstance(order, str):
            order = SortOrder(order)
        self._orders.append(QueryOrder(field, order))
        return self
    
    def limit(self, limit: int, offset: int = 0) -> 'QueryBuilder':
        """Add LIMIT clause."""
        self._limit = QueryLimit(limit, offset)
        return self
    
    def distinct(self, enabled: bool = True) -> 'QueryBuilder':
        """Enable DISTINCT."""
        self._distinct = enabled
        return self
    
    def join(self, table: str, on_field: str, on_target: str) -> 'QueryBuilder':
        """Add JOIN clause."""
        self._join = (table, on_field, on_target)
        return self
    
    def group_by(self, *fields: str) -> 'QueryBuilder':
        """Add GROUP BY clause."""
        self._group_by = list(fields)
        return self
    
    def having(self, field: str, operator: Union[Operator, str], value: Any) -> 'QueryBuilder':
        """Add HAVING condition."""
        if isinstance(operator, str):
            operator = Operator(operator)
        self._having.append(QueryCondition(field, operator, value, is_or=False))
        return self
    
    def build(self) -> Tuple[str, Tuple]:
        """
        Build the SQL query.
        
        Returns:
            Tuple of (sql, params)
        
        Raises:
            ValueError: If query cannot be built
        """
        parts = []
        params = []
        
        # SELECT
        distinct_str = "DISTINCT " if self._distinct else ""
        if self._select_fields:
            fields = ", ".join(self._select_fields)
        else:
            fields = "*"
        
        sql = f"SELECT {distinct_str}{fields} FROM {self._table}"
        
        # JOIN
        if self._join:
            join_table, on_field, on_target = self._join
            sql += f" JOIN {join_table} ON {on_field} = {on_target}"
        
        # WHERE - with proper AND/OR handling
        if self._conditions:
            where_sql, where_params = self._build_where_clause(self._conditions)
            sql += f" WHERE {where_sql}"
            params.extend(where_params)
        
        # GROUP BY
        if self._group_by:
            sql += f" GROUP BY {', '.join(self._group_by)}"
            
            # HAVING
            if self._having:
                having_sql, having_params = self._build_where_clause(self._having)
                sql += f" HAVING {having_sql}"
                params.extend(having_params)
        
        # ORDER BY
        if self._orders:
            order_parts = [f"{o.field} {o.order.value}" for o in self._orders]
            sql += f" ORDER BY {', '.join(order_parts)}"
        
        # LIMIT
        if self._limit:
            sql += f" LIMIT {self._limit.limit} OFFSET {self._limit.offset}"
        
        return sql, tuple(params)
    
    def _build_where_clause(self, conditions: List[QueryCondition]) -> Tuple[str, List]:
        """
        Build WHERE clause with proper AND/OR precedence.
        
        Returns:
            Tuple of (sql, params)
        """
        if not conditions:
            return "", []
        
        # Group conditions by OR
        groups = []
        current_group = []
        
        for cond in conditions:
            if cond.is_or and current_group:
                groups.append(current_group)
                current_group = []
            current_group.append(cond)
        
        if current_group:
            groups.append(current_group)
        
        # Build each group
        group_sqls = []
        params = []
        
        for group in groups:
            group_parts = []
            for cond in group:
                if cond.operator == Operator.IN:
                    placeholders = ",".join("?" for _ in cond.value)
                    group_parts.append(f"{cond.field} IN ({placeholders})")
                    params.extend(cond.value)
                elif cond.operator == Operator.NOT_IN:
                    placeholders = ",".join("?" for _ in cond.value)
                    group_parts.append(f"{cond.field} NOT IN ({placeholders})")
                    params.extend(cond.value)
                elif cond.operator == Operator.BETWEEN:
                    group_parts.append(f"{cond.field} BETWEEN ? AND ?")
                    params.extend(cond.value)
                elif cond.operator == Operator.IS_NULL:
                    group_parts.append(f"{cond.field} IS NULL")
                elif cond.operator == Operator.IS_NOT_NULL:
                    group_parts.append(f"{cond.field} IS NOT NULL")
                else:
                    group_parts.append(f"{cond.field} {cond.operator.value} ?")
                    params.append(cond.value)
            
            group_sqls.append("(" + " AND ".join(group_parts) + ")")
        
        return " OR ".join(group_sqls), params


# ==============================================================================
# BASE REPOSITORY
# ==============================================================================

class BaseRepository(Generic[T]):
    """
    Base repository class with CRUD operations.
    
    Provides a generic repository pattern implementation.
    
    USAGE:
        class MarketRepository(BaseRepository[Market]):
            TABLE = "markets"
            MODEL = Market
            
            def __init__(self, db: DatabaseManager):
                super().__init__(db)
    
    DIFFERENCE FROM ADAPTER:
        - Adapter: How to talk to the database
        - Repository: What data to store/retrieve
    """
    
    # Table name (must be overridden by subclasses)
    TABLE: str = None
    
    # Model class (must be overridden by subclasses)
    MODEL: Type[T] = None
    
    def __init__(self, db_manager: DatabaseManager):
        """
        Initialize the repository.
        
        Args:
            db_manager: DatabaseManager instance (injected)
        """
        self.db = db_manager
        self.adapter: DatabaseAdapter = db_manager.get_adapter()
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        
        # Validate
        if not self.TABLE:
            raise ValueError("TABLE must be defined in subclass")
        
        if not self.MODEL:
            raise ValueError("MODEL must be defined in subclass")
    
    # ==========================================================================
    # QUERY BUILDER
    # ==========================================================================
    
    def query(self) -> QueryBuilder:
        """
        Create a new query builder.
        
        Returns:
            QueryBuilder instance
        """
        return QueryBuilder(self.TABLE)
    
    # ==========================================================================
    # CRUD OPERATIONS
    # ==========================================================================
    
    def insert(self, entity: T) -> ID:
        """
        Insert an entity.
        
        Args:
            entity: Entity instance
            
        Returns:
            ID of the inserted entity
        """
        data = self._entity_to_dict(entity)
        return self.insert_dict(data)
    
    def insert_dict(self, data: Dict[str, Any]) -> ID:
        """
        Insert a dictionary as entity.
        
        Args:
            data: Entity data
            
        Returns:
            ID of the inserted entity
        """
        if not data:
            raise ValueError("Cannot insert empty data")
        
        # Remove ID if present (auto-increment)
        data = {k: v for k, v in data.items() if k != 'id'}
        
        fields = list(data.keys())
        placeholders = ", ".join("?" for _ in fields)
        field_names = ", ".join(fields)
        
        sql = f"INSERT INTO {self.TABLE} ({field_names}) VALUES ({placeholders})"
        
        self.adapter.execute(sql, tuple(data.values()))
        return self.adapter.get_last_insert_id()
    
    def insert_many(self, entities: List[T]) -> int:
        """
        Insert multiple entities.
        
        Args:
            entities: List of entity instances
            
        Returns:
            Number of rows inserted
        """
        data_list = [self._entity_to_dict(e) for e in entities]
        return self.insert_many_dict(data_list)
    
    def insert_many_dict(self, data_list: List[Dict[str, Any]]) -> int:
        """
        Insert multiple dictionaries as entities.
        
        Args:
            data_list: List of entity data
            
        Returns:
            Number of rows inserted
        """
        if not data_list:
            return 0
        
        # Remove ID from all items
        data_list = [{k: v for k, v in d.items() if k != 'id'} for d in data_list]
        
        fields = list(data_list[0].keys())
        placeholders = ", ".join("?" for _ in fields)
        field_names = ", ".join(fields)
        
        sql = f"INSERT INTO {self.TABLE} ({field_names}) VALUES ({placeholders})"
        params = [tuple(d.get(f) for f in fields) for d in data_list]
        
        return self.adapter.execute_many(sql, params)
    
    def upsert(self, data: Dict[str, Any], conflict_fields: List[str]) -> ID:
        """
        Insert or update an entity (UPSERT).
        
        Args:
            data: Entity data
            conflict_fields: Fields to check for conflict
            
        Returns:
            ID of the entity
        """
        # Check if exists
        where_clause = " AND ".join(f"{f} = ?" for f in conflict_fields)
        params = [data[f] for f in conflict_fields]
        existing = self.find_one_where(where_clause, tuple(params))
        
        if existing:
            # Update
            update_data = {k: v for k, v in data.items() if k not in conflict_fields}
            if update_data:
                self.update_where(where_clause, tuple(params), update_data)
            return self._get_id(existing)
        else:
            # Insert
            return self.insert_dict(data)
    
    def get_by_id(self, entity_id: ID, id_field: str = "id") -> Optional[T]:
        """
        Get entity by ID.
        
        Args:
            entity_id: Entity ID
            id_field: ID field name
            
        Returns:
            Entity instance or None if not found
        """
        row = self.adapter.fetch_one(
            f"SELECT * FROM {self.TABLE} WHERE {id_field} = ?",
            (entity_id,)
        )
        return self._row_to_entity(row) if row else None
    
    def find_one(self, **kwargs) -> Optional[T]:
        """
        Find one entity by field values.
        
        Args:
            **kwargs: Field name -> value mappings
            
        Returns:
            Entity instance or None
        """
        if not kwargs:
            return None
        
        where_clause = " AND ".join(f"{k} = ?" for k in kwargs.keys())
        params = tuple(kwargs.values())
        return self.find_one_where(where_clause, params)
    
    def find_one_where(self, where_clause: str, params: tuple = ()) -> Optional[T]:
        """
        Find one entity with custom WHERE clause.
        
        Args:
            where_clause: WHERE clause (without WHERE keyword)
            params: Query parameters
            
        Returns:
            Entity instance or None
        """
        sql = f"SELECT * FROM {self.TABLE} WHERE {where_clause} LIMIT 1"
        row = self.adapter.fetch_one(sql, params)
        return self._row_to_entity(row) if row else None
    
    def find_all(self, **kwargs) -> List[T]:
        """
        Find all entities by field values.
        
        Args:
            **kwargs: Field name -> value mappings
            
        Returns:
            List of entity instances
        """
        if not kwargs:
            return self.find_all_where()
        
        where_clause = " AND ".join(f"{k} = ?" for k in kwargs.keys())
        params = tuple(kwargs.values())
        return self.find_all_where(where_clause, params)
    
    def find_all_where(
        self,
        where_clause: Optional[str] = None,
        params: tuple = (),
        order_by: Optional[str] = None,
        order: SortOrder = SortOrder.ASC,
        limit: Optional[int] = None,
        offset: int = 0,
    ) -> List[T]:
        """
        Find all entities with custom WHERE clause.
        
        Args:
            where_clause: WHERE clause (without WHERE keyword)
            params: Query parameters
            order_by: Field to order by
            order: Sort order
            limit: Maximum number of rows
            offset: Offset for pagination
            
        Returns:
            List of entity instances
        """
        sql = f"SELECT * FROM {self.TABLE}"
        
        if where_clause:
            sql += f" WHERE {where_clause}"
        
        if order_by:
            sql += f" ORDER BY {order_by} {order.value}"
        
        if limit is not None:
            sql += f" LIMIT {limit} OFFSET {offset}"
        
        rows = self.adapter.fetch_all(sql, params)
        return [self._row_to_entity(row) for row in rows]
    
    def count(self, where_clause: Optional[str] = None, params: tuple = ()) -> int:
        """
        Count entities.
        
        Args:
            where_clause: WHERE clause (without WHERE keyword)
            params: Query parameters
            
        Returns:
            Count of entities
        """
        sql = f"SELECT COUNT(*) as count FROM {self.TABLE}"
        if where_clause:
            sql += f" WHERE {where_clause}"
        return self.adapter.fetch_count(sql, params)
    
    def exists(self, **kwargs) -> bool:
        """
        Check if entity exists.
        
        Args:
            **kwargs: Field name -> value mappings
            
        Returns:
            True if exists, False otherwise
        """
        if not kwargs:
            return False
        
        where_clause = " AND ".join(f"{k} = ?" for k in kwargs.keys())
        params = tuple(kwargs.values())
        return self.count(where_clause, params) > 0
    
    def update(self, entity_id: ID, data: Dict[str, Any], id_field: str = "id") -> bool:
        """
        Update an entity by ID.
        
        Args:
            entity_id: Entity ID
            data: Data to update
            id_field: ID field name
            
        Returns:
            True if updated, False if not found
        """
        if not data:
            return False
        
        # Remove ID if present
        data = {k: v for k, v in data.items() if k != id_field}
        
        set_clause = ", ".join(f"{k} = ?" for k in data.keys())
        sql = f"UPDATE {self.TABLE} SET {set_clause} WHERE {id_field} = ?"
        params = tuple(data.values()) + (entity_id,)
        
        self.adapter.execute(sql, params)
        return True
    
    def update_where(self, where_clause: str, params: tuple, data: Dict[str, Any]) -> int:
        """
        Update entities by WHERE clause.
        
        Args:
            where_clause: WHERE clause (without WHERE keyword)
            params: Query parameters for WHERE
            data: Data to update
            
        Returns:
            Number of rows updated
        """
        if not data:
            return 0
        
        set_clause = ", ".join(f"{k} = ?" for k in data.keys())
        sql = f"UPDATE {self.TABLE} SET {set_clause} WHERE {where_clause}"
        query_params = tuple(data.values()) + params
        
        self.adapter.execute(sql, query_params)
        return True
    
    def delete(self, entity_id: ID, id_field: str = "id") -> bool:
        """
        Delete an entity by ID.
        
        Args:
            entity_id: Entity ID
            id_field: ID field name
            
        Returns:
            True if deleted, False if not found
        """
        sql = f"DELETE FROM {self.TABLE} WHERE {id_field} = ?"
        self.adapter.execute(sql, (entity_id,))
        return True
    
    def delete_where(self, where_clause: str, params: tuple) -> int:
        """
        Delete entities by WHERE clause.
        
        Args:
            where_clause: WHERE clause (without WHERE keyword)
            params: Query parameters
            
        Returns:
            Number of rows deleted
        """
        sql = f"DELETE FROM {self.TABLE} WHERE {where_clause}"
        self.adapter.execute(sql, params)
        return True
    
    def delete_all(self) -> int:
        """Delete all entities."""
        self.adapter.execute(f"DELETE FROM {self.TABLE}")
        return True
    
    # ==========================================================================
    # BULK OPERATIONS
    # ==========================================================================
    
    def bulk_insert(self, data_list: List[Dict[str, Any]], chunk_size: int = 1000) -> int:
        """
        Bulk insert with chunking.
        
        Args:
            data_list: List of entity data
            chunk_size: Size of each chunk
            
        Returns:
            Total rows inserted
        """
        if not data_list:
            return 0
        
        total = 0
        for i in range(0, len(data_list), chunk_size):
            chunk = data_list[i:i+chunk_size]
            total += self.insert_many_dict(chunk)
        
        return total
    
    def bulk_upsert(
        self,
        data_list: List[Dict[str, Any]],
        conflict_fields: List[str],
        chunk_size: int = 500,
    ) -> int:
        """
        Bulk upsert with chunking.
        
        Args:
            data_list: List of entity data
            conflict_fields: Fields to check for conflict
            chunk_size: Size of each chunk
            
        Returns:
            Total rows processed
        """
        if not data_list:
            return 0
        
        total = 0
        for i in range(0, len(data_list), chunk_size):
            chunk = data_list[i:i+chunk_size]
            for data in chunk:
                self.upsert(data, conflict_fields)
                total += 1
        
        return total
    
    # ==========================================================================
    # TRANSACTION SUPPORT
    # ==========================================================================
    
    def transaction(self):
        """
        Execute operations in a transaction.
        
        USAGE:
            with repo.transaction():
                repo.insert(data1)
                repo.insert(data2)
        """
        return self.adapter.transaction()
    
    # ==========================================================================
    # UTILITY METHODS
    # ==========================================================================
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get table statistics."""
        count = self.count()
        
        return {
            'table': self.TABLE,
            'row_count': count,
            'model': self.MODEL.__name__ if self.MODEL else None,
        }
    
    def truncate(self):
        """Truncate the table."""
        self.adapter.execute(f"DELETE FROM {self.TABLE}")
    
    def vacuum(self):
        """Vacuum the database."""
        self.adapter.vacuum()
    
    # ==========================================================================
    # ABSTRACT METHODS (To be overridden)
    # ==========================================================================
    
    def _entity_to_dict(self, entity: T) -> Dict[str, Any]:
        """
        Convert entity to dictionary.
        
        Must be overridden by subclasses.
        """
        raise NotImplementedError("Subclasses must implement _entity_to_dict")
    
    def _row_to_entity(self, row: Dict[str, Any]) -> T:
        """
        Convert database row to entity.
        
        Must be overridden by subclasses.
        """
        raise NotImplementedError("Subclasses must implement _row_to_entity")
    
    def _get_id(self, entity: T) -> ID:
        """
        Get ID from entity.
        
        Must be overridden by subclasses.
        """
        raise NotImplementedError("Subclasses must implement _get_id")


# ==============================================================================
# REPOSITORY REGISTRY
# ==============================================================================

class RepositoryRegistry:
    """
    Registry for repository instances.
    
    Provides centralized access to all repositories.
    
    USAGE:
        registry = RepositoryRegistry.get_instance()
        registry.register('markets', market_repo)
        repo = registry.get('markets')
    """
    
    _instance: Optional['RepositoryRegistry'] = None
    _lock = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._repositories = {}
        return cls._instance
    
    def __init__(self):
        self._repositories: Dict[str, BaseRepository] = {}
    
    def register(self, name: str, repository: BaseRepository):
        """Register a repository."""
        self._repositories[name] = repository
    
    def get(self, name: str) -> Optional[BaseRepository]:
        """Get a repository by name."""
        return self._repositories.get(name)
    
    def get_all(self) -> Dict[str, BaseRepository]:
        """Get all registered repositories."""
        return self._repositories.copy()
    
    def clear(self):
        """Clear all registered repositories."""
        self._repositories.clear()
    
    def has(self, name: str) -> bool:
        """Check if a repository is registered."""
        return name in self._repositories