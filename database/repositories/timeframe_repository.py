"""
database/repositories/timeframe_repository.py - Timeframe Repository (Production Grade)

RESPONSIBILITY:
Manage timeframe data in the database with advanced trading infrastructure support.

ARCHITECTURAL PRINCIPLES:
1. Single Responsibility - CRUD + DB operations only
2. Service Layer - Business logic separated
3. Cache Invalidation - Stale data prevention
4. Index Optimization - Performance-critical queries
5. DB-Driven Hierarchy - Extensible, not hardcoded

SCALABILITY VISION:
This repository is the foundation for all timeframe operations in MarketAI.
It supports multi-market, multi-strategy, and multi-timeframe analysis.

VERSION: 2.0.0
"""

import json
import logging
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Any, Set, Tuple, Generator
from uuid import uuid4

from database.core.connection import DatabaseManager
from database.repositories.base_repository import BaseRepository
from database.models.timeframe import Timeframe, TimeframeCategory, TimeframeStatus


logger = logging.getLogger(__name__)


class TimeframeRepository(BaseRepository[Timeframe]):
    """
    Repository for timeframe data.
    
    Provides CRUD operations and timeframe-specific queries.
    Business logic moved to TimeframeService.
    
    USAGE:
        repo = TimeframeRepository(db_manager)
        
        # Create a timeframe
        h1 = repo.create(
            name="H1",
            seconds=3600,
            sort_order=4,
            description="1 Hour",
            category=TimeframeCategory.HOURLY,
        )
        
        # Find common timeframes
        common = repo.find_common()
    """
    
    TABLE = "timeframes"
    MODEL = Timeframe
    
    # Standard timeframe definitions with sort_order
    STANDARD_TIMEFRAMES = [
        {"name": "M1", "seconds": 60, "sort_order": 1, "description": "1 Minute", "category": TimeframeCategory.MINUTE},
        {"name": "M5", "seconds": 300, "sort_order": 2, "description": "5 Minutes", "category": TimeframeCategory.MINUTE},
        {"name": "M15", "seconds": 900, "sort_order": 3, "description": "15 Minutes", "category": TimeframeCategory.MINUTE},
        {"name": "M30", "seconds": 1800, "sort_order": 4, "description": "30 Minutes", "category": TimeframeCategory.MINUTE},
        {"name": "H1", "seconds": 3600, "sort_order": 5, "description": "1 Hour", "category": TimeframeCategory.HOURLY},
        {"name": "H2", "seconds": 7200, "sort_order": 6, "description": "2 Hours", "category": TimeframeCategory.HOURLY},
        {"name": "H3", "seconds": 10800, "sort_order": 7, "description": "3 Hours", "category": TimeframeCategory.HOURLY},
        {"name": "H4", "seconds": 14400, "sort_order": 8, "description": "4 Hours", "category": TimeframeCategory.HOURLY},
        {"name": "H6", "seconds": 21600, "sort_order": 9, "description": "6 Hours", "category": TimeframeCategory.HOURLY},
        {"name": "H8", "seconds": 28800, "sort_order": 10, "description": "8 Hours", "category": TimeframeCategory.HOURLY},
        {"name": "H12", "seconds": 43200, "sort_order": 11, "description": "12 Hours", "category": TimeframeCategory.HOURLY},
        {"name": "D1", "seconds": 86400, "sort_order": 12, "description": "1 Day", "category": TimeframeCategory.DAILY},
        {"name": "W1", "seconds": 604800, "sort_order": 13, "description": "1 Week", "category": TimeframeCategory.WEEKLY},
        {"name": "MN1", "seconds": 2592000, "sort_order": 14, "description": "1 Month", "category": TimeframeCategory.MONTHLY},
    ]
    
    def __init__(self, db_manager: DatabaseManager):
        """Initialize the timeframe repository."""
        super().__init__(db_manager)
        self.logger = logging.getLogger(__name__)
        
        # Cache for O(1) lookups
        self._name_cache: Dict[str, int] = {}       # name -> sort_order
        self._seconds_cache: Dict[int, str] = {}    # seconds -> name
        self._id_cache: Dict[int, str] = {}         # id -> name
        
        # Ensure indexes exist
        self._ensure_indexes()
    
    # ==========================================================================
    # DATABASE INDEXES
    # ==========================================================================
    
    def _ensure_indexes(self):
        """Ensure performance-critical indexes exist."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            indexes = [
                "CREATE INDEX IF NOT EXISTS idx_timeframes_name ON timeframes(name)",
                "CREATE INDEX IF NOT EXISTS idx_timeframes_seconds ON timeframes(seconds)",
                "CREATE INDEX IF NOT EXISTS idx_timeframes_category ON timeframes(category)",
                "CREATE INDEX IF NOT EXISTS idx_timeframes_sort_order ON timeframes(sort_order)",
                "CREATE INDEX IF NOT EXISTS idx_timeframes_status ON timeframes(status)",
            ]
            
            for idx_sql in indexes:
                try:
                    cursor.execute(idx_sql)
                except sqlite3.Error as e:
                    self.logger.warning(f"Failed to create index: {e}")
    
    # ==========================================================================
    # CREATE OPERATIONS
    # ==========================================================================
    
    def create(
        self,
        name: str,
        seconds: int,
        sort_order: Optional[int] = None,
        description: Optional[str] = None,
        category: TimeframeCategory = TimeframeCategory.CUSTOM,
        status: TimeframeStatus = TimeframeStatus.ACTIVE,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Timeframe:
        """
        Create a new timeframe.
        
        Args:
            name: Timeframe name (e.g., "H1")
            seconds: Duration in seconds
            sort_order: Hierarchy position (auto-calculated if not provided)
            description: Timeframe description
            category: Timeframe category
            status: Timeframe status
            metadata: Additional metadata
            
        Returns:
            Created Timeframe object
        """
        # Auto-calculate sort_order if not provided
        if sort_order is None:
            sort_order = self._calculate_next_sort_order()
        
        timeframe = Timeframe(
            timeframe_id=None,
            timeframe_uuid=str(uuid4()),
            name=name.upper(),
            seconds=seconds,
            sort_order=sort_order,
            description=description,
            category=category,
            status=status,
            metadata=metadata or {},
            created_at=datetime.now(),
            updated_at=datetime.now(),
        )
        
        # Insert into database
        data = self._entity_to_dict(timeframe)
        timeframe_id = self.insert_dict(data)
        timeframe.timeframe_id = timeframe_id
        
        # Update caches
        self._update_caches(timeframe)
        
        self.logger.info(f"✅ Timeframe created: {name} ({seconds}s, order={sort_order})")
        return timeframe
    
    def create_or_update(
        self,
        name: str,
        data: Dict[str, Any],
    ) -> Timeframe:
        """
        Create or update a timeframe by name.
        
        Args:
            name: Timeframe name
            data: Timeframe data
            
        Returns:
            Timeframe object
        """
        existing = self.find_by_name(name)
        
        if existing:
            # Update existing
            self.update(existing.timeframe_id, data)
            return self.find_by_name(name)
        else:
            # Create new
            return self.create(name=name, **data)
    
    def create_standard_timeframes(self) -> List[Timeframe]:
        """
        Create all standard timeframe definitions.
        
        Returns:
            List of created Timeframe objects
        """
        results = []
        for data in self.STANDARD_TIMEFRAMES:
            try:
                timeframe = self.create(**data)
                results.append(timeframe)
            except Exception as e:
                self.logger.warning(f"Failed to create timeframe {data.get('name')}: {e}")
        
        self.logger.info(f"✅ Created {len(results)} standard timeframes")
        return results
    
    # ==========================================================================
    # READ OPERATIONS
    # ==========================================================================
    
    def find_by_name(self, name: str) -> Optional[Timeframe]:
        """Find a timeframe by name (with cache)."""
        name = name.upper()
        
        # Check cache first
        if name in self._name_cache:
            timeframe_id = self._name_cache[name]
            return self.get_by_id(timeframe_id)
        
        # Query database
        row = self.adapter.fetch_one(
            f"SELECT * FROM {self.TABLE} WHERE name = ?",
            (name,)
        )
        
        if row:
            timeframe = self._row_to_entity(row)
            self._update_caches(timeframe)
            return timeframe
        
        return None
    
    def find_by_seconds(self, seconds: int) -> Optional[Timeframe]:
        """Find a timeframe by duration in seconds (with cache)."""
        # Check cache first
        if seconds in self._seconds_cache:
            name = self._seconds_cache[seconds]
            return self.find_by_name(name)
        
        # Query database
        row = self.adapter.fetch_one(
            f"SELECT * FROM {self.TABLE} WHERE seconds = ?",
            (seconds,)
        )
        
        if row:
            timeframe = self._row_to_entity(row)
            self._update_caches(timeframe)
            return timeframe
        
        return None
    
    def find_by_uuid(self, timeframe_uuid: str) -> Optional[Timeframe]:
        """Find a timeframe by UUID."""
        row = self.adapter.fetch_one(
            f"SELECT * FROM {self.TABLE} WHERE timeframe_uuid = ?",
            (timeframe_uuid,)
        )
        return self._row_to_entity(row) if row else None
    
    def find_by_category(self, category: TimeframeCategory) -> List[Timeframe]:
        """Find all timeframes of a specific category."""
        rows = self.adapter.fetch_all(
            f"SELECT * FROM {self.TABLE} WHERE category = ? ORDER BY sort_order ASC",
            (category.value,)
        )
        return [self._row_to_entity(row) for row in rows]
    
    def find_active(self) -> List[Timeframe]:
        """Find all active timeframes."""
        rows = self.adapter.fetch_all(
            f"SELECT * FROM {self.TABLE} WHERE status = ? ORDER BY sort_order ASC",
            (TimeframeStatus.ACTIVE.value,)
        )
        return [self._row_to_entity(row) for row in rows]
    
    def find_common(self) -> List[Timeframe]:
        """Find common trading timeframes (M5, M15, M30, H1, H4, D1, W1, MN1)."""
        common_names = ['M5', 'M15', 'M30', 'H1', 'H4', 'D1', 'W1', 'MN1']
        placeholders = ','.join('?' for _ in common_names)
        
        rows = self.adapter.fetch_all(
            f"SELECT * FROM {self.TABLE} WHERE name IN ({placeholders}) AND status = ? ORDER BY sort_order ASC",
            tuple(common_names) + (TimeframeStatus.ACTIVE.value,)
        )
        return [self._row_to_entity(row) for row in rows]
    
    def find_by_sort_range(self, min_order: int, max_order: int) -> List[Timeframe]:
        """Find timeframes within a sort order range."""
        rows = self.adapter.fetch_all(
            f"SELECT * FROM {self.TABLE} WHERE sort_order BETWEEN ? AND ? AND status = ? ORDER BY sort_order ASC",
            (min_order, max_order, TimeframeStatus.ACTIVE.value)
        )
        return [self._row_to_entity(row) for row in rows]
    
    def find_by_market_structure(self, structure: str) -> List[Timeframe]:
        """
        Find timeframes by market structure group.
        
        Args:
            structure: 'scalping', 'intraday', 'swing', 'long_term'
            
        Returns:
            List of Timeframe objects
        """
        structure_mapping = {
            'scalping': ['M1', 'M5', 'M15'],
            'intraday': ['M30', 'H1', 'H4'],
            'swing': ['D1', 'W1'],
            'long_term': ['MN1'],
        }
        
        names = structure_mapping.get(structure.lower(), [])
        if not names:
            return []
        
        placeholders = ','.join('?' for _ in names)
        rows = self.adapter.fetch_all(
            f"SELECT * FROM {self.TABLE} WHERE name IN ({placeholders}) AND status = ? ORDER BY sort_order ASC",
            tuple(names) + (TimeframeStatus.ACTIVE.value,)
        )
        return [self._row_to_entity(row) for row in rows]
    
    # ==========================================================================
    # UPDATE OPERATIONS
    # ==========================================================================
    
    def update(self, timeframe_id: int, data: Dict[str, Any]) -> bool:
        """Update a timeframe and invalidate cache."""
        # Get existing entity for cache invalidation
        existing = self.get_by_id(timeframe_id)
        
        # Update database
        result = super().update(timeframe_id, data)
        
        # Invalidate cache if updated
        if result and existing:
            self._invalidate_caches(existing)
            
            # Refresh cache with updated data
            updated = self.get_by_id(timeframe_id)
            if updated:
                self._update_caches(updated)
        
        return result
    
    def update_status(self, timeframe_id: int, status: TimeframeStatus) -> bool:
        """Update timeframe status and invalidate cache."""
        return self.update(timeframe_id, {'status': status.value})
    
    def activate(self, timeframe_id: int) -> bool:
        """Activate a timeframe."""
        return self.update_status(timeframe_id, TimeframeStatus.ACTIVE)
    
    def deactivate(self, timeframe_id: int) -> bool:
        """Deactivate a timeframe."""
        return self.update_status(timeframe_id, TimeframeStatus.INACTIVE)
    
    def reorder_hierarchy(self, order_map: Dict[str, int]) -> bool:
        """
        Reorder the timeframe hierarchy.
        
        Args:
            order_map: Dictionary mapping timeframe name to new sort_order
            
        Returns:
            True if all updates succeeded
        """
        with self.transaction():
            for name, order in order_map.items():
                timeframe = self.find_by_name(name)
                if timeframe:
                    self.update(timeframe.timeframe_id, {'sort_order': order})
        
        # Clear cache after reorder
        self.clear_cache()
        self.logger.info("🔄 Timeframe hierarchy reordered")
        return True
    
    # ==========================================================================
    # DELETE OPERATIONS
    # ==========================================================================
    
    def delete(self, timeframe_id: int) -> bool:
        """Delete a timeframe and invalidate cache."""
        # Get existing for cache invalidation
        existing = self.get_by_id(timeframe_id)
        
        # Delete from database
        result = super().delete(timeframe_id)
        
        # Invalidate cache
        if result and existing:
            self._invalidate_caches(existing)
        
        return result
    
    def delete_by_name(self, name: str) -> bool:
        """Delete a timeframe by name."""
        timeframe = self.find_by_name(name)
        if not timeframe:
            return False
        return self.delete(timeframe.timeframe_id)
    
    # ==========================================================================
    # CACHE MANAGEMENT
    # ==========================================================================
    
    def _update_caches(self, timeframe: Timeframe):
        """Update all caches with timeframe data."""
        self._name_cache[timeframe.name] = timeframe.timeframe_id
        self._seconds_cache[timeframe.seconds] = timeframe.name
        self._id_cache[timeframe.timeframe_id] = timeframe.name
    
    def _invalidate_caches(self, timeframe: Timeframe):
        """Invalidate all caches for a timeframe."""
        self._name_cache.pop(timeframe.name, None)
        self._seconds_cache.pop(timeframe.seconds, None)
        self._id_cache.pop(timeframe.timeframe_id, None)
    
    def clear_cache(self):
        """Clear all caches."""
        self._name_cache.clear()
        self._seconds_cache.clear()
        self._id_cache.clear()
        self.logger.debug("Timeframe cache cleared")
    
    # ==========================================================================
    # CONVERSION OPERATIONS
    # ==========================================================================
    
    def get_seconds(self, name: str) -> Optional[int]:
        """Get duration in seconds for a timeframe name (O(1) lookup)."""
        if name.upper() in self._name_cache:
            timeframe = self.get_by_id(self._name_cache[name.upper()])
            return timeframe.seconds if timeframe else None
        
        timeframe = self.find_by_name(name)
        return timeframe.seconds if timeframe else None
    
    def get_name(self, seconds: int) -> Optional[str]:
        """Get timeframe name for a duration in seconds (O(1) lookup)."""
        if seconds in self._seconds_cache:
            return self._seconds_cache[seconds]
        
        timeframe = self.find_by_seconds(seconds)
        return timeframe.name if timeframe else None
    
    def get_candles_per_timeframe(self, from_name: str, to_name: str) -> Optional[int]:
        """Get number of candles of timeframe A in timeframe B."""
        seconds_from = self.get_seconds(from_name)
        seconds_to = self.get_seconds(to_name)
        
        if seconds_from and seconds_to:
            return seconds_to // seconds_from
        
        return None
    
    def convert_timeframe(self, from_name: str, to_name: str) -> Optional[float]:
        """
        Convert timeframe duration.
        
        Example:
            convert_timeframe("H1", "M15") -> 4.0 (1 H1 = 4 M15 candles)
        """
        return self.get_candles_per_timeframe(from_name, to_name)
    
    # ==========================================================================
    # UTILITY METHODS
    # ==========================================================================
    
    def _calculate_next_sort_order(self) -> int:
        """Calculate the next available sort_order."""
        rows = self.adapter.fetch_all(
            f"SELECT MAX(sort_order) as max_order FROM {self.TABLE}"
        )
        max_order = rows[0]['max_order'] if rows and rows[0]['max_order'] else 0
        return max_order + 1
    
    def timeframe_exists(self, name: str) -> bool:
        """Check if a timeframe exists."""
        return name.upper() in self._name_cache or self.count(f"name = ?", (name.upper(),)) > 0
    
    def get_or_create(self, name: str, seconds: int) -> Timeframe:
        """
        Get a timeframe or create it if it doesn't exist.
        
        Args:
            name: Timeframe name
            seconds: Duration in seconds (required)
            
        Returns:
            Timeframe object
            
        Raises:
            ValueError: If seconds is not provided
        """
        if not seconds or seconds <= 0:
            raise ValueError("Timeframe must include valid seconds")
        
        existing = self.find_by_name(name)
        if existing:
            return existing
        
        return self.create(name=name, seconds=seconds)
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get repository statistics."""
        base_stats = super().get_statistics()
        
        active_count = self.count("status = ?", (TimeframeStatus.ACTIVE.value,))
        inactive_count = self.count("status = ?", (TimeframeStatus.INACTIVE.value,))
        category_counts = self._get_category_counts()
        
        return {
            **base_stats,
            'active_timeframes': active_count,
            'inactive_timeframes': inactive_count,
            'categories': category_counts,
            'cache_size': {
                'name_cache': len(self._name_cache),
                'seconds_cache': len(self._seconds_cache),
                'id_cache': len(self._id_cache),
            },
            'total_hierarchy_depth': self.count("status = ?", (TimeframeStatus.ACTIVE.value,)),
            'scalping_timeframes': len(self.find_by_market_structure('scalping')),
            'intraday_timeframes': len(self.find_by_market_structure('intraday')),
            'swing_timeframes': len(self.find_by_market_structure('swing')),
            'long_term_timeframes': len(self.find_by_market_structure('long_term')),
        }
    
    def get_timeframe_summary(self) -> Dict[str, Any]:
        """Get a summary of all timeframes."""
        active = self.find_active()
        
        return {
            'total_active': len(active),
            'by_category': self._get_category_counts(),
            'hierarchy': [{'name': t.name, 'order': t.sort_order, 'seconds': t.seconds} for t in active],
            'common_timeframes': [t.name for t in self.find_common()],
            'market_structure': {
                'scalping': [t.name for t in self.find_by_market_structure('scalping')],
                'intraday': [t.name for t in self.find_by_market_structure('intraday')],
                'swing': [t.name for t in self.find_by_market_structure('swing')],
                'long_term': [t.name for t in self.find_by_market_structure('long_term')],
            },
        }
    
    # ==========================================================================
    # PRIVATE METHODS
    # ==========================================================================
    
    def _get_category_counts(self) -> Dict[str, int]:
        """Get count of timeframes by category."""
        rows = self.adapter.fetch_all(f"""
            SELECT category, COUNT(*) as count 
            FROM {self.TABLE} 
            WHERE status = ?
            GROUP BY category
        """, (TimeframeStatus.ACTIVE.value,))
        
        return {row['category']: row['count'] for row in rows}
    
    def _entity_to_dict(self, timeframe: Timeframe) -> Dict[str, Any]:
        """Convert Timeframe entity to dictionary."""
        return {
            'timeframe_id': timeframe.timeframe_id,
            'timeframe_uuid': timeframe.timeframe_uuid,
            'name': timeframe.name,
            'seconds': timeframe.seconds,
            'sort_order': timeframe.sort_order,
            'description': timeframe.description,
            'category': timeframe.category.value if timeframe.category else None,
            'status': timeframe.status.value if timeframe.status else None,
            'metadata': json.dumps(timeframe.metadata) if timeframe.metadata else '{}',
            'created_at': timeframe.created_at.isoformat() if timeframe.created_at else None,
            'updated_at': timeframe.updated_at.isoformat() if timeframe.updated_at else None,
        }
    
    def _row_to_entity(self, row: Dict[str, Any]) -> Timeframe:
        """Convert database row to Timeframe entity."""
        return Timeframe(
            timeframe_id=row['timeframe_id'],
            timeframe_uuid=row['timeframe_uuid'],
            name=row['name'],
            seconds=row['seconds'],
            sort_order=row.get('sort_order', 0),
            description=row['description'],
            category=TimeframeCategory(row['category']) if row['category'] else None,
            status=TimeframeStatus(row['status']) if row['status'] else None,
            metadata=json.loads(row['metadata']) if row['metadata'] else {},
            created_at=datetime.fromisoformat(row['created_at']) if row['created_at'] else None,
            updated_at=datetime.fromisoformat(row['updated_at']) if row['updated_at'] else None,
        )
    
    def _get_id(self, timeframe: Timeframe) -> int:
        """Get ID from Timeframe entity."""
        return timeframe.timeframe_id


# ==============================================================================
# TIMEFRAME SERVICE (Business Logic Layer)
# ==============================================================================

class TimeframeService:
    """
    Timeframe business logic service.
    
    Separates business logic from repository CRUD operations.
    
    USAGE:
        service = TimeframeService(repo)
        
        # Get next higher timeframe
        higher = service.get_next_higher("H1")
        
        # Get hierarchy chain
        chain = service.get_hierarchy_chain("M5", "H4")
        
        # Get market structure
        structure = service.get_market_structure()
    """
    
    # Market structure groups
    MARKET_STRUCTURE = {
        'scalping': ['M1', 'M5', 'M15'],
        'intraday': ['M30', 'H1', 'H4'],
        'swing': ['D1', 'W1'],
        'long_term': ['MN1'],
    }
    
    def __init__(self, repo: TimeframeRepository):
        """
        Initialize the timeframe service.
        
        Args:
            repo: TimeframeRepository instance
        """
        self.repo = repo
        self.logger = logging.getLogger(__name__)
    
    def get_next_higher(self, name: str) -> Optional[Timeframe]:
        """
        Get the next higher timeframe using DB-driven hierarchy.
        
        Args:
            name: Current timeframe name
            
        Returns:
            Next higher Timeframe or None
        """
        timeframe = self.repo.find_by_name(name)
        if not timeframe:
            return None
        
        # Find next higher by sort_order
        rows = self.repo.adapter.fetch_all(
            f"SELECT * FROM {self.repo.TABLE} "
            f"WHERE sort_order > ? AND status = ? "
            f"ORDER BY sort_order ASC LIMIT 1",
            (timeframe.sort_order, TimeframeStatus.ACTIVE.value)
        )
        
        if rows:
            return self.repo._row_to_entity(rows[0])
        return None
    
    def get_next_lower(self, name: str) -> Optional[Timeframe]:
        """
        Get the next lower timeframe using DB-driven hierarchy.
        
        Args:
            name: Current timeframe name
            
        Returns:
            Next lower Timeframe or None
        """
        timeframe = self.repo.find_by_name(name)
        if not timeframe:
            return None
        
        # Find next lower by sort_order
        rows = self.repo.adapter.fetch_all(
            f"SELECT * FROM {self.repo.TABLE} "
            f"WHERE sort_order < ? AND status = ? "
            f"ORDER BY sort_order DESC LIMIT 1",
            (timeframe.sort_order, TimeframeStatus.ACTIVE.value)
        )
        
        if rows:
            return self.repo._row_to_entity(rows[0])
        return None
    
    def get_hierarchy_chain(self, from_name: str, to_name: str) -> List[Timeframe]:
        """
        Get the hierarchy chain between two timeframes.
        
        Args:
            from_name: Starting timeframe
            to_name: Ending timeframe
            
        Returns:
            List of Timeframe objects in the chain
        """
        from_tf = self.repo.find_by_name(from_name)
        to_tf = self.repo.find_by_name(to_name)
        
        if not from_tf or not to_tf:
            return []
        
        min_order = min(from_tf.sort_order, to_tf.sort_order)
        max_order = max(from_tf.sort_order, to_tf.sort_order)
        
        rows = self.repo.adapter.fetch_all(
            f"SELECT * FROM {self.repo.TABLE} "
            f"WHERE sort_order BETWEEN ? AND ? AND status = ? "
            f"ORDER BY sort_order ASC",
            (min_order, max_order, TimeframeStatus.ACTIVE.value)
        )
        
        return [self.repo._row_to_entity(row) for row in rows]
    
    def is_higher(self, name1: str, name2: str) -> bool:
        """
        Check if timeframe1 is higher than timeframe2.
        
        Args:
            name1: First timeframe
            name2: Second timeframe
            
        Returns:
            True if timeframe1 is higher
        """
        tf1 = self.repo.find_by_name(name1)
        tf2 = self.repo.find_by_name(name2)
        
        if not tf1 or not tf2:
            return False
        
        return tf1.sort_order > tf2.sort_order
    
    def is_lower(self, name1: str, name2: str) -> bool:
        """
        Check if timeframe1 is lower than timeframe2.
        
        Args:
            name1: First timeframe
            name2: Second timeframe
            
        Returns:
            True if timeframe1 is lower
        """
        tf1 = self.repo.find_by_name(name1)
        tf2 = self.repo.find_by_name(name2)
        
        if not tf1 or not tf2:
            return False
        
        return tf1.sort_order < tf2.sort_order
    
    def get_hierarchy_level(self, name: str) -> int:
        """
        Get the hierarchy level of a timeframe.
        
        Args:
            name: Timeframe name
            
        Returns:
            Hierarchy level (0 = lowest, higher = higher)
        """
        timeframe = self.repo.find_by_name(name)
        if not timeframe:
            return -1
        
        return timeframe.sort_order
    
    def get_market_structure(self) -> Dict[str, List[str]]:
        """
        Get market structure grouping.
        
        Returns:
            Dictionary with structure groups
        """
        structure = {}
        for group, names in self.MARKET_STRUCTURE.items():
            # Only return names that exist in DB
            structure[group] = [n for n in names if self.repo.timeframe_exists(n)]
        
        return structure
    
    def get_fast_timeframes(self) -> List[str]:
        """
        Get fast timeframes (scalping + intraday low).
        
        Returns:
            List of fast timeframe names
        """
        fast = []
        for group in ['scalping', 'intraday']:
            fast.extend(self.MARKET_STRUCTURE.get(group, []))
        
        # Filter to only existing timeframes
        return [n for n in fast if self.repo.timeframe_exists(n)]
    
    def get_slow_timeframes(self) -> List[str]:
        """
        Get slow timeframes (swing + long_term).
        
        Returns:
            List of slow timeframe names
        """
        slow = []
        for group in ['swing', 'long_term']:
            slow.extend(self.MARKET_STRUCTURE.get(group, []))
        
        # Filter to only existing timeframes
        return [n for n in slow if self.repo.timeframe_exists(n)]
    
    def get_hierarchy_depth(self) -> int:
        """Get the total hierarchy depth."""
        active = self.repo.find_active()
        return len(active)
    
    def get_timeframe_by_market_context(
        self,
        current_timeframe: str,
        market_condition: str,
    ) -> Optional[str]:
        """
        Get appropriate timeframe based on market condition.
        
        Args:
            current_timeframe: Current timeframe
            market_condition: 'trending', 'ranging', 'volatile', 'quiet'
            
        Returns:
            Recommended timeframe name or None
        """
        if market_condition == 'trending':
            # Use higher timeframe for trends
            return self.get_next_higher(current_timeframe)
        elif market_condition == 'ranging':
            # Use lower timeframe for ranges
            return self.get_next_lower(current_timeframe)
        elif market_condition == 'volatile':
            # Use lower timeframe for volatility
            return self.get_next_lower(current_timeframe)
        elif market_condition == 'quiet':
            # Use higher timeframe for quiet markets
            return self.get_next_higher(current_timeframe)
        
        return current_timeframe


# ==============================================================================
# REGISTER IN REPOSITORY MANAGER
# ==============================================================================

# To be added to REPOSITORIES list in repository_manager.py:
# ('timeframes', TimeframeRepository, None)