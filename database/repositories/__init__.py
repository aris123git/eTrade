"""
database/repositories/__init__.py - Repository Package

RESPONSIBILITY:
Export repository classes for external use.
This file is intentionally small - only imports and exports.

VERSION: 1.0.0
"""

# ==============================================================================
# BASE
# ==============================================================================

from database.repositories.base_repository import (
    BaseRepository,
    QueryBuilder,
    QueryCondition,
    QueryOrder,
    QueryLimit,
    SortOrder,
    Operator,
)

# ==============================================================================
# MANAGER
# ==============================================================================

from database.repositories.repository_manager import RepositoryManager

# ==============================================================================
# FACTORY
# ==============================================================================

from database.repositories.factory import (
    create_repository,
    create_repository_manager,
)

# ==============================================================================
# EXPORTS
# ==============================================================================

__all__ = [
    # Base
    'BaseRepository',
    'QueryBuilder',
    'QueryCondition',
    'QueryOrder',
    'QueryLimit',
    'SortOrder',
    'Operator',
    
    # Manager
    'RepositoryManager',
    
    # Factory
    'create_repository',
    'create_repository_manager',
]