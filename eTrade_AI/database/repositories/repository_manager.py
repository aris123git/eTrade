"""
database/repositories/repository_manager.py - Repository Manager

RESPONSIBILITY:
Centralized access to all repositories via dynamic property resolution.

ARCHITECTURAL PRINCIPLES:
1. Single Responsibility - Only manages repository access
2. Dynamic Properties - No hardcoded property methods
3. Lazy Initialization - Repositories created on first access
4. Dependency Injection - Receives DatabaseManager

SCALABILITY VISION:
This manager will grow to handle 100+ repositories without modification.
Adding a new repository is a one-line change in REPOSITORIES list.

VERSION: 1.0.0
"""

import logging
from typing import Dict, Any, Optional, List, Tuple, Type, Union

from database.core.connection import DatabaseManager
from database.repositories.base_repository import BaseRepository


logger = logging.getLogger(__name__)


class RepositoryManager:
    """
    Centralized repository manager.
    
    Provides access to all repositories through dynamic properties.
    
    USAGE:
        repo_manager = RepositoryManager(db_manager)
        
        # Dynamic property access
        markets = repo_manager.markets
        candles = repo_manager.candles
        
        # Or via get() method
        patterns = repo_manager.get('patterns')
    
    SCALABILITY:
        This manager supports any number of repositories without
        adding new properties. Adding a repository is a one-line
        change in REPOSITORIES list.
    """
    
    # Registry of all repositories
    # Each entry: (name, repository_class, dependency_name)
    # The dependency is resolved automatically at initialization
    REPOSITORIES: List[Tuple[str, Type[BaseRepository], Optional[str]]] = [
        # ('markets', MarketRepository, None),
        # ('candles', CandleRepository, None),
        # ('brokers', BrokerRepository, None),
        # ('currencies', CurrencyRepository, None),
        # ('timeframes', TimeframeRepository, None),
        # ('patterns', PatternRepository, None),
        # ('correlations', CorrelationRepository, None),
        # ('hypotheses', HypothesisRepository, None),
        # ('knowledge', KnowledgeRepository, None),
        # ('validation', ValidationRepository, None),
        # ('trades', TradeRepository, None),
        # ('positions', PositionRepository, None),
        # ('orders', OrderRepository, None),
        # ('accounts', AccountRepository, None),
        # ('statistics', StatisticsRepository, None),
        # ('cache', CacheRepository, None),
    ]
    
    def __init__(self, db_manager: DatabaseManager):
        """
        Initialize the repository manager.
        
        Args:
            db_manager: DatabaseManager instance
        """
        self._db_manager = db_manager
        self._repositories: Dict[str, BaseRepository] = {}
        self._initialized = False
        
        # Initialize repositories
        self._init_repositories()
        
        logger.info(f"✅ RepositoryManager initialized with {len(self._repositories)} repositories")
    
    # ==========================================================================
    # INITIALIZATION
    # ==========================================================================
    
    def _init_repositories(self):
        """
        Initialize all repositories from REPOSITORIES list.
        
        To add a new repository:
            1. Import it at the top of this file
            2. Add it to REPOSITORIES list
        """
        # Import repositories dynamically (lazy imports)
        self._lazy_imports()
        
        for name, repo_class, dependency_name in self.REPOSITORIES:
            try:
                # Check if this repository depends on another
                if dependency_name:
                    dependency = self._repositories.get(dependency_name)
                    if dependency is None:
                        logger.warning(
                            f"Dependency '{dependency_name}' not found for '{name}', "
                            f"creating without dependency"
                        )
                        repo = repo_class(self._db_manager)
                    else:
                        repo = repo_class(self._db_manager, dependency)
                else:
                    repo = repo_class(self._db_manager)
                
                self._repositories[name] = repo
                logger.debug(f"Registered repository: {name}")
                
            except Exception as e:
                logger.error(f"Failed to register repository '{name}': {e}")
        
        self._initialized = True
    
    def _lazy_imports(self):
        """
        Lazy import repositories to avoid circular imports.
        
        This is called before initialization to ensure all repository
        classes are available.
        """
        # These imports will be uncommented as repositories are created
        # from database.repositories.market_repository import MarketRepository
        # from database.repositories.candle_repository import CandleRepository
        # from database.repositories.broker_repository import BrokerRepository
        # from database.repositories.currency_repository import CurrencyRepository
        # from database.repositories.timeframe_repository import TimeframeRepository
        # from database.repositories.pattern_repository import PatternRepository
        # from database.repositories.correlation_repository import CorrelationRepository
        # from database.repositories.hypothesis_repository import HypothesisRepository
        # from database.repositories.knowledge_repository import KnowledgeRepository
        # from database.repositories.validation_repository import ValidationRepository
        # from database.repositories.trade_repository import TradeRepository
        # from database.repositories.position_repository import PositionRepository
        # from database.repositories.order_repository import OrderRepository
        # from database.repositories.account_repository import AccountRepository
        # from database.repositories.statistics_repository import StatisticsRepository
        # from database.repositories.cache_repository import CacheRepository
        
        # If imports are commented, REPOSITORIES must be empty
        if self.REPOSITORIES:
            logger.warning(
                "REPOSITORIES list is not empty but lazy imports are commented. "
                "Uncomment the imports or remove REPOSITORIES entries."
            )
    
    # ==========================================================================
    # PUBLIC METHODS
    # ==========================================================================
    
    def get(self, name: str) -> Optional[BaseRepository]:
        """
        Get a repository by name.
        
        Args:
            name: Repository name ('markets', 'candles', etc.)
            
        Returns:
            Repository instance or None if not found
        """
        return self._repositories.get(name)
    
    def has(self, name: str) -> bool:
        """
        Check if a repository is registered.
        
        Args:
            name: Repository name
            
        Returns:
            True if registered, False otherwise
        """
        return name in self._repositories
    
    def get_all(self) -> Dict[str, BaseRepository]:
        """
        Get all registered repositories.
        
        Returns:
            Dictionary mapping name to repository instance
        """
        return self._repositories.copy()
    
    def get_names(self) -> List[str]:
        """
        Get all registered repository names.
        
        Returns:
            List of repository names
        """
        return list(self._repositories.keys())
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get statistics for all repositories.
        
        Returns:
            Dictionary with repository statistics
        """
        stats = {}
        for name, repo in self._repositories.items():
            try:
                stats[name] = repo.get_statistics()
            except Exception as e:
                stats[name] = {'error': str(e)}
        return stats
    
    def register(self, name: str, repository: BaseRepository):
        """
        Register a repository dynamically.
        
        Args:
            name: Repository name
            repository: Repository instance
        """
        self._repositories[name] = repository
        logger.debug(f"Dynamically registered repository: {name}")
    
    def unregister(self, name: str) -> bool:
        """
        Unregister a repository.
        
        Args:
            name: Repository name
            
        Returns:
            True if removed, False if not found
        """
        if name in self._repositories:
            del self._repositories[name]
            logger.debug(f"Unregistered repository: {name}")
            return True
        return False
    
    def reload(self):
        """Reload all repositories."""
        self._repositories.clear()
        self._init_repositories()
        logger.info("Repositories reloaded")
    
    def close(self):
        """Close all repository connections."""
        logger.info("Closing all repositories...")
        # Repositories use shared DatabaseManager connection
        # No individual close needed
    
    # ==========================================================================
    # DYNAMIC PROPERTY ACCESS
    # ==========================================================================
    
    def __getattr__(self, name: str):
        """
        Dynamic property access for repositories.
        
        This allows:
            repo_manager.markets
            repo_manager.candles
            
        Without defining 100+ property methods.
        
        Raises:
            AttributeError: If repository not found
        """
        repo = self._repositories.get(name)
        if repo is None:
            raise AttributeError(
                f"'{self.__class__.__name__}' has no attribute '{name}'. "
                f"Available repositories: {list(self._repositories.keys())}"
            )
        return repo
    
    def __dir__(self):
        """Custom dir() for better autocomplete."""
        items = list(self.__dict__.keys())
        items.extend(self._repositories.keys())
        return sorted(items)
    
    def __contains__(self, name: str) -> bool:
        """Check if a repository exists."""
        return name in self._repositories
    
    def __len__(self) -> int:
        """Get the number of registered repositories."""
        return len(self._repositories)
    
    def __iter__(self):
        """Iterate over repository names."""
        return iter(self._repositories.keys())
    
    def __repr__(self) -> str:
        """String representation."""
        return f"<RepositoryManager repositories={list(self._repositories.keys())}>"