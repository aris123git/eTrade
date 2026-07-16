"""
core/connection.py - Connection Manager for MarketAI

RESPONSIBILITY:
Manage database and MT5 connections with health monitoring and auto-reconnect.

ARCHITECTURAL DECISIONS:
1. Singleton pattern - Single connection manager for the entire application
2. Lazy initialization - Connect on first use
3. Heartbeat monitoring - Periodic health checks every 30 seconds
4. Auto-reconnect - Exponential backoff on failure
5. Graceful shutdown - Clean close of all connections
6. Thread-safe - All operations protected by threading.Lock

USAGE:
    from core.connection import ConnectionManager
    from config import config
    from core.logger import get_logger
    
    conn_mgr = ConnectionManager(config, get_logger())
    
    # Get connections
    db = conn_mgr.get_database()
    mt5 = conn_mgr.get_mt5()
    
    # Check health
    if conn_mgr.is_healthy():
        # All connections are alive
        pass

VERSION: 1.0.1
"""

import logging
import threading
import time
import sqlite3
from datetime import datetime
from typing import Optional, Dict, Any

from core.config import Config
from core.exceptions import (
    DatabaseConnectionError,
    MT5ConnectionError,
    ConnectionError,
)
from core.utils import exponential_backoff
from mt5.manager import MT5Manager

logger = logging.getLogger(__name__)


# ==============================================================================
# CONNECTION MANAGER
# ==============================================================================

class ConnectionManager:
    """
    Thread-safe singleton connection manager.
    
    Manages database and MT5 connections with health monitoring.
    """
    
    _instance: Optional['ConnectionManager'] = None
    _lock = threading.Lock()
    
    # Default configuration
    _DEFAULT_HEALTH_CHECK_INTERVAL = 30  # seconds
    _DEFAULT_RECONNECT_ATTEMPTS = 3
    _DEFAULT_RECONNECT_DELAY = 1.0
    _DEFAULT_RECONNECT_BACKOFF = 2.0
    _DEFAULT_DB_TIMEOUT = 30
    _DEFAULT_DB_MAX_RETRIES = 3
    
    def __new__(cls, config: Config = None, logger: logging.Logger = None):
        """Singleton pattern with thread safety."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance
    
    def __init__(self, config: Config = None, logger: logging.Logger = None):
        """
        Initialize the connection manager.
        
        Args:
            config: Configuration instance
            logger: Logger instance
        """
        if self._initialized:
            return
        
        self._config = config or Config()
        self._logger = logger or logging.getLogger(__name__)
        
        # Database connection
        self._db_conn: Optional[sqlite3.Connection] = None
        self._db_lock = threading.Lock()
        self._db_connected = False
        
        # MT5 Manager
        self._mt5_manager: Optional[MT5Manager] = None
        self._mt5_lock = threading.Lock()
        
        # Health monitoring
        self._health_check_interval = self._DEFAULT_HEALTH_CHECK_INTERVAL
        self._health_check_thread: Optional[threading.Thread] = None
        self._health_check_running = False
        self._health_check_stop_event = threading.Event()
        
        # Metrics
        self._start_time = datetime.now()
        self._db_reconnect_count = 0
        self._mt5_reconnect_count = 0
        self._health_checks_performed = 0
        self._health_check_failures = 0
        self._last_health_check: Optional[datetime] = None
        
        # Shutdown flag
        self._shutdown = False
        
        self._initialized = True
        
        # Start health monitor
        self._start_health_monitor()
        
        self._logger.info("✅ ConnectionManager initialized")
    
    # ==========================================================================
    # PUBLIC METHODS
    # ==========================================================================
    
    def get_database(self) -> sqlite3.Connection:
        """
        Get a database connection.
        
        Returns:
            SQLite database connection
        
        Raises:
            DatabaseConnectionError: If connection fails
        """
        if self._shutdown:
            raise ConnectionError("ConnectionManager is shutdown")
        
        with self._db_lock:
            if not self._db_connected or self._db_conn is None:
                self._connect_database()
            return self._db_conn
    
    def get_mt5(self) -> MT5Manager:
        """
        Get the MT5 Manager instance.
        
        Returns:
            MT5Manager instance
        
        Raises:
            MT5ConnectionError: If MT5 is not available
        """
        if self._shutdown:
            raise ConnectionError("ConnectionManager is shutdown")
        
        with self._mt5_lock:
            if self._mt5_manager is None:
                self._mt5_manager = MT5Manager(self._config, self._logger)
            
            # Ensure MT5 is connected
            if not self._mt5_manager.is_connected():
                self._mt5_reconnect_count += 1
                success = self._mt5_manager.initialize()
                if not success:
                    raise MT5ConnectionError("Failed to initialize MT5 connection")
            
            return self._mt5_manager
    
    def is_healthy(self) -> bool:
        """
        Check if all connections are healthy.
        
        Returns:
            True if all connections are alive, False otherwise
        """
        db_ok = self._check_database_health()
        mt5_ok = self._check_mt5_health()
        
        return db_ok and mt5_ok
    
    def reconnect_database(self) -> bool:
        """
        Reconnect to the database.
        
        Returns:
            True if reconnection successful, False otherwise
        """
        with self._db_lock:
            return self._reconnect_database()
    
    def reconnect_mt5(self) -> bool:
        """
        Reconnect to MT5.
        
        Returns:
            True if reconnection successful, False otherwise
        """
        with self._mt5_lock:
            return self._reconnect_mt5()
    
    def close_all(self) -> None:
        """
        Close all connections gracefully.
        """
        with self._lock:
            if self._shutdown:
                return
            
            self._shutdown = True
            
            # Stop health monitor
            self._stop_health_monitor()
            
            # Close database
            with self._db_lock:
                self._close_database()
            
            # Close MT5
            with self._mt5_lock:
                self._close_mt5()
            
            self._logger.info("⏹️ All connections closed")
    
    def get_status(self) -> Dict[str, Any]:
        """
        Get connection status and metrics.
        
        Returns:
            Dictionary with status information
        """
        uptime = (datetime.now() - self._start_time).total_seconds()
        
        db_status = {
            'connected': self._db_connected,
            'reconnect_count': self._db_reconnect_count,
        }
        
        mt5_status = {
            'connected': self._mt5_manager.is_connected() if self._mt5_manager else False,
            'reconnect_count': self._mt5_reconnect_count,
        }
        
        return {
            'uptime_seconds': uptime,
            'is_shutdown': self._shutdown,
            'database': db_status,
            'mt5': mt5_status,
            'health_monitor': {
                'running': self._health_check_running,
                'checks_performed': self._health_checks_performed,
                'failures': self._health_check_failures,
                'last_check': self._last_health_check.isoformat() if self._last_health_check else None,
                'interval': self._health_check_interval,
            },
        }
    
    # ==========================================================================
    # PRIVATE METHODS
    # ==========================================================================
    
    def _connect_database(self) -> None:
        """
        Internal method to connect to database.
        
        Raises:
            DatabaseConnectionError: If connection fails
        """
        try:
            db_path = self._config.DB_PATH
            self._logger.debug(f"🔌 Connecting to database: {db_path}")
            
            self._db_conn = sqlite3.connect(
                db_path,
                timeout=self._DEFAULT_DB_TIMEOUT,
                check_same_thread=False,
            )
            self._db_conn.row_factory = sqlite3.Row
            self._db_connected = True
            
            self._logger.info(f"✅ Database connected: {db_path}")
            
        except Exception as e:
            self._db_connected = False
            self._db_conn = None
            self._logger.error(f"❌ Database connection failed: {e}")
            raise DatabaseConnectionError(str(db_path), original_error=e)
    
    def _close_database(self) -> None:
        """Internal method to close database connection."""
        if self._db_conn is not None:
            try:
                self._db_conn.close()
                self._logger.debug("Database connection closed")
            except Exception as e:
                self._logger.warning(f"⚠️ Error closing database: {e}")
            finally:
                self._db_conn = None
                self._db_connected = False
    
    def _reconnect_database(self) -> bool:
        """
        Internal method to reconnect database with backoff.
        
        Returns:
            True if reconnection successful, False otherwise
        """
        self._db_reconnect_count += 1
        attempt = 0
        max_attempts = self._DEFAULT_DB_MAX_RETRIES
        
        while attempt < max_attempts:
            try:
                self._close_database()
                self._connect_database()
                self._logger.info(
                    f"✅ Database reconnected successfully (attempt {attempt + 1})"
                )
                return True
            except Exception as e:
                attempt += 1
                self._logger.warning(
                    f"⚠️ Database reconnect attempt {attempt}/{max_attempts} failed: {e}"
                )
                
                if attempt < max_attempts:
                    delay = exponential_backoff(attempt - 1, base_delay=1.0)
                    time.sleep(delay)
        
        self._logger.error("❌ Database reconnection failed after all attempts")
        return False
    
    def _close_mt5(self) -> None:
        """Internal method to close MT5 connection."""
        if self._mt5_manager is not None:
            try:
                self._mt5_manager.close()
                self._logger.debug("MT5 connection closed")
            except Exception as e:
                self._logger.warning(f"⚠️ Error closing MT5: {e}")
    
    def _reconnect_mt5(self) -> bool:
        """
        Internal method to reconnect MT5 with backoff.
        
        Returns:
            True if reconnection successful, False otherwise
        """
        self._mt5_reconnect_count += 1
        
        try:
            if self._mt5_manager is None:
                self._mt5_manager = MT5Manager(self._config, self._logger)
            
            success = self._mt5_manager.initialize()
            if success:
                self._logger.info("✅ MT5 reconnected successfully")
                return True
            
            self._logger.error("❌ MT5 reconnection failed")
            return False
            
        except Exception as e:
            self._logger.error(f"❌ MT5 reconnection error: {e}")
            return False
    
    def _check_database_health(self) -> bool:
        """
        Check if database connection is healthy.
        
        Returns:
            True if healthy, False otherwise
        """
        if not self._db_connected or self._db_conn is None:
            return False
        
        try:
            # Simple query to verify connection
            cursor = self._db_conn.execute("SELECT 1")
            cursor.fetchone()
            return True
        except Exception:
            return False
    
    def _check_mt5_health(self) -> bool:
        """
        Check if MT5 connection is healthy.
        
        Returns:
            True if healthy, False otherwise
        """
        if self._mt5_manager is None:
            return False
        
        try:
            return self._mt5_manager.is_connected()
        except Exception:
            return False
    
    def _health_check(self) -> None:
        """
        Perform health check on all connections.
        
        This is called periodically by the health monitor.
        """
        self._health_checks_performed += 1
        self._last_health_check = datetime.now()
        
        # Check database
        if not self._check_database_health():
            self._health_check_failures += 1
            self._logger.warning("⚠️ Database health check failed, reconnecting...")
            self.reconnect_database()
        
        # Check MT5
        if not self._check_mt5_health():
            self._health_check_failures += 1
            self._logger.warning("⚠️ MT5 health check failed, reconnecting...")
            self.reconnect_mt5()
    
    def _start_health_monitor(self) -> None:
        """
        Start the health monitor thread.
        """
        if self._health_check_running:
            return
        
        self._health_check_running = True
        self._health_check_stop_event.clear()
        self._health_check_thread = threading.Thread(
            target=self._health_monitor_loop,
            name="connection-health-monitor",
            daemon=True,
        )
        self._health_check_thread.start()
        
        self._logger.debug("✅ Health monitor started")
    
    def _stop_health_monitor(self) -> None:
        """
        Stop the health monitor thread.
        """
        if not self._health_check_running:
            return
        
        self._health_check_running = False
        self._health_check_stop_event.set()
        
        if self._health_check_thread and self._health_check_thread.is_alive():
            self._health_check_thread.join(timeout=5)
        
        self._logger.debug("⏹️ Health monitor stopped")
    
    def _health_monitor_loop(self) -> None:
        """
        Main health monitor loop.
        
        Runs in background thread, checking connections periodically.
        """
        self._logger.debug("🔄 Health monitor loop started")
        
        while not self._shutdown and self._health_check_running:
            # Check if we should stop
            if self._health_check_stop_event.wait(self._health_check_interval):
                break
            
            try:
                self._health_check()
            except Exception as e:
                self._logger.error(f"❌ Health check error: {e}")
        
        self._logger.debug("🔄 Health monitor loop ended")
    
    # ==========================================================================
    # DUNDER METHODS
    # ==========================================================================
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close_all()
    
    def __del__(self):
        """Cleanup on destruction."""
        try:
            self.close_all()
        except Exception:
            pass


# ==============================================================================
# FACTORY FUNCTION
# ==============================================================================

def create_connection_manager(
    config: Config,
    logger: logging.Logger = None,
) -> ConnectionManager:
    """
    Factory function for ConnectionManager creation.
    
    Args:
        config: Configuration instance
        logger: Optional logger instance
    
    Returns:
        ConnectionManager instance
    """
    return ConnectionManager(config, logger)