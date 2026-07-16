"""
mt5/manager.py - Thread-Safe MT5 Connection Manager

RESPONSIBILITY:
Manage the lifecycle of MetaTrader 5 connections for MarketAI.

ARCHITECTURAL DECISIONS:
1. Singleton pattern - Single MT5 connection for the entire application
2. Thread-safe - All operations protected by threading.Lock
3. Auto-reconnect - Exponential backoff on disconnection
4. Account switching - Support multiple accounts
5. Lazy initialization - Connect on first use
6. Error tracking - Maintain last_error for debugging
7. No destructor shutdown - Application controls lifecycle

USAGE:
    from mt5.manager import MT5Manager
    from config import config
    from core.logger import get_logger
    
    mt5_manager = MT5Manager(config, get_logger())
    
    # Connect with credentials
    mt5_manager.initialize(login=12345, password="pass", server="Demo")
    
    # Download data
    candles, meta = mt5_manager.download_candles(
        symbol="EURUSD",
        timeframe="H1",
        start_date=datetime(2024, 1, 1),
        end_date=datetime.now(),
    )

VERSION: 1.0.0
"""

import logging
import threading
import time
from datetime import datetime, timedelta
from typing import Optional, Dict, List, Any, Tuple, Union

# Import config and exceptions
from core.config import Config
from core.exceptions import (
    MT5Error,
    MT5ConnectionError,
    MT5DisconnectedError,
    MT5DownloadError,
    MT5TimeoutError,
    MT5SymbolNotFoundError,
)
from core.utils import retry, exponential_backoff

logger = logging.getLogger(__name__)


# ==============================================================================
# MT5 MANAGER
# ==============================================================================

class MT5Manager:
    """
    Thread-safe singleton manager for MetaTrader 5 connections.
    
    Manages MT5 connection lifecycle, auto-reconnect, and account switching.
    """
    
    _instance: Optional['MT5Manager'] = None
    _lock = threading.Lock()
    
    # Default configuration
    _DEFAULT_RECONNECT_ATTEMPTS = 3
    _DEFAULT_RECONNECT_DELAY = 1.0
    _DEFAULT_RECONNECT_BACKOFF = 2.0
    _DEFAULT_MAX_RETRY_ATTEMPTS = 3
    
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
        Initialize the MT5 manager.
        
        Args:
            config: Configuration instance
            logger: Logger instance for this manager
        """
        if self._initialized:
            return
        
        self._config = config or Config()
        self._logger = logger or logging.getLogger(__name__)
        
        # MT5 module (loaded lazily)
        self._mt5 = None
        
        # Connection state
        self._connected = False
        self._connection_lock = threading.Lock()
        self._last_error: Optional[str] = None
        
        # Connection parameters
        self._login: Optional[int] = None
        self._password: Optional[str] = None
        self._server: Optional[str] = None
        
        # Metrics
        self._connection_attempts = 0
        self._reconnects = 0
        self._last_error_time: Optional[datetime] = None
        
        # Flag to prevent re-initialization
        self._initialized = True
        
        self._logger.info("✅ MT5Manager initialized")
    
    # ==========================================================================
    # PUBLIC METHODS
    # ==========================================================================
    
    def is_connected(self) -> bool:
        """
        Check if MT5 is currently connected.
        
        Returns:
            True if connected, False otherwise
        """
        with self._connection_lock:
            return self._connected
    
    def initialize(
        self,
        login: Optional[int] = None,
        password: Optional[str] = None,
        server: Optional[str] = None,
    ) -> bool:
        """
        Initialize MT5 connection with credentials.
        
        Args:
            login: Account login (optional)
            password: Account password (optional)
            server: Server name (optional)
        
        Returns:
            True if connection successful, False otherwise
        
        Raises:
            MT5ConnectionError: If connection fails
        """
        with self._connection_lock:
            # Store credentials for reconnection
            if login:
                self._login = login
            if password:
                self._password = password
            if server:
                self._server = server
            
            return self._connect()
    
    def login(
        self,
        login: int,
        password: str,
        server: Optional[str] = None,
    ) -> bool:
        """
        Switch to a different MT5 account.
        
        Args:
            login: Account login
            password: Account password
            server: Server name (optional)
        
        Returns:
            True if login successful, False otherwise
        
        Raises:
            MT5ConnectionError: If login fails
        """
        with self._connection_lock:
            # Close existing connection
            self._disconnect()
            
            # Update credentials
            self._login = login
            self._password = password
            if server:
                self._server = server
            
            # Connect with new credentials
            return self._connect()
    
    def get(self) -> Any:
        """
        Get the MT5 module (auto-reconnect if needed).
        
        Returns:
            MT5 module instance
        
        Raises:
            MT5DisconnectedError: If not connected and reconnection fails
        """
        if not self._connected:
            with self._connection_lock:
                if not self._connected:
                    self._logger.warning("⚠️ MT5 disconnected, attempting reconnect...")
                    if self._connect():
                        return self._mt5
                    raise MT5DisconnectedError("MT5 is not connected and reconnection failed")
        
        return self._mt5
    
    @retry(
        max_attempts=3,
        delay=1.0,
        backoff=2.0,
        exceptions=(MT5DownloadError, MT5TimeoutError),
    )
    def download_candles(
        self,
        symbol: str,
        timeframe: str,
        start_date: datetime,
        end_date: Optional[datetime] = None,
        max_candles: int = 100000,
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """
        Download candles from MT5.
        
        Args:
            symbol: Symbol to download (e.g., "EURUSD")
            timeframe: Timeframe (e.g., "H1")
            start_date: Start date for download
            end_date: End date (default: now)
            max_candles: Maximum candles to download
        
        Returns:
            Tuple of (candles_list, metadata_dict)
        
        Raises:
            MT5SymbolNotFoundError: If symbol not found in MT5
            MT5DownloadError: If download fails
            MT5TimeoutError: If download times out
        """
        self._logger.debug(f"📥 Downloading candles: {symbol} {timeframe}")
        
        # Validate inputs
        if not symbol:
            raise ValueError("symbol cannot be None or empty")
        if not timeframe:
            raise ValueError("timeframe cannot be None or empty")
        
        if end_date is None:
            end_date = datetime.now()
        
        # Get MT5 module
        mt5 = self.get()
        
        try:
            # Convert timeframe string to MT5 constant
            tf_constant = self._get_timeframe_constant(timeframe)
            
            # Select symbol
            if not mt5.symbol_select(symbol, True):
                self._last_error = f"Symbol not found: {symbol}"
                raise MT5SymbolNotFoundError(symbol)
            
            # Convert dates to MT5 format
            start_timestamp = int(start_date.timestamp())
            end_timestamp = int(end_date.timestamp())
            
            # Calculate max candles if not specified
            # Approximate: timeframe seconds * max_candles
            tf_seconds = self._get_timeframe_seconds(timeframe)
            if not max_candles:
                max_candles = self._DEFAULT_MAX_RETRY_ATTEMPTS
            
            # Download candles
            rates = mt5.copy_rates_range(
                symbol,
                tf_constant,
                start_timestamp,
                end_timestamp
            )
            
            if rates is None:
                error = mt5.last_error()
                self._last_error = str(error)
                raise MT5DownloadError(
                    symbol,
                    timeframe,
                    f"MT5 returned None: {error}",
                    original_error=RuntimeError(str(error))
                )
            
            # Convert to list of dicts
            candles = self._convert_rates(rates)
            
            # Build metadata
            metadata = {
                'symbol': symbol,
                'timeframe': timeframe,
                'start_date': start_date.isoformat(),
                'end_date': end_date.isoformat(),
                'candle_count': len(candles),
                'start_timestamp': start_timestamp,
                'end_timestamp': end_timestamp,
                'downloaded_at': datetime.now().isoformat(),
            }
            
            self._logger.debug(
                f"✅ Downloaded {len(candles)} candles: {symbol} {timeframe}"
            )
            
            return candles, metadata
            
        except MT5SymbolNotFoundError:
            raise
        except MT5DownloadError:
            raise
        except Exception as e:
            self._last_error = str(e)
            raise MT5DownloadError(
                symbol,
                timeframe,
                f"Failed to download: {e}",
                original_error=e
            )
    
    def get_symbol_info(self, symbol: str) -> Dict[str, Any]:
        """
        Get symbol information from MT5.
        
        Args:
            symbol: Symbol name
        
        Returns:
            Dictionary with symbol information
        
        Raises:
            MT5SymbolNotFoundError: If symbol not found
        """
        mt5 = self.get()
        
        try:
            # Select symbol
            if not mt5.symbol_select(symbol, True):
                raise MT5SymbolNotFoundError(symbol)
            
            # Get symbol info
            symbol_info = mt5.symbol_info(symbol)
            
            if symbol_info is None:
                raise MT5SymbolNotFoundError(symbol)
            
            # Convert to dict
            return self._symbol_info_to_dict(symbol_info)
            
        except MT5SymbolNotFoundError:
            raise
        except Exception as e:
            self._last_error = str(e)
            raise MT5Error(f"Failed to get symbol info for {symbol}: {e}")
    
    def get_account_info(self) -> Optional[Dict[str, Any]]:
        """
        Get account information from MT5.
        
        Returns:
            Dictionary with account information, or None if not connected
        """
        if not self._connected:
            return None
        
        mt5 = self.get()
        
        try:
            account = mt5.account_info()
            if account is None:
                return None
            
            return {
                'login': account.login,
                'server': account.server,
                'name': account.name,
                'company': account.company,
                'currency': account.currency,
                'balance': account.balance,
                'equity': account.equity,
                'margin': account.margin,
                'free_margin': account.margin_free,
                'profit': account.profit,
                'leverage': account.leverage,
            }
            
        except Exception as e:
            self._logger.error(f"❌ Failed to get account info: {e}")
            return None
    
    def get_terminal_info(self) -> Optional[Dict[str, Any]]:
        """
        Get terminal information from MT5.
        
        Returns:
            Dictionary with terminal information, or None if not connected
        """
        if not self._connected:
            return None
        
        mt5 = self.get()
        
        try:
            terminal = mt5.terminal_info()
            if terminal is None:
                return None
            
            return {
                'name': terminal.name,
                'company': terminal.company,
                'path': terminal.path,
                'build': terminal.build,
                'language': terminal.language,
                'trade_allowed': terminal.trade_allowed,
                'trade_mode': terminal.trade_mode,
            }
            
        except Exception as e:
            self._logger.error(f"❌ Failed to get terminal info: {e}")
            return None
    
    def get_last_error(self) -> Optional[str]:
        """
        Get the last error that occurred.
        
        Returns:
            Last error message or None
        """
        return self._last_error
    
    def get_metrics(self) -> Dict[str, Any]:
        """
        Get connection metrics.
        
        Returns:
            Dictionary with metrics
        """
        return {
            'is_connected': self._connected,
            'connection_attempts': self._connection_attempts,
            'reconnects': self._reconnects,
            'last_error': self._last_error,
            'last_error_time': self._last_error_time.isoformat() if self._last_error_time else None,
            'login': self._login,
            'server': self._server,
        }
    
    def close(self) -> None:
        """
        Close the MT5 connection.
        
        This should be called during application shutdown.
        """
        with self._connection_lock:
            self._disconnect()
            self._logger.info("⏹️ MT5 connection closed")
    
    # ==========================================================================
    # PRIVATE METHODS
    # ==========================================================================
    
    def _connect(self) -> bool:
        """
        Internal method to establish MT5 connection.
        
        Returns:
            True if connection successful, False otherwise
        """
        self._connection_attempts += 1
        self._logger.debug(f"🔄 Connecting to MT5 (attempt {self._connection_attempts})...")
        
        try:
            # Import MT5
            if self._mt5 is None:
                import MetaTrader5 as mt5
                self._mt5 = mt5
            
            # Initialize MT5
            if self._login and self._password:
                # Initialize with credentials
                initialized = self._mt5.initialize(
                    login=self._login,
                    password=self._password,
                    server=self._server,
                )
            else:
                # Initialize without credentials (use existing terminal)
                initialized = self._mt5.initialize()
            
            if not initialized:
                error = self._mt5.last_error()
                self._last_error = str(error)
                self._last_error_time = datetime.now()
                self._logger.error(f"❌ MT5 initialization failed: {error}")
                return False
            
            # Verify connection
            account = self._mt5.account_info()
            if account is None:
                self._last_error = "Account info not available"
                self._last_error_time = datetime.now()
                self._logger.error(f"❌ {self._last_error}")
                return False
            
            self._connected = True
            self._last_error = None
            self._last_error_time = None
            
            self._logger.info(
                f"✅ MT5 connected: login={account.login}, "
                f"server={account.server}, currency={account.currency}"
            )
            
            return True
            
        except Exception as e:
            self._last_error = str(e)
            self._last_error_time = datetime.now()
            self._logger.error(f"❌ MT5 connection error: {e}")
            return False
    
    def _disconnect(self) -> None:
        """
        Internal method to disconnect from MT5.
        """
        if self._mt5 is not None and self._connected:
            try:
                self._mt5.shutdown()
                self._logger.debug("MT5 shutdown completed")
            except Exception as e:
                self._logger.warning(f"⚠️ Error during MT5 shutdown: {e}")
        
        self._connected = False
    
    def _reconnect(self) -> bool:
        """
        Reconnect to MT5 with exponential backoff.
        
        Returns:
            True if reconnection successful, False otherwise
        """
        self._reconnects += 1
        self._logger.warning(f"🔄 Reconnecting to MT5 (attempt {self._reconnects})...")
        
        # Disconnect first
        self._disconnect()
        
        # Exponential backoff
        delay = exponential_backoff(self._reconnects - 1, base_delay=1.0)
        if delay > 0:
            self._logger.debug(f"⏳ Waiting {delay:.2f}s before reconnect...")
            time.sleep(delay)
        
        return self._connect()
    
    def _get_timeframe_constant(self, timeframe: str) -> int:
        """
        Convert timeframe string to MT5 constant.
        
        Args:
            timeframe: Timeframe string (e.g., "H1")
        
        Returns:
            MT5 timeframe constant
        
        Raises:
            ValueError: If timeframe is invalid
        """
        tf_map = {
            "M1": 1,
            "M2": 2,
            "M3": 3,
            "M4": 4,
            "M5": 5,
            "M6": 6,
            "M10": 10,
            "M12": 12,
            "M15": 15,
            "M20": 20,
            "M30": 30,
            "H1": 16385,
            "H2": 16386,
            "H3": 16387,
            "H4": 16388,
            "H6": 16390,
            "H8": 16392,
            "H12": 16396,
            "D1": 16408,
            "W1": 16409,
            "MN1": 16410,
        }
        
        if timeframe not in tf_map:
            raise ValueError(f"Invalid timeframe: {timeframe}")
        
        return tf_map[timeframe]
    
    def _get_timeframe_seconds(self, timeframe: str) -> int:
        """
        Get timeframe duration in seconds.
        
        Args:
            timeframe: Timeframe string
        
        Returns:
            Duration in seconds
        
        Raises:
            ValueError: If timeframe is invalid
        """
        tf_seconds = {
            "M1": 60,
            "M2": 120,
            "M3": 180,
            "M4": 240,
            "M5": 300,
            "M6": 360,
            "M10": 600,
            "M12": 720,
            "M15": 900,
            "M20": 1200,
            "M30": 1800,
            "H1": 3600,
            "H2": 7200,
            "H3": 10800,
            "H4": 14400,
            "H6": 21600,
            "H8": 28800,
            "H12": 43200,
            "D1": 86400,
            "W1": 604800,
            "MN1": 2592000,
        }
        
        if timeframe not in tf_seconds:
            raise ValueError(f"Invalid timeframe: {timeframe}")
        
        return tf_seconds[timeframe]
    
    def _convert_rates(self, rates) -> List[Dict[str, Any]]:
        """
        Convert MT5 rate array to list of dictionaries.
        
        Args:
            rates: MT5 rate array (numpy array-like)
        
        Returns:
            List of candle dictionaries
        """
        candles = []
        
        for rate in rates:
            candle = {
                'time': int(rate[0]),
                'open': float(rate[1]),
                'high': float(rate[2]),
                'low': float(rate[3]),
                'close': float(rate[4]),
                'tick_volume': int(rate[5]),
                'spread': int(rate[6]) if len(rate) > 6 else 0,
                'real_volume': int(rate[7]) if len(rate) > 7 else 0,
            }
            candles.append(candle)
        
        return candles
    
    def _symbol_info_to_dict(self, symbol_info) -> Dict[str, Any]:
        """
        Convert MT5 symbol info to dictionary.
        
        Args:
            symbol_info: MT5 symbol info object
        
        Returns:
            Dictionary with symbol information
        """
        return {
            'name': symbol_info.name,
            'path': symbol_info.path,
            'description': symbol_info.description,
            'currency_base': symbol_info.currency_base,
            'currency_profit': symbol_info.currency_profit,
            'point': symbol_info.point,
            'digits': symbol_info.digits,
            'trade_contract_size': symbol_info.trade_contract_size,
            'trade_tick_size': symbol_info.trade_tick_size,
            'trade_tick_value': symbol_info.trade_tick_value,
            'trade_mode': symbol_info.trade_mode,
            'trade_calc_mode': symbol_info.trade_calc_mode,
            'swap_mode': symbol_info.swap_mode,
            'swap_long': symbol_info.swap_long,
            'swap_short': symbol_info.swap_short,
            'margin_initial': symbol_info.margin_initial,
            'margin_maintenance': symbol_info.margin_maintenance,
            'margin_hedged': symbol_info.margin_hedged,
            'time_start': symbol_info.time_start,
            'time_expiration': symbol_info.time_expiration,
            'lot_min': symbol_info.lot_min,
            'lot_max': symbol_info.lot_max,
            'lot_step': symbol_info.lot_step,
            'volume_min': symbol_info.volume_min,
            'volume_max': symbol_info.volume_max,
            'volume_step': symbol_info.volume_step,
        }


# ==============================================================================
# FACTORY FUNCTION
# ==============================================================================

def create_mt5_manager(
    config: Config,
    logger: logging.Logger = None,
) -> MT5Manager:
    """
    Factory function for MT5Manager creation.
    
    Args:
        config: Configuration instance
        logger: Optional logger instance
    
    Returns:
        MT5Manager instance
    """
    return MT5Manager(config, logger)