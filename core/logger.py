"""
core/logger.py - Global Logger Factory

RESPONSIBILITY:
Initialize and manage the global logging system for MarketAI.

ARCHITECTURAL DECISIONS:
1. Singleton pattern - Single logger instance
2. Lazy initialization - Setup on first call
3. Rotating file handler - Prevents disk overflow
4. Colored console output - Better readability
5. Module-level log levels - Granular control
6. Thread-safe - Uses threading.Lock for protection
7. Environment variable support - LOG_LEVEL, LOG_DIR

USAGE:
    from core.logger import setup_logger
    
    logger = setup_logger()
    logger.info("MarketAI initialized")
    
    # Module-specific log level
    logger = setup_logger(module_levels={
        'mt5.manager': 'DEBUG',
        'collector.updater': 'INFO',
    })

VERSION: 1.0.1
"""

import os
import sys
import logging
import logging.handlers
import threading
from pathlib import Path
from typing import Dict, Optional, Union, Any
from datetime import datetime

# Try to import colorama for colored output
try:
    import colorama
    colorama.init(autoreset=True)
    HAS_COLORAMA = True
except ImportError:
    HAS_COLORAMA = False

# Try to import tqdm for progress bar logging
try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False


# ==============================================================================
# CONSTANTS
# ==============================================================================

# Environment variable support
ENV_LOG_LEVEL = os.environ.get("LOG_LEVEL", "INFO").upper()
ENV_LOG_DIR = os.environ.get("LOG_DIR", "logs")

DEFAULT_LOG_LEVEL = getattr(logging, ENV_LOG_LEVEL, logging.INFO)
DEFAULT_LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
DEFAULT_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

LOG_DIR = ENV_LOG_DIR
LOG_FILE = "market_ai.log"
MAX_FILE_SIZE = 100 * 1024 * 1024  # 100 MB
BACKUP_COUNT = 5

# Noisy libraries to silence
NOISY_LOGGERS = {
    "MetaTrader5": logging.WARNING,
    "urllib3": logging.WARNING,
    "requests": logging.WARNING,
    "matplotlib": logging.WARNING,
    "tensorflow": logging.WARNING,
    "torch": logging.WARNING,
    "numexpr": logging.WARNING,
    "werkzeug": logging.WARNING,
    "asyncio": logging.WARNING,
}


# ==============================================================================
# COLORED FORMATTER (Fixed - No LogRecord Mutation)
# ==============================================================================

class ColoredFormatter(logging.Formatter):
    """
    Custom formatter with colors for console output.
    
    FIXED: Does NOT mutate LogRecord - stores colors separately.
    """
    
    # ANSI color codes
    COLORS = {
        'DEBUG': '\033[36m',      # Cyan
        'INFO': '\033[32m',       # Green
        'WARNING': '\033[33m',    # Yellow
        'ERROR': '\033[31m',      # Red
        'CRITICAL': '\033[35m',   # Magenta
        'RESET': '\033[0m',
    }
    
    # Bold variants for emphasis
    BOLD_COLORS = {
        'DEBUG': '\033[1;36m',
        'INFO': '\033[1;32m',
        'WARNING': '\033[1;33m',
        'ERROR': '\033[1;31m',
        'CRITICAL': '\033[1;35m',
    }
    
    def __init__(
        self,
        fmt: str = DEFAULT_LOG_FORMAT,
        datefmt: str = DEFAULT_DATE_FORMAT,
        use_colors: bool = True,
    ):
        super().__init__(fmt, datefmt)
        self.use_colors = use_colors and HAS_COLORAMA
    
    def format(self, record: logging.LogRecord) -> str:
        """
        Format log record with colors.
        
        FIXED: Does NOT mutate the record - stores colors separately.
        """
        # If colors are disabled, use standard formatting
        if not self.use_colors:
            return super().format(record)
        
        # Create a copy of the record to avoid mutation
        record_copy = logging.LogRecord(
            name=record.name,
            level=record.levelno,
            pathname=record.pathname,
            lineno=record.lineno,
            msg=record.msg,
            args=record.args,
            exc_info=record.exc_info,
            func=record.funcName,
            sinfo=None,
        )
        
        # Copy additional attributes
        for key, value in record.__dict__.items():
            if key not in record_copy.__dict__:
                setattr(record_copy, key, value)
        
        # Apply color to levelname (on the copy)
        levelname = record.levelname
        if levelname in self.COLORS:
            color = self.COLORS.get(levelname, '')
            reset = self.COLORS['RESET']
            record_copy.levelname = f"{color}{levelname}{reset}"
        
        # Apply color to message if it's an error
        if record.levelno >= logging.ERROR:
            msg = record_copy.getMessage()
            color = self.COLORS.get('ERROR', '')
            reset = self.COLORS['RESET']
            # Store colored message separately
            record_copy.msg = f"{color}{msg}{reset}"
        
        # Format using the copy
        return super().format(record_copy)


# ==============================================================================
# LOGGER FACTORY
# ==============================================================================

class LoggerFactory:
    """
    Thread-safe logger factory singleton.
    
    Provides lazy initialization and module-level log level control.
    FIXED: Uses threading.Lock instead of logging._lock.
    """
    
    _instance: Optional['LoggerFactory'] = None
    _lock = threading.Lock()
    
    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._initialized = False
            return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
        
        self._root_logger: Optional[logging.Logger] = None
        self._module_levels: Dict[str, int] = {}
        self._initialized = False
        self._factory_lock = threading.Lock()
    
    def setup(
        self,
        level: Union[str, int] = None,
        log_file: Optional[str] = None,
        log_dir: Optional[str] = None,
        module_levels: Optional[Dict[str, Union[str, int]]] = None,
        console_enabled: bool = True,
        file_enabled: bool = True,
        use_colors: bool = True,
        silence_noisy: bool = True,
    ) -> logging.Logger:
        """
        Setup the global logger.
        
        Args:
            level: Default log level (string or int)
            log_file: Log file name (default: market_ai.log)
            log_dir: Log directory (default: logs/ or env LOG_DIR)
            module_levels: Module-specific log levels
            console_enabled: Enable console output
            file_enabled: Enable file output
            use_colors: Use colored output (if available)
            silence_noisy: Silence noisy libraries
        
        Returns:
            Root logger instance
        """
        with self._factory_lock:
            # Use default from env if not specified
            if level is None:
                level = DEFAULT_LOG_LEVEL
            
            # Convert level to int
            if isinstance(level, str):
                level = getattr(logging, level.upper(), DEFAULT_LOG_LEVEL)
            
            # Use env log dir if not specified
            if log_dir is None:
                log_dir = LOG_DIR
            
            # Get root logger
            root_logger = logging.getLogger()
            root_logger.setLevel(level)
            
            # Clear existing handlers
            root_logger.handlers.clear()
            
            # Create formatter
            fmt = DEFAULT_LOG_FORMAT
            datefmt = DEFAULT_DATE_FORMAT
            
            # Console handler
            if console_enabled:
                console_handler = self._create_console_handler(
                    fmt=fmt,
                    datefmt=datefmt,
                    use_colors=use_colors,
                )
                root_logger.addHandler(console_handler)
            
            # File handler
            if file_enabled:
                file_handler = self._create_file_handler(
                    log_file=log_file,
                    log_dir=log_dir,
                    fmt=fmt,
                    datefmt=datefmt,
                )
                root_logger.addHandler(file_handler)
            
            # Set module-specific levels
            self._module_levels = {}
            if module_levels:
                for module, mod_level in module_levels.items():
                    if isinstance(mod_level, str):
                        mod_level = getattr(logging, mod_level.upper(), level)
                    self._module_levels[module] = mod_level
                    logger_instance = logging.getLogger(module)
                    logger_instance.setLevel(mod_level)
            
            # Silence noisy libraries
            if silence_noisy:
                for logger_name, log_level in NOISY_LOGGERS.items():
                    logging.getLogger(logger_name).setLevel(log_level)
            
            # Store root logger
            self._root_logger = root_logger
            self._initialized = True
            
            # Log initialization
            root_logger.info(
                f"✅ Logger initialized (level={logging.getLevelName(level)}, "
                f"log_dir={log_dir})"
            )
            
            return root_logger
    
    def get_logger(self, name: Optional[str] = None) -> logging.Logger:
        """
        Get a logger instance.
        
        Args:
            name: Logger name (optional)
        
        Returns:
            Logger instance
        """
        if not self._initialized:
            # Lazy initialization
            self.setup()
        
        logger_instance = logging.getLogger(name)
        
        # Apply module-specific level if set
        if name and name in self._module_levels:
            logger_instance.setLevel(self._module_levels[name])
        
        return logger_instance
    
    def set_module_level(self, module: str, level: Union[str, int]) -> None:
        """
        Set log level for a specific module.
        
        Args:
            module: Module name
            level: Log level (string or int)
        """
        with self._factory_lock:
            if isinstance(level, str):
                level = getattr(logging, level.upper(), DEFAULT_LOG_LEVEL)
            
            self._module_levels[module] = level
            logger_instance = logging.getLogger(module)
            logger_instance.setLevel(level)
            
            if self._root_logger:
                self._root_logger.info(
                    f"📊 Module level set: {module} -> {logging.getLevelName(level)}"
                )
    
    def get_module_level(self, module: str) -> Optional[int]:
        """Get log level for a specific module."""
        return self._module_levels.get(module)
    
    def _create_console_handler(
        self,
        fmt: str,
        datefmt: str,
        use_colors: bool = True,
    ) -> logging.Handler:
        """Create console handler."""
        handler = logging.StreamHandler(sys.stdout)
        
        if use_colors and HAS_COLORAMA:
            formatter = ColoredFormatter(fmt, datefmt, use_colors=True)
        else:
            formatter = logging.Formatter(fmt, datefmt)
        
        handler.setFormatter(formatter)
        return handler
    
    def _create_file_handler(
        self,
        log_file: Optional[str] = None,
        log_dir: Optional[str] = None,
        fmt: str = DEFAULT_LOG_FORMAT,
        datefmt: str = DEFAULT_DATE_FORMAT,
    ) -> logging.Handler:
        """Create rotating file handler."""
        # Determine log file path
        log_file = log_file or LOG_FILE
        log_dir = log_dir or LOG_DIR
        
        # Create log directory
        Path(log_dir).mkdir(parents=True, exist_ok=True)
        
        # Full path
        log_path = os.path.join(log_dir, log_file)
        
        # Create rotating file handler
        handler = logging.handlers.RotatingFileHandler(
            filename=log_path,
            maxBytes=MAX_FILE_SIZE,
            backupCount=BACKUP_COUNT,
            encoding='utf-8',
        )
        
        formatter = logging.Formatter(fmt, datefmt)
        handler.setFormatter(formatter)
        
        return handler


# ==============================================================================
# GLOBAL INSTANCE
# ==============================================================================

# Global logger factory instance
_factory = LoggerFactory()


# ==============================================================================
# PUBLIC API
# ==============================================================================

def setup_logger(
    level: Union[str, int] = None,
    log_file: Optional[str] = None,
    log_dir: Optional[str] = None,
    module_levels: Optional[Dict[str, Union[str, int]]] = None,
    console_enabled: bool = True,
    file_enabled: bool = True,
    use_colors: bool = True,
    silence_noisy: bool = True,
) -> logging.Logger:
    """
    Setup the global logger.
    
    Environment Variables:
        LOG_LEVEL: Default log level (e.g., "INFO", "DEBUG")
        LOG_DIR: Log directory (default: logs/)
    
    Args:
        level: Default log level (e.g., "INFO", "DEBUG")
        log_file: Log file name (default: market_ai.log)
        log_dir: Log directory (default: from env LOG_DIR or logs/)
        module_levels: Module-specific log levels
        console_enabled: Enable console output
        file_enabled: Enable file output
        use_colors: Use colored output (if available)
        silence_noisy: Silence noisy libraries
    
    Returns:
        Root logger instance
    
    Examples:
        # Basic usage
        logger = setup_logger()
        logger.info("Hello world")
        
        # With module levels
        logger = setup_logger(
            level="DEBUG",
            module_levels={
                'mt5.manager': 'DEBUG',
                'collector.updater': 'INFO',
            }
        )
        
        # Production setup with env vars
        # LOG_LEVEL=INFO LOG_DIR=logs/prod python main.py
        logger = setup_logger()
        
        # Custom log directory
        logger = setup_logger(log_dir="/var/log/market_ai")
    """
    return _factory.setup(
        level=level,
        log_file=log_file,
        log_dir=log_dir,
        module_levels=module_levels,
        console_enabled=console_enabled,
        file_enabled=file_enabled,
        use_colors=use_colors,
        silence_noisy=silence_noisy,
    )


def get_logger(name: Optional[str] = None) -> logging.Logger:
    """
    Get a logger instance.
    
    Args:
        name: Logger name (optional)
    
    Returns:
        Logger instance
    
    Examples:
        logger = get_logger(__name__)
        logger.info("Hello from module")
        
        # Default logger
        logger = get_logger()
        logger.info("Hello from main")
    """
    return _factory.get_logger(name)


def set_module_level(module: str, level: Union[str, int]) -> None:
    """
    Set log level for a specific module.
    
    Args:
        module: Module name (e.g., 'mt5.manager')
        level: Log level (e.g., 'DEBUG', 'INFO')
    
    Examples:
        set_module_level('mt5.manager', 'DEBUG')
        set_module_level('collector.updater', 'WARNING')
    """
    _factory.set_module_level(module, level)


def get_module_level(module: str) -> Optional[int]:
    """
    Get log level for a specific module.
    
    Args:
        module: Module name
    
    Returns:
        Log level integer or None if not set
    """
    return _factory.get_module_level(module)


def silence_library(library: str, level: Union[str, int] = logging.WARNING) -> None:
    """
    Silence a specific library logger.
    
    Args:
        library: Library name (e.g., 'urllib3')
        level: Log level (default: WARNING)
    """
    if isinstance(level, str):
        level = getattr(logging, level.upper(), logging.WARNING)
    logging.getLogger(library).setLevel(level)


def get_log_level_from_env() -> int:
    """
    Get log level from environment variable LOG_LEVEL.
    
    Returns:
        Log level integer (default: INFO)
    """
    level_str = os.environ.get("LOG_LEVEL", "INFO").upper()
    return getattr(logging, level_str, logging.INFO)


# ==============================================================================
# CONVENIENCE FUNCTIONS
# ==============================================================================

def debug_enabled() -> bool:
    """Check if debug logging is enabled."""
    return get_logger().isEnabledFor(logging.DEBUG)


def trace_enabled() -> bool:
    """Check if trace (DEBUG - 5) logging is enabled."""
    return get_logger().isEnabledFor(logging.DEBUG - 5)


def log_execution_time(logger: logging.Logger = None) -> callable:
    """
    Decorator to log execution time of a function.
    
    Args:
        logger: Logger instance (default: get_logger())
    
    Returns:
        Decorator
    
    Examples:
        @log_execution_time()
        def my_function():
            pass
        
        @log_execution_time(get_logger(__name__))
        def my_function():
            pass
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            log = logger or get_logger(func.__module__)
            start = datetime.now()
            try:
                result = func(*args, **kwargs)
                elapsed = (datetime.now() - start).total_seconds()
                if elapsed > 1.0:
                    log.warning(f"⏱️ {func.__name__} took {elapsed:.2f}s")
                else:
                    log.debug(f"⏱️ {func.__name__} took {elapsed:.3f}s")
                return result
            except Exception as e:
                elapsed = (datetime.now() - start).total_seconds()
                log.error(f"❌ {func.__name__} failed after {elapsed:.2f}s: {e}")
                raise
        return wrapper
    return decorator


# ==============================================================================
# PRE-CONFIGURED LOGGERS
# ==============================================================================

def get_rotating_logger(
    name: str,
    log_file: str,
    log_dir: str = None,
    max_bytes: int = MAX_FILE_SIZE,
    backup_count: int = BACKUP_COUNT,
    level: Union[str, int] = logging.INFO,
) -> logging.Logger:
    """
    Get a logger with a custom rotating file handler.
    
    Useful for modules that need separate log files.
    
    Args:
        name: Logger name
        log_file: Log file name
        log_dir: Log directory (default: from env LOG_DIR or logs/)
        max_bytes: Maximum file size
        backup_count: Number of backups
        level: Log level
    
    Returns:
        Logger instance
    
    Examples:
        collector_logger = get_rotating_logger(
            'collector',
            'collector.log',
            level='DEBUG',
        )
    """
    if log_dir is None:
        log_dir = LOG_DIR
    
    logger_instance = logging.getLogger(name)
    logger_instance.setLevel(level)
    logger_instance.propagate = False
    
    # Create directory
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    
    # File handler
    handler = logging.handlers.RotatingFileHandler(
        filename=os.path.join(log_dir, log_file),
        maxBytes=max_bytes,
        backupCount=backup_count,
        encoding='utf-8',
    )
    
    formatter = logging.Formatter(DEFAULT_LOG_FORMAT, DEFAULT_DATE_FORMAT)
    handler.setFormatter(formatter)
    logger_instance.addHandler(handler)
    
    return logger_instance


# ==============================================================================
# INITIALIZATION
# ==============================================================================

# Lazy initialization is triggered by setup_logger() or get_logger()
__all__ = [
    'setup_logger',
    'get_logger',
    'set_module_level',
    'get_module_level',
    'silence_library',
    'debug_enabled',
    'trace_enabled',
    'log_execution_time',
    'get_rotating_logger',
    'get_log_level_from_env',
]