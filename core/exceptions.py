"""
core/exceptions.py - Custom Exception Hierarchy for MarketAI

RESPONSIBILITY:
Define a comprehensive exception hierarchy for the entire MarketAI system.

ARCHITECTURAL DECISIONS:
1. All exceptions inherit from MarketAIException (base)
2. Clear exception categories for different system components
3. Rich context information (symbol, timeframe, operation, etc.)
4. Error codes for monitoring and alerting
5. Proper __str__() overrides for clear error messages
6. Automatic context tracking via kwargs

USAGE:
    try:
        updater.update("EURUSD", "H1")
    except MT5DownloadError as e:
        logger.error(f"Download failed: {e.error_code} - {e}")
        # e.symbol, e.timeframe, e.original_error available

VERSION: 1.0.0
"""

from typing import Optional, Any, Dict, Tuple
from datetime import datetime


# ==============================================================================
# BASE EXCEPTION
# ==============================================================================

class MarketAIException(Exception):
    """
    Base exception for all MarketAI errors.
    
    All custom exceptions in MarketAI inherit from this class.
    Provides common functionality for error codes and context.
    """
    
    # Default error code (overridden by subclasses)
    error_code: str = "MARKETAI_ERROR"
    
    def __init__(
        self,
        message: str,
        error_code: Optional[str] = None,
        original_error: Optional[Exception] = None,
        context: Optional[Dict[str, Any]] = None,
        **kwargs,
    ):
        """
        Initialize the exception.
        
        Args:
            message: Human-readable error message
            error_code: Optional error code (overrides class default)
            original_error: Original exception that caused this error
            context: Additional context information
            **kwargs: Additional attributes to store
        """
        self.message = message
        self.error_code = error_code or self.error_code
        self.original_error = original_error
        self.context = context or {}
        self.timestamp = datetime.now()
        
        # Store additional kwargs as attributes
        for key, value in kwargs.items():
            setattr(self, key, value)
        
        super().__init__(message)
    
    def __str__(self) -> str:
        """Return formatted error message."""
        if self.error_code:
            return f"[{self.error_code}] {self.message}"
        return self.message
    
    def __repr__(self) -> str:
        """Return detailed representation."""
        base = f"{self.__class__.__name__}(message='{self.message}')"
        if self.error_code:
            base = f"{base}, error_code='{self.error_code}'"
        if self.context:
            base = f"{base}, context={self.context}"
        return base
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert exception to dictionary for logging/monitoring.
        
        Returns:
            Dict with error details
        """
        return {
            'exception': self.__class__.__name__,
            'error_code': self.error_code,
            'message': self.message,
            'timestamp': self.timestamp.isoformat(),
            'context': self.context,
            'original_error': str(self.original_error) if self.original_error else None,
            'type': self.original_error.__class__.__name__ if self.original_error else None,
        }


MarketAIError = MarketAIException


# ==============================================================================
# CONFIGURATION ERRORS
# ==============================================================================

class ConfigurationError(MarketAIException):
    """
    Invalid or missing configuration.
    
    Raised when configuration values are missing, invalid, or cannot be loaded.
    """
    error_code = "CONFIG_ERROR"


class ConfigNotFoundError(ConfigurationError):
    """
    Configuration file not found.
    """
    error_code = "CONFIG_NOT_FOUND"
    
    def __init__(self, path: str, message: Optional[str] = None):
        self.path = path
        super().__init__(
            message or f"Configuration file not found: {path}",
            context={'path': path},
        )


class ConfigValidationError(ConfigurationError):
    """
    Configuration validation failed.
    """
    error_code = "CONFIG_VALIDATION_ERROR"
    
    def __init__(self, key: str, value: Any, reason: str):
        self.key = key
        self.value = value
        self.reason = reason
        super().__init__(
            f"Invalid configuration for '{key}': {value} - {reason}",
            context={'key': key, 'value': value, 'reason': reason},
        )


# ==============================================================================
# DATABASE ERRORS
# ==============================================================================

class DatabaseError(MarketAIException):
    """
    Database operation failed.
    
    Base exception for all database-related errors.
    """
    error_code = "DB_ERROR"


class DatabaseConnectionError(DatabaseError):
    """
    Cannot connect to the database.
    """
    error_code = "DB_CONNECTION_ERROR"
    
    def __init__(self, path: str, original_error: Optional[Exception] = None):
        self.path = path
        super().__init__(
            f"Cannot connect to database: {path}",
            original_error=original_error,
            context={'path': path},
        )


class DatabaseQueryError(DatabaseError):
    """
    Database query execution failed.
    """
    error_code = "DB_QUERY_ERROR"
    
    def __init__(
        self,
        sql: str,
        message: str,
        params: Optional[Tuple] = None,
        original_error: Optional[Exception] = None,
    ):
        self.sql = sql
        self.params = params
        super().__init__(
            f"Query failed: {message}\nSQL: {sql[:500]}",
            original_error=original_error,
            context={'sql': sql[:500], 'params': params},
        )


class DatabaseSchemaError(DatabaseError):
    """
    Database schema migration or validation failed.
    """
    error_code = "DB_SCHEMA_ERROR"
    
    def __init__(self, table: str, message: str):
        self.table = table
        super().__init__(
            f"Schema error in '{table}': {message}",
            context={'table': table},
        )


# ==============================================================================
# MT5 ERRORS
# ==============================================================================

class MT5Error(MarketAIException):
    """
    MT5 operation failed.
    
    Base exception for all MetaTrader 5 related errors.
    """
    error_code = "MT5_ERROR"


class MT5ConnectionError(MT5Error):
    """
    Cannot connect to MT5 terminal.
    """
    error_code = "MT5_CONNECTION_ERROR"
    
    def __init__(
        self,
        message: str = "Cannot connect to MT5 terminal",
        original_error: Optional[Exception] = None,
    ):
        super().__init__(
            message,
            original_error=original_error,
            context={'message': message},
        )


ConnectionError = MT5ConnectionError


class MT5DisconnectedError(MT5Error):
    """
    MT5 connection was lost.
    """
    error_code = "MT5_DISCONNECTED"
    
    def __init__(self, message: str = "MT5 connection lost"):
        super().__init__(message)


class MT5DownloadError(MT5Error):
    """
    Failed to download data from MT5.
    """
    error_code = "MT5_DOWNLOAD_ERROR"
    
    def __init__(
        self,
        symbol: str,
        timeframe: str,
        message: str,
        original_error: Optional[Exception] = None,
    ):
        self.symbol = symbol
        self.timeframe = timeframe
        super().__init__(
            f"Failed to download {symbol} {timeframe}: {message}",
            original_error=original_error,
            context={'symbol': symbol, 'timeframe': timeframe},
        )


class MT5TimeoutError(MT5Error):
    """
    MT5 operation timed out.
    """
    error_code = "MT5_TIMEOUT_ERROR"
    
    def __init__(
        self,
        operation: str,
        timeout: float,
        original_error: Optional[Exception] = None,
    ):
        self.operation = operation
        self.timeout = timeout
        super().__init__(
            f"MT5 operation '{operation}' timed out after {timeout}s",
            original_error=original_error,
            context={'operation': operation, 'timeout': timeout},
        )


class MT5SymbolNotFoundError(MT5Error):
    """
    Symbol not found in MT5.
    """
    error_code = "MT5_SYMBOL_NOT_FOUND"
    
    def __init__(self, symbol: str):
        self.symbol = symbol
        super().__init__(
            f"Symbol not found in MT5: {symbol}",
            context={'symbol': symbol},
        )


# ==============================================================================
# DATA VALIDATION ERRORS
# ==============================================================================

class DataValidationError(MarketAIException):
    """
    Invalid data or format.
    """
    error_code = "DATA_VALIDATION_ERROR"
    
    def __init__(
        self,
        field: str,
        value: Any,
        reason: str,
    ):
        self.field = field
        self.value = value
        self.reason = reason
        super().__init__(
            f"Invalid {field} '{value}': {reason}",
            context={'field': field, 'value': value, 'reason': reason},
        )


class DataMissingError(DataValidationError):
    """
    Required data is missing.
    """
    error_code = "DATA_MISSING_ERROR"
    
    def __init__(self, field: str, message: Optional[str] = None):
        self.field = field
        super().__init__(
            field=field,
            value=None,
            reason=message or f"Required field '{field}' is missing",
        )


class DataFormatError(DataValidationError):
    """
    Data is in wrong format.
    """
    error_code = "DATA_FORMAT_ERROR"
    
    def __init__(
        self,
        field: str,
        value: Any,
        expected_format: str,
    ):
        self.expected_format = expected_format
        super().__init__(
            field=field,
            value=value,
            reason=f"Expected format: {expected_format}",
            context={'expected_format': expected_format},
        )


# ==============================================================================
# MARKET ERRORS
# ==============================================================================

class MarketError(MarketAIException):
    """
    Market operation failed.
    
    Base exception for market-related errors.
    """
    error_code = "MARKET_ERROR"


class MarketNotFoundError(MarketError):
    """
    Market/symbol not found in the system.
    """
    error_code = "MARKET_NOT_FOUND"
    
    def __init__(self, symbol: str):
        self.symbol = symbol
        super().__init__(
            f"Market not found: {symbol}",
            context={'symbol': symbol},
        )


class MarketInvalidError(MarketError):
    """
    Market/symbol is invalid or unsupported.
    """
    error_code = "MARKET_INVALID"
    
    def __init__(self, symbol: str, reason: str):
        self.symbol = symbol
        self.reason = reason
        super().__init__(
            f"Invalid market {symbol}: {reason}",
            context={'symbol': symbol, 'reason': reason},
        )


class TimeframeError(MarketError):
    """
    Timeframe operation failed.
    
    Base exception for timeframe-related errors.
    """
    error_code = "TIMEFRAME_ERROR"


class TimeframeNotFoundError(TimeframeError):
    """
    Timeframe not found in the system.
    """
    error_code = "TIMEFRAME_NOT_FOUND"
    
    def __init__(self, timeframe: str):
        self.timeframe = timeframe
        super().__init__(
            f"Timeframe not found: {timeframe}",
            context={'timeframe': timeframe},
        )


class TimeframeInvalidError(TimeframeError):
    """
    Timeframe is invalid or unsupported.
    """
    error_code = "TIMEFRAME_INVALID"
    
    def __init__(self, timeframe: str, supported: list):
        self.timeframe = timeframe
        self.supported = supported
        super().__init__(
            f"Invalid timeframe: {timeframe}. Supported: {supported}",
            context={'timeframe': timeframe, 'supported': supported},
        )


# ==============================================================================
# COLLECTOR ERRORS
# ==============================================================================

class CollectorError(MarketAIException):
    """
    Data collection failed.
    
    Base exception for all collector-related errors.
    """
    error_code = "COLLECTOR_ERROR"


class DownloadQueueError(CollectorError):
    """
    Download queue operation failed.
    """
    error_code = "QUEUE_ERROR"
    
    def __init__(
        self,
        operation: str,
        symbol: Optional[str] = None,
        timeframe: Optional[str] = None,
        message: Optional[str] = None,
    ):
        self.operation = operation
        self.symbol = symbol
        self.timeframe = timeframe
        context = {'operation': operation}
        if symbol:
            context['symbol'] = symbol
        if timeframe:
            context['timeframe'] = timeframe
        
        msg = f"Queue operation '{operation}' failed"
        if message:
            msg += f": {message}"
        if symbol and timeframe:
            msg += f" ({symbol} {timeframe})"
        
        super().__init__(msg, context=context)


class QueueFullError(DownloadQueueError):
    """
    Download queue is full.
    """
    error_code = "QUEUE_FULL"
    
    def __init__(
        self,
        max_size: int,
        symbol: Optional[str] = None,
        timeframe: Optional[str] = None,
    ):
        self.max_size = max_size
        super().__init__(
            operation="enqueue",
            symbol=symbol,
            timeframe=timeframe,
            message=f"Queue full (max_size={max_size})",
        )


class DuplicateTaskError(DownloadQueueError):
    """
    Task already exists in the queue.
    """
    error_code = "DUPLICATE_TASK"
    
    def __init__(self, symbol: str, timeframe: str):
        super().__init__(
            operation="enqueue",
            symbol=symbol,
            timeframe=timeframe,
            message="Task already exists in queue",
        )


class SchedulerError(MarketAIException):
    """
    Scheduler operation failed.
    """
    error_code = "SCHEDULER_ERROR"
    
    def __init__(
        self,
        operation: str,
        symbol: Optional[str] = None,
        timeframe: Optional[str] = None,
        message: Optional[str] = None,
    ):
        self.operation = operation
        self.symbol = symbol
        self.timeframe = timeframe
        context = {'operation': operation}
        if symbol:
            context['symbol'] = symbol
        if timeframe:
            context['timeframe'] = timeframe
        
        msg = f"Scheduler operation '{operation}' failed"
        if message:
            msg += f": {message}"
        if symbol and timeframe:
            msg += f" ({symbol} {timeframe})"
        
        super().__init__(msg, context=context)


# ==============================================================================
# DISCOVERY ERRORS
# ==============================================================================

class DiscoveryError(MarketAIException):
    """
    Market discovery failed.
    
    Base exception for all discovery-related errors.
    """
    error_code = "DISCOVERY_ERROR"


class PatternDiscoveryError(DiscoveryError):
    """
    Pattern discovery failed.
    """
    error_code = "PATTERN_DISCOVERY_ERROR"
    
    def __init__(
        self,
        symbol: str,
        timeframe: str,
        message: str,
        original_error: Optional[Exception] = None,
    ):
        self.symbol = symbol
        self.timeframe = timeframe
        super().__init__(
            f"Pattern discovery failed for {symbol} {timeframe}: {message}",
            original_error=original_error,
            context={'symbol': symbol, 'timeframe': timeframe},
        )


class CorrelationDiscoveryError(DiscoveryError):
    """
    Correlation discovery failed.
    """
    error_code = "CORRELATION_DISCOVERY_ERROR"
    
    def __init__(
        self,
        market1: str,
        market2: str,
        message: str,
        original_error: Optional[Exception] = None,
    ):
        self.market1 = market1
        self.market2 = market2
        super().__init__(
            f"Correlation discovery failed between {market1} and {market2}: {message}",
            original_error=original_error,
            context={'market1': market1, 'market2': market2},
        )


# ==============================================================================
# VALIDATION ERRORS
# ==============================================================================

class ValidationError(MarketAIException):
    """
    Validation operation failed.
    
    Base exception for all validation-related errors.
    """
    error_code = "VALIDATION_ERROR"


class LiveValidationError(ValidationError):
    """
    Live validation failed.
    """
    error_code = "LIVE_VALIDATION_ERROR"
    
    def __init__(
        self,
        pattern_id: str,
        message: str,
        original_error: Optional[Exception] = None,
    ):
        self.pattern_id = pattern_id
        super().__init__(
            f"Live validation failed for pattern {pattern_id}: {message}",
            original_error=original_error,
            context={'pattern_id': pattern_id},
        )


# ==============================================================================
# RETRY ERRORS
# ==============================================================================

class RetryExhaustedError(MarketAIException):
    """
    Maximum retry attempts exceeded.
    """
    error_code = "RETRY_EXHAUSTED"
    
    def __init__(
        self,
        operation: str,
        attempts: int,
        last_error: Optional[Exception] = None,
        symbol: Optional[str] = None,
        timeframe: Optional[str] = None,
    ):
        self.operation = operation
        self.attempts = attempts
        self.symbol = symbol
        self.timeframe = timeframe
        context = {'operation': operation, 'attempts': attempts}
        if symbol:
            context['symbol'] = symbol
        if timeframe:
            context['timeframe'] = timeframe
        
        msg = f"Operation '{operation}' exhausted after {attempts} attempts"
        if last_error:
            msg += f": {last_error}"
        if symbol and timeframe:
            msg += f" ({symbol} {timeframe})"
        
        super().__init__(
            msg,
            original_error=last_error,
            context=context,
        )


# ==============================================================================
# SYSTEM ERRORS
# ==============================================================================

class SystemError(MarketAIException):
    """
    System-level error.
    """
    error_code = "SYSTEM_ERROR"


class ShutdownError(SystemError):
    """
    System shutdown failed.
    """
    error_code = "SHUTDOWN_ERROR"
    
    def __init__(self, component: str, message: str):
        self.component = component
        super().__init__(
            f"Shutdown error in '{component}': {message}",
            context={'component': component},
        )


class InitializationError(SystemError):
    """
    System initialization failed.
    """
    error_code = "INITIALIZATION_ERROR"
    
    def __init__(self, component: str, message: str):
        self.component = component
        super().__init__(
            f"Initialization error in '{component}': {message}",
            context={'component': component},
        )


# ==============================================================================
# EXCEPTION HELPERS
# ==============================================================================

def format_exception(e: Exception) -> str:
    """
    Format an exception for logging.
    
    Args:
        e: Exception to format
    
    Returns:
        Formatted string with exception details
    """
    if isinstance(e, MarketAIException):
        parts = [f"[{e.error_code}] {e.__class__.__name__}: {e.message}"]
        if e.context:
            parts.append(f"Context: {e.context}")
        if e.original_error:
            parts.append(f"Original: {e.original_error}")
        return "\n".join(parts)
    
    return f"{e.__class__.__name__}: {e}"


def get_exception_context(e: Exception) -> Dict[str, Any]:
    """
    Get context from an exception if available.
    
    Args:
        e: Exception to inspect
    
    Returns:
        Dict with context information
    """
    if isinstance(e, MarketAIException):
        return e.context
    return {}