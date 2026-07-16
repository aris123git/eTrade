"""
core/utils.py - Utility Functions for MarketAI

RESPONSIBILITY:
Provide common utility functions used across MarketAI components.

ARCHITECTURAL DECISIONS:
1. Pure functions - No side effects
2. No external dependencies - Standard library only
3. Comprehensive type hints - For IDE support
4. Extensive docstrings - For maintainability
5. Production-ready - Thorough error handling

VERSION: 1.0.0
"""

import re
import time
import math
from datetime import datetime, timedelta
from typing import Any, List, Dict, Optional, Iterator, Tuple, Callable, TypeVar, Union
from functools import wraps
from decimal import Decimal, ROUND_HALF_UP

# ==============================================================================
# CONSTANTS
# ==============================================================================

# Valid timeframes in seconds
TIMEFRAME_MAP = {
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

# Valid timeframe names
VALID_TIMEFRAMES = set(TIMEFRAME_MAP.keys())

# Symbol validation pattern (basic)
SYMBOL_PATTERN = re.compile(r'^[A-Z0-9]{2,12}([._][A-Z0-9]{2,6})?$')

# Candle field validation
CANDLE_REQUIRED_FIELDS = {"open", "high", "low", "close"}
CANDLE_OPTIONAL_FIELDS = {"volume", "tick_volume", "time", "spread", "real_volume"}


# ==============================================================================
# TIMESTAMP UTILITIES
# ==============================================================================

def to_datetime(value: Any) -> Optional[datetime]:
    """
    Convert various timestamp formats to datetime.
    
    Handles:
    - int/float: Unix timestamp (seconds since epoch)
    - datetime: Return as-is
    - str: ISO format string
    - None: Return None
    
    Args:
        value: Timestamp value to convert
        
    Returns:
        datetime object or None if value is None
        
    Raises:
        ValueError: If value cannot be converted
        TypeError: If value type is unsupported
    
    Examples:
        >>> to_datetime(1700000000.0)
        datetime.datetime(2023, 11, 14, 16, 13, 20)
        >>> to_datetime("2023-11-14T16:13:20")
        datetime.datetime(2023, 11, 14, 16, 13, 20)
    """
    if value is None:
        return None
    
    if isinstance(value, datetime):
        return value
    
    if isinstance(value, (int, float)):
        # Handle milliseconds (13+ digits)
        if value > 1000000000000:  # Year 2000 is ~946684800000
            value = value / 1000.0
        return datetime.fromtimestamp(value)
    
    if isinstance(value, str):
        # Try common formats
        formats = [
            "%Y-%m-%d %H:%M:%S",
            "%Y-%m-%dT%H:%M:%S",
            "%Y-%m-%d %H:%M:%S.%f",
            "%Y-%m-%dT%H:%M:%S.%f",
            "%Y-%m-%d",
            "%Y%m%d",
        ]
        for fmt in formats:
            try:
                return datetime.strptime(value, fmt)
            except ValueError:
                continue
        
        # Try parsing as timestamp string
        try:
            return datetime.fromtimestamp(float(value))
        except (ValueError, TypeError):
            pass
        
        raise ValueError(f"Unable to parse datetime string: {value}")
    
    raise TypeError(f"Unsupported type for datetime conversion: {type(value)}")


def to_timestamp(dt: Optional[datetime]) -> Optional[float]:
    """
    Convert datetime to Unix timestamp.
    
    Args:
        dt: datetime object or None
        
    Returns:
        Unix timestamp as float, or None if dt is None
        
    Examples:
        >>> to_timestamp(datetime(2023, 11, 14, 16, 13, 20))
        1700000000.0
    """
    if dt is None:
        return None
    if isinstance(dt, datetime):
        return dt.timestamp()
    raise TypeError(f"Expected datetime, got {type(dt)}")


def format_datetime(
    dt: Optional[datetime],
    format_str: str = "%Y-%m-%d %H:%M:%S",
) -> Optional[str]:
    """
    Format datetime as string.
    
    Args:
        dt: datetime object or None
        format_str: Format string (default: "%Y-%m-%d %H:%M:%S")
        
    Returns:
        Formatted string or None if dt is None
        
    Examples:
        >>> format_datetime(datetime(2023, 11, 14, 16, 13, 20))
        '2023-11-14 16:13:20'
    """
    if dt is None:
        return None
    return dt.strftime(format_str)


def parse_timeframe_to_seconds(timeframe: str) -> Optional[int]:
    """
    Parse timeframe string to seconds.
    
    Args:
        timeframe: Timeframe string (e.g., "M1", "H1", "D1")
        
    Returns:
        Number of seconds, or None if invalid
        
    Examples:
        >>> parse_timeframe_to_seconds("H1")
        3600
        >>> parse_timeframe_to_seconds("D1")
        86400
    """
    return TIMEFRAME_MAP.get(timeframe)


def get_timeframe_label(seconds: int) -> Optional[str]:
    """
    Get timeframe label from seconds.
    
    Args:
        seconds: Number of seconds
        
    Returns:
        Timeframe string, or None if not found
        
    Examples:
        >>> get_timeframe_label(3600)
        'H1'
        >>> get_timeframe_label(86400)
        'D1'
    """
    reverse_map = {v: k for k, v in TIMEFRAME_MAP.items()}
    return reverse_map.get(seconds)


def get_timeframe_multiplier(timeframe: str) -> Optional[int]:
    """
    Get the multiplier for a timeframe relative to M1.
    
    Args:
        timeframe: Timeframe string
        
    Returns:
        Multiplier (e.g., H1 = 60, D1 = 1440), or None if invalid
    """
    seconds = parse_timeframe_to_seconds(timeframe)
    if seconds is None:
        return None
    return seconds // 60


# ==============================================================================
# RETRY UTILITIES
# ==============================================================================

def exponential_backoff(attempt: int, base_delay: float = 1.0) -> float:
    """
    Calculate exponential backoff delay.
    
    Args:
        attempt: Attempt number (0-indexed)
        base_delay: Base delay in seconds
        
    Returns:
        Delay in seconds
        
    Examples:
        >>> exponential_backoff(0)
        1.0
        >>> exponential_backoff(1)
        2.0
        >>> exponential_backoff(2)
        4.0
    """
    return base_delay * (2 ** attempt)


def retry(
    max_attempts: int = 3,
    delay: float = 1.0,
    backoff: float = 2.0,
    exceptions: Tuple[Exception, ...] = (Exception,),
    on_retry: Optional[Callable[[int, Exception], None]] = None,
) -> Callable:
    """
    Retry decorator with exponential backoff.
    
    Args:
        max_attempts: Maximum number of attempts
        delay: Initial delay in seconds
        backoff: Backoff multiplier
        exceptions: Tuple of exceptions to retry on
        on_retry: Callback called on each retry (attempt, exception)
        
    Returns:
        Decorated function
        
    Examples:
        @retry(max_attempts=3, delay=1.0)
        def unreliable_function():
            # May raise exception
            pass
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None
            
            for attempt in range(max_attempts):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    
                    if on_retry:
                        on_retry(attempt + 1, e)
                    
                    if attempt < max_attempts - 1:
                        wait_time = exponential_backoff(attempt, delay)
                        time.sleep(wait_time)
            
            raise last_exception
        
        return wrapper
    return decorator


def retry_async(
    max_attempts: int = 3,
    delay: float = 1.0,
    backoff: float = 2.0,
    exceptions: Tuple[Exception, ...] = (Exception,),
    on_retry: Optional[Callable[[int, Exception], None]] = None,
) -> Callable:
    """
    Async retry decorator with exponential backoff.
    
    Args:
        max_attempts: Maximum number of attempts
        delay: Initial delay in seconds
        backoff: Backoff multiplier
        exceptions: Tuple of exceptions to retry on
        on_retry: Callback called on each retry (attempt, exception)
        
    Returns:
        Decorated async function
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(*args, **kwargs):
            last_exception = None
            
            for attempt in range(max_attempts):
                try:
                    return await func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    
                    if on_retry:
                        on_retry(attempt + 1, e)
                    
                    if attempt < max_attempts - 1:
                        wait_time = exponential_backoff(attempt, delay)
                        await asyncio.sleep(wait_time)
            
            raise last_exception
        
        return wrapper
    return decorator


# ==============================================================================
# VALIDATION UTILITIES
# ==============================================================================

def is_valid_symbol(symbol: str) -> bool:
    """
    Validate a market symbol.
    
    Args:
        symbol: Symbol string to validate
        
    Returns:
        True if valid, False otherwise
        
    Examples:
        >>> is_valid_symbol("EURUSD")
        True
        >>> is_valid_symbol("BTCUSD")
        True
        >>> is_valid_symbol("invalid")
        False
    """
    if not symbol or not isinstance(symbol, str):
        return False
    
    # Basic length check
    if len(symbol) < 2 or len(symbol) > 20:
        return False
    
    # Pattern match
    return bool(SYMBOL_PATTERN.match(symbol))


def is_valid_timeframe(timeframe: str) -> bool:
    """
    Validate a timeframe string.
    
    Args:
        timeframe: Timeframe string to validate
        
    Returns:
        True if valid, False otherwise
        
    Examples:
        >>> is_valid_timeframe("H1")
        True
        >>> is_valid_timeframe("M5")
        True
        >>> is_valid_timeframe("invalid")
        False
    """
    return timeframe in VALID_TIMEFRAMES


def validate_candle(candle: Dict[str, Any]) -> bool:
    """
    Validate a candle dictionary has required fields.
    
    Args:
        candle: Candle dictionary with OHLCV data
        
    Returns:
        True if valid, False otherwise
        
    Examples:
        >>> validate_candle({'open': 1.0, 'high': 1.1, 'low': 0.9, 'close': 1.05})
        True
        >>> validate_candle({'open': 1.0, 'high': 0.9, 'low': 1.1, 'close': 1.05})
        False  # high < low
    """
    if not candle or not isinstance(candle, dict):
        return False
    
    # Check required fields
    if not all(field in candle for field in CANDLE_REQUIRED_FIELDS):
        return False
    
    try:
        open_price = float(candle['open'])
        high = float(candle['high'])
        low = float(candle['low'])
        close = float(candle['close'])
        
        # Basic price validation
        if high < low:
            return False
        if open_price <= 0 or high <= 0 or low <= 0 or close <= 0:
            return False
        
        # Volume (optional but should be >= 0 if present)
        if 'volume' in candle:
            volume = candle['volume']
            if not isinstance(volume, (int, float)) or volume < 0:
                return False
        
        return True
        
    except (ValueError, TypeError):
        return False


def validate_candles(candles: List[Dict[str, Any]]) -> bool:
    """
    Validate a list of candles.
    
    Args:
        candles: List of candle dictionaries
        
    Returns:
        True if all valid, False otherwise
    """
    if not candles:
        return False
    
    return all(validate_candle(c) for c in candles)


def validate_symbol_name(symbol: str) -> Tuple[bool, Optional[str]]:
    """
    Validate symbol name with detailed error message.
    
    Args:
        symbol: Symbol string to validate
        
    Returns:
        Tuple of (is_valid, error_message)
        
    Examples:
        >>> validate_symbol_name("EURUSD")
        (True, None)
        >>> validate_symbol_name("")
        (False, "Symbol cannot be empty")
    """
    if not symbol:
        return False, "Symbol cannot be empty"
    
    if not isinstance(symbol, str):
        return False, "Symbol must be a string"
    
    if len(symbol) < 2:
        return False, f"Symbol '{symbol}' is too short (min 2 characters)"
    
    if len(symbol) > 20:
        return False, f"Symbol '{symbol}' is too long (max 20 characters)"
    
    if not SYMBOL_PATTERN.match(symbol):
        return False, f"Symbol '{symbol}' contains invalid characters"
    
    return True, None


# ==============================================================================
# BATCH UTILITIES
# ==============================================================================

def batch_iterator(items: List[Any], batch_size: int) -> Iterator[List[Any]]:
    """
    Iterate over items in batches.
    
    Args:
        items: List of items to batch
        batch_size: Size of each batch
        
    Yields:
        Batches as lists
        
    Examples:
        >>> list(batch_iterator([1,2,3,4,5], 2))
        [[1, 2], [3, 4], [5]]
    """
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    
    for i in range(0, len(items), batch_size):
        yield items[i:i + batch_size]


def chunk_list(lst: List[Any], chunk_size: int) -> List[List[Any]]:
    """
    Split list into chunks.
    
    Args:
        lst: List to split
        chunk_size: Size of each chunk
        
    Returns:
        List of chunks
        
    Examples:
        >>> chunk_list([1,2,3,4,5], 2)
        [[1, 2], [3, 4], [5]]
    """
    return list(batch_iterator(lst, chunk_size))


def batch_process(
    items: List[Any],
    process_func: Callable[[Any], Any],
    batch_size: int,
) -> List[Any]:
    """
    Process items in batches.
    
    Args:
        items: List of items to process
        process_func: Function to apply to each item
        batch_size: Size of each batch
        
    Returns:
        List of results in original order
    """
    results = []
    for batch in batch_iterator(items, batch_size):
        batch_results = [process_func(item) for item in batch]
        results.extend(batch_results)
    return results


# ==============================================================================
# METRIC UTILITIES
# ==============================================================================

def format_duration(seconds: float) -> str:
    """
    Format duration as human-readable string.
    
    Args:
        seconds: Duration in seconds
        
    Returns:
        Formatted string (e.g., "1h 23m 45s")
        
    Examples:
        >>> format_duration(3665)
        '1h 1m 5s'
        >>> format_duration(65)
        '1m 5s'
        >>> format_duration(5)
        '5s'
    """
    if seconds < 0:
        return f"-{format_duration(abs(seconds))}"
    
    if seconds < 1:
        return f"{seconds * 1000:.0f}ms"
    
    minutes, secs = divmod(int(seconds), 60)
    hours, minutes = divmod(minutes, 60)
    days, hours = divmod(hours, 24)
    
    parts = []
    if days:
        parts.append(f"{days}d")
    if hours:
        parts.append(f"{hours}h")
    if minutes:
        parts.append(f"{minutes}m")
    if secs:
        parts.append(f"{secs}s")
    
    return " ".join(parts) if parts else "0s"


def format_bytes(bytes_count: int) -> str:
    """
    Format bytes as human-readable string.
    
    Args:
        bytes_count: Number of bytes
        
    Returns:
        Formatted string (e.g., "1.5 MB")
        
    Examples:
        >>> format_bytes(1500000)
        '1.43 MB'
        >>> format_bytes(500)
        '500 B'
    """
    if bytes_count < 0:
        return f"-{format_bytes(abs(bytes_count))}"
    
    if bytes_count < 1024:
        return f"{bytes_count} B"
    
    units = ["KB", "MB", "GB", "TB", "PB"]
    value = bytes_count / 1024.0
    
    for unit in units:
        if value < 1024.0:
            return f"{value:.2f} {unit}"
        value /= 1024.0
    
    return f"{value:.2f} PB"


def calculate_percentage(value: float, total: float) -> float:
    """
    Calculate percentage.
    
    Args:
        value: Current value
        total: Total value
        
    Returns:
        Percentage (0-100), or 0.0 if total is 0
        
    Examples:
        >>> calculate_percentage(25, 100)
        25.0
        >>> calculate_percentage(0, 100)
        0.0
    """
    if total == 0:
        return 0.0
    return (value / total) * 100


def round_price(value: float, digits: int = 5) -> float:
    """
    Round price to specific number of decimal places.
    
    Args:
        value: Price value
        digits: Number of decimal places
        
    Returns:
        Rounded price
        
    Examples:
        >>> round_price(1.234567, 5)
        1.23457
    """
    if digits < 0:
        return value
    
    return Decimal(str(value)).quantize(
        Decimal("0." + "0" * digits),
        rounding=ROUND_HALF_UP
    )


# ==============================================================================
# STRING UTILITIES
# ==============================================================================

def truncate_string(text: str, max_length: int = 100, suffix: str = "...") -> str:
    """
    Truncate a string to maximum length.
    
    Args:
        text: String to truncate
        max_length: Maximum length
        suffix: Suffix to add (default: "...")
        
    Returns:
        Truncated string
    """
    if not text or len(text) <= max_length:
        return text
    
    trunc_len = max_length - len(suffix)
    if trunc_len <= 0:
        return text[:max_length]
    
    return text[:trunc_len] + suffix


def to_snake_case(text: str) -> str:
    """
    Convert text to snake_case.
    
    Args:
        text: String to convert
        
    Returns:
        snake_case string
        
    Examples:
        >>> to_snake_case("HelloWorld")
        'hello_world'
        >>> to_snake_case("Hello World")
        'hello_world'
    """
    # Replace spaces and hyphens with underscores
    text = re.sub(r'[\s-]+', '_', text)
    
    # Convert CamelCase to snake_case
    text = re.sub(r'(?<=[a-z])(?=[A-Z])', '_', text)
    text = re.sub(r'(?<=[A-Z])(?=[A-Z][a-z])', '_', text)
    
    return text.lower()


# ==============================================================================
# DICTIONARY UTILITIES
# ==============================================================================

def safe_get(dictionary: Dict, key: str, default: Any = None) -> Any:
    """
    Safe dictionary access with dot notation support.
    
    Args:
        dictionary: Dict to access
        key: Key to access (supports dot notation)
        default: Default value if not found
        
    Returns:
        Value or default
        
    Examples:
        >>> safe_get({'a': {'b': 1}}, 'a.b')
        1
        >>> safe_get({'a': {'b': 1}}, 'a.c', 0)
        0
    """
    if not dictionary or not key:
        return default
    
    parts = key.split('.')
    current = dictionary
    
    for part in parts:
        if not isinstance(current, dict):
            return default
        if part not in current:
            return default
        current = current[part]
    
    return current


def deep_merge(dict1: Dict, dict2: Dict) -> Dict:
    """
    Deep merge two dictionaries.
    
    Args:
        dict1: First dictionary (base)
        dict2: Second dictionary (override)
        
    Returns:
        Merged dictionary
        
    Examples:
        >>> deep_merge({'a': 1, 'b': {'c': 2}}, {'b': {'d': 3}})
        {'a': 1, 'b': {'c': 2, 'd': 3}}
    """
    result = dict1.copy()
    
    for key, value in dict2.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = value
    
    return result


# ==============================================================================
# RATE LIMITING UTILITIES
# ==============================================================================

class RateLimiter:
    """
    Simple rate limiter for controlling operation frequency.
    
    Examples:
        >>> limiter = RateLimiter(max_calls=10, period=60)
        >>> if limiter.allow():
        ...     # Do something
    """
    
    def __init__(self, max_calls: int, period: float):
        """
        Initialize rate limiter.
        
        Args:
            max_calls: Maximum number of calls in the period
            period: Time period in seconds
        """
        self.max_calls = max_calls
        self.period = period
        self.calls: List[float] = []
    
    def allow(self) -> bool:
        """
        Check if a call is allowed.
        
        Returns:
            True if call is allowed, False otherwise
        """
        now = time.time()
        self.calls = [t for t in self.calls if now - t < self.period]
        
        if len(self.calls) < self.max_calls:
            self.calls.append(now)
            return True
        
        return False
    
    def remaining(self) -> int:
        """
        Get remaining calls in the current period.
        
        Returns:
            Number of remaining calls
        """
        now = time.time()
        self.calls = [t for t in self.calls if now - t < self.period]
        return max(0, self.max_calls - len(self.calls))
    
    def reset(self) -> None:
        """Reset the rate limiter."""
        self.calls.clear()


# ==============================================================================
# TIME UTILITIES
# ==============================================================================

def get_current_unix_timestamp() -> int:
    """
    Get current Unix timestamp as integer.
    
    Returns:
        Current Unix timestamp in seconds
    """
    return int(time.time())


def get_current_datetime() -> datetime:
    """
    Get current datetime with timezone awareness.
    
    Returns:
        Current datetime
    """
    return datetime.now()


def get_date_range(
    start: datetime,
    end: datetime,
    interval_seconds: int,
) -> List[datetime]:
    """
    Generate a list of datetimes between start and end at intervals.
    
    Args:
        start: Start datetime
        end: End datetime
        interval_seconds: Interval in seconds
        
    Returns:
        List of datetime objects
    """
    if start > end:
        start, end = end, start
    
    dates = []
    current = start
    
    while current <= end:
        dates.append(current)
        current += timedelta(seconds=interval_seconds)
    
    return dates


# ==============================================================================
# ID GENERATION
# ==============================================================================

def generate_id(prefix: str = "", length: int = 8) -> str:
    """
    Generate a unique ID string.
    
    Args:
        prefix: Optional prefix
        length: Length of the random part
        
    Returns:
        Unique ID string
        
    Examples:
        >>> generate_id("task", 6)
        'task_a1b2c3'
    """
    import random
    import string
    
    chars = string.ascii_lowercase + string.digits
    random_part = ''.join(random.choice(chars) for _ in range(length))
    
    if prefix:
        return f"{prefix}_{random_part}"
    return random_part


# ==============================================================================
# IMPORT EXCEPTIONS FOR ASYNC
# ==============================================================================

# Lazy import for asyncio (not always needed)
try:
    import asyncio
    HAS_ASYNCIO = True
except ImportError:
    HAS_ASYNCIO = False


# ==============================================================================
# EXPORTS
# ==============================================================================

__all__ = [
    # Timestamp utilities
    'to_datetime',
    'to_timestamp',
    'format_datetime',
    'parse_timeframe_to_seconds',
    'get_timeframe_label',
    'get_timeframe_multiplier',
    # Retry utilities
    'exponential_backoff',
    'retry',
    'retry_async',
    # Validation utilities
    'is_valid_symbol',
    'is_valid_timeframe',
    'validate_candle',
    'validate_candles',
    'validate_symbol_name',
    # Batch utilities
    'batch_iterator',
    'chunk_list',
    'batch_process',
    # Metric utilities
    'format_duration',
    'format_bytes',
    'calculate_percentage',
    'round_price',
    # String utilities
    'truncate_string',
    'to_snake_case',
    # Dictionary utilities
    'safe_get',
    'deep_merge',
    # Rate limiting
    'RateLimiter',
    # Time utilities
    'get_current_unix_timestamp',
    'get_current_datetime',
    'get_date_range',
    # ID generation
    'generate_id',
    # Constants
    'VALID_TIMEFRAMES',
    'TIMEFRAME_MAP',
]