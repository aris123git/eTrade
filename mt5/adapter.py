"""
mt5/adapter.py - Data Transformation Adapter for MT5

RESPONSIBILITY:
Transform MT5's raw data into clean, typed, predictable Python objects.

ARCHITECTURAL PRINCIPLES:
1. Pure data transformation - No side effects, no I/O, no business logic
2. Validation at boundaries - Ensure data quality before passing inward
3. Type safety - All conversions preserve type information
4. Cache optimization - Symbol data cached for performance
5. SOLID compliance - Single responsibility, open/closed

WHAT IT NEVER DOES:
- ❌ Connect to MT5
- ❌ Download data
- ❌ Discover symbols
- ❌ Store data
- ❌ Analyze data
- ❌ Make decisions

VERSION: 1.0.1
"""

import logging
from datetime import datetime, timezone
from typing import List, Optional, Dict, Any, Tuple, Union
from dataclasses import dataclass, field

from core.config import Config
from core.exceptions import MarketAIError


# ==============================================================================
# EXCEPTIONS
# ==============================================================================

class AdapterError(MarketAIError):
    """Base exception for adapter errors."""
    error_code = "ADAPTER_ERROR"


class ValidationError(AdapterError):
    """Raised when data validation fails."""
    error_code = "VALIDATION_ERROR"


class NormalizationError(AdapterError):
    """Raised when data normalization fails."""
    error_code = "NORMALIZATION_ERROR"


# ==============================================================================
# DATA MODELS (Immutable)
# ==============================================================================

@dataclass(frozen=True)
class Symbol:
    """Normalized symbol information."""
    symbol: str
    description: str
    digits: int
    tick_size: float
    tick_value: float
    min_volume: float
    max_volume: float
    volume_step: float
    trade_mode: int
    trade_calc_mode: int
    currency: str
    margin_currency: str
    currency_base: str
    currency_profit: str
    path: str
    spread: int
    spread_float: bool
    point: float
    timezone: str


@dataclass(frozen=True)
class Candle:
    """Normalized candle data."""
    symbol: str
    timeframe: str
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: int
    spread: Optional[int] = None
    real_volume: Optional[int] = None
    
    def __post_init__(self):
        """Validate candle data after initialization."""
        if self.high < self.low:
            raise ValidationError(
                f"High ({self.high}) < Low ({self.low}) in candle for {self.symbol} "
                f"at {self.timestamp}"
            )
        if self.open <= 0 or self.high <= 0 or self.low <= 0 or self.close <= 0:
            raise ValidationError(
                f"Invalid price values in candle for {self.symbol} at {self.timestamp}: "
                f"O={self.open}, H={self.high}, L={self.low}, C={self.close}"
            )
        # Validate that high is >= both open and close, low is <= both
        if self.high < self.open or self.high < self.close:
            raise ValidationError(
                f"High ({self.high}) must be >= Open ({self.open}) and Close ({self.close})"
            )
        if self.low > self.open or self.low > self.close:
            raise ValidationError(
                f"Low ({self.low}) must be <= Open ({self.open}) and Close ({self.close})"
            )


@dataclass(frozen=True)
class Order:
    """Normalized order data."""
    ticket: int
    symbol: str
    order_type: str
    volume: float
    price_open: float
    price_current: float
    price_stop_loss: Optional[float]
    price_take_profit: Optional[float]
    state: str
    timestamp_open: datetime
    comment: str
    magic: int


@dataclass(frozen=True)
class Position:
    """Normalized position data."""
    ticket: int
    symbol: str
    position_type: str
    volume: float
    price_open: float
    price_current: float
    price_stop_loss: Optional[float]
    price_take_profit: Optional[float]
    profit: float
    swap: float
    timestamp_open: datetime
    comment: str
    magic: int


@dataclass(frozen=True)
class Account:
    """Normalized account data."""
    login: int
    balance: float
    equity: float
    margin: float
    margin_free: float
    margin_level: float
    profit: float
    currency: str
    leverage: int
    server: str


# ==============================================================================
# CONSTANTS
# ==============================================================================

# MT5 Timeframe mapping (MT5 constant → string)
MT5_TIMEFRAMES: Dict[int, str] = {
    1: "M1",
    5: "M5",
    15: "M15",
    30: "M30",
    60: "H1",
    240: "H4",
    1440: "D1",
    10080: "W1",
    43200: "MN1",
}

# Reverse mapping (string → MT5 constant)
MT5_TIMEFRAMES_REVERSE: Dict[str, int] = {v: k for k, v in MT5_TIMEFRAMES.items()}

# MT5 Order types
MT5_ORDER_TYPES: Dict[int, str] = {
    0: "buy",
    1: "sell",
    2: "buy_limit",
    3: "sell_limit",
    4: "buy_stop",
    5: "sell_stop",
}

# MT5 Position types
MT5_POSITION_TYPES: Dict[int, str] = {
    0: "buy",
    1: "sell",
}

# MT5 Order states
MT5_ORDER_STATES: Dict[int, str] = {
    0: "started",
    1: "placed",
    2: "canceled",
    3: "partial",
    4: "filled",
    5: "rejected",
    6: "expired",
    7: "removed",
}

# MT5 Position states
MT5_POSITION_STATES: Dict[int, str] = {
    0: "opened",
    1: "closed",
    2: "partially_closed",
}


# ==============================================================================
# MAIN ADAPTER CLASS
# ==============================================================================

class MT5Adapter:
    """
    Data transformation adapter for MT5.
    
    Converts raw MT5 data structures to clean, typed Python objects.
    Pure transformation - no side effects, no I/O, no business logic.
    
    Attributes:
        config: Application configuration
        logger: Logger instance
        _symbol_cache: Cache for Symbol objects by symbol name
        _timezone: Timezone for datetime conversion (from config)
    """
    
    def __init__(self, config: Config):
        """
        Initialize the MT5 adapter.
        
        Args:
            config: Application configuration
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        self._symbol_cache: Dict[str, Symbol] = {}
        # Timezone from config, default to UTC
        self._timezone = getattr(config, 'MT5_TIMEZONE', 'UTC')
    
    # ==========================================================================
    # SYMBOL CONVERSION
    # ==========================================================================
    
    def to_symbol(self, mt5_symbol: Any) -> Symbol:
        """
        Convert MT5 symbol object to normalized Symbol.
        
        Args:
            mt5_symbol: MT5 symbol object from mt5.symbol_info()
            
        Returns:
            Normalized Symbol dataclass
            
        Raises:
            ValidationError: If symbol data is invalid
            NormalizationError: If conversion fails
        """
        self.logger.debug(f"Converting symbol: {getattr(mt5_symbol, 'name', 'unknown')}")
        
        try:
            symbol = Symbol(
                symbol=mt5_symbol.name,
                description=getattr(mt5_symbol, 'description', mt5_symbol.name),
                digits=mt5_symbol.digits,
                tick_size=mt5_symbol.trade_tick_size or 0.0,
                tick_value=mt5_symbol.trade_tick_value or 0.0,
                min_volume=mt5_symbol.volume_min or 0.0,
                max_volume=mt5_symbol.volume_max or 0.0,
                volume_step=mt5_symbol.volume_step or 0.0,
                trade_mode=mt5_symbol.trade_mode or 0,
                trade_calc_mode=mt5_symbol.trade_calc_mode or 0,
                currency=getattr(mt5_symbol, 'currency', ''),
                margin_currency=getattr(mt5_symbol, 'currency_margin', ''),
                currency_base=getattr(mt5_symbol, 'currency_base', ''),
                currency_profit=getattr(mt5_symbol, 'currency_profit', ''),
                path=getattr(mt5_symbol, 'path', ''),
                spread=mt5_symbol.spread or 0,
                spread_float=mt5_symbol.spread_float or False,
                point=mt5_symbol.point or 0.0,
                timezone=self._timezone,
            )
            
            # Cache the symbol
            self._symbol_cache[symbol.symbol] = symbol
            
            self.logger.debug(f"✅ Symbol converted: {symbol.symbol}")
            return symbol
            
        except AttributeError as e:
            raise NormalizationError(f"Missing attribute in MT5 symbol: {e}")
        except Exception as e:
            raise NormalizationError(f"Failed to convert symbol: {e}")
    
    def to_symbols(self, mt5_symbols: List[Any]) -> List[Symbol]:
        """
        Bulk convert MT5 symbols to normalized Symbols.
        
        Args:
            mt5_symbols: List of MT5 symbol objects
            
        Returns:
            List of normalized Symbol dataclasses
        """
        if not mt5_symbols:
            return []
        
        self.logger.debug(f"Converting {len(mt5_symbols)} symbols")
        
        symbols = []
        errors = 0
        
        for mt5_symbol in mt5_symbols:
            try:
                symbols.append(self.to_symbol(mt5_symbol))
            except Exception as e:
                errors += 1
                self.logger.warning(
                    f"⚠️ Failed to convert symbol {getattr(mt5_symbol, 'name', 'unknown')}: {e}"
                )
        
        if errors > 0:
            self.logger.warning(f"⚠️ {errors} symbols failed conversion out of {len(mt5_symbols)}")
        
        self.logger.debug(f"✅ Converted {len(symbols)} symbols")
        return symbols
    
    def get_cached_symbol(self, symbol: str) -> Optional[Symbol]:
        """
        Get a cached symbol by name.
        
        Args:
            symbol: Symbol name
            
        Returns:
            Cached Symbol or None
        """
        return self._symbol_cache.get(symbol)
    
    def clear_cache(self) -> None:
        """Clear the symbol cache."""
        self._symbol_cache.clear()
        self.logger.debug("Symbol cache cleared")
    
    def cache_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics.
        
        Returns:
            Dictionary with cache statistics
        """
        return {
            'cache_size': len(self._symbol_cache),
            'cached_symbols': list(self._symbol_cache.keys()),
        }
    
    # ==========================================================================
    # CANDLE CONVERSION
    # ==========================================================================
    
    def to_candle(
        self,
        mt5_rate: Tuple,
        symbol: str,
        timeframe: str,
    ) -> Candle:
        """
        Convert MT5 rate tuple to normalized Candle.
        
        MT5 rate tuple format: (time, open, high, low, close, tick_volume, spread, real_volume)
        
        Args:
            mt5_rate: Tuple from mt5.copy_rates_*()
            symbol: Symbol name
            timeframe: Timeframe string (e.g., "H1")
            
        Returns:
            Normalized Candle dataclass
            
        Raises:
            ValidationError: If candle data is invalid
            NormalizationError: If conversion fails
        """
        self.logger.debug(f"Converting candle: {symbol} {timeframe}")
        
        try:
            # Validate input
            if not mt5_rate or len(mt5_rate) < 5:
                raise ValidationError(f"MT5 rate tuple has {len(mt5_rate) if mt5_rate else 0} fields, expected at least 5")
            
            # Extract values
            timestamp = self._normalize_datetime(mt5_rate[0])
            open_price = float(mt5_rate[1])
            high = float(mt5_rate[2])
            low = float(mt5_rate[3])
            close = float(mt5_rate[4])
            volume = int(mt5_rate[5]) if len(mt5_rate) > 5 else 0
            spread = int(mt5_rate[6]) if len(mt5_rate) > 6 else None
            real_volume = int(mt5_rate[7]) if len(mt5_rate) > 7 else None
            
            # Validate OHLC logic (handled by Candle.__post_init__)
            candle = Candle(
                symbol=symbol,
                timeframe=timeframe,
                timestamp=timestamp,
                open=open_price,
                high=high,
                low=low,
                close=close,
                volume=volume,
                spread=spread,
                real_volume=real_volume,
            )
            
            self.logger.debug(f"✅ Candle converted: {symbol} {timeframe} at {timestamp}")
            return candle
            
        except ValidationError:
            raise
        except Exception as e:
            raise NormalizationError(f"Failed to convert candle for {symbol} {timeframe}: {e}")
    
    def to_candles(
        self,
        mt5_rates: List[Tuple],
        symbol: str,
        timeframe: str,
    ) -> List[Candle]:
        """
        Bulk convert MT5 rates to normalized Candles.
        
        Args:
            mt5_rates: List of MT5 rate tuples
            symbol: Symbol name
            timeframe: Timeframe string
            
        Returns:
            List of normalized Candle dataclasses
        """
        if not mt5_rates:
            return []
        
        self.logger.debug(f"Converting {len(mt5_rates)} candles: {symbol} {timeframe}")
        
        candles = []
        errors = 0
        
        for rate in mt5_rates:
            try:
                candles.append(self.to_candle(rate, symbol, timeframe))
            except Exception as e:
                errors += 1
                self.logger.warning(f"⚠️ Failed to convert candle: {e}")
        
        if errors > 0:
            self.logger.warning(f"⚠️ {errors} candles failed conversion out of {len(mt5_rates)}")
        
        self.logger.debug(f"✅ Converted {len(candles)} candles")
        return candles
    
    # ==========================================================================
    # ORDER CONVERSION
    # ==========================================================================
    
    def to_order(self, mt5_order: Any) -> Order:
        """
        Convert MT5 order object to normalized Order.
        
        Args:
            mt5_order: MT5 order object from mt5.orders_get()
            
        Returns:
            Normalized Order dataclass
            
        Raises:
            NormalizationError: If conversion fails
        """
        self.logger.debug(f"Converting order: {getattr(mt5_order, 'ticket', 'unknown')}")
        
        try:
            order_type = MT5_ORDER_TYPES.get(mt5_order.type, "unknown")
            state = MT5_ORDER_STATES.get(mt5_order.state, "unknown")
            
            order = Order(
                ticket=mt5_order.ticket,
                symbol=mt5_order.symbol,
                order_type=order_type,
                volume=float(mt5_order.volume_initial or 0.0),
                price_open=float(mt5_order.price_open or 0.0),
                price_current=float(mt5_order.price_current or 0.0),
                price_stop_loss=float(mt5_order.sl) if mt5_order.sl else None,
                price_take_profit=float(mt5_order.tp) if mt5_order.tp else None,
                state=state,
                timestamp_open=self._normalize_datetime(mt5_order.time_setup),
                comment=getattr(mt5_order, 'comment', ''),
                magic=mt5_order.magic or 0,
            )
            
            self.logger.debug(f"✅ Order converted: {order.ticket}")
            return order
            
        except AttributeError as e:
            raise NormalizationError(f"Missing attribute in MT5 order: {e}")
        except Exception as e:
            raise NormalizationError(f"Failed to convert order: {e}")
    
    def to_orders(self, mt5_orders: List[Any]) -> List[Order]:
        """
        Bulk convert MT5 orders to normalized Orders.
        
        Args:
            mt5_orders: List of MT5 order objects
            
        Returns:
            List of normalized Order dataclasses
        """
        if not mt5_orders:
            return []
        
        self.logger.debug(f"Converting {len(mt5_orders)} orders")
        
        orders = []
        errors = 0
        
        for mt5_order in mt5_orders:
            try:
                orders.append(self.to_order(mt5_order))
            except Exception as e:
                errors += 1
                self.logger.warning(f"⚠️ Failed to convert order: {e}")
        
        if errors > 0:
            self.logger.warning(f"⚠️ {errors} orders failed conversion out of {len(mt5_orders)}")
        
        return orders
    
    # ==========================================================================
    # POSITION CONVERSION
    # ==========================================================================
    
    def to_position(self, mt5_position: Any) -> Position:
        """
        Convert MT5 position object to normalized Position.
        
        Args:
            mt5_position: MT5 position object from mt5.positions_get()
            
        Returns:
            Normalized Position dataclass
            
        Raises:
            NormalizationError: If conversion fails
        """
        self.logger.debug(f"Converting position: {getattr(mt5_position, 'ticket', 'unknown')}")
        
        try:
            position_type = MT5_POSITION_TYPES.get(mt5_position.type, "unknown")
            
            position = Position(
                ticket=mt5_position.ticket,
                symbol=mt5_position.symbol,
                position_type=position_type,
                volume=float(mt5_position.volume or 0.0),
                price_open=float(mt5_position.price_open or 0.0),
                price_current=float(mt5_position.price_current or 0.0),
                price_stop_loss=float(mt5_position.sl) if mt5_position.sl else None,
                price_take_profit=float(mt5_position.tp) if mt5_position.tp else None,
                profit=float(mt5_position.profit or 0.0),
                swap=float(mt5_position.swap or 0.0),
                timestamp_open=self._normalize_datetime(mt5_position.time),
                comment=getattr(mt5_position, 'comment', ''),
                magic=mt5_position.magic or 0,
            )
            
            self.logger.debug(f"✅ Position converted: {position.ticket}")
            return position
            
        except AttributeError as e:
            raise NormalizationError(f"Missing attribute in MT5 position: {e}")
        except Exception as e:
            raise NormalizationError(f"Failed to convert position: {e}")
    
    def to_positions(self, mt5_positions: List[Any]) -> List[Position]:
        """
        Bulk convert MT5 positions to normalized Positions.
        
        Args:
            mt5_positions: List of MT5 position objects
            
        Returns:
            List of normalized Position dataclasses
        """
        if not mt5_positions:
            return []
        
        self.logger.debug(f"Converting {len(mt5_positions)} positions")
        
        positions = []
        errors = 0
        
        for mt5_position in mt5_positions:
            try:
                positions.append(self.to_position(mt5_position))
            except Exception as e:
                errors += 1
                self.logger.warning(f"⚠️ Failed to convert position: {e}")
        
        if errors > 0:
            self.logger.warning(f"⚠️ {errors} positions failed conversion out of {len(mt5_positions)}")
        
        return positions
    
    # ==========================================================================
    # ACCOUNT CONVERSION
    # ==========================================================================
    
    def to_account(self, mt5_account: Any) -> Account:
        """
        Convert MT5 account object to normalized Account.
        
        Args:
            mt5_account: MT5 account object from mt5.account_info()
            
        Returns:
            Normalized Account dataclass
            
        Raises:
            NormalizationError: If conversion fails
        """
        self.logger.debug(f"Converting account: {getattr(mt5_account, 'login', 'unknown')}")
        
        try:
            margin_level = 0.0
            if mt5_account.margin and mt5_account.margin > 0:
                margin_level = (mt5_account.equity / mt5_account.margin) * 100.0
            
            account = Account(
                login=mt5_account.login,
                balance=float(mt5_account.balance or 0.0),
                equity=float(mt5_account.equity or 0.0),
                margin=float(mt5_account.margin or 0.0),
                margin_free=float(mt5_account.margin_free or 0.0),
                margin_level=float(margin_level),
                profit=float(mt5_account.profit or 0.0),
                currency=getattr(mt5_account, 'currency', 'USD'),
                leverage=mt5_account.leverage or 0,
                server=getattr(mt5_account, 'server', 'unknown'),
            )
            
            self.logger.debug(f"✅ Account converted: {account.login}")
            return account
            
        except AttributeError as e:
            raise NormalizationError(f"Missing attribute in MT5 account: {e}")
        except Exception as e:
            raise NormalizationError(f"Failed to convert account: {e}")
    
    # ==========================================================================
    # INTERNAL UTILITIES
    # ==========================================================================
    
    def _normalize_timeframe(self, mt5_timeframe: int) -> str:
        """
        Convert MT5 timeframe constant to string.
        
        Args:
            mt5_timeframe: MT5 timeframe constant (e.g., 16385 for H1)
            
        Returns:
            Timeframe string (e.g., "H1")
            
        Raises:
            NormalizationError: If timeframe is unknown
        """
        timeframe = MT5_TIMEFRAMES.get(mt5_timeframe)
        if timeframe is None:
            raise NormalizationError(f"Unknown MT5 timeframe constant: {mt5_timeframe}")
        return timeframe
    
    def _normalize_datetime(self, mt5_time: int) -> datetime:
        """
        Convert MT5 timestamp to UTC datetime.
        
        Args:
            mt5_time: Unix timestamp in seconds
            
        Returns:
            UTC datetime
        """
        return datetime.fromtimestamp(mt5_time, tz=timezone.utc)
    
    def _validate_candle_data(self, mt5_rate: Tuple) -> bool:
        """
        Validate MT5 rate tuple has required fields.
        
        Args:
            mt5_rate: MT5 rate tuple
            
        Returns:
            True if valid, False otherwise
        """
        if not mt5_rate or len(mt5_rate) < 5:
            return False
        
        try:
            open_price = float(mt5_rate[1])
            high = float(mt5_rate[2])
            low = float(mt5_rate[3])
            close = float(mt5_rate[4])
            
            if open_price <= 0 or high <= 0 or low <= 0 or close <= 0:
                return False
            
            return self._validate_ohlc_logic(open_price, high, low, close)
            
        except (ValueError, TypeError):
            return False
    
    def _validate_ohlc_logic(self, open_price: float, high: float, low: float, close: float) -> bool:
        """
        Validate OHLC logic: high >= low, high >= open/close, low <= open/close.
        
        Args:
            open_price: Open price
            high: High price
            low: Low price
            close: Close price
            
        Returns:
            True if valid, False otherwise
        """
        if high < low:
            return False
        if high < open_price or high < close:
            return False
        if low > open_price or low > close:
            return False
        return True


# ==============================================================================
# FACTORY FUNCTION
# ==============================================================================

def create_adapter(config: Config) -> MT5Adapter:
    """
    Factory function for MT5Adapter creation.
    
    Args:
        config: Application configuration
        
    Returns:
        MT5Adapter instance
    """
    return MT5Adapter(config)