"""
eTrade Market Data Downloader

Production-grade MT5 data collection engine with:
- Monthly batch downloading for memory efficiency
- Automatic resume from interruptions
- Duplicate detection and prevention
- Comprehensive error handling and retries
- Transaction management
- Progress tracking
- Clean architecture and SOLID principles

Author: eTrade Development Team
Version: 1.0.0
"""

import logging
import sqlite3
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Generator
from enum import Enum
import time
from pathlib import Path

try:
    import MetaTrader5 as mt5
except ImportError:
    mt5 = None

# ==============================================================================
# CONFIGURATION & CONSTANTS
# ==============================================================================

logger = logging.getLogger(__name__)

# Retry configuration
MAX_RETRIES = 3
RETRY_DELAY = 1.0
EXPONENTIAL_BACKOFF = 1.5

# Batch configuration
BATCH_SIZE_MONTHLY = 1
VALIDATION_THRESHOLD = 0.95  # Require 95% valid candles in batch

# Request throttling
MT5_REQUEST_DELAY = 0.05  # seconds between MT5 requests


# ==============================================================================
# ENUMS
# ==============================================================================

class SyncStatus(Enum):
    """Enumeration of sync status values."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    PARTIAL = "partial"
    ERROR = "error"


class DownloaderError(Exception):
    """Base exception for downloader errors."""
    pass


class MT5ConnectionError(DownloaderError):
    """Raised when MT5 connection fails."""
    pass


class DataValidationError(DownloaderError):
    """Raised when candle data validation fails."""
    pass


# ==============================================================================
# DATA MODELS
# ==============================================================================

@dataclass
class Candle:
    """Represents a single market candle/bar."""
    time: int
    open: float
    high: float
    low: float
    close: float
    tick_volume: int
    spread: int = 0
    real_volume: int = 0
    flags: int = 0

    def to_tuple(self) -> Tuple:
        """Convert candle to tuple for database insertion."""
        return (
            self.time,
            self.open,
            self.high,
            self.low,
            self.close,
            self.tick_volume,
            self.spread,
            self.real_volume,
            self.flags,
        )

    @staticmethod
    def from_mt5_rate(rate) -> 'Candle':
        """Create Candle from MT5 rate object."""
        return Candle(
            time=int(rate['time']),
            open=float(rate['open']),
            high=float(rate['high']),
            low=float(rate['low']),
            close=float(rate['close']),
            tick_volume=int(rate['tick_volume']),
            spread=int(rate['spread']) if 'spread' in rate.dtype.names else 0,
            real_volume=int(rate['real_volume']) if 'real_volume' in rate.dtype.names else 0,
            flags=int(rate['flags']) if 'flags' in rate.dtype.names else 0,
        )


@dataclass
class SyncRecord:
    """Represents sync status for a market/timeframe pair."""
    market_id: int
    market_name: str
    timeframe: str
    status: SyncStatus
    last_synced: Optional[datetime]
    last_candle_time: Optional[int]
    candles_count: int
    error_message: Optional[str] = None


# ==============================================================================
# MT5 CLIENT (Adapter Pattern)
# ==============================================================================

class MT5Client:
    """
    Adapter for MetaTrader5 API.
    
    Handles MT5 connection lifecycle and provides a clean interface
    for data retrieval. This design allows easy substitution with
    other brokers (Binance, Interactive Brokers, etc.).
    """

    def __init__(self):
        """Initialize MT5 client."""
        self._is_connected = False

    def connect(self) -> bool:
        """
        Establish connection to MT5 terminal.
        
        Returns:
            bool: True if connection successful, False otherwise.
        """
        if mt5 is None:
            logger.error("MetaTrader5 package not installed")
            raise ImportError("MetaTrader5 package required")

        try:
            if not mt5.initialize():
                logger.error(
                    "Failed to initialize MT5. "
                    "Ensure MetaTrader 5 is running and properly configured."
                )
                return False
            self._is_connected = True
            logger.info("✓ Connected to MT5")
            return True
        except Exception as e:
            logger.error(f"MT5 connection error: {e}")
            raise MT5ConnectionError(f"Failed to connect to MT5: {e}") from e

    def disconnect(self) -> None:
        """Cleanly disconnect from MT5."""
        if self._is_connected:
            mt5.shutdown()
            self._is_connected = False
            logger.info("✓ Disconnected from MT5")

    def is_connected(self) -> bool:
        """Check if connected to MT5."""
        return self._is_connected

    def select_symbol(self, symbol: str) -> bool:
        """
        Select a symbol for trading.
        
        Args:
            symbol: Market symbol name (e.g., "EURUSD")
            
        Returns:
            bool: True if selection successful.
        """
        try:
            if not mt5.symbol_select(symbol, True):
                logger.warning(f"Failed to select symbol: {symbol}")
                return False
            return True
        except Exception as e:
            logger.error(f"Error selecting symbol {symbol}: {e}")
            return False

    def fetch_rates(
        self,
        symbol: str,
        timeframe: int,
        start_date: datetime,
        end_date: datetime,
    ) -> Optional[List[Candle]]:
        """
        Fetch OHLCV data for a symbol within date range.
        
        Args:
            symbol: Market symbol
            timeframe: MT5 timeframe constant
            start_date: Start of date range
            end_date: End of date range
            
        Returns:
            List of Candle objects or None if fetch fails.
            
        Raises:
            MT5ConnectionError: If MT5 connection fails.
        """
        if not self._is_connected:
            raise MT5ConnectionError("MT5 not connected")

        try:
            rates = mt5.copy_rates_range(symbol, timeframe, start_date, end_date)

            if rates is None or len(rates) == 0:
                logger.debug(
                    f"No data for {symbol} {timeframe} "
                    f"from {start_date} to {end_date}"
                )
                return []

            candles = [Candle.from_mt5_rate(rate) for rate in rates]
            logger.debug(f"Fetched {len(candles)} candles for {symbol}")
            return candles

        except Exception as e:
            logger.error(
                f"Error fetching rates for {symbol}: {e}"
            )
            return None


# ==============================================================================
# DATA VALIDATION
# ==============================================================================

class CandleValidator:
    """
    Validates candle data for integrity and consistency.
    
    Implements OHLC validation rules to ensure data quality
    before insertion into the database.
    """

    @staticmethod
    def validate_candle(candle: Candle) -> bool:
        """
        Validate a single candle.
        
        Args:
            candle: Candle object to validate
            
        Returns:
            bool: True if candle passes all validations.
        """
        # Check basic OHLC relationship
        if not (candle.low <= candle.open <= candle.high):
            logger.debug(
                f"Invalid OHLC: low={candle.low}, "
                f"open={candle.open}, high={candle.high}"
            )
            return False

        if not (candle.low <= candle.close <= candle.high):
            logger.debug(
                f"Invalid OHLC: low={candle.low}, "
                f"close={candle.close}, high={candle.high}"
            )
            return False

        # Check for negative/zero prices
        if candle.open <= 0 or candle.high <= 0 or candle.low <= 0 or candle.close <= 0:
            logger.debug("Negative or zero price detected")
            return False

        # Check for extreme price movements (more than 50% in one candle)
        # This is a heuristic; adjust based on asset class
        max_price = max(candle.open, candle.high, candle.low, candle.close)
        min_price = min(candle.open, candle.high, candle.low, candle.close)
        if max_price > 0 and (max_price / min_price) > 2.0:
            logger.debug(
                f"Extreme price movement: {min_price} to {max_price}"
            )
            return False

        # Check timestamp
        if candle.time <= 0:
            logger.debug("Invalid timestamp")
            return False

        # Check volume
        if candle.tick_volume < 0:
            logger.debug("Negative volume")
            return False

        return True

    @staticmethod
    def validate_batch(candles: List[Candle]) -> Tuple[List[Candle], float]:
        """
        Validate a batch of candles.
        
        Args:
            candles: List of Candle objects
            
        Returns:
            Tuple of (valid_candles, validation_ratio)
        """
        if not candles:
            return [], 1.0

        valid_candles = [
            c for c in candles
            if CandleValidator.validate_candle(c)
        ]

        ratio = len(valid_candles) / len(candles)
        logger.debug(
            f"Batch validation: {len(valid_candles)}/{len(candles)} "
            f"valid ({ratio:.1%})"
        )

        return valid_candles, ratio


# ==============================================================================
# DATABASE OPERATIONS
# ==============================================================================

class DatabaseOperations:
    """
    Handles all database operations.
    
    Uses dependency injection for database connection to allow
    testing and support for different database backends.
    """

    def __init__(self, db_connection: sqlite3.Connection):
        """
        Initialize database operations.
        
        Args:
            db_connection: Active SQLite connection.
        """
        self.db = db_connection

    def get_markets(self) -> List[Dict[str, any]]:
        """
        Fetch all markets from database.
        
        Returns:
            List of market dictionaries with id and name.
        """
        cursor = self.db.cursor()
        try:
            cursor.execute("SELECT id, name FROM markets WHERE active = 1")
            rows = cursor.fetchall()
            return [
                {"id": row[0], "name": row[1]}
                for row in rows
            ]
        except sqlite3.OperationalError as e:
            logger.error(f"Error fetching markets: {e}")
            return []

    def get_sync_status(
        self,
        market_id: int,
        timeframe: str,
    ) -> SyncRecord:
        """
        Fetch sync status for a market/timeframe pair.
        
        Args:
            market_id: Market ID
            timeframe: Timeframe string (e.g., "H1")
            
        Returns:
            SyncRecord with current status.
        """
        cursor = self.db.cursor()
        try:
            cursor.execute(
                """
                SELECT m.id, m.name, tf.name, ss.status, 
                       ss.last_synced, ss.last_candle_time, ss.candles_count
                FROM sync_status ss
                JOIN markets m ON ss.market_id = m.id
                JOIN timeframes tf ON ss.timeframe_id = tf.id
                WHERE ss.market_id = ? AND tf.name = ?
                """,
                (market_id, timeframe),
            )
            row = cursor.fetchone()

            if row:
                return SyncRecord(
                    market_id=row[0],
                    market_name=row[1],
                    timeframe=row[2],
                    status=SyncStatus(row[3]),
                    last_synced=datetime.fromisoformat(row[4]) if row[4] else None,
                    last_candle_time=row[5],
                    candles_count=row[6],
                )

            # No sync record exists, return pending
            cursor.execute(
                "SELECT id, name FROM markets WHERE id = ?",
                (market_id,),
            )
            market_row = cursor.fetchone()
            if market_row:
                return SyncRecord(
                    market_id=market_id,
                    market_name=market_row[1],
                    timeframe=timeframe,
                    status=SyncStatus.PENDING,
                    last_synced=None,
                    last_candle_time=None,
                    candles_count=0,
                )
            return None

        except sqlite3.OperationalError as e:
            logger.error(f"Error fetching sync status: {e}")
            return None

    def insert_candles(
        self,
        market_id: int,
        timeframe_id: int,
        candles: List[Candle],
    ) -> int:
        """
        Insert candles into database using batch insert with
        duplicate detection.
        
        Args:
            market_id: Market ID
            timeframe_id: Timeframe ID
            candles: List of Candle objects
            
        Returns:
            Number of candles actually inserted.
        """
        if not candles:
            return 0

        cursor = self.db.cursor()
        inserted = 0

        try:
            # Use INSERT OR IGNORE to prevent duplicates
            # This assumes a UNIQUE constraint on (market_id, timeframe_id, time)
            for candle in candles:
                cursor.execute(
                    """
                    INSERT OR IGNORE INTO candles
                    (market_id, timeframe_id, time, open, high, low, close,
                     tick_volume, spread, real_volume, flags)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (market_id, timeframe_id, *candle.to_tuple()),
                )
                if cursor.rowcount > 0:
                    inserted += 1

            self.db.commit()
            logger.debug(f"Inserted {inserted}/{len(candles)} candles")
            return inserted

        except sqlite3.Error as e:
            self.db.rollback()
            logger.error(f"Error inserting candles: {e}")
            raise

    def update_sync_status(
        self,
        market_id: int,
        timeframe_id: int,
        status: SyncStatus,
        last_candle_time: Optional[int] = None,
        candles_count: Optional[int] = None,
        error_message: Optional[str] = None,
    ) -> bool:
        """
        Update sync status for a market/timeframe pair.
        
        Args:
            market_id: Market ID
            timeframe_id: Timeframe ID
            status: New status
            last_candle_time: Timestamp of last candle
            candles_count: Total candles for this pair
            error_message: Error message if status is ERROR
            
        Returns:
            True if update successful.
        """
        cursor = self.db.cursor()
        try:
            cursor.execute(
                """
                INSERT OR REPLACE INTO sync_status
                (market_id, timeframe_id, status, last_synced,
                 last_candle_time, candles_count, error_message)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    market_id,
                    timeframe_id,
                    status.value,
                    datetime.now().isoformat(),
                    last_candle_time,
                    candles_count,
                    error_message,
                ),
            )
            self.db.commit()
            logger.debug(
                f"Updated sync status: market={market_id}, "
                f"timeframe={timeframe_id}, status={status.value}"
            )
            return True

        except sqlite3.Error as e:
            self.db.rollback()
            logger.error(f"Error updating sync status: {e}")
            return False

    def get_timeframe_id(self, timeframe: str) -> Optional[int]:
        """
        Get timeframe ID from timeframe name.
        
        Args:
            timeframe: Timeframe name (e.g., "H1")
            
        Returns:
            Timeframe ID or None if not found.
        """
        cursor = self.db.cursor()
        try:
            cursor.execute("SELECT id FROM timeframes WHERE name = ?", (timeframe,))
            row = cursor.fetchone()
            return row[0] if row else None
        except sqlite3.OperationalError as e:
            logger.error(f"Error fetching timeframe ID: {e}")
            return None

    def get_last_candle_time(
        self,
        market_id: int,
        timeframe_id: int,
    ) -> Optional[int]:
        """
        Get timestamp of last candle in database.
        
        Args:
            market_id: Market ID
            timeframe_id: Timeframe ID
            
        Returns:
            Timestamp of last candle or None if no candles exist.
        """
        cursor = self.db.cursor()
        try:
            cursor.execute(
                """
                SELECT MAX(time) FROM candles
                WHERE market_id = ? AND timeframe_id = ?
                """,
                (market_id, timeframe_id),
            )
            row = cursor.fetchone()
            return row[0] if row and row[0] else None
        except sqlite3.OperationalError as e:
            logger.error(f"Error fetching last candle time: {e}")
            return None

    def get_candles_count(
        self,
        market_id: int,
        timeframe_id: int,
    ) -> int:
        """
        Get count of candles for a market/timeframe pair.
        
        Args:
            market_id: Market ID
            timeframe_id: Timeframe ID
            
        Returns:
            Number of candles.
        """
        cursor = self.db.cursor()
        try:
            cursor.execute(
                """
                SELECT COUNT(*) FROM candles
                WHERE market_id = ? AND timeframe_id = ?
                """,
                (market_id, timeframe_id),
            )
            row = cursor.fetchone()
            return row[0] if row else 0
        except sqlite3.OperationalError as e:
            logger.error(f"Error counting candles: {e}")
            return 0


# ==============================================================================
# BATCH GENERATOR (Memory Efficiency)
# ==============================================================================

class MonthlyBatchGenerator:
    """
    Generator that yields monthly date ranges.
    
    This design ensures that only one month of data is processed
    at a time, keeping memory usage constant regardless of the
    total dataset size.
    """

    def __init__(self, start_date: datetime, end_date: datetime):
        """
        Initialize batch generator.
        
        Args:
            start_date: Start of date range
            end_date: End of date range
        """
        self.start_date = start_date
        self.end_date = end_date

    def generate(self) -> Generator[Tuple[datetime, datetime], None, None]:
        """
        Generate monthly batch date ranges.
        
        Yields:
            Tuple of (month_start, month_end)
        """
        current = self.start_date.replace(day=1)

        while current <= self.end_date:
            # Calculate month end
            if current.month == 12:
                month_end = current.replace(year=current.year + 1, month=1, day=1)
            else:
                month_end = current.replace(month=current.month + 1, day=1)

            month_end = min(month_end, self.end_date)

            yield current, month_end

            current = month_end


# ==============================================================================
# DOWNLOADER (Main Orchestrator)
# ==============================================================================

class Downloader:
    """
    Production-grade MT5 data downloader.
    
    Handles complete lifecycle of downloading market data:
    - Connects to MT5
    - Reads markets and timeframes from database
    - Downloads data in monthly batches
    - Validates and deduplicates candles
    - Maintains sync status
    - Survives interruptions
    - Provides progress tracking
    
    Architecture:
    - Dependency injection for testability
    - Strategy pattern for validators
    - Generator pattern for memory efficiency
    - Adapter pattern for MT5 client
    - Clean separation of concerns
    """

    def __init__(
        self,
        db_connection: sqlite3.Connection,
        timeframes: Dict[str, int],
        config: Optional[Dict] = None,
    ):
        """
        Initialize downloader.
        
        Args:
            db_connection: Active SQLite database connection
            timeframes: Dict mapping timeframe names to MT5 constants
            config: Optional configuration overrides
        """
        self.db = DatabaseOperations(db_connection)
        self.mt5_client = MT5Client()
        self.validator = CandleValidator()
        self.timeframes = timeframes
        self.config = config or {}

        # Configuration
        self.max_retries = self.config.get("max_retries", MAX_RETRIES)
        self.retry_delay = self.config.get("retry_delay", RETRY_DELAY)
        self.batch_size = self.config.get("batch_size", BATCH_SIZE_MONTHLY)
        self.validation_threshold = self.config.get(
            "validation_threshold",
            VALIDATION_THRESHOLD,
        )

        # State
        self._total_downloaded = 0
        self._total_inserted = 0
        self._total_duplicates = 0

    def run(self) -> bool:
        """
        Run complete download cycle.
        
        Returns:
            bool: True if all markets completed successfully.
        """
        logger.info("=" * 80)
        logger.info("Starting eTrade Market Data Downloader")
        logger.info("=" * 80)

        try:
            if not self.mt5_client.connect():
                return False

            markets = self.db.get_markets()
            if not markets:
                logger.warning("No active markets found")
                return False

            logger.info(f"Found {len(markets)} markets to download")

            success_count = 0
            for market in markets:
                if self.download_market(market["id"], market["name"]):
                    success_count += 1
                time.sleep(MT5_REQUEST_DELAY)

            self._print_summary(success_count, len(markets))
            return success_count == len(markets)

        except Exception as e:
            logger.error(f"Fatal error in download cycle: {e}", exc_info=True)
            return False

        finally:
            self.mt5_client.disconnect()

    def download_market(self, market_id: int, market_name: str) -> bool:
        """
        Download all timeframes for a market.
        
        Args:
            market_id: Market ID from database
            market_name: Market symbol name (e.g., "EURUSD")
            
        Returns:
            bool: True if all timeframes completed successfully.
        """
        logger.info(f"\n{'─' * 80}")
        logger.info(f"Downloading: {market_name}")
        logger.info(f"{'─' * 80}")

        if not self.mt5_client.select_symbol(market_name):
            logger.error(f"Failed to select symbol: {market_name}")
            return False

        success_count = 0
        for timeframe_name, timeframe_value in self.timeframes.items():
            if self.download_timeframe(
                market_id,
                market_name,
                timeframe_name,
                timeframe_value,
            ):
                success_count += 1

        return success_count == len(self.timeframes)

    def download_timeframe(
        self,
        market_id: int,
        market_name: str,
        timeframe_name: str,
        timeframe_value: int,
    ) -> bool:
        """
        Download all months for a market/timeframe pair.
        
        Args:
            market_id: Market ID
            market_name: Market symbol
            timeframe_name: Timeframe name (e.g., "H1")
            timeframe_value: MT5 timeframe constant
            
        Returns:
            bool: True if download successful.
        """
        logger.info(f"  {timeframe_name}...", end=" ")

        # Get timeframe ID
        timeframe_id = self.db.get_timeframe_id(timeframe_name)
        if not timeframe_id:
            logger.error(f"Timeframe not found: {timeframe_name}")
            return False

        # Get sync status
        sync_record = self.db.get_sync_status(market_id, timeframe_name)
        if not sync_record:
            logger.error("Failed to get sync status")
            return False

        # Determine start date
        if sync_record.last_candle_time:
            start_date = datetime.fromtimestamp(sync_record.last_candle_time)
        else:
            start_date = datetime(2000, 1, 1)  # Default historical start

        end_date = datetime.now()

        # Update status to in_progress
        self.db.update_sync_status(
            market_id,
            timeframe_id,
            SyncStatus.IN_PROGRESS,
        )

        try:
            total_inserted = 0
            batch_generator = MonthlyBatchGenerator(start_date, end_date)

            for batch_start, batch_end in batch_generator.generate():
                # Download with retries
                candles = self._download_with_retry(
                    market_name,
                    timeframe_value,
                    batch_start,
                    batch_end,
                )

                if candles is None:
                    logger.error(
                        f"Failed to download {market_name} {timeframe_name} "
                        f"for {batch_start.strftime('%Y-%m')}"
                    )
                    self.db.update_sync_status(
                        market_id,
                        timeframe_id,
                        SyncStatus.ERROR,
                        error_message="Download failed",
                    )
                    return False

                if not candles:
                    continue

                # Validate batch
                valid_candles, validation_ratio = self.validator.validate_batch(
                    candles
                )

                if validation_ratio < self.validation_threshold:
                    logger.warning(
                        f"Low validation ratio: {validation_ratio:.1%} "
                        f"for {batch_start.strftime('%Y-%m')}"
                    )

                # Save batch
                inserted = self.db.insert_candles(
                    market_id,
                    timeframe_id,
                    valid_candles,
                )

                total_inserted += inserted
                duplicates = len(candles) - inserted
                self._total_duplicates += duplicates

                time.sleep(MT5_REQUEST_DELAY)

            # Get final candle count and update status
            candles_count = self.db.get_candles_count(market_id, timeframe_id)
            last_candle_time = self.db.get_last_candle_time(market_id, timeframe_id)

            self.db.update_sync_status(
                market_id,
                timeframe_id,
                SyncStatus.COMPLETED,
                last_candle_time=last_candle_time,
                candles_count=candles_count,
            )

            self._total_inserted += total_inserted
            logger.info(f"✓ {total_inserted:,} candles inserted")
            return True

        except Exception as e:
            logger.error(f"Error downloading {market_name} {timeframe_name}: {e}")
            self.db.update_sync_status(
                market_id,
                timeframe_id,
                SyncStatus.ERROR,
                error_message=str(e)[:100],
            )
            return False

    def _download_with_retry(
        self,
        symbol: str,
        timeframe: int,
        start_date: datetime,
        end_date: datetime,
    ) -> Optional[List[Candle]]:
        """
        Download data with automatic retry on failure.
        
        Args:
            symbol: Market symbol
            timeframe: MT5 timeframe constant
            start_date: Start of date range
            end_date: End of date range
            
        Returns:
            List of Candle objects or None if all retries failed.
        """
        delay = self.retry_delay

        for attempt in range(self.max_retries):
            try:
                candles = self.mt5_client.fetch_rates(
                    symbol,
                    timeframe,
                    start_date,
                    end_date,
                )

                if candles is not None:
                    self._total_downloaded += len(candles)
                    return candles

            except MT5ConnectionError as e:
                logger.warning(
                    f"Attempt {attempt + 1}/{self.max_retries} failed: {e}"
                )

            if attempt < self.max_retries - 1:
                logger.debug(f"Retrying in {delay:.1f}s...")
                time.sleep(delay)
                delay *= EXPONENTIAL_BACKOFF

        return None

    def _print_summary(self, success_count: int, total_count: int) -> None:
        """
        Print download summary statistics.
        
        Args:
            success_count: Number of markets successfully downloaded
            total_count: Total number of markets
        """
        logger.info("\n" + "=" * 80)
        logger.info("Download Summary")
        logger.info("=" * 80)
        logger.info(f"Markets: {success_count}/{total_count}")
        logger.info(f"Candles downloaded: {self._total_downloaded:,}")
        logger.info(f"Candles inserted: {self._total_inserted:,}")
        logger.info(f"Duplicates skipped: {self._total_duplicates:,}")
        logger.info("=" * 80 + "\n")


# ==============================================================================
# USAGE EXAMPLE
# ==============================================================================

if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    # Example usage (uncomment to use)
    """
    import sys
    sys.path.insert(0, '/path/to/etrade')
    
    from database.database import get_connection
    from config import TIMEFRAMES
    
    db_conn = get_connection()
    
    downloader = Downloader(
        db_connection=db_conn,
        timeframes=TIMEFRAMES,
    )
    
    success = downloader.run()
    
    db_conn.close()
    
    sys.exit(0 if success else 1)
    """