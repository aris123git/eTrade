"""
database/repositories/candle_repository.py - Candle Repository (Production Ready)

RESPONSIBILITY:
Manage candle/OHLCV data with time-series optimizations.

DESIGN PHILOSOPHY:
- Simple where possible, optimized where it matters
- No over-engineering for "billions of candles" (SQLite handles millions well)
- Every optimization justifies its complexity
- Progressive evolution, not premature architecture

VERSION: 3.0.4
"""

import json
import logging
import sqlite3
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Any, Tuple, Iterator, Union
from uuid import uuid4
from functools import lru_cache

from database.core.connection import DatabaseManager
from database.repositories.base_repository import BaseRepository
from database.models.candle import Candle, CandleStatus


logger = logging.getLogger(__name__)


class CandleRepository(BaseRepository[Candle]):
    """
    Candle repository for time-series OHLCV data.
    
    KEY FEATURES:
    - Atomic UPSERT (ON CONFLICT) - 1 query instead of SELECT + INSERT/UPDATE
    - Keyset pagination - No OFFSET for large datasets
    - Streaming generator - Memory-efficient for AI training
    - LRU cache with invalidation - For frequently accessed latest candles
    - Transaction support - Bulk operations are atomic
    - Fast validation mode - For trusted data sources (MT5)
    
    SIMPLICITY FIRST:
    - Single table (no manual partitioning - SQLite handles millions)
    - 4 essential indexes (not 10+)
    - No metrics (add when needed)
    - No distributed architecture (not needed yet)
    """
    
    TABLE = "candles"
    MODEL = Candle
    
    # Performance constants
    DEFAULT_BATCH_SIZE = 10000
    MAX_BATCH_SIZE = 50000
    DEFAULT_LIMIT = 1000
    MAX_LIMIT = 100000
    STREAM_BATCH_SIZE = 10000
    
    def __init__(self, db_manager: DatabaseManager):
        """Initialize the candle repository."""
        super().__init__(db_manager)
        self.logger = logging.getLogger(__name__)
        
        # Ensure indexes exist
        self._ensure_indexes()
    
    # ==========================================================================
    # DATABASE INDEXES
    # ==========================================================================
    
    def _ensure_indexes(self):
        """Create essential indexes."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            indexes = [
                "CREATE INDEX IF NOT EXISTS idx_candles_primary ON candles(symbol, timeframe, timestamp)",
                "CREATE INDEX IF NOT EXISTS idx_candles_market ON candles(market_id)",
                "CREATE INDEX IF NOT EXISTS idx_candles_broker ON candles(broker_id)",
                "CREATE INDEX IF NOT EXISTS idx_candles_active ON candles(symbol, timeframe, timestamp) WHERE status = 'active'",
            ]
            
            for idx_sql in indexes:
                try:
                    cursor.execute(idx_sql)
                except sqlite3.Error as e:
                    self.logger.warning(f"Failed to create index: {e}")
    
    # ==========================================================================
    # CREATE / UPSERT
    # ==========================================================================
    
    def create(
        self,
        symbol: str,
        timeframe: str,
        timestamp: datetime,
        open_price: float,
        high: float,
        low: float,
        close: float,
        volume: float,
        market_id: Optional[int] = None,
        broker_id: Optional[int] = None,
        spread: Optional[float] = None,
        tick_volume: Optional[int] = None,
        status: CandleStatus = CandleStatus.ACTIVE,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Candle:
        """Create a new candle with atomic UPSERT."""
        if timestamp is None:
            raise ValueError("Timestamp is required")
        
        candle = Candle(
            candle_id=None,
            candle_uuid=str(uuid4()),
            symbol=symbol.upper(),
            timeframe=timeframe.upper(),
            timestamp=timestamp,
            open=open_price,
            high=high,
            low=low,
            close=close,
            volume=volume,
            market_id=market_id,
            broker_id=broker_id,
            spread=spread,
            tick_volume=tick_volume,
            status=status,
            metadata=metadata or {},
            created_at=datetime.now(),
            updated_at=datetime.now(),
        )
        
        self._validate_candle(candle)
        
        data = self._entity_to_dict(candle)
        data.pop('candle_id', None)
        if data.get('broker_id') is None:
            data['broker_id'] = 0
        
        fields = list(data.keys())
        placeholders = ", ".join("?" for _ in fields)
        field_names = ", ".join(fields)
        update_clause = ", ".join(
            f"{f} = excluded.{f}"
            for f in fields
            if f not in ['broker_id', 'symbol', 'timeframe', 'timestamp']
        )
        
        sql = f"""
            INSERT INTO {self.TABLE} ({field_names})
            VALUES ({placeholders})
            ON CONFLICT(broker_id, symbol, timeframe, timestamp) DO UPDATE SET {update_clause}
            RETURNING candle_id
        """
        
        with self._get_connection() as conn:
            try:
                cursor = conn.execute(sql, tuple(data.values()))
                row = cursor.fetchone()
                candle_id = row[0] if row else None
            except sqlite3.OperationalError as e:
                # SQLite version may not support RETURNING
                if "near 'RETURNING'" in str(e):
                    self.logger.warning("RETURNING not supported, using fallback")
                    # Fallback: insert then select
                    conn.execute(sql.replace(" RETURNING candle_id", ""), tuple(data.values()))
                    candle_id = conn.execute("SELECT last_insert_rowid()").fetchone()[0]
                else:
                    raise
            
            if candle_id:
                candle.candle_id = candle_id
                self._invalidate_cache(symbol, timeframe)
                self.logger.debug(f"✅ Candle upserted: {symbol} {timeframe} at {timestamp}")
                return candle
            
            raise RuntimeError("Failed to upsert candle")
    
    def bulk_upsert(self, candles: List[Dict[str, Any]], batch_size: int = None, trusted_source: bool = False) -> int:
        """Bulk UPSERT candles with atomic transaction."""
        if not candles:
            return 0
        
        batch_size = batch_size or self.DEFAULT_BATCH_SIZE
        batch_size = min(batch_size, self.MAX_BATCH_SIZE)
        
        total = 0
        symbols_timeframes = set()
        
        with self.transaction():
            for i in range(0, len(candles), batch_size):
                chunk = candles[i:i+batch_size]
                count, affected_symbols = self._upsert_chunk(chunk, trusted_source)
                total += count
                symbols_timeframes.update(affected_symbols)
        
        # Invalidate cache for all affected symbols/timeframes
        for symbol, timeframe in symbols_timeframes:
            self._invalidate_cache(symbol, timeframe)
        
        self.logger.info(f"✅ Bulk upserted {total} candles")
        return total
    
    def _upsert_chunk(self, chunk: List[Dict[str, Any]], trusted_source: bool) -> Tuple[int, set]:
        """Upsert a single chunk."""
        if not chunk:
            return 0, set()
        
        data_list = []
        symbols_timeframes = set()
        
        for c in chunk:
            if trusted_source:
                self._fast_validate_candle(c)
            else:
                self._validate_candle_data(c)
            
            symbol = c.get('symbol', '').upper()
            timeframe = c.get('timeframe', '').upper()
            symbols_timeframes.add((symbol, timeframe))
            ts = c.get('timestamp', datetime.now())
            if isinstance(ts, datetime):
                ts = ts.isoformat(timespec="seconds")
            
            data_list.append({
                'candle_uuid': str(uuid4()),
                'symbol': symbol,
                'timeframe': timeframe,
                'timestamp': ts,
                'open': float(c.get('open', 0)),
                'high': float(c.get('high', 0)),
                'low': float(c.get('low', 0)),
                'close': float(c.get('close', 0)),
                'volume': float(c.get('volume', 0)),
                'market_id': c.get('market_id'),
                # NULL broker_id breaks UNIQUE/ON CONFLICT matching in SQLite
                'broker_id': c.get('broker_id') if c.get('broker_id') is not None else 0,
                'spread': c.get('spread'),
                'tick_volume': c.get('tick_volume'),
                'status': c.get('status', CandleStatus.ACTIVE.value),
                'metadata': json.dumps(c.get('metadata', {})),
                'created_at': datetime.now().isoformat(),
                'updated_at': datetime.now().isoformat(),
            })
        
        fields = list(data_list[0].keys())
        placeholders = ", ".join("?" for _ in fields)
        field_names = ", ".join(fields)
        update_fields = [
            f for f in fields
            if f not in ['broker_id', 'symbol', 'timeframe', 'timestamp'] and f != 'candle_uuid'
        ]
        update_clause = ", ".join(f"{f} = excluded.{f}" for f in update_fields)
        
        sql = f"""
            INSERT INTO {self.TABLE} ({field_names})
            VALUES ({placeholders})
            ON CONFLICT(broker_id, symbol, timeframe, timestamp) DO UPDATE SET {update_clause}
        """
        
        params = [tuple(d.values()) for d in data_list]
        try:
            return self.adapter.execute_many(sql, params), symbols_timeframes
        except sqlite3.IntegrityError as e:
            self.logger.error(f"Integrity error in upsert: {e}")
            count = self._fallback_upsert(data_list)
            return count, symbols_timeframes
    
    def _fallback_upsert(self, data_list: List[Dict[str, Any]]) -> int:
        """Fallback to individual upserts."""
        count = 0
        for data in data_list:
            try:
                fields = list(data.keys())
                placeholders = ", ".join("?" for _ in fields)
                field_names = ", ".join(fields)
                update_fields = [
                    f for f in fields
                    if f not in ['broker_id', 'symbol', 'timeframe', 'timestamp']
                ]
                update_clause = ", ".join(f"{f} = excluded.{f}" for f in update_fields)
                
                sql = f"""
                    INSERT INTO {self.TABLE} ({field_names})
                    VALUES ({placeholders})
                    ON CONFLICT(broker_id, symbol, timeframe, timestamp) DO UPDATE SET {update_clause}
                """
                
                self.adapter.execute(sql, tuple(data.values()))
                count += 1
            except sqlite3.IntegrityError as e:
                self.logger.warning(f"Integrity error on individual upsert: {e}")
            except Exception as e:
                self.logger.warning(f"Failed to upsert individual candle: {e}")
        return count
    
    # ==========================================================================
    # READ OPERATIONS
    # ==========================================================================
    
    def find_by_time_range(
        self,
        symbol: str,
        timeframe: str,
        start_time: datetime,
        end_time: datetime,
        limit: int = None,
        after_timestamp: Optional[datetime] = None,
        order: str = "ASC",
    ) -> List[Candle]:
        """Find candles in a time range with keyset pagination."""
        order = order.upper()
        if order not in ("ASC", "DESC"):
            order = "ASC"
        
        limit = min(limit or self.DEFAULT_LIMIT, self.MAX_LIMIT)
        
        sql = f"""
            SELECT * FROM {self.TABLE} 
            WHERE symbol = ? 
            AND timeframe = ? 
            AND timestamp BETWEEN ? AND ?
            AND status = ?
        """
        params = [symbol.upper(), timeframe.upper(), self._sql_ts(start_time), self._sql_ts(end_time), CandleStatus.ACTIVE.value]
        
        if after_timestamp:
            if order == "ASC":
                sql += " AND timestamp > ?"
                params.append(after_timestamp)
            else:
                sql += " AND timestamp < ?"
                params.append(after_timestamp)
        
        sql += f" ORDER BY timestamp {order} LIMIT {limit}"
        
        rows = self.adapter.fetch_all(sql, tuple(params))
        return [self._row_to_entity(row) for row in rows]
    
    def find_latest(self, symbol: str, timeframe: str, limit: int = 1) -> List[Candle]:
        """Find latest candles."""
        limit = min(limit, self.MAX_LIMIT)
        
        sql = f"""
            SELECT * FROM {self.TABLE} 
            WHERE symbol = ? AND timeframe = ? AND status = ?
            ORDER BY timestamp DESC
            LIMIT {limit}
        """
        rows = self.adapter.fetch_all(sql, (symbol.upper(), timeframe.upper(), CandleStatus.ACTIVE.value))
        return [self._row_to_entity(row) for row in rows]
    
    def get_last_n(self, symbol: str, timeframe: str, n: int = 500) -> List[Candle]:
        """Get last N candles (used for indicators)."""
        return self.find_latest(symbol, timeframe, n)
    
    def exists(self, symbol: str, timeframe: str, timestamp: datetime) -> bool:
        """Check if a candle exists."""
        return self.count(
            "symbol = ? AND timeframe = ? AND timestamp = ? AND status = ?",
            (symbol.upper(), timeframe.upper(), self._sql_ts(timestamp), CandleStatus.ACTIVE.value)
        ) > 0
    
    def count_between(self, symbol: str, timeframe: str, start_time: datetime, end_time: datetime) -> int:
        """Count candles between two dates."""
        return self.count(
            "symbol = ? AND timeframe = ? AND timestamp BETWEEN ? AND ? AND status = ?",
            (symbol.upper(), timeframe.upper(), self._sql_ts(start_time), self._sql_ts(end_time), CandleStatus.ACTIVE.value)
        )
    
    def first_timestamp(self, symbol: str, timeframe: str) -> Optional[datetime]:
        """Get the earliest timestamp for a symbol/timeframe."""
        row = self.adapter.fetch_one(f"""
            SELECT MIN(timestamp) as first_ts
            FROM {self.TABLE} 
            WHERE symbol = ? AND timeframe = ? AND status = ?
        """, (symbol.upper(), timeframe.upper(), CandleStatus.ACTIVE.value))
        
        return row['first_ts'] if row and row['first_ts'] else None
    
    def last_timestamp(self, symbol: str, timeframe: str) -> Optional[datetime]:
        """Get the latest timestamp for a symbol/timeframe."""
        row = self.adapter.fetch_one(f"""
            SELECT MAX(timestamp) as last_ts
            FROM {self.TABLE} 
            WHERE symbol = ? AND timeframe = ? AND status = ?
        """, (symbol.upper(), timeframe.upper(), CandleStatus.ACTIVE.value))
        
        return row['last_ts'] if row and row['last_ts'] else None
    
    def has_gap(self, symbol: str, timeframe: str, start_time: datetime, end_time: datetime) -> bool:
        """Check if there are gaps in the data."""
        expected = self.count_between(symbol, timeframe, start_time, end_time)
        
        # Calculate expected number of candles based on timeframe
        tf_seconds = {
            'M1': 60, 'M5': 300, 'M15': 900, 'M30': 1800,
            'H1': 3600, 'H4': 14400, 'D1': 86400, 'W1': 604800, 'MN1': 2592000
        }
        
        seconds = tf_seconds.get(timeframe.upper())
        if not seconds:
            return False
        
        total_seconds = (end_time - start_time).total_seconds()
        expected_count = int(total_seconds / seconds) + 1
        
        return expected < expected_count
    
    def find_missing_ranges(
        self,
        symbol: str,
        timeframe: str,
        start_time: datetime,
        end_time: datetime,
    ) -> List[Tuple[datetime, datetime]]:
        """
        Find missing data ranges.
        
        Returns:
            List of (start, end) tuples for missing ranges.
        """
        candles = self.find_by_time_range(symbol, timeframe, start_time, end_time, order='ASC')
        
        if not candles:
            return [(start_time, end_time)]
        
        missing_ranges = []
        current = start_time
        
        tf_seconds = {
            'M1': 60, 'M5': 300, 'M15': 900, 'M30': 1800,
            'H1': 3600, 'H4': 14400, 'D1': 86400, 'W1': 604800, 'MN1': 2592000
        }
        
        step = tf_seconds.get(timeframe.upper(), 60)
        
        for candle in candles:
            if candle.timestamp > current:
                missing_ranges.append((current, candle.timestamp))
            current = candle.timestamp + timedelta(seconds=step)
        
        if current < end_time:
            missing_ranges.append((current, end_time))
        
        return missing_ranges
    
    @lru_cache(maxsize=128)
    def find_latest_cached(self, symbol: str, timeframe: str) -> Optional[Candle]:
        """Cached latest candle (automatically invalidated on writes)."""
        result = self.find_latest(symbol, timeframe)
        return result[0] if result else None
    
    def _invalidate_cache(self, symbol: str, timeframe: str):
        """Invalidate cache for a specific symbol/timeframe."""
        # LRU cache doesn't support per-key invalidation
        # We clear the entire cache (128 entries max, cheap)
        self.find_latest_cached.cache_clear()
    
    def find_latest_by_symbols(self, symbols: List[str], timeframe: str) -> Dict[str, Optional[Candle]]:
        """Find latest candle for multiple symbols in a SINGLE query."""
        if not symbols:
            return {}
        
        placeholders = ",".join("?" for _ in symbols)
        
        sql = f"""
            WITH ranked AS (
                SELECT *,
                    ROW_NUMBER() OVER (PARTITION BY symbol ORDER BY timestamp DESC) as rn
                FROM {self.TABLE}
                WHERE symbol IN ({placeholders})
                AND timeframe = ?
                AND status = ?
            )
            SELECT * FROM ranked WHERE rn = 1
        """
        
        params = list(symbols) + [timeframe.upper(), CandleStatus.ACTIVE.value]
        rows = self.adapter.fetch_all(sql, tuple(params))
        
        result = {symbol: None for symbol in symbols}
        for row in rows:
            candle = self._row_to_entity(row)
            result[candle.symbol] = candle
        
        return result
    

    @staticmethod
    def _sql_ts(value):
        """Normalize datetimes to ISO-8601 text for SQLite comparisons."""
        if isinstance(value, datetime):
            return value.isoformat(timespec="seconds")
        return value

    # ==========================================================================
    # STREAMING
    # ==========================================================================
    
    def stream_candles(
        self,
        symbol: str,
        timeframe: str,
        start_time: datetime,
        end_time: datetime,
        batch_size: int = None,
        order: str = "ASC",
    ) -> Iterator[Candle]:
        """Stream candles as a generator using keyset pagination."""
        order = order.upper()
        if order not in ("ASC", "DESC"):
            order = "ASC"
        
        batch_size = batch_size or self.STREAM_BATCH_SIZE
        
        if order == "ASC":
            last_timestamp = start_time
            first_page = True
            while True:
                op = ">=" if first_page else ">"
                sql = f"""
                    SELECT * FROM {self.TABLE} 
                    WHERE symbol = ? 
                    AND timeframe = ? 
                    AND timestamp {op} ?
                    AND timestamp <= ?
                    AND status = ?
                    ORDER BY timestamp ASC
                    LIMIT {batch_size}
                """
                params = (symbol.upper(), timeframe.upper(), self._sql_ts(last_timestamp), self._sql_ts(end_time), CandleStatus.ACTIVE.value)
                
                rows = self.adapter.fetch_all(sql, params)
                if not rows:
                    break
                
                for row in rows:
                    candle = self._row_to_entity(row)
                    last_timestamp = candle.timestamp
                    yield candle
                
                first_page = False
                if len(rows) < batch_size:
                    break
        else:
            last_timestamp = end_time
            first_page = True
            while True:
                op = "<=" if first_page else "<"
                sql = f"""
                    SELECT * FROM {self.TABLE} 
                    WHERE symbol = ? 
                    AND timeframe = ? 
                    AND timestamp >= ?
                    AND timestamp {op} ?
                    AND status = ?
                    ORDER BY timestamp DESC
                    LIMIT {batch_size}
                """
                params = (symbol.upper(), timeframe.upper(), self._sql_ts(start_time), self._sql_ts(last_timestamp), CandleStatus.ACTIVE.value)
                
                rows = self.adapter.fetch_all(sql, params)
                if not rows:
                    break
                
                for row in rows:
                    candle = self._row_to_entity(row)
                    last_timestamp = candle.timestamp
                    yield candle
                
                first_page = False
                if len(rows) < batch_size:
                    break
    
    # ==========================================================================
    # AGGREGATION
    # ==========================================================================
    
    def get_ohlc(
        self,
        symbol: str,
        timeframe: str,
        start_time: datetime,
        end_time: datetime,
    ) -> Optional[Dict[str, float]]:
        """Get aggregated OHLC for a time range."""
        row = self.adapter.fetch_one(f"""
            SELECT 
                MIN(timestamp) as first_ts,
                MAX(timestamp) as last_ts,
                (SELECT open FROM {self.TABLE} 
                 WHERE symbol = ? AND timeframe = ? AND timestamp >= ? AND timestamp <= ? AND status = ?
                 ORDER BY timestamp ASC LIMIT 1) as open,
                MAX(high) as high,
                MIN(low) as low,
                (SELECT close FROM {self.TABLE} 
                 WHERE symbol = ? AND timeframe = ? AND timestamp >= ? AND timestamp <= ? AND status = ?
                 ORDER BY timestamp DESC LIMIT 1) as close,
                SUM(volume) as volume
            FROM {self.TABLE} 
            WHERE symbol = ? AND timeframe = ? AND timestamp >= ? AND timestamp <= ? AND status = ?
        """, (
            symbol.upper(), timeframe.upper(), start_time, end_time, CandleStatus.ACTIVE.value,
            symbol.upper(), timeframe.upper(), start_time, end_time, CandleStatus.ACTIVE.value,
            symbol.upper(), timeframe.upper(), start_time, end_time, CandleStatus.ACTIVE.value,
        ))
        
        return dict(row) if row else None
    
    def get_volume_by_time_range(
        self,
        symbol: str,
        timeframe: str,
        start_time: datetime,
        end_time: datetime,
    ) -> float:
        """Get total volume for a time range."""
        row = self.adapter.fetch_one(f"""
            SELECT SUM(volume) as total_volume
            FROM {self.TABLE} 
            WHERE symbol = ? AND timeframe = ? 
            AND timestamp BETWEEN ? AND ? 
            AND status = ?
        """, (symbol.upper(), timeframe.upper(), self._sql_ts(start_time), self._sql_ts(end_time), CandleStatus.ACTIVE.value))
        
        return row['total_volume'] if row and row['total_volume'] else 0.0
    
    # ==========================================================================
    # DELETE
    # ==========================================================================
    
    def delete_by_time_range(
        self,
        symbol: str,
        timeframe: str,
        start_time: datetime,
        end_time: datetime,
    ) -> int:
        """Delete candles in a time range."""
        with self.transaction():
            result = self.delete_where(
                "symbol = ? AND timeframe = ? AND timestamp BETWEEN ? AND ?",
                (symbol.upper(), timeframe.upper(), start_time, end_time),
            )
            self._invalidate_cache(symbol, timeframe)
            return result
    
    def bulk_delete(self, symbol: str, timeframe: str, timestamps: List[datetime]) -> int:
        """Delete multiple candles by timestamp."""
        if not timestamps:
            return 0
        
        placeholders = ",".join("?" for _ in timestamps)
        with self.transaction():
            result = self.delete_where(
                f"symbol = ? AND timeframe = ? AND timestamp IN ({placeholders})",
                (symbol.upper(), timeframe.upper()) + tuple(timestamps),
            )
            self._invalidate_cache(symbol, timeframe)
            return result
    
    def replace_candles(
        self,
        symbol: str,
        timeframe: str,
        new_candles: List[Dict[str, Any]],
    ) -> int:
        """Replace all candles for a symbol/timeframe atomically."""
        with self.transaction():
            self.delete_where(
                "symbol = ? AND timeframe = ?",
                (symbol.upper(), timeframe.upper()),
            )
            result = self.bulk_upsert(new_candles)
            self._invalidate_cache(symbol, timeframe)
            return result
    
    # ==========================================================================
    # VALIDATION
    # ==========================================================================
    
    def _fast_validate_candle(self, data: Dict[str, Any]):
        """Fast validation for trusted sources (MT5)."""
        required = ['symbol', 'timeframe', 'timestamp', 'open', 'high', 'low', 'close']
        for field in required:
            if field not in data:
                raise ValueError(f"Missing required field: {field}")
    
    def _validate_candle(self, candle: Candle):
        """Validate a candle entity."""
        self._validate_candle_data({
            'symbol': candle.symbol,
            'timeframe': candle.timeframe,
            'timestamp': candle.timestamp,
            'open': candle.open,
            'high': candle.high,
            'low': candle.low,
            'close': candle.close,
            'volume': candle.volume,
        })
    
    def _validate_candle_data(self, data: Dict[str, Any]):
        """Validate candle data with comprehensive checks."""
        import math
        
        symbol = data.get('symbol', '')
        if not symbol:
            raise ValueError("Symbol is required")
        
        timeframe = data.get('timeframe', '')
        if not timeframe:
            raise ValueError("Timeframe is required")
        
        timestamp = data.get('timestamp')
        if not timestamp:
            raise ValueError("Timestamp is required")
        if not isinstance(timestamp, datetime):
            raise ValueError(f"Timestamp must be datetime, got {type(timestamp)}")
        
        try:
            open_price = float(data.get('open', 0))
            high = float(data.get('high', 0))
            low = float(data.get('low', 0))
            close = float(data.get('close', 0))
        except (TypeError, ValueError) as e:
            raise ValueError(f"Invalid price values: {e}")
        
        for name, value in [('open', open_price), ('high', high), ('low', low), ('close', close)]:
            if math.isnan(value):
                raise ValueError(f"{name} price is NaN")
            if math.isinf(value):
                raise ValueError(f"{name} price is infinite")
            if value <= 0:
                raise ValueError(f"{name} price must be positive: {value}")
        
        if high < low:
            raise ValueError(f"High ({high}) cannot be less than Low ({low})")
        if high < open_price:
            raise ValueError(f"High ({high}) cannot be less than Open ({open_price})")
        if high < close:
            raise ValueError(f"High ({high}) cannot be less than Close ({close})")
        if low > open_price:
            raise ValueError(f"Low ({low}) cannot be greater than Open ({open_price})")
        if low > close:
            raise ValueError(f"Low ({low}) cannot be greater than Close ({close})")
        
        try:
            volume = float(data.get('volume', 0))
        except (TypeError, ValueError) as e:
            raise ValueError(f"Invalid volume: {e}")
        
        if volume < 0:
            raise ValueError(f"Volume cannot be negative: {volume}")
    
    # ==========================================================================
    # STATISTICS
    # ==========================================================================
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get repository statistics."""
        base_stats = super().get_statistics()
        
        return {
            **base_stats,
            'active_candles': self.count("status = ?", (CandleStatus.ACTIVE.value,)),
            'archived_candles': self.count("status = ?", (CandleStatus.ARCHIVED.value,)),
            'invalid_candles': self.count("status = ?", (CandleStatus.INVALID.value,)),
        }
    
    # ==========================================================================
    # CONVERSION
    # ==========================================================================
    
    def _entity_to_dict(self, candle: Candle) -> Dict[str, Any]:
        """Convert Candle entity to dictionary."""
        ts = candle.timestamp
        if isinstance(ts, datetime):
            ts = ts.isoformat(timespec="seconds")
        created = candle.created_at
        updated = candle.updated_at
        if isinstance(created, datetime):
            created = created.isoformat(timespec="seconds")
        if isinstance(updated, datetime):
            updated = updated.isoformat(timespec="seconds")
        return {
            'candle_id': candle.candle_id,
            'candle_uuid': candle.candle_uuid,
            'symbol': candle.symbol,
            'timeframe': candle.timeframe,
            'timestamp': ts,
            'open': candle.open,
            'high': candle.high,
            'low': candle.low,
            'close': candle.close,
            'volume': candle.volume,
            'market_id': candle.market_id,
            'broker_id': candle.broker_id,
            'spread': candle.spread,
            'tick_volume': candle.tick_volume,
            'status': candle.status.value if candle.status else None,
            'metadata': json.dumps(candle.metadata) if candle.metadata else '{}',
            'created_at': created,
            'updated_at': updated,
        }
    
    def _row_to_entity(self, row: Dict[str, Any]) -> Candle:
        """Convert database row to Candle entity."""
        ts = row['timestamp']
        if isinstance(ts, str):
            ts = datetime.fromisoformat(ts)
        created = row.get('created_at')
        updated = row.get('updated_at')
        if isinstance(created, str):
            created = datetime.fromisoformat(created)
        if isinstance(updated, str):
            updated = datetime.fromisoformat(updated)
        return Candle(
            candle_id=row['candle_id'],
            candle_uuid=row['candle_uuid'],
            symbol=row['symbol'],
            timeframe=row['timeframe'],
            timestamp=ts,
            open=row['open'],
            high=row['high'],
            low=row['low'],
            close=row['close'],
            volume=row['volume'],
            market_id=row['market_id'],
            broker_id=row['broker_id'],
            spread=row['spread'],
            tick_volume=row['tick_volume'],
            status=CandleStatus(row['status']) if row['status'] else None,
            metadata=json.loads(row['metadata']) if row['metadata'] else {},
            created_at=created,
            updated_at=updated,
        )
    
    def _get_id(self, candle: Candle) -> int:
        """Get ID from Candle entity."""
        return candle.candle_id


# ==============================================================================
# REGISTER IN REPOSITORY MANAGER
# ==============================================================================

# To be added to REPOSITORIES list in repository_manager.py:
# ('candles', CandleRepository, None)