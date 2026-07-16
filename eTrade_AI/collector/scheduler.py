"""
collector/scheduler.py - Data Collection Scheduler

RESPONSIBILITY:
Manage periodic data collection for multiple symbol/timeframe pairs.

VERSION: 1.0.1
"""

import logging
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Set, Any, Tuple
from enum import Enum
from concurrent.futures import ThreadPoolExecutor, Future, as_completed

from core.config import Config
from database.database import DatabaseManager
from mt5.manager import MT5Manager
from collector.updater import Updater, UpdateResult, UpdateStatus

logger = logging.getLogger(__name__)


# ==============================================================================
# DATA CLASSES
# ==============================================================================

@dataclass
class Schedule:
    """
    A data collection schedule.
    
    Defines when and how often to update a symbol/timeframe.
    """
    symbol: str
    timeframe: str
    interval_seconds: int
    lookback_days: Optional[int] = None
    last_run: Optional[datetime] = None
    next_run: Optional[datetime] = None
    is_active: bool = True
    retry_count: int = 0
    max_retries: int = 3
    last_status: str = "pending"  # pending, success, failed, skipped
    last_error: Optional[str] = None
    consecutive_failures: int = 0
    total_downloads: int = 0
    total_skipped: int = 0
    total_errors: int = 0
    last_duration: float = 0.0
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    
    def should_run(self, current_time: datetime) -> bool:
        """Check if this schedule should run."""
        if not self.is_active:
            return False
        if self.next_run is None:
            return True
        return current_time >= self.next_run
    
    def update_next_run(self, current_time: datetime) -> None:
        """Update the next run time."""
        if self.last_run:
            self.next_run = self.last_run + timedelta(seconds=self.interval_seconds)
        else:
            self.next_run = current_time + timedelta(seconds=self.interval_seconds)
        
        # Ensure we don't fall behind
        while self.next_run < current_time:
            self.next_run += timedelta(seconds=self.interval_seconds)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'symbol': self.symbol,
            'timeframe': self.timeframe,
            'interval_seconds': self.interval_seconds,
            'lookback_days': self.lookback_days,
            'last_run': self.last_run.isoformat() if self.last_run else None,
            'next_run': self.next_run.isoformat() if self.next_run else None,
            'is_active': self.is_active,
            'retry_count': self.retry_count,
            'max_retries': self.max_retries,
            'last_status': self.last_status,
            'last_error': self.last_error,
            'consecutive_failures': self.consecutive_failures,
            'total_downloads': self.total_downloads,
            'total_skipped': self.total_skipped,
            'total_errors': self.total_errors,
            'last_duration': self.last_duration,
            'created_at': self.created_at.isoformat(),
            'updated_at': self.updated_at.isoformat(),
        }


class SchedulerStatus(Enum):
    """Scheduler running status."""
    STOPPED = "stopped"
    RUNNING = "running"
    PAUSED = "paused"
    STOPPING = "stopping"
    ERROR = "error"


# ==============================================================================
# SCHEDULER CLASS
# ==============================================================================

class DataCollectionScheduler:
    """
    Data collection scheduler - Thread-safe singleton.
    
    Manages periodic data collection for multiple symbol/timeframe pairs.
    
    THREAD-SAFETY:
    - All shared state is protected by threading.Lock
    - Schedule modifications are atomic
    - Status reads are consistent
    
    USAGE:
        scheduler = DataCollectionScheduler(
            db_manager=db,
            mt5_manager=mt5,
            updater=updater,
            config=config,
            logger=logger
        )
        scheduler.add_schedule("EURUSD", "H1", 3600)
        scheduler.start()
        
        # Later...
        scheduler.stop()
        scheduler.start()  # Can restart after stop
    """
    
    _instance: Optional['DataCollectionScheduler'] = None
    _instance_lock = threading.Lock()
    
    # Default configuration
    _DEFAULT_MAX_CONCURRENT = 3
    _DEFAULT_CHECK_INTERVAL = 10  # seconds
    _DEFAULT_MAX_RETRIES = 3
    
    def __new__(cls, *args, **kwargs):
        """Singleton pattern with thread safety."""
        if cls._instance is None:
            with cls._instance_lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(
        self,
        db_manager: DatabaseManager,
        mt5_manager: MT5Manager,
        updater: Updater,
        config: Config,
        logger: logging.Logger = None,
        max_concurrent: int = None,
        check_interval: int = None,
        max_retries: int = None,
    ):
        """
        Initialize the scheduler.
        
        Args:
            db_manager: DatabaseManager instance
            mt5_manager: MT5Manager instance
            updater: Updater instance
            config: Config instance
            logger: Optional logger instance
            max_concurrent: Maximum concurrent downloads
            check_interval: How often to check schedules (seconds)
            max_retries: Maximum retries per schedule
        """
        # Skip if already initialized
        if hasattr(self, '_initialized') and self._initialized:
            return
        
        # Store dependencies
        self._db_manager = db_manager
        self._mt5_manager = mt5_manager
        self._updater = updater
        self._config = config
        self._logger = logger or logging.getLogger(__name__)
        
        # Configuration
        self._max_concurrent = max_concurrent or self._DEFAULT_MAX_CONCURRENT
        self._check_interval = check_interval or self._DEFAULT_CHECK_INTERVAL
        self._max_retries = max_retries or self._DEFAULT_MAX_RETRIES
        
        # State
        self._schedules: Dict[str, Schedule] = {}
        self._lock = threading.Lock()
        self._status = SchedulerStatus.STOPPED
        self._thread: Optional[threading.Thread] = None
        self._executor: Optional[ThreadPoolExecutor] = None
        self._futures: List[Future] = []
        self._stop_event = threading.Event()
        self._pause_event = threading.Event()
        self._pause_event.set()  # Not paused by default
        
        # Metrics
        self._start_time: Optional[datetime] = None
        self._total_downloads = 0
        self._total_skipped = 0
        self._total_errors = 0
        
        # Flag to prevent re-initialization
        self._initialized = True
        
        self._logger.info(
            f"✅ Scheduler initialized: max_concurrent={self._max_concurrent}, "
            f"check_interval={self._check_interval}s, max_retries={self._max_retries}"
        )
    
    # ==========================================================================
    # PUBLIC METHODS
    # ==========================================================================
    
    def add_schedule(
        self,
        symbol: str,
        timeframe: str,
        interval_seconds: int,
        lookback_days: Optional[int] = None,
        max_retries: Optional[int] = None,
    ) -> bool:
        """
        Add a new schedule for symbol/timeframe.
        
        Args:
            symbol: Symbol to update (e.g., "EURUSD")
            timeframe: Timeframe to update (e.g., "H1")
            interval_seconds: How often to run (seconds)
            lookback_days: For first-time download, how far back to go
            max_retries: Override default max retries
        
        Returns:
            True if added successfully, False if already exists
        """
        with self._lock:
            key = self._get_schedule_key(symbol, timeframe)
            
            if key in self._schedules:
                self._logger.warning(
                    f"⚠️ Schedule already exists: {symbol} {timeframe}"
                )
                return False
            
            # Create schedule
            schedule = Schedule(
                symbol=symbol,
                timeframe=timeframe,
                interval_seconds=interval_seconds,
                lookback_days=lookback_days,
                max_retries=max_retries or self._max_retries,
                next_run=datetime.now() + timedelta(seconds=interval_seconds),
            )
            
            self._schedules[key] = schedule
            self._logger.info(
                f"✅ Schedule added: {symbol} {timeframe} (every {interval_seconds}s)"
            )
            
            return True
    
    def remove_schedule(self, symbol: str, timeframe: str) -> bool:
        """
        Remove a schedule.
        
        Args:
            symbol: Symbol
            timeframe: Timeframe
        
        Returns:
            True if removed, False if not found
        """
        with self._lock:
            key = self._get_schedule_key(symbol, timeframe)
            
            if key not in self._schedules:
                self._logger.warning(
                    f"⚠️ Schedule not found: {symbol} {timeframe}"
                )
                return False
            
            del self._schedules[key]
            self._logger.info(f"✅ Schedule removed: {symbol} {timeframe}")
            return True
    
    def start(self) -> None:
        """
        Start the scheduler.
        
        Begins the background thread that checks schedules.
        Can be called after stop() to restart.
        """
        with self._lock:
            if self._status == SchedulerStatus.RUNNING:
                self._logger.warning("⚠️ Scheduler already running")
                return
            
            # Reset state for restart
            if self._status == SchedulerStatus.STOPPED:
                self._stop_event.clear()
                self._pause_event.set()
                self._futures.clear()
                
                # Create new thread pool
                if self._executor is None or self._executor._shutdown:
                    self._executor = ThreadPoolExecutor(
                        max_workers=self._max_concurrent,
                        thread_name_prefix="scheduler-worker"
                    )
            
            self._status = SchedulerStatus.RUNNING
            self._start_time = datetime.now()
            
            # Create main thread if needed
            if self._thread is None or not self._thread.is_alive():
                self._thread = threading.Thread(
                    target=self._run_loop,
                    name="scheduler-main",
                    daemon=True,
                )
                self._thread.start()
            
            self._logger.info("▶️ Scheduler started")
    
    def stop(self, timeout: int = 30) -> None:
        """
        Stop the scheduler gracefully.
        
        Args:
            timeout: Maximum time to wait for pending downloads (seconds)
        
        Note:
            After stop(), the scheduler can be restarted with start()
        """
        with self._lock:
            if self._status in (SchedulerStatus.STOPPED, SchedulerStatus.STOPPING):
                return
            
            self._status = SchedulerStatus.STOPPING
            self._logger.info("⏹️ Stopping scheduler...")
        
        # Signal stop
        self._stop_event.set()
        self._pause_event.set()  # Resume if paused
        
        # Wait for pending downloads
        if self._futures:
            self._logger.info(f"⏳ Waiting for {len(self._futures)} pending downloads...")
            for future in as_completed(self._futures, timeout=timeout):
                try:
                    future.result(timeout=1)
                except Exception as e:
                    self._logger.warning(f"⚠️ Pending download error: {e}")
        
        # Shutdown executor
        if self._executor:
            self._executor.shutdown(wait=True, cancel_futures=False)
            self._executor = None
        
        # Clear futures
        self._futures.clear()
        
        # Wait for thread
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=5)
        
        with self._lock:
            self._status = SchedulerStatus.STOPPED
            self._thread = None
            self._start_time = None
            self._stop_event.clear()  # Reset for future start()
        
        self._logger.info("⏹️ Scheduler stopped")
    
    def pause(self) -> None:
        """Pause the scheduler."""
        with self._lock:
            if self._status != SchedulerStatus.RUNNING:
                self._logger.warning("⚠️ Scheduler not running, cannot pause")
                return
            
            self._status = SchedulerStatus.PAUSED
            self._pause_event.clear()
            self._logger.info("⏸️ Scheduler paused")
    
    def resume(self) -> None:
        """Resume the scheduler."""
        with self._lock:
            if self._status != SchedulerStatus.PAUSED:
                self._logger.warning("⚠️ Scheduler not paused, cannot resume")
                return
            
            self._status = SchedulerStatus.RUNNING
            self._pause_event.set()
            self._logger.info("▶️ Scheduler resumed")
    
    def get_status(self) -> Dict[str, Any]:
        """
        Get current scheduler status.
        
        Returns:
            Dict with status, schedules, metrics
        """
        with self._lock:
            schedules = []
            for schedule in self._schedules.values():
                schedules.append(schedule.to_dict())
            
            return {
                'status': self._status.value,
                'is_running': self._status == SchedulerStatus.RUNNING,
                'is_paused': self._status == SchedulerStatus.PAUSED,
                'total_schedules': len(self._schedules),
                'active_schedules': sum(1 for s in self._schedules.values() if s.is_active),
                'schedules': schedules,
                'pending_downloads': len(self._futures),
                'start_time': self._start_time.isoformat() if self._start_time else None,
                'uptime_seconds': self._get_uptime(),
            }
    
    def get_metrics(self) -> Dict[str, Any]:
        """
        Get scheduler metrics.
        
        Returns:
            Dict with metrics
        """
        with self._lock:
            return {
                'total_downloads': self._total_downloads,
                'total_skipped': self._total_skipped,
                'total_errors': self._total_errors,
                'uptime_seconds': self._get_uptime(),
                'schedules_count': len(self._schedules),
                'active_schedules': sum(1 for s in self._schedules.values() if s.is_active),
                'pending_downloads': len(self._futures),
                'max_concurrent': self._max_concurrent,
                'check_interval': self._check_interval,
            }
    
    def get_schedule_status(self, symbol: str, timeframe: str) -> Optional[Dict]:
        """
        Get status for a specific schedule.
        
        Args:
            symbol: Symbol
            timeframe: Timeframe
        
        Returns:
            Schedule dict or None
        """
        with self._lock:
            key = self._get_schedule_key(symbol, timeframe)
            if key in self._schedules:
                return self._schedules[key].to_dict()
            return None
    
    def reload_from_config(self) -> None:
        """Reload schedules from configuration."""
        self._logger.info("🔄 Reloading schedules from config...")
        # Implementation depends on Config structure
        # Placeholder for future implementation
    
    # ==========================================================================
    # PRIVATE METHODS
    # ==========================================================================
    
    def _run_loop(self) -> None:
        """
        Main scheduler loop.
        
        Runs in background thread, checks schedules periodically.
        
        FIXED: Simplified pause/resume logic - single wait at loop start.
        """
        self._logger.info("🔄 Scheduler loop started")
        
        while not self._stop_event.is_set():
            try:
                # Check pause - blocks until resumed or stopped
                self._pause_event.wait()
                if self._stop_event.is_set():
                    break
                
                # Process schedules
                self._process_schedules()
                
            except Exception as e:
                self._logger.exception(f"❌ Scheduler loop error: {e}")
            
            # Wait before next check (check stop event every second)
            for _ in range(self._check_interval):
                if self._stop_event.is_set():
                    break
                time.sleep(1)
        
        self._logger.info("🔄 Scheduler loop ended")
    
    def _process_schedules(self) -> None:
        """Process all active schedules."""
        current_time = datetime.now()
        schedules_to_run = []
        
        with self._lock:
            # Collect schedules that should run
            for schedule in self._schedules.values():
                if schedule.should_run(current_time):
                    schedules_to_run.append(schedule)
            
            if not schedules_to_run:
                return
            
            self._logger.debug(
                f"📊 {len(schedules_to_run)} schedules ready for processing"
            )
        
        # Submit downloads
        for schedule in schedules_to_run:
            self._submit_download(schedule, current_time)
    
    def _submit_download(self, schedule: Schedule, current_time: datetime) -> None:
        """
        Submit a download task to the thread pool.
        
        Args:
            schedule: Schedule to process
            current_time: Current time
        """
        if self._executor is None or self._executor._shutdown:
            self._logger.warning("⚠️ Executor not available, cannot submit download")
            return
        
        # Update schedule status
        with self._lock:
            schedule.last_run = current_time
            schedule.retry_count = 0
            schedule.update_next_run(current_time)
        
        # Submit to thread pool
        future = self._executor.submit(
            self._execute_download,
            schedule.symbol,
            schedule.timeframe,
            schedule.lookback_days,
            schedule.max_retries,
        )
        
        # Add callback
        future.add_done_callback(
            lambda f: self._on_download_complete(f, schedule.symbol, schedule.timeframe)
        )
        
        with self._lock:
            self._futures.append(future)
        
        self._logger.debug(
            f"📥 Download submitted: {schedule.symbol} {schedule.timeframe}"
        )
    
    def _execute_download(
        self,
        symbol: str,
        timeframe: str,
        lookback_days: Optional[int],
        max_retries: int,
    ) -> UpdateResult:
        """
        Execute a download with retry logic.
        
        This runs in a thread pool worker.
        
        Args:
            symbol: Symbol to update
            timeframe: Timeframe to update
            lookback_days: Lookback days for first-time download
            max_retries: Maximum retry attempts
        
        Returns:
            UpdateResult
        """
        attempt = 0
        last_error = None
        
        while attempt <= max_retries:
            try:
                if attempt > 0:
                    delay = min(60, 2 ** attempt)  # Exponential backoff
                    self._logger.info(
                        f"🔄 Retry {attempt}/{max_retries}: {symbol} {timeframe} "
                        f"(waiting {delay}s)"
                    )
                    time.sleep(delay)
                
                result = self._updater.update(
                    symbol=symbol,
                    timeframe=timeframe,
                    lookback_days=lookback_days,
                    force_full=False,
                )
                
                # Update metrics
                with self._lock:
                    self._total_downloads += result.candles_downloaded
                    self._total_skipped += result.candles_skipped
                    if result.status == UpdateStatus.FAILED.value:
                        self._total_errors += 1
                
                return result
                
            except Exception as e:
                last_error = str(e)
                attempt += 1
                self._logger.warning(
                    f"⚠️ Download attempt {attempt}/{max_retries} failed: "
                    f"{symbol} {timeframe}: {e}"
                )
        
        # All retries failed
        return UpdateResult(
            symbol=symbol,
            timeframe=timeframe,
            status=UpdateStatus.FAILED.value,
            error_message=f"All {max_retries} retries failed: {last_error}",
        )
    
    def _on_download_complete(self, future: Future, symbol: str, timeframe: str) -> None:
        """
        Callback when download completes.
        
        Args:
            future: Completed future
            symbol: Symbol
            timeframe: Timeframe
        """
        # Remove future from list
        with self._lock:
            if future in self._futures:
                self._futures.remove(future)
        
        try:
            result = future.result(timeout=1)
        except Exception as e:
            self._logger.error(
                f"❌ Download failed: {symbol} {timeframe}: {e}"
            )
            return
        
        # Log result
        if result.status == UpdateStatus.SUCCESS.value:
            self._logger.info(
                f"✅ Download complete: {symbol} {timeframe} | "
                f"Downloaded: {result.candles_downloaded}, "
                f"Skipped: {result.candles_skipped}, "
                f"Elapsed: {result.elapsed_time:.2f}s"
            )
        elif result.status == UpdateStatus.NO_NEW_DATA.value:
            self._logger.debug(
                f"⏱️ No new data: {symbol} {timeframe}"
            )
        else:
            self._logger.warning(
                f"⚠️ Download failed: {symbol} {timeframe} | "
                f"Status: {result.status}, Error: {result.error_message}"
            )
        
        # Update schedule
        with self._lock:
            key = self._get_schedule_key(symbol, timeframe)
            if key in self._schedules:
                schedule = self._schedules[key]
                schedule.last_status = result.status
                schedule.last_duration = result.elapsed_time
                
                if result.status == UpdateStatus.SUCCESS.value:
                    schedule.total_downloads += result.candles_downloaded
                    schedule.total_skipped += result.candles_skipped
                    schedule.consecutive_failures = 0
                    schedule.last_error = None
                elif result.status == UpdateStatus.NO_NEW_DATA.value:
                    schedule.total_skipped += result.candles_skipped
                    schedule.consecutive_failures = 0
                else:
                    schedule.total_errors += 1
                    schedule.consecutive_failures += 1
                    schedule.last_error = result.error_message
                    schedule.last_status = result.status
                
                schedule.updated_at = datetime.now()
    
    def _get_schedule_key(self, symbol: str, timeframe: str) -> str:
        """Generate schedule key."""
        return f"{symbol}|{timeframe}"
    
    def _get_uptime(self) -> Optional[float]:
        """Get uptime in seconds."""
        if self._start_time:
            return (datetime.now() - self._start_time).total_seconds()
        return None
    
    # ==========================================================================
    # DUNDER METHODS
    # ==========================================================================
    
    def __enter__(self):
        """Context manager entry."""
        self.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.stop()
    
    def __del__(self):
        """Cleanup on destruction."""
        try:
            self.stop(timeout=5)
        except Exception:
            pass


# ==============================================================================
# FACTORY FUNCTION
# ==============================================================================

def create_scheduler(
    db_manager: DatabaseManager,
    mt5_manager: MT5Manager,
    updater: Updater,
    config: Config,
    logger: logging.Logger = None,
) -> DataCollectionScheduler:
    """
    Factory function for scheduler creation.
    
    Ensures all dependencies are properly injected.
    
    Args:
        db_manager: DatabaseManager instance
        mt5_manager: MT5Manager instance
        updater: Updater instance
        config: Config instance
        logger: Optional logger instance
    
    Returns:
        DataCollectionScheduler instance
    """
    return DataCollectionScheduler(
        db_manager=db_manager,
        mt5_manager=mt5_manager,
        updater=updater,
        config=config,
        logger=logger,
    )