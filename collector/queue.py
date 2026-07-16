"""
collector/queue.py - Priority-based Download Queue

RESPONSIBILITY:
Manage a priority-based work queue for data collection tasks.

This is the task management layer for the collector system.
It DOES NOT:
- Download data (delegates to Updater)
- Manage connections (uses DatabaseManager)
- Schedule tasks (used by Scheduler)

ARCHITECTURAL DECISIONS:
1. Priority-based ordering (higher priority = first out)
2. Persistent storage for crash recovery
3. Thread-safe operations
4. Duplicate prevention
5. Status tracking (queued, processing, completed, failed)
6. Auto-priority calculation based on data staleness

USAGE:
    queue = DownloadQueue(db_manager, config, logger)
    queue.enqueue("EURUSD", "H1", priority=80)
    task = queue.dequeue()
    if task:
        # Process task
        queue.mark_completed(task.symbol, task.timeframe)

VERSION: 1.0.0
"""

import json
import logging
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Any, Tuple
from queue import PriorityQueue

from core.config import Config
from database.database import DatabaseManager

logger = logging.getLogger(__name__)


# ==============================================================================
# ENUMS
# ==============================================================================

class TaskStatus(Enum):
    """Status of a download task."""
    QUEUED = "queued"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class TaskPriority(Enum):
    """Predefined priority levels."""
    CRITICAL = 100
    HIGH = 80
    NORMAL = 50
    LOW = 20
    BACKGROUND = 10


# ==============================================================================
# DATA CLASSES
# ==============================================================================

@dataclass
class DownloadTask:
    """
    A data collection task.
    
    Represents a single symbol/timeframe download request.
    """
    symbol: str
    timeframe: str
    priority: int = TaskPriority.NORMAL.value
    added_at: datetime = field(default_factory=datetime.now)
    attempts: int = 0
    max_attempts: int = 3
    status: str = TaskStatus.QUEUED.value
    error_message: Optional[str] = None
    completed_at: Optional[datetime] = None
    last_attempt: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'symbol': self.symbol,
            'timeframe': self.timeframe,
            'priority': self.priority,
            'added_at': self.added_at.isoformat() if self.added_at else None,
            'attempts': self.attempts,
            'max_attempts': self.max_attempts,
            'status': self.status,
            'error_message': self.error_message,
            'completed_at': self.completed_at.isoformat() if self.completed_at else None,
            'last_attempt': self.last_attempt.isoformat() if self.last_attempt else None,
            'metadata': self.metadata,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'DownloadTask':
        """Create DownloadTask from dictionary."""
        return cls(
            symbol=data['symbol'],
            timeframe=data['timeframe'],
            priority=data.get('priority', TaskPriority.NORMAL.value),
            added_at=datetime.fromisoformat(data['added_at']) if data.get('added_at') else datetime.now(),
            attempts=data.get('attempts', 0),
            max_attempts=data.get('max_attempts', 3),
            status=data.get('status', TaskStatus.QUEUED.value),
            error_message=data.get('error_message'),
            completed_at=datetime.fromisoformat(data['completed_at']) if data.get('completed_at') else None,
            last_attempt=datetime.fromisoformat(data['last_attempt']) if data.get('last_attempt') else None,
            metadata=data.get('metadata', {}),
        )
    
    def __lt__(self, other: 'DownloadTask') -> bool:
        """
        Compare tasks for priority queue ordering.
        
        Higher priority = first out.
        If same priority, older tasks first.
        """
        if self.priority != other.priority:
            return self.priority > other.priority
        return self.added_at < other.added_at


# ==============================================================================
# MAIN QUEUE CLASS
# ==============================================================================

class DownloadQueue:
    """
    Priority-based download queue - Thread-safe singleton.
    
    Manages data collection tasks with priority ordering and persistence.
    
    THREAD-SAFETY:
    - All shared state is protected by threading.Lock
    - PriorityQueue is thread-safe internally
    - Status updates are atomic
    
    PERSISTENCE:
    - Incomplete tasks are saved to database
    - Crash recovery loads pending tasks on initialization
    - Completed tasks are removed from persistence
    
    USAGE:
        queue = DownloadQueue(db_manager, config, logger)
        queue.enqueue("EURUSD", "H1", priority=80)
        queue.enqueue("BTCUSD", "M5", priority=100)
        
        task = queue.dequeue()
        if task:
            try:
                # Download data...
                queue.mark_completed(task.symbol, task.timeframe)
            except Exception as e:
                queue.mark_failed(task.symbol, task.timeframe, str(e))
    
    VERSION: 1.0.0
    """
    
    _instance: Optional['DownloadQueue'] = None
    _instance_lock = threading.Lock()
    
    # Default configuration
    _DEFAULT_MAX_SIZE = 10000
    _DEFAULT_MAX_ATTEMPTS = 3
    _PERSISTENCE_TABLE = "download_queue"
    
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
        config: Config,
        logger: logging.Logger = None,
        max_size: int = None,
        max_attempts: int = None,
        auto_load: bool = True,
    ):
        """
        Initialize the download queue.
        
        Args:
            db_manager: DatabaseManager instance
            config: Config instance
            logger: Optional logger instance
            max_size: Maximum queue size (prevents memory bloat)
            max_attempts: Maximum retry attempts per task
            auto_load: Automatically load pending tasks from database
        """
        # Skip if already initialized
        if hasattr(self, '_initialized') and self._initialized:
            return
        
        # Store dependencies
        self._db_manager = db_manager
        self._config = config
        self._logger = logger or logging.getLogger(__name__)
        
        # Configuration
        self._max_size = max_size or self._DEFAULT_MAX_SIZE
        self._max_attempts = max_attempts or self._DEFAULT_MAX_ATTEMPTS
        
        # Internal queue (thread-safe PriorityQueue)
        self._queue: PriorityQueue = PriorityQueue()
        self._lock = threading.Lock()
        
        # Task tracking
        self._active_tasks: Dict[str, DownloadTask] = {}  # symbol|timeframe -> task
        self._processing_tasks: Dict[str, DownloadTask] = {}  # symbol|timeframe -> task
        self._completed_tasks: Dict[str, DownloadTask] = {}  # symbol|timeframe -> task
        self._failed_tasks: Dict[str, DownloadTask] = {}  # symbol|timeframe -> task
        
        # Metrics
        self._enqueued_count = 0
        self._completed_count = 0
        self._failed_count = 0
        self._cancelled_count = 0
        self._start_time = datetime.now()
        
        # Flag to prevent re-initialization
        self._initialized = True
        
        # Load pending tasks from database
        if auto_load:
            self.load_from_db()
        
        self._logger.info(
            f"✅ DownloadQueue initialized: max_size={self._max_size}, "
            f"max_attempts={self._max_attempts}, active_tasks={len(self._active_tasks)}"
        )
    
    # ==========================================================================
    # PUBLIC METHODS
    # ==========================================================================
    
    def enqueue(
        self,
        symbol: str,
        timeframe: str,
        priority: int = TaskPriority.NORMAL.value,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """
        Add a task to the queue.
        
        Args:
            symbol: Symbol to download
            timeframe: Timeframe to download
            priority: Priority (0-100, higher = first out)
            metadata: Optional metadata for the task
        
        Returns:
            True if enqueued successfully, False if duplicate or queue full
        
        Raises:
            ValueError: If symbol or timeframe is invalid
        """
        if not symbol or not timeframe:
            raise ValueError(f"Invalid symbol/timeframe: {symbol}/{timeframe}")
        
        # Validate priority
        if not 0 <= priority <= 100:
            raise ValueError(f"Priority must be between 0 and 100: {priority}")
        
        with self._lock:
            # Check for duplicates
            key = self._get_task_key(symbol, timeframe)
            if key in self._active_tasks:
                self._logger.warning(
                    f"⚠️ Task already enqueued: {symbol} {timeframe}"
                )
                return False
            
            if key in self._processing_tasks:
                self._logger.warning(
                    f"⚠️ Task is currently processing: {symbol} {timeframe}"
                )
                return False
            
            # Check queue size
            if self._queue.qsize() >= self._max_size:
                self._logger.warning(
                    f"⚠️ Queue full ({self._max_size}), cannot enqueue: {symbol} {timeframe}"
                )
                return False
            
            # Create task
            task = DownloadTask(
                symbol=symbol,
                timeframe=timeframe,
                priority=priority,
                max_attempts=self._max_attempts,
                metadata=metadata or {},
            )
            
            # Add to queue and tracking
            self._queue.put(task)
            self._active_tasks[key] = task
            self._enqueued_count += 1
            
            self._logger.debug(
                f"📥 Task enqueued: {symbol} {timeframe} (priority={priority})"
            )
            
            # Persist to database
            self._persist_task(task)
            
            return True
    
    def dequeue(self) -> Optional[DownloadTask]:
        """
        Get the highest priority task from the queue.
        
        Returns:
            DownloadTask or None if queue is empty
        
        Note:
            Task is moved from active to processing state.
            Call mark_completed() or mark_failed() when done.
        """
        with self._lock:
            if self._queue.empty():
                return None
            
            # Get highest priority task
            task = self._queue.get()
            key = self._get_task_key(task.symbol, task.timeframe)
            
            # Move from active to processing
            if key in self._active_tasks:
                del self._active_tasks[key]
            
            task.status = TaskStatus.PROCESSING.value
            task.last_attempt = datetime.now()
            task.attempts += 1
            self._processing_tasks[key] = task
            
            # Update persistence
            self._persist_task(task)
            
            self._logger.debug(
                f"🔄 Task dequeued: {task.symbol} {task.timeframe} "
                f"(attempt {task.attempts}/{task.max_attempts})"
            )
            
            return task
    
    def peek(self) -> Optional[DownloadTask]:
        """
        View the highest priority task without removing it.
        
        Returns:
            DownloadTask or None if queue is empty
        """
        with self._lock:
            if self._queue.empty():
                return None
            
            # Get highest priority task without removing
            task = self._queue.get()
            self._queue.put(task)
            return task
    
    def get_size(self) -> int:
        """Get the number of tasks in the queue."""
        with self._lock:
            return self._queue.qsize() + len(self._processing_tasks)
    
    def get_all_tasks(self) -> List[DownloadTask]:
        """
        Get all tasks across all states.
        
        Returns:
            List of all DownloadTask objects
        """
        with self._lock:
            tasks = []
            
            # Active tasks
            tasks.extend(self._active_tasks.values())
            
            # Processing tasks
            tasks.extend(self._processing_tasks.values())
            
            # Completed tasks (limited to recent)
            tasks.extend(list(self._completed_tasks.values())[-100:])
            
            # Failed tasks (limited to recent)
            tasks.extend(list(self._failed_tasks.values())[-100:])
            
            return tasks
    
    def mark_completed(self, symbol: str, timeframe: str) -> bool:
        """
        Mark a task as completed.
        
        Args:
            symbol: Symbol
            timeframe: Timeframe
        
        Returns:
            True if task was found and marked
        """
        key = self._get_task_key(symbol, timeframe)
        
        with self._lock:
            if key in self._processing_tasks:
                task = self._processing_tasks.pop(key)
                task.status = TaskStatus.COMPLETED.value
                task.completed_at = datetime.now()
                self._completed_tasks[key] = task
                self._completed_count += 1
                
                # Remove from persistence
                self._remove_persisted_task(task)
                
                self._logger.debug(
                    f"✅ Task completed: {symbol} {timeframe}"
                )
                return True
            
            self._logger.warning(
                f"⚠️ Task not in processing state: {symbol} {timeframe}"
            )
            return False
    
    def mark_failed(self, symbol: str, timeframe: str, error: str) -> bool:
        """
        Mark a task as failed.
        
        Args:
            symbol: Symbol
            timeframe: Timeframe
            error: Error message
        
        Returns:
            True if task was found and marked
        
        Note:
            If attempts < max_attempts, task is re-enqueued.
            Otherwise, it's moved to failed state.
        """
        key = self._get_task_key(symbol, timeframe)
        
        with self._lock:
            if key not in self._processing_tasks:
                self._logger.warning(
                    f"⚠️ Task not in processing state: {symbol} {timeframe}"
                )
                return False
            
            task = self._processing_tasks.pop(key)
            task.error_message = error
            task.status = TaskStatus.FAILED.value
            task.last_attempt = datetime.now()
            
            # Check if we should retry
            if task.attempts < task.max_attempts:
                # Re-enqueue with adjusted priority
                new_priority = max(0, task.priority + 10)  # Boost priority
                self._logger.info(
                    f"🔄 Retrying task: {symbol} {timeframe} "
                    f"(attempt {task.attempts + 1}/{task.max_attempts}, "
                    f"priority={new_priority})"
                )
                
                # Reset status and re-enqueue
                task.status = TaskStatus.QUEUED.value
                task.priority = new_priority
                self._queue.put(task)
                self._active_tasks[key] = task
                
                # Update persistence
                self._persist_task(task)
                
            else:
                # Max attempts reached, mark as failed
                self._failed_tasks[key] = task
                self._failed_count += 1
                self._logger.error(
                    f"❌ Task failed permanently: {symbol} {timeframe} "
                    f"(attempts={task.attempts}, error={error})"
                )
                
                # Update persistence
                self._persist_task(task)
            
            return True
    
    def reprioritize(self, symbol: str, timeframe: str, new_priority: int) -> bool:
        """
        Change priority of an existing task.
        
        Args:
            symbol: Symbol
            timeframe: Timeframe
            new_priority: New priority (0-100)
        
        Returns:
            True if task was found and reprioritized
        """
        if not 0 <= new_priority <= 100:
            raise ValueError(f"Priority must be between 0 and 100: {new_priority}")
        
        key = self._get_task_key(symbol, timeframe)
        
        with self._lock:
            # Check active tasks
            if key in self._active_tasks:
                task = self._active_tasks[key]
                task.priority = new_priority
                self._logger.debug(
                    f"📊 Task reprioritized: {symbol} {timeframe} -> {new_priority}"
                )
                return True
            
            # Check processing tasks
            if key in self._processing_tasks:
                task = self._processing_tasks[key]
                task.priority = new_priority
                self._logger.debug(
                    f"📊 Processing task reprioritized: {symbol} {timeframe} -> {new_priority}"
                )
                return True
            
            self._logger.warning(
                f"⚠️ Task not found for reprioritization: {symbol} {timeframe}"
            )
            return False
    
    def cancel_task(self, symbol: str, timeframe: str) -> bool:
        """
        Cancel a queued or processing task.
        
        Args:
            symbol: Symbol
            timeframe: Timeframe
        
        Returns:
            True if task was found and cancelled
        """
        key = self._get_task_key(symbol, timeframe)
        
        with self._lock:
            # Check active tasks
            if key in self._active_tasks:
                task = self._active_tasks.pop(key)
                task.status = TaskStatus.CANCELLED.value
                self._cancelled_count += 1
                self._remove_persisted_task(task)
                self._logger.info(
                    f"⏹️ Task cancelled: {symbol} {timeframe}"
                )
                return True
            
            # Check processing tasks
            if key in self._processing_tasks:
                task = self._processing_tasks.pop(key)
                task.status = TaskStatus.CANCELLED.value
                self._cancelled_count += 1
                self._remove_persisted_task(task)
                self._logger.info(
                    f"⏹️ Processing task cancelled: {symbol} {timeframe}"
                )
                return True
            
            self._logger.warning(
                f"⚠️ Task not found for cancellation: {symbol} {timeframe}"
            )
            return False
    
    def clear(self) -> None:
        """Clear all tasks from the queue (keeps processing tasks)."""
        with self._lock:
            # Clear queue
            while not self._queue.empty():
                try:
                    task = self._queue.get_nowait()
                    self._remove_persisted_task(task)
                except:
                    pass
            
            # Clear active tasks
            for task in list(self._active_tasks.values()):
                self._remove_persisted_task(task)
            self._active_tasks.clear()
            
            # Don't clear processing tasks
            self._logger.info(
                f"🗑️ Queue cleared: {len(self._processing_tasks)} tasks still processing"
            )
    
    def load_from_db(self) -> int:
        """
        Load pending tasks from database.
        
        Returns:
            Number of tasks loaded
        
        Note:
            This is called automatically during initialization.
        """
        try:
            with self._lock:
                # Query pending tasks
                query = f"""
                    SELECT task_data FROM {self._PERSISTENCE_TABLE}
                    WHERE status IN ('{TaskStatus.QUEUED.value}', '{TaskStatus.PROCESSING.value}')
                    ORDER BY priority DESC, added_at ASC
                """
                
                results = self._db_manager.fetchall(query)
                
                loaded_count = 0
                for row in results:
                    try:
                        data = json.loads(row['task_data'])
                        task = DownloadTask.from_dict(data)
                        key = self._get_task_key(task.symbol, task.timeframe)
                        
                        # Only load if not already present
                        if key not in self._active_tasks and key not in self._processing_tasks:
                            if task.status == TaskStatus.QUEUED.value:
                                self._queue.put(task)
                                self._active_tasks[key] = task
                            else:
                                self._processing_tasks[key] = task
                            
                            loaded_count += 1
                    
                    except Exception as e:
                        self._logger.warning(f"⚠️ Failed to load task: {e}")
                
                self._logger.info(
                    f"📂 Loaded {loaded_count} pending tasks from database"
                )
                return loaded_count
                
        except Exception as e:
            self._logger.warning(f"⚠️ Could not load tasks from database: {e}")
            return 0
    
    def persist_to_db(self) -> int:
        """
        Persist all pending tasks to database.
        
        Returns:
            Number of tasks persisted
        
        Note:
            This is called automatically on task changes.
        """
        with self._lock:
            # Collect all pending tasks
            pending_tasks = []
            pending_tasks.extend(self._active_tasks.values())
            pending_tasks.extend(self._processing_tasks.values())
            
            if not pending_tasks:
                return 0
            
            # Clear existing pending tasks
            self._db_manager.execute(
                f"DELETE FROM {self._PERSISTENCE_TABLE}"
            )
            
            # Insert current pending tasks
            for task in pending_tasks:
                self._persist_task(task)
            
            self._logger.debug(
                f"💾 Persisted {len(pending_tasks)} tasks to database"
            )
            return len(pending_tasks)
    
    def get_metrics(self) -> Dict[str, Any]:
        """
        Get queue metrics.
        
        Returns:
            Dict with metrics
        """
        with self._lock:
            return {
                'queue_size': self._queue.qsize(),
                'active_tasks': len(self._active_tasks),
                'processing_tasks': len(self._processing_tasks),
                'completed_tasks': len(self._completed_tasks),
                'failed_tasks': len(self._failed_tasks),
                'total_enqueued': self._enqueued_count,
                'total_completed': self._completed_count,
                'total_failed': self._failed_count,
                'total_cancelled': self._cancelled_count,
                'max_size': self._max_size,
                'max_attempts': self._max_attempts,
                'uptime_seconds': (datetime.now() - self._start_time).total_seconds(),
            }
    
    def get_queue_status(self) -> Dict[str, Any]:
        """
        Get detailed queue status.
        
        Returns:
            Dict with queue status
        """
        with self._lock:
            # Get next few tasks (without removing)
            tasks = []
            temp_queue = []
            
            # Peek at up to 5 tasks
            for _ in range(min(5, self._queue.qsize())):
                try:
                    task = self._queue.get()
                    tasks.append(task.to_dict())
                    temp_queue.append(task)
                except:
                    break
            
            # Put tasks back
            for task in temp_queue:
                self._queue.put(task)
            
            return {
                'queue_size': self._queue.qsize(),
                'processing_count': len(self._processing_tasks),
                'pending_count': len(self._active_tasks),
                'next_tasks': tasks,
                'queue_full': self._queue.qsize() >= self._max_size,
                'metrics': self.get_metrics(),
            }
    
    # ==========================================================================
    # PRIVATE METHODS
    # ==========================================================================
    
    def _get_task_key(self, symbol: str, timeframe: str) -> str:
        """Generate task key."""
        return f"{symbol}|{timeframe}"
    
    def _persist_task(self, task: DownloadTask) -> None:
        """
        Persist a single task to database.
        
        Args:
            task: Task to persist
        """
        try:
            data = task.to_dict()
            json_data = json.dumps(data, default=str)
            
            self._db_manager.execute(
                f"""
                INSERT OR REPLACE INTO {self._PERSISTENCE_TABLE}
                (task_id, symbol, timeframe, priority, status, task_data, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    self._get_task_key(task.symbol, task.timeframe),
                    task.symbol,
                    task.timeframe,
                    task.priority,
                    task.status,
                    json_data,
                    datetime.now().isoformat(),
                )
            )
        except Exception as e:
            self._logger.warning(f"⚠️ Failed to persist task: {e}")
    
    def _remove_persisted_task(self, task: DownloadTask) -> None:
        """
        Remove a task from persistence.
        
        Args:
            task: Task to remove
        """
        try:
            self._db_manager.execute(
                f"DELETE FROM {self._PERSISTENCE_TABLE} "
                "WHERE task_id = ?",
                (self._get_task_key(task.symbol, task.timeframe),)
            )
        except Exception as e:
            self._logger.warning(f"⚠️ Failed to remove persisted task: {e}")
    
    # ==========================================================================
    # DUNDER METHODS
    # ==========================================================================
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.persist_to_db()
    
    def __del__(self):
        try:
            self.persist_to_db()
        except:
            pass


# ==============================================================================
# FACTORY FUNCTION
# ==============================================================================

def create_download_queue(
    db_manager: DatabaseManager,
    config: Config,
    logger: logging.Logger = None,
    max_size: int = None,
    max_attempts: int = None,
) -> DownloadQueue:
    """
    Factory function for queue creation.
    
    Args:
        db_manager: DatabaseManager instance
        config: Config instance
        logger: Optional logger instance
        max_size: Maximum queue size
        max_attempts: Maximum retry attempts
    
    Returns:
        DownloadQueue instance
    """
    return DownloadQueue(
        db_manager=db_manager,
        config=config,
        logger=logger,
        max_size=max_size,
        max_attempts=max_attempts,
    )