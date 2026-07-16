"""
core/events.py - Event Pub/Sub System for MarketAI

RESPONSIBILITY:
Provide a thread-safe event bus for communication between MarketAI components.

ARCHITECTURAL DECISIONS:
1. Singleton pattern - Single event bus for the entire system
2. Thread-safe - Uses threading.Lock for all operations
3. Priority-based - Higher priority listeners called first
4. Async emission - Non-blocking via ThreadPoolExecutor
5. Error isolation - Listener exceptions don't crash the bus
6. Event types enum - Type-safe event classification
7. Metrics tracking - For monitoring and debugging

USAGE:
    # Get bus instance
    bus = EventBus.get_instance()
    
    # Register listener
    bus.register_listener(EventType.MARKET_DISCOVERED, on_market_discovered)
    
    # Emit event
    bus.emit(Event(
        event_type=EventType.MARKET_DISCOVERED,
        source="seed",
        data={'symbol': 'EURUSD', 'market_id': 123}
    ))

VERSION: 1.0.0
"""

import logging
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from typing import Dict, List, Optional, Set, Any, Callable, Tuple, Union
from concurrent.futures import ThreadPoolExecutor, Future

from core.logger import get_logger

logger = get_logger(__name__)


# ==============================================================================
# EVENT TYPES
# ==============================================================================

class EventType(Enum):
    """
    Standard event types for MarketAI.
    
    These are the primary event types used for communication between components.
    """
    
    # ====== MARKET EVENTS ======
    MARKET_DISCOVERED = "market.discovered"
    MARKET_REMOVED = "market.removed"
    MARKET_UPDATED = "market.updated"
    MARKET_ACTIVATED = "market.activated"
    MARKET_DEACTIVATED = "market.deactivated"
    
    # ====== DATA EVENTS ======
    DATA_DOWNLOADED = "data.downloaded"
    DATA_VALIDATED = "data.validated"
    DATA_CLEANED = "data.cleaned"
    DATA_MERGED = "data.merged"
    DATA_AVAILABILITY_CHANGED = "data.availability.changed"
    
    # ====== DISCOVERY EVENTS ======
    PATTERN_DISCOVERED = "pattern.discovered"
    PATTERN_VALIDATED = "pattern.validated"
    PATTERN_DEPRECATED = "pattern.deprecated"
    CORRELATION_FOUND = "correlation.found"
    CORRELATION_LOST = "correlation.lost"
    CLUSTER_DISCOVERED = "cluster.discovered"
    
    # ====== PREDICTION EVENTS ======
    PREDICTION_MADE = "prediction.made"
    PREDICTION_CONFIRMED = "prediction.confirmed"
    PREDICTION_FAILED = "prediction.failed"
    
    # ====== VALIDATION EVENTS ======
    VALIDATION_STARTED = "validation.started"
    VALIDATION_COMPLETE = "validation.complete"
    VALIDATION_FAILED = "validation.failed"
    VALIDATION_CONFIDENCE_CHANGED = "validation.confidence.changed"
    
    # ====== COLLECTOR EVENTS ======
    COLLECTOR_STARTED = "collector.started"
    COLLECTOR_STOPPED = "collector.stopped"
    COLLECTOR_TASK_STARTED = "collector.task.started"
    COLLECTOR_TASK_COMPLETED = "collector.task.completed"
    COLLECTOR_TASK_FAILED = "collector.task.failed"
    
    # ====== SCHEDULER EVENTS ======
    SCHEDULER_STARTED = "scheduler.started"
    SCHEDULER_STOPPED = "scheduler.stopped"
    SCHEDULER_PAUSED = "scheduler.paused"
    SCHEDULER_RESUMED = "scheduler.resumed"
    SCHEDULE_ADDED = "schedule.added"
    SCHEDULE_REMOVED = "schedule.removed"
    
    # ====== QUEUE EVENTS ======
    QUEUE_TASK_ENQUEUED = "queue.task.enqueued"
    QUEUE_TASK_DEQUEUED = "queue.task.dequeued"
    QUEUE_TASK_COMPLETED = "queue.task.completed"
    QUEUE_TASK_FAILED = "queue.task.failed"
    QUEUE_FULL = "queue.full"
    QUEUE_EMPTY = "queue.empty"
    
    # ====== SYSTEM EVENTS ======
    SYSTEM_STARTED = "system.started"
    SYSTEM_STOPPED = "system.stopped"
    SYSTEM_ERROR = "system.error"
    SYSTEM_WARNING = "system.warning"
    SYSTEM_STATUS_CHANGED = "system.status.changed"
    
    # ====== ERROR EVENTS ======
    ERROR_OCCURRED = "error.occurred"
    ERROR_RECOVERED = "error.recovered"
    RETRY_ATTEMPT = "retry.attempt"
    RETRY_EXHAUSTED = "retry.exhausted"


# ==============================================================================
# EVENT CLASS
# ==============================================================================

@dataclass
class Event:
    """
    Event object for the pub/sub system.
    
    Attributes:
        event_type: The type of event (from EventType enum)
        timestamp: When the event was created
        source: The component that created the event
        data: Event payload data
        priority: Priority (0-100, higher = processed first)
        id: Unique event ID (auto-generated)
        correlation_id: For tracking related events
    """
    event_type: Union[EventType, str]
    source: str
    data: Dict[str, Any] = field(default_factory=dict)
    priority: int = 50
    timestamp: datetime = field(default_factory=datetime.now)
    correlation_id: Optional[str] = None
    id: str = field(default_factory=lambda: f"evt_{int(time.time()*1000)}_{id(object())}")
    
    def __post_init__(self):
        """Validate event type."""
        if isinstance(self.event_type, EventType):
            self.event_type_str = self.event_type.value
        else:
            self.event_type_str = str(self.event_type)
    
    @property
    def type_str(self) -> str:
        """Get string representation of event type."""
        return self.event_type_str
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert event to dictionary."""
        return {
            'id': self.id,
            'event_type': self.type_str,
            'timestamp': self.timestamp.isoformat(),
            'source': self.source,
            'data': self.data,
            'priority': self.priority,
            'correlation_id': self.correlation_id,
        }
    
    def __lt__(self, other: 'Event') -> bool:
        """For priority queue ordering."""
        if self.priority != other.priority:
            return self.priority > other.priority
        return self.timestamp < other.timestamp


# ==============================================================================
# LISTENER WRAPPER
# ==============================================================================

@dataclass
class ListenerWrapper:
    """
    Wrapper for event listeners with priority.
    
    Attributes:
        callback: The callback function
        priority: Priority (higher = called first)
        name: Optional name for the listener
    """
    callback: Callable[[Event], None]
    priority: int = 50
    name: Optional[str] = None
    
    def __call__(self, event: Event) -> None:
        """Call the wrapped callback."""
        return self.callback(event)
    
    def __lt__(self, other: 'ListenerWrapper') -> bool:
        """For priority ordering."""
        if self.priority != other.priority:
            return self.priority > other.priority
        return (self.name or "") < (other.name or "")
    
    def __hash__(self) -> int:
        return hash((self.callback, self.priority))


# ==============================================================================
# EVENT BUS (Singleton)
# ==============================================================================

class EventBus:
    """
    Thread-safe event bus for MarketAI.
    
    Singleton pattern ensures a single bus for the entire application.
    Supports priority-based listeners, async emission, and error handling.
    
    USAGE:
        bus = EventBus.get_instance()
        
        # Register listener
        bus.register_listener(EventType.MARKET_DISCOVERED, on_market_discovered)
        
        # Emit event (blocking)
        bus.emit(Event(EventType.MARKET_DISCOVERED, "seed", {'symbol': 'EURUSD'}))
        
        # Emit event (non-blocking)
        bus.emit_async(Event(EventType.MARKET_DISCOVERED, "seed", {'symbol': 'EURUSD'}))
    """
    
    _instance: Optional['EventBus'] = None
    _lock = threading.Lock()
    
    # Default configuration
    _DEFAULT_MAX_WORKERS = 4
    _DEFAULT_QUEUE_SIZE = 10000
    
    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._initialized = False
            return cls._instance
    
    def __init__(self):
        """Initialize the event bus."""
        if self._initialized:
            return
        
        self._listeners: Dict[str, List[ListenerWrapper]] = {}
        self._wildcard_listeners: List[ListenerWrapper] = []
        self._bus_lock = threading.RLock()
        
        # Async execution
        self._executor: Optional[ThreadPoolExecutor] = None
        self._max_workers = self._DEFAULT_MAX_WORKERS
        
        # Metrics
        self._events_emitted = 0
        self._events_async = 0
        self._events_failed = 0
        self._start_time = datetime.now()
        
        # Shutdown flag
        self._shutdown = False
        
        self._initialized = True
        logger.info("✅ EventBus initialized")
    
    # ==========================================================================
    # SINGLETON ACCESS
    # ==========================================================================
    
    @classmethod
    def get_instance(cls) -> 'EventBus':
        """Get the singleton EventBus instance."""
        return cls()
    
    # ==========================================================================
    # PUBLIC METHODS
    # ==========================================================================
    
    def register_listener(
        self,
        event_type: Union[EventType, str],
        callback: Callable[[Event], None],
        priority: int = 50,
        name: Optional[str] = None,
    ) -> None:
        """
        Register a listener for a specific event type.
        
        Args:
            event_type: Event type to listen for
            callback: Callback function (accepts Event)
            priority: Priority (higher = called first, default 50)
            name: Optional name for the listener (for debugging)
        
        Raises:
            ValueError: If callback is not callable or priority invalid
        """
        if not callable(callback):
            raise ValueError("Callback must be callable")
        
        if not 0 <= priority <= 100:
            raise ValueError(f"Priority must be between 0 and 100: {priority}")
        
        event_type_str = self._normalize_event_type(event_type)
        wrapper = ListenerWrapper(callback=callback, priority=priority, name=name)
        
        with self._bus_lock:
            if event_type_str not in self._listeners:
                self._listeners[event_type_str] = []
            
            # Add listener
            self._listeners[event_type_str].append(wrapper)
            
            # Sort by priority (highest first)
            self._listeners[event_type_str].sort(key=lambda w: w.priority, reverse=True)
            
            logger.debug(
                f"📢 Listener registered: {event_type_str} "
                f"(priority={priority}, name={name or 'unnamed'})"
            )
    
    def register_wildcard(
        self,
        callback: Callable[[Event], None],
        priority: int = 50,
        name: Optional[str] = None,
    ) -> None:
        """
        Register a listener that receives ALL events.
        
        Args:
            callback: Callback function (accepts Event)
            priority: Priority (higher = called first)
            name: Optional name for the listener
        """
        if not callable(callback):
            raise ValueError("Callback must be callable")
        
        if not 0 <= priority <= 100:
            raise ValueError(f"Priority must be between 0 and 100: {priority}")
        
        wrapper = ListenerWrapper(callback=callback, priority=priority, name=name)
        
        with self._bus_lock:
            self._wildcard_listeners.append(wrapper)
            self._wildcard_listeners.sort(key=lambda w: w.priority, reverse=True)
            
            logger.debug(
                f"📢 Wildcard listener registered (priority={priority}, name={name or 'unnamed'})"
            )
    
    def unregister_listener(
        self,
        event_type: Union[EventType, str],
        callback: Callable[[Event], None],
        priority: Optional[int] = None,
    ) -> bool:
        """
        Unregister a listener.
        
        Args:
            event_type: Event type
            callback: The callback to unregister
            priority: Specific priority to unregister (if multiple with same callback)
        
        Returns:
            True if listener was found and removed
        """
        event_type_str = self._normalize_event_type(event_type)
        
        with self._bus_lock:
            if event_type_str not in self._listeners:
                return False
            
            original_count = len(self._listeners[event_type_str])
            
            if priority is not None:
                self._listeners[event_type_str] = [
                    w for w in self._listeners[event_type_str]
                    if w.callback != callback or w.priority != priority
                ]
            else:
                self._listeners[event_type_str] = [
                    w for w in self._listeners[event_type_str]
                    if w.callback != callback
                ]
            
            removed = original_count - len(self._listeners[event_type_str])
            
            if removed > 0:
                logger.debug(f"📢 Listener unregistered: {event_type_str}")
                return True
            
            return False
    
    def unregister_wildcard(
        self,
        callback: Callable[[Event], None],
        priority: Optional[int] = None,
    ) -> bool:
        """
        Unregister a wildcard listener.
        
        Args:
            callback: The callback to unregister
            priority: Specific priority to unregister
        
        Returns:
            True if listener was found and removed
        """
        with self._bus_lock:
            original_count = len(self._wildcard_listeners)
            
            if priority is not None:
                self._wildcard_listeners = [
                    w for w in self._wildcard_listeners
                    if w.callback != callback or w.priority != priority
                ]
            else:
                self._wildcard_listeners = [
                    w for w in self._wildcard_listeners
                    if w.callback != callback
                ]
            
            removed = original_count - len(self._wildcard_listeners)
            
            if removed > 0:
                logger.debug("📢 Wildcard listener unregistered")
                return True
            
            return False
    
    def emit(self, event: Event) -> None:
        """
        Emit an event to all registered listeners (blocking).
        
        Args:
            event: Event to emit
        
        Raises:
            ValueError: If event is None or invalid
        """
        if event is None:
            raise ValueError("Event cannot be None")
        
        if self._shutdown:
            logger.warning("⚠️ EventBus is shutdown, event not emitted")
            return
        
        event_type_str = event.type_str
        listeners_called = 0
        
        with self._bus_lock:
            # Get listeners for this event type
            listeners = self._listeners.get(event_type_str, []).copy()
            wildcard_listeners = self._wildcard_listeners.copy()
        
        # Combine listeners: event-specific first, then wildcards
        all_listeners = listeners + wildcard_listeners
        
        if not all_listeners:
            logger.debug(f"📢 No listeners for event: {event_type_str}")
        
        # Call each listener
        for wrapper in all_listeners:
            try:
                wrapper(event)
                listeners_called += 1
            except Exception as e:
                self._events_failed += 1
                logger.error(
                    f"❌ Listener error for {event_type_str}: "
                    f"{wrapper.name or 'unnamed'}: {e}",
                    exc_info=True
                )
        
        self._events_emitted += 1
        
        logger.debug(
            f"📢 Event emitted: {event_type_str} "
            f"(listeners={listeners_called}, source={event.source})"
        )
    
    def emit_async(self, event: Event) -> Optional[Future]:
        """
        Emit an event asynchronously (non-blocking).
        
        Args:
            event: Event to emit
        
        Returns:
            Future for the async operation, or None if executor not available
        
        Raises:
            ValueError: If event is None or invalid
        """
        if event is None:
            raise ValueError("Event cannot be None")
        
        if self._shutdown:
            logger.warning("⚠️ EventBus is shutdown, async event not emitted")
            return None
        
        # Ensure executor is running
        self._ensure_executor()
        
        if self._executor is None:
            logger.warning("⚠️ No executor available, falling back to sync emit")
            self.emit(event)
            return None
        
        self._events_async += 1
        
        def _emit_async():
            self.emit(event)
        
        future = self._executor.submit(_emit_async)
        
        logger.debug(
            f"📢 Async event submitted: {event.type_str} (source={event.source})"
        )
        
        return future
    
    def get_listeners(self, event_type: Union[EventType, str]) -> List[Callable]:
        """
        Get all listeners for a specific event type.
        
        Args:
            event_type: Event type
        
        Returns:
            List of callback functions
        """
        event_type_str = self._normalize_event_type(event_type)
        
        with self._bus_lock:
            if event_type_str not in self._listeners:
                return []
            return [w.callback for w in self._listeners[event_type_str]]
    
    def get_listener_count(self, event_type: Union[EventType, str]) -> int:
        """
        Get the number of listeners for a specific event type.
        
        Args:
            event_type: Event type
        
        Returns:
            Number of listeners
        """
        event_type_str = self._normalize_event_type(event_type)
        
        with self._bus_lock:
            return len(self._listeners.get(event_type_str, []))
    
    def clear_listeners(self, event_type: Optional[Union[EventType, str]] = None) -> None:
        """
        Clear listeners for a specific event type or all listeners.
        
        Args:
            event_type: Event type to clear (None = clear all)
        """
        with self._bus_lock:
            if event_type is None:
                self._listeners.clear()
                self._wildcard_listeners.clear()
                logger.info("🗑️ All listeners cleared")
            else:
                event_type_str = self._normalize_event_type(event_type)
                if event_type_str in self._listeners:
                    self._listeners.pop(event_type_str)
                    logger.info(f"🗑️ Listeners cleared for: {event_type_str}")
    
    def set_max_workers(self, max_workers: int) -> None:
        """
        Set the maximum number of threads for async emission.
        
        Args:
            max_workers: Maximum number of threads
        """
        if max_workers < 1:
            raise ValueError("max_workers must be at least 1")
        
        with self._bus_lock:
            self._max_workers = max_workers
            # Restart executor with new settings
            if self._executor:
                self._executor.shutdown(wait=False)
                self._executor = ThreadPoolExecutor(
                    max_workers=self._max_workers,
                    thread_name_prefix="event-bus"
                )
            logger.info(f"⚙️ EventBus max_workers set to {max_workers}")
    
    def get_metrics(self) -> Dict[str, Any]:
        """
        Get event bus metrics.
        
        Returns:
            Dict with metrics
        """
        with self._bus_lock:
            total_listeners = sum(len(l) for l in self._listeners.values())
            total_listeners += len(self._wildcard_listeners)
            
            return {
                'events_emitted': self._events_emitted,
                'events_async': self._events_async,
                'events_failed': self._events_failed,
                'listener_count': total_listeners,
                'event_types': len(self._listeners),
                'wildcard_listeners': len(self._wildcard_listeners),
                'max_workers': self._max_workers,
                'uptime_seconds': (datetime.now() - self._start_time).total_seconds(),
                'is_shutdown': self._shutdown,
            }
    
    def get_event_types(self) -> List[str]:
        """
        Get all registered event types.
        
        Returns:
            List of event type strings
        """
        with self._bus_lock:
            return sorted(self._listeners.keys())
    
    def shutdown(self, wait: bool = True) -> None:
        """
        Shutdown the event bus.
        
        Args:
            wait: Wait for pending async events to complete
        """
        with self._bus_lock:
            if self._shutdown:
                return
            
            self._shutdown = True
            
            if self._executor:
                self._executor.shutdown(wait=wait)
                self._executor = None
            
            logger.info("⏹️ EventBus shutdown complete")
    
    # ==========================================================================
    # PRIVATE METHODS
    # ==========================================================================
    
    def _normalize_event_type(self, event_type: Union[EventType, str]) -> str:
        """Convert event type to string."""
        if isinstance(event_type, EventType):
            return event_type.value
        return str(event_type)
    
    def _ensure_executor(self) -> None:
        """Ensure the thread pool executor is running."""
        with self._bus_lock:
            if self._executor is None and not self._shutdown:
                self._executor = ThreadPoolExecutor(
                    max_workers=self._max_workers,
                    thread_name_prefix="event-bus"
                )
    
    # ==========================================================================
    # CONTEXT MANAGER
    # ==========================================================================
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.shutdown(wait=True)


# ==============================================================================
# CONVENIENCE FUNCTIONS
# ==============================================================================

def get_event_bus() -> EventBus:
    """
    Get the global event bus instance.
    
    Returns:
        EventBus singleton instance
    """
    return EventBus.get_instance()


def emit_event(
    event_type: Union[EventType, str],
    source: str,
    data: Optional[Dict[str, Any]] = None,
    priority: int = 50,
    correlation_id: Optional[str] = None,
) -> None:
    """
    Convenience function to emit an event.
    
    Args:
        event_type: Event type
        source: Source component name
        data: Event data
        priority: Priority (0-100)
        correlation_id: Optional correlation ID
    """
    event = Event(
        event_type=event_type,
        source=source,
        data=data or {},
        priority=priority,
        correlation_id=correlation_id,
    )
    get_event_bus().emit(event)


def emit_event_async(
    event_type: Union[EventType, str],
    source: str,
    data: Optional[Dict[str, Any]] = None,
    priority: int = 50,
    correlation_id: Optional[str] = None,
) -> Optional[Future]:
    """
    Convenience function to emit an event asynchronously.
    
    Args:
        event_type: Event type
        source: Source component name
        data: Event data
        priority: Priority (0-100)
        correlation_id: Optional correlation ID
    
    Returns:
        Future for the async operation
    """
    event = Event(
        event_type=event_type,
        source=source,
        data=data or {},
        priority=priority,
        correlation_id=correlation_id,
    )
    return get_event_bus().emit_async(event)


# ==============================================================================
# DECORATORS
# ==============================================================================

def on_event(event_type: Union[EventType, str], priority: int = 50):
    """
    Decorator to register a function as an event listener.
    
    Args:
        event_type: Event type to listen for
        priority: Priority (higher = called first)
    
    Returns:
        Decorator function
    
    Usage:
        @on_event(EventType.MARKET_DISCOVERED, priority=90)
        def handle_market_discovered(event: Event):
            print(f"Market discovered: {event.data['symbol']}")
    """
    def decorator(func: Callable[[Event], None]):
        bus = get_event_bus()
        bus.register_listener(event_type, func, priority, name=func.__name__)
        return func
    return decorator


def on_any_event(priority: int = 50):
    """
    Decorator to register a function as a wildcard listener.
    
    Args:
        priority: Priority (higher = called first)
    
    Returns:
        Decorator function
    
    Usage:
        @on_any_event(priority=10)
        def log_all_events(event: Event):
            print(f"Event: {event.type_str} from {event.source}")
    """
    def decorator(func: Callable[[Event], None]):
        bus = get_event_bus()
        bus.register_wildcard(func, priority, name=func.__name__)
        return func
    return decorator