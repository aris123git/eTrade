"""
ai/scheduler/jobs.py - Scheduler job contracts.

VERSION: 1.0.0
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Callable, Dict
import uuid


JobCallable = Callable[..., Any]


@dataclass(frozen=True)
class JobResult:
    """Result from one job execution."""

    job_id: str
    name: str
    started_at: datetime
    finished_at: datetime
    success: bool
    result: Any = None
    error: str | None = None

    @property
    def duration_seconds(self) -> float:
        return float((self.finished_at - self.started_at).total_seconds())


@dataclass
class Job:
    """Periodic scheduler job."""

    name: str
    func: JobCallable
    interval_seconds: float
    args: tuple[Any, ...] = field(default_factory=tuple)
    kwargs: Dict[str, Any] = field(default_factory=dict)
    job_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    enabled: bool = True
    run_immediately: bool = False
    next_run: datetime | None = None

    def __post_init__(self) -> None:
        if self.interval_seconds <= 0.0:
            raise ValueError("interval_seconds must be > 0")
        now = datetime.now(timezone.utc)
        self.next_run = now if self.run_immediately else now + timedelta(seconds=float(self.interval_seconds))

    def due(self, now: datetime | None = None) -> bool:
        if not self.enabled:
            return False
        current = now or datetime.now(timezone.utc)
        return self.next_run is not None and current >= self.next_run

    def run(self) -> JobResult:
        started = datetime.now(timezone.utc)
        try:
            result = self.func(*self.args, **self.kwargs)
            success = True
            error = None
        except Exception as exc:
            result = None
            success = False
            error = f"{exc.__class__.__name__}: {exc}"
        finished = datetime.now(timezone.utc)
        self.next_run = finished + timedelta(seconds=float(self.interval_seconds))
        return JobResult(
            job_id=self.job_id,
            name=self.name,
            started_at=started,
            finished_at=finished,
            success=success,
            result=result,
            error=error,
        )
