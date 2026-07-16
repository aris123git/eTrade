"""
ai/scheduler/scheduler.py - Threading-based AI scheduler.

VERSION: 1.0.0
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from threading import Event, Lock, Thread
from typing import Callable, Dict, List

from ai.scheduler.jobs import Job, JobCallable, JobResult


@dataclass
class AIScheduler:
    """Lightweight periodic scheduler for retrain, predict, and monitor jobs."""

    poll_seconds: float = 1.0
    jobs: Dict[str, Job] = field(default_factory=dict)
    history: List[JobResult] = field(default_factory=list)

    def __post_init__(self) -> None:
        self._stop = Event()
        self._lock = Lock()
        self._thread: Thread | None = None

    def add_job(
        self,
        name: str,
        func: JobCallable,
        interval_seconds: float,
        *args: object,
        run_immediately: bool = False,
        **kwargs: object,
    ) -> Job:
        """Register a periodic job."""

        job = Job(
            name=name,
            func=func,
            interval_seconds=interval_seconds,
            args=tuple(args),
            kwargs=dict(kwargs),
            run_immediately=run_immediately,
        )
        with self._lock:
            self.jobs[job.job_id] = job
        return job

    def add_retrain_job(self, func: JobCallable, interval_seconds: float, **kwargs: object) -> Job:
        return self.add_job("retrain", func, interval_seconds, **kwargs)

    def add_predict_job(self, func: JobCallable, interval_seconds: float, **kwargs: object) -> Job:
        return self.add_job("predict", func, interval_seconds, **kwargs)

    def add_monitor_job(self, func: JobCallable, interval_seconds: float, **kwargs: object) -> Job:
        return self.add_job("monitor", func, interval_seconds, **kwargs)

    def add_data_download_job(
        self,
        pipeline: object,
        interval_seconds: float | None = None,
        *,
        run_immediately: bool = True,
    ) -> Job:
        """
        Periodically let the AI re-download all configured symbols × timeframes.
        """

        def _run() -> object:
            ensure = getattr(pipeline, "ensure_market_data", None)
            if ensure is None:
                raise RuntimeError("pipeline does not expose ensure_market_data()")
            return ensure(force=False)

        seconds = interval_seconds
        if seconds is None:
            cfg = getattr(pipeline, "config", None)
            data_cfg = getattr(cfg, "data", None) if cfg is not None else None
            seconds = float(getattr(data_cfg, "refresh_interval_seconds", 3600.0) or 3600.0)
        return self.add_job(
            "data_download",
            _run,
            float(seconds),
            run_immediately=run_immediately,
        )

    def remove_job(self, job_id: str) -> bool:
        with self._lock:
            return self.jobs.pop(job_id, None) is not None

    def run_pending(self) -> List[JobResult]:
        """Run all due jobs once."""

        now = datetime.now(timezone.utc)
        with self._lock:
            due_jobs = [job for job in self.jobs.values() if job.due(now)]
        results: List[JobResult] = []
        for job in due_jobs:
            result = job.run()
            with self._lock:
                self.history.append(result)
            results.append(result)
        return results

    def start(self) -> None:
        """Start the scheduler loop in a daemon thread."""

        if self._thread is not None and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = Thread(target=self._run_loop, name="ai-scheduler", daemon=True)
        self._thread.start()

    def stop(self, timeout: float | None = 5.0) -> None:
        """Stop the scheduler loop."""

        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=timeout)

    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def _run_loop(self) -> None:
        while not self._stop.is_set():
            self.run_pending()
            self._stop.wait(max(0.1, float(self.poll_seconds)))


def create_scheduler(configure: Callable[[AIScheduler], None] | None = None) -> AIScheduler:
    """Factory for AIScheduler."""

    scheduler = AIScheduler()
    if configure is not None:
        configure(scheduler)
    return scheduler
