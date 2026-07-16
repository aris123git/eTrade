"""
ai/research/autonomous_scheduler.py - Calendar-aware autonomous jobs.

Hourly:  download new data, update features, generate predictions
Daily:   retrain when needed, validate, compare models
Weekly:  walk-forward evaluation, robustness testing
Monthly: complete research cycle
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional

from ai.scheduler.jobs import Job, JobResult
from ai.scheduler.scheduler import AIScheduler

logger = logging.getLogger(__name__)

SECONDS = {
    "hourly": 3600.0,
    "daily": 86400.0,
    "weekly": 604800.0,
    "monthly": 2592000.0,  # 30d nominal
}


@dataclass
class SchedulePlan:
    """Declarative autonomous schedule."""

    hourly_enabled: bool = True
    daily_enabled: bool = True
    weekly_enabled: bool = True
    monthly_enabled: bool = True
    run_immediately: bool = True


class AutonomousScheduler:
    """
    Wraps AIScheduler with institutional cadence.

    Jobs call into AutonomousResearchPlatform methods:
      run_hourly / run_daily / run_weekly / run_monthly
    """

    def __init__(
        self,
        platform: Any,
        *,
        plan: SchedulePlan | None = None,
        scheduler: AIScheduler | None = None,
    ):
        self.platform = platform
        self.plan = plan or SchedulePlan()
        self.scheduler = scheduler or AIScheduler(poll_seconds=5.0)
        self._job_ids: Dict[str, str] = {}

    def install(self) -> Dict[str, Job]:
        """Register cadence jobs on the underlying scheduler."""
        installed: Dict[str, Job] = {}
        if self.plan.hourly_enabled and hasattr(self.platform, "run_hourly"):
            job = self.scheduler.add_job(
                "hourly_research",
                self.platform.run_hourly,
                SECONDS["hourly"],
                run_immediately=self.plan.run_immediately,
            )
            installed["hourly"] = job
            self._job_ids["hourly"] = job.job_id
        if self.plan.daily_enabled and hasattr(self.platform, "run_daily"):
            job = self.scheduler.add_job(
                "daily_research",
                self.platform.run_daily,
                SECONDS["daily"],
                run_immediately=False,
            )
            installed["daily"] = job
            self._job_ids["daily"] = job.job_id
        if self.plan.weekly_enabled and hasattr(self.platform, "run_weekly"):
            job = self.scheduler.add_job(
                "weekly_research",
                self.platform.run_weekly,
                SECONDS["weekly"],
                run_immediately=False,
            )
            installed["weekly"] = job
            self._job_ids["weekly"] = job.job_id
        if self.plan.monthly_enabled and hasattr(self.platform, "run_monthly"):
            job = self.scheduler.add_job(
                "monthly_research",
                self.platform.run_monthly,
                SECONDS["monthly"],
                run_immediately=False,
            )
            installed["monthly"] = job
            self._job_ids["monthly"] = job.job_id
        return installed

    def start(self) -> None:
        if not self._job_ids:
            self.install()
        self.scheduler.start()

    def stop(self) -> None:
        self.scheduler.stop()

    def run_pending(self) -> List[JobResult]:
        return self.scheduler.run_pending()

    def status(self) -> Dict[str, Any]:
        return {
            "jobs": {
                name: {
                    "job_id": job_id,
                    "next_run": (
                        self.scheduler.jobs[job_id].next_run.isoformat()
                        if job_id in self.scheduler.jobs and self.scheduler.jobs[job_id].next_run
                        else None
                    ),
                }
                for name, job_id in self._job_ids.items()
            },
            "history_len": len(self.scheduler.history),
            "now": datetime.now(timezone.utc).isoformat(),
        }
