"""
ai/research/report.py - Cycle reports for the autonomous research platform.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class StageResult:
    """Timed stage outcome inside one research cycle."""

    name: str
    status: str  # ok | skipped | failed
    started_at: str
    finished_at: str
    detail: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None

    @property
    def duration_seconds(self) -> float:
        try:
            start = datetime.fromisoformat(self.started_at)
            end = datetime.fromisoformat(self.finished_at)
            return float((end - start).total_seconds())
        except Exception:
            return 0.0

    def to_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        payload["duration_seconds"] = self.duration_seconds
        return payload


@dataclass
class CycleReport:
    """Full autonomous research cycle artifact."""

    cycle_id: str
    started_at: str
    finished_at: str = ""
    status: str = "running"
    stages: List[StageResult] = field(default_factory=list)
    champions: Dict[str, Any] = field(default_factory=dict)
    rejected: List[Dict[str, Any]] = field(default_factory=list)
    hypotheses: List[Dict[str, Any]] = field(default_factory=list)
    metrics_by_symbol: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    notes: List[str] = field(default_factory=list)
    data_disclaimer: str = (
        "The AI automates process; it cannot invent market history. "
        "Coverage is limited to what brokers/vendors supply (MT5, Dukascopy, "
        "Polygon, Binance, IB, CSV, …)."
    )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "cycle_id": self.cycle_id,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "status": self.status,
            "stages": [s.to_dict() for s in self.stages],
            "champions": self.champions,
            "rejected": self.rejected,
            "hypotheses": self.hypotheses,
            "metrics_by_symbol": self.metrics_by_symbol,
            "notes": self.notes,
            "data_disclaimer": self.data_disclaimer,
        }

    def save(self, path: Path | str) -> Path:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(self.to_dict(), indent=2, default=str), encoding="utf-8")
        return target

    def print_summary(self) -> None:
        print()
        print("=" * 64)
        print(f"Autonomous Research Cycle {self.cycle_id}")
        print("=" * 64)
        print(f"Status: {self.status}")
        print(f"Started: {self.started_at}")
        print(f"Finished: {self.finished_at or '-'}")
        for stage in self.stages:
            print(f"  [{stage.status:7}] {stage.name} ({stage.duration_seconds:.2f}s)")
            if stage.error:
                print(f"           error: {stage.error}")
        if self.champions:
            print("Champions:")
            for key, meta in self.champions.items():
                print(f"  {key}: {meta}")
        if self.rejected:
            print(f"Rejected challengers: {len(self.rejected)}")
        if self.hypotheses:
            print("Next hypotheses:")
            for hyp in self.hypotheses[:8]:
                print(f"  - ({hyp.get('priority', 0):.2f}) {hyp.get('rationale')}")
        for note in self.notes:
            print(f"note: {note}")
        print(self.data_disclaimer)
        print()


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()
