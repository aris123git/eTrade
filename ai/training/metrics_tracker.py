"""
ai/training/metrics_tracker.py - Training metric history containers.

RESPONSIBILITY:
Store epoch-level metrics in serializable dataclasses for trainers and
experiment trackers.

VERSION: 1.0.0
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional
import json


# ==============================================================================
# HISTORY
# ==============================================================================


@dataclass
class TrainingHistory:
    """Append-only metric history for a single training run."""

    records: List[Dict[str, Any]] = field(default_factory=list)

    def add_epoch(
        self,
        epoch: int,
        train_metrics: Dict[str, float],
        val_metrics: Optional[Dict[str, float]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Append one epoch record."""
        self.records.append(
            {
                "epoch": int(epoch),
                "train": dict(train_metrics),
                "val": dict(val_metrics or {}),
                "metadata": dict(metadata or {}),
            }
        )

    def best(self, metric: str, minimize: bool = False, split: str = "val") -> Dict[str, Any] | None:
        """Return the record with the best metric value."""
        candidates = [record for record in self.records if metric in record.get(split, {})]
        if not candidates:
            return None
        key = lambda record: float(record[split][metric])
        return min(candidates, key=key) if minimize else max(candidates, key=key)

    def latest(self) -> Dict[str, Any] | None:
        """Return the latest record."""
        return self.records[-1] if self.records else None

    def to_dict(self) -> Dict[str, Any]:
        """Serialize history to a dictionary."""
        return {"records": list(self.records)}

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TrainingHistory":
        """Create history from a dictionary."""
        return cls(records=list(data.get("records", [])))

    def save(self, path: Path | str) -> None:
        """Write history JSON."""
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("w", encoding="utf-8") as handle:
            json.dump(self.to_dict(), handle, indent=2, default=str)

    @classmethod
    def load(cls, path: Path | str) -> "TrainingHistory":
        """Read history JSON."""
        with Path(path).open("r", encoding="utf-8") as handle:
            return cls.from_dict(json.load(handle))
