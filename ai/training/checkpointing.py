"""
ai/training/checkpointing.py - Training checkpoint persistence.

RESPONSIBILITY:
Save and load model, scaler, and metadata artifacts as portable checkpoints.

VERSION: 1.0.0
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional
import json
import pickle
import time

from ai.models.base import BaseModel


# ==============================================================================
# CHECKPOINTS
# ==============================================================================


@dataclass
class Checkpoint:
    """Loaded checkpoint payload."""

    model: BaseModel
    scaler: Any = None
    meta: Dict[str, Any] = field(default_factory=dict)
    path: Path | None = None


def save_checkpoint(
    path: Path | str,
    model: BaseModel,
    scaler: Any = None,
    meta: Optional[Dict[str, Any]] = None,
) -> Path:
    """Persist a checkpoint directory and return its path."""
    target = Path(path)
    target.mkdir(parents=True, exist_ok=True)
    metadata = dict(meta or {})
    metadata.setdefault("created_at", time.time())
    with (target / "model.pkl").open("wb") as handle:
        pickle.dump(model, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with (target / "scaler.pkl").open("wb") as handle:
        pickle.dump(scaler, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with (target / "meta.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, default=str)
    return target


def load_checkpoint(path: Path | str) -> Checkpoint:
    """Load a checkpoint created by save_checkpoint."""
    source = Path(path)
    with (source / "model.pkl").open("rb") as handle:
        model = pickle.load(handle)
    if not isinstance(model, BaseModel):
        raise TypeError(f"Checkpoint model is not a BaseModel: {type(model)!r}")
    scaler = None
    scaler_path = source / "scaler.pkl"
    if scaler_path.exists():
        with scaler_path.open("rb") as handle:
            scaler = pickle.load(handle)
    meta_path = source / "meta.json"
    meta: Dict[str, Any] = {}
    if meta_path.exists():
        with meta_path.open("r", encoding="utf-8") as handle:
            meta = json.load(handle)
    return Checkpoint(model=model, scaler=scaler, meta=meta, path=source)


def latest_checkpoint(directory: Path | str) -> Path | None:
    """Return the newest checkpoint directory under a root directory."""
    root = Path(directory)
    if not root.exists():
        return None
    candidates = [path for path in root.iterdir() if path.is_dir() and (path / "model.pkl").exists()]
    if not candidates:
        return None
    return max(candidates, key=lambda path: path.stat().st_mtime)
