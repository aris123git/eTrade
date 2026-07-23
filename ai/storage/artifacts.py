"""
ai/storage/artifacts.py - Model artifact persistence.

VERSION: 1.0.0
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, is_dataclass
from datetime import datetime, timezone
import json
import pickle
from pathlib import Path
from typing import Any, Dict, Sequence

import numpy as np

from ai.storage.versioning import timestamp_version
from ai.utils.hashing import content_hash

try:
    import joblib
except ModuleNotFoundError:  # pragma: no cover - depends on runtime environment.
    joblib = None


MODEL_JOBLIB = "model.joblib"
MODEL_PICKLE = "model.pkl"


@dataclass(frozen=True)
class ArtifactManifest:
    """Paths and hashes for a saved model artifact directory."""

    directory: Path
    version: str
    model_path: Path
    scaler_path: Path
    features_path: Path
    meta_path: Path
    metrics_path: Path
    hashes_path: Path
    encoder_path: Path | None
    hashes: Dict[str, str]

    def to_dict(self) -> Dict[str, Any]:
        """Serialize the manifest to JSON-compatible primitives."""
        return {
            "directory": str(self.directory),
            "version": self.version,
            "model_path": str(self.model_path),
            "scaler_path": str(self.scaler_path),
            "features_path": str(self.features_path),
            "meta_path": str(self.meta_path),
            "metrics_path": str(self.metrics_path),
            "hashes_path": str(self.hashes_path),
            "encoder_path": str(self.encoder_path) if self.encoder_path else None,
            "hashes": dict(self.hashes),
        }


def save_model(path: Path | str, model: Any, compress: bool = True) -> Path:
    """Persist a model with joblib when available, otherwise pickle."""
    base = Path(path)
    if base.suffix:
        base.parent.mkdir(parents=True, exist_ok=True)
    else:
        base.mkdir(parents=True, exist_ok=True)
    if joblib is not None and base.suffix != ".pkl":
        model_path = base if base.suffix else base / MODEL_JOBLIB
        joblib.dump(model, model_path, compress=3 if compress else 0)
        return model_path
    model_path = base if base.suffix else base / MODEL_PICKLE
    with model_path.open("wb") as handle:
        pickle.dump(model, handle, protocol=pickle.HIGHEST_PROTOCOL)
    return model_path


def load_model(path: Path | str) -> Any:
    """Load a model saved by save_model or save_artifacts."""
    model_path = _resolve_model_path(Path(path))
    if model_path.suffix == ".joblib":
        if joblib is None:
            raise RuntimeError("joblib is required to load joblib model artifacts")
        return joblib.load(model_path)
    with model_path.open("rb") as handle:
        return pickle.load(handle)


def save_artifacts(
    directory: Path | str,
    model: Any,
    scaler: Any = None,
    encoder: Any = None,
    features: Sequence[str] | None = None,
    metadata: Dict[str, Any] | None = None,
    metrics: Dict[str, Any] | None = None,
    version: str | None = None,
    params: Dict[str, Any] | None = None,
    compress: bool = True,
) -> ArtifactManifest:
    """Save a complete model artifact directory."""
    artifact_dir = Path(directory)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    active_version = version or timestamp_version()

    model_path = save_model(artifact_dir / (MODEL_JOBLIB if joblib is not None else MODEL_PICKLE), model, compress=compress)
    scaler_path = save_json(artifact_dir / "scaler.json", _object_payload(scaler))
    features_path = save_json(artifact_dir / "features.json", {"features": list(features or [])})
    meta_payload = {
        "version": active_version,
        "date": datetime.now(timezone.utc).isoformat(),
        "params": params or {},
        "metadata": metadata or {},
        "model_file": model_path.name,
    }
    meta_path = save_json(artifact_dir / "meta.json", meta_payload)
    metrics_path = save_json(artifact_dir / "metrics.json", metrics or {})
    encoder_path = save_json(artifact_dir / "encoder.json", _object_payload(encoder)) if encoder is not None else None
    hashes = artifact_hashes(artifact_dir, include_hashes_file=False)
    hashes_path = save_json(artifact_dir / "hashes.json", hashes)
    return ArtifactManifest(
        directory=artifact_dir,
        version=active_version,
        model_path=model_path,
        scaler_path=scaler_path,
        features_path=features_path,
        meta_path=meta_path,
        metrics_path=metrics_path,
        hashes_path=hashes_path,
        encoder_path=encoder_path,
        hashes=hashes,
    )


def load_artifacts(directory: Path | str) -> Dict[str, Any]:
    """Load a complete artifact directory."""
    artifact_dir = Path(directory)
    features_payload = load_json(artifact_dir / "features.json", default={"features": []})
    return {
        "model": load_model(artifact_dir),
        "scaler": load_json(artifact_dir / "scaler.json", default={}),
        "encoder": load_json(artifact_dir / "encoder.json", default=None),
        "features": list(features_payload.get("features", [])),
        "meta": load_json(artifact_dir / "meta.json", default={}),
        "metrics": load_json(artifact_dir / "metrics.json", default={}),
        "hashes": load_json(artifact_dir / "hashes.json", default={}),
        "directory": artifact_dir,
    }


def save_json(path: Path | str, payload: Any) -> Path:
    """Write JSON with stable formatting."""
    json_path = Path(path)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(_to_jsonable(payload), handle, indent=2, sort_keys=True, default=str)
    return json_path


def load_json(path: Path | str, default: Any = None) -> Any:
    """Load JSON, returning default when the file does not exist."""
    json_path = Path(path)
    if not json_path.exists():
        return default
    with json_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def artifact_hashes(directory: Path | str, include_hashes_file: bool = True) -> Dict[str, str]:
    """Compute deterministic content hashes for files in an artifact directory."""
    artifact_dir = Path(directory)
    hashes: Dict[str, str] = {}
    for path in sorted(item for item in artifact_dir.iterdir() if item.is_file()):
        if not include_hashes_file and path.name == "hashes.json":
            continue
        hashes[path.name] = content_hash(path.name, path.read_bytes())
    return hashes


def _resolve_model_path(path: Path) -> Path:
    if path.is_dir():
        for name in (MODEL_JOBLIB, MODEL_PICKLE):
            candidate = path / name
            if candidate.exists():
                return candidate
        raise FileNotFoundError(f"No model artifact found in {path}")
    return path


def _object_payload(value: Any) -> Dict[str, Any]:
    if value is None:
        return {}
    payload = _to_jsonable(value)
    if isinstance(payload, dict):
        return payload
    return {"value": payload}


def _to_jsonable(value: Any) -> Any:
    if is_dataclass(value):
        return _to_jsonable(asdict(value))
    if isinstance(value, dict):
        return {str(key): _to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_to_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, np.floating, np.bool_)):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "to_dict") and callable(value.to_dict):
        return _to_jsonable(value.to_dict())
    if hasattr(value, "__dict__") and not isinstance(value, type):
        public = {key: item for key, item in vars(value).items() if not key.startswith("_")}
        return _to_jsonable(public)
    return value
