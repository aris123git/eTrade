"""
ai/storage/registry.py - Filesystem model registry.

VERSION: 1.0.0
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
import shutil
from typing import Any, Dict, List, Sequence

from ai.config.settings import AIConfig
from ai.storage.artifacts import ArtifactManifest, load_artifacts, load_json, save_artifacts, save_json
from ai.storage.versioning import latest_version, timestamp_version, version_sort_key


@dataclass(frozen=True)
class RegisteredModel:
    """Metadata for a registered model version."""

    name: str
    version: str
    path: Path
    meta: Dict[str, Any] = field(default_factory=dict)
    metrics: Dict[str, Any] = field(default_factory=dict)
    hashes: Dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize the model registration metadata."""
        return {
            "name": self.name,
            "version": self.version,
            "path": str(self.path),
            "meta": self.meta,
            "metrics": self.metrics,
            "hashes": self.hashes,
        }


@dataclass
class ModelRegistry:
    """Production filesystem registry for versioned model artifacts."""

    config: AIConfig = field(default_factory=AIConfig)
    root_dir: Path | str | None = None

    def __post_init__(self) -> None:
        root = Path(self.root_dir) if self.root_dir is not None else Path(self.config.storage.root_dir)
        if self.root_dir is None:
            root = root / self.config.storage.models_dir
        self.root_dir = root
        Path(self.root_dir).mkdir(parents=True, exist_ok=True)

    def register(
        self,
        name: str,
        model: Any,
        scaler: Any = None,
        encoder: Any = None,
        features: Sequence[str] | None = None,
        metadata: Dict[str, Any] | None = None,
        metrics: Dict[str, Any] | None = None,
        params: Dict[str, Any] | None = None,
        version: str | None = None,
        overwrite: bool = False,
    ) -> RegisteredModel:
        """Register a complete model artifact version."""
        model_name = _safe_name(name)
        active_version = version or timestamp_version()
        artifact_dir = self._artifact_dir(model_name, active_version)
        if artifact_dir.exists() and not overwrite:
            raise FileExistsError(f"Model version already exists: {model_name}/{active_version}")
        if artifact_dir.exists() and overwrite:
            shutil.rmtree(artifact_dir)
        merged_metadata = dict(metadata or {})
        merged_metadata.update({"model_name": model_name})
        manifest = save_artifacts(
            artifact_dir,
            model=model,
            scaler=scaler,
            encoder=encoder,
            features=features,
            metadata=merged_metadata,
            metrics=metrics,
            version=active_version,
            params=params,
            compress=self.config.storage.compress,
        )
        self._write_index(model_name)
        return self._registered_model(model_name, active_version, manifest=manifest)

    def load(self, name: str, version: str | None = None) -> Dict[str, Any]:
        """Load model artifacts for a named version or the latest version."""
        model_name = _safe_name(name)
        active_version = version or self._require_latest(model_name)
        artifacts = load_artifacts(self._artifact_dir(model_name, active_version))
        artifacts["name"] = model_name
        artifacts["version"] = active_version
        return artifacts

    def list_versions(self, name: str) -> List[str]:
        """List registered versions for a model sorted from oldest to newest."""
        model_dir = self._model_dir(_safe_name(name))
        if not model_dir.exists():
            return []
        versions = [path.name for path in model_dir.iterdir() if path.is_dir()]
        return sorted(versions, key=version_sort_key)

    def get_latest(self, name: str) -> RegisteredModel | None:
        """Return metadata for the latest version of a model."""
        model_name = _safe_name(name)
        versions = self.list_versions(model_name)
        if not versions:
            return None
        return self._registered_model(model_name, latest_version(versions))

    def delete_old(self, name: str, keep_last_n: int | None = None) -> List[str]:
        """Delete old versions, preserving the newest keep_last_n versions."""
        model_name = _safe_name(name)
        keep = self.config.storage.keep_last_n_versions if keep_last_n is None else keep_last_n
        if keep < 0:
            raise ValueError("keep_last_n must be >= 0")
        versions = self.list_versions(model_name)
        delete_versions = versions[: max(len(versions) - keep, 0)]
        deleted: List[str] = []
        for version in delete_versions:
            shutil.rmtree(self._artifact_dir(model_name, version), ignore_errors=True)
            deleted.append(version)
        if deleted:
            self._write_index(model_name)
        return deleted

    def list_models(self) -> List[str]:
        """List registered model names."""
        root = Path(self.root_dir)
        return sorted(path.name for path in root.iterdir() if path.is_dir())

    def _model_dir(self, name: str) -> Path:
        return Path(self.root_dir) / name

    def _artifact_dir(self, name: str, version: str) -> Path:
        return self._model_dir(name) / version

    def _require_latest(self, name: str) -> str:
        latest = self.get_latest(name)
        if latest is None:
            raise FileNotFoundError(f"No registered versions for model: {name}")
        return latest.version

    def _registered_model(
        self,
        name: str,
        version: str,
        manifest: ArtifactManifest | None = None,
    ) -> RegisteredModel:
        artifact_dir = self._artifact_dir(name, version)
        meta = load_json(artifact_dir / "meta.json", default={})
        metrics = load_json(artifact_dir / "metrics.json", default={})
        hashes = manifest.hashes if manifest is not None else load_json(artifact_dir / "hashes.json", default={})
        return RegisteredModel(name=name, version=version, path=artifact_dir, meta=meta, metrics=metrics, hashes=hashes)

    def _write_index(self, name: str) -> None:
        versions = self.list_versions(name)
        payload = {
            "name": name,
            "versions": versions,
            "latest": latest_version(versions) if versions else None,
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }
        save_json(self._model_dir(name) / "registry.json", payload)


def create_model_registry(config: AIConfig | None = None) -> ModelRegistry:
    """Factory for ModelRegistry."""
    return ModelRegistry(config=config or AIConfig())


def _safe_name(name: str) -> str:
    clean = name.strip().replace("\\", "/").strip("/")
    if not clean or "/" in clean or clean in {".", ".."}:
        raise ValueError(f"Invalid model name: {name!r}")
    return clean
