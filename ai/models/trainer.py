"""
ai/models/trainer.py - Model training facade for the autonomous trading system.

Supports RandomForest, LightGBM, XGBoost, and Neural Networks via the existing
model registry. Loads training data from CandleRepository and persists models
under ai/artifacts/models/ (configurable via AIConfig.storage).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from ai.config.settings import AIConfig
from ai.data.candle_adapter import CandleRepositoryAdapter
from ai.models import MODEL_REGISTRY
from ai.storage.registry import ModelRegistry
from ai.utils.types import CandleDict

logger = logging.getLogger(__name__)

# Lazy typing to avoid circular import with ai.services.pipeline
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ai.services.pipeline import AIPipeline

DEFAULT_ARTIFACT_ROOT = Path("ai/artifacts")

SUPPORTED_MODELS = (
    "random_forest",
    "lightgbm",
    "xgboost",
    "mlp",
    "neural_mlp",
    "lstm",
    "gru",
    "transformer",
)


@dataclass
class ModelTrainer:
    """
    Train production models on CandleRepository history.

    Paper/live agnostic — only builds and registers models.
    """

    config: AIConfig = field(default_factory=AIConfig)
    candle_repository: Any = None
    pipeline: Any = None
    registry: ModelRegistry | None = None

    def __post_init__(self) -> None:
        from ai.services.pipeline import AIPipeline

        # Phase 4 default artifact root unless caller already customized storage.
        if Path(self.config.storage.root_dir) == Path("ai_artifacts"):
            self.config.storage.root_dir = DEFAULT_ARTIFACT_ROOT
        self.config.ensure_directories()
        self.registry = self.registry or ModelRegistry(config=self.config)
        if self.pipeline is None:
            source = (
                CandleRepositoryAdapter(self.candle_repository)
                if self.candle_repository is not None
                else None
            )
            self.pipeline = AIPipeline(
                config=self.config,
                candle_source=source,
            )
            self.pipeline.registry = self.registry
        logger.info(
            "ModelTrainer ready models=%s artifacts=%s",
            sorted(m for m in SUPPORTED_MODELS if m in MODEL_REGISTRY),
            self.config.storage.root_dir / self.config.storage.models_dir,
        )

    def supported_models(self) -> List[str]:
        return [m for m in SUPPORTED_MODELS if m in MODEL_REGISTRY]

    def load_training_candles(
        self,
        symbol: str,
        timeframe: str | None = None,
        *,
        limit: int = 5000,
    ) -> List[CandleDict]:
        tf = timeframe or self.config.primary_timeframe
        assert self.pipeline is not None
        if self.candle_repository is not None and self.pipeline.candle_source is None:
            self.pipeline.candle_source = CandleRepositoryAdapter(self.candle_repository)
        candles = self.pipeline.load_candles(
            symbol=symbol,
            timeframe=tf,
            limit=limit,
            auto_download=bool(self.config.data.auto_download),
        )
        logger.info("loaded %s candles for %s %s", len(candles), symbol, tf)
        return candles

    def train(
        self,
        *,
        symbol: str,
        model_type: str = "random_forest",
        timeframe: str | None = None,
        candles: Sequence[CandleDict] | None = None,
        limit: int = 5000,
        register: bool = True,
    ) -> Dict[str, Any]:
        """Train one model type and optionally register the artifact."""
        name = str(model_type).lower().strip()
        if name not in MODEL_REGISTRY:
            raise ValueError(f"Unsupported model_type={model_type!r}. Supported={self.supported_models()}")

        active = list(candles) if candles is not None else self.load_training_candles(
            symbol, timeframe=timeframe, limit=limit
        )
        if len(active) < 100:
            raise ValueError(f"Need >=100 candles to train, got {len(active)}")

        from ai.services.pipeline import AIPipeline

        cfg = self.config.copy()
        cfg.model.model_type = name
        assert self.pipeline is not None
        local = AIPipeline(
            config=cfg,
            candle_source=self.pipeline.candle_source,
            data_service=self.pipeline.data_service,
            registry=self.registry,
        )
        run = local.run_training(
            candles=active,
            register=register,
            auto_download=False,
            model_name=f"{symbol}_{name}",
        )
        artifact_dir = Path(self.config.storage.root_dir) / self.config.storage.models_dir
        result = {
            "symbol": symbol.upper(),
            "timeframe": timeframe or self.config.primary_timeframe,
            "model_type": name,
            "metrics": {**run.train.metrics, **run.evaluation},
            "n_candles": len(active),
            "n_features": len(run.dataset.bundle.feature_names),
            "registered": run.registration.to_dict() if run.registration else None,
            "artifact_root": str(artifact_dir),
            "trained_at": datetime.now(timezone.utc).isoformat(),
        }
        logger.info(
            "trained %s on %s metrics=%s",
            name,
            symbol,
            {k: result["metrics"].get(k) for k in ("train_f1", "val_f1", "test_f1", "f1", "accuracy") if k in result["metrics"]},
        )
        return result

    def train_candidates(
        self,
        *,
        symbol: str,
        model_types: Sequence[str] | None = None,
        timeframe: str | None = None,
        limit: int = 5000,
    ) -> List[Dict[str, Any]]:
        """Train multiple candidates; returns one result per successful model."""
        candles = self.load_training_candles(symbol, timeframe=timeframe, limit=limit)
        types = list(model_types or self.supported_models()[:4])
        results: List[Dict[str, Any]] = []
        for model_type in types:
            try:
                results.append(
                    self.train(
                        symbol=symbol,
                        model_type=model_type,
                        timeframe=timeframe,
                        candles=candles,
                        register=True,
                    )
                )
            except Exception as exc:
                logger.exception("candidate %s failed", model_type)
                results.append(
                    {
                        "symbol": symbol.upper(),
                        "model_type": model_type,
                        "error": f"{exc.__class__.__name__}: {exc}",
                    }
                )
        return results


def create_model_trainer(
    config: AIConfig | None = None,
    candle_repository: Any = None,
) -> ModelTrainer:
    return ModelTrainer(config=config or AIConfig(), candle_repository=candle_repository)
