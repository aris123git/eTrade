"""
ai/research/discovery.py - Automatic hypothesis and feature-set discovery.

Searches relationships across existing feature groups (no new indicators):
multi-timeframe alignment, volatility regimes, session behaviour,
inter-market correlations, feature interactions, temporal effects.

Candidate feature groups are evaluated; weak sets are discarded.
"""

from __future__ import annotations

import itertools
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from ai.config.settings import AIConfig
from ai.features.engine import FeatureEngine, FeatureGroup
from ai.labels.generator import LabelGenerator
from ai.models import create_model
from ai.research.gate import extract_metric
from ai.training.trainer import Trainer
from ai.utils.types import CandleDict

logger = logging.getLogger(__name__)

# Search space built from existing FeatureEngine groups — nothing hardcoded as "the" edge
DISCOVERY_GROUPS: Tuple[str, ...] = (
    FeatureGroup.PRICE.value,
    FeatureGroup.RETURNS.value,
    FeatureGroup.MOVING_AVERAGES.value,
    FeatureGroup.MOMENTUM.value,
    FeatureGroup.VOLATILITY.value,
    FeatureGroup.CHANNELS.value,
    FeatureGroup.VOLUME.value,
    FeatureGroup.CANDLE_STRUCTURE.value,
    FeatureGroup.SESSION.value,
    FeatureGroup.REGIME.value,
    FeatureGroup.STRUCTURE.value,
    FeatureGroup.PATTERNS.value,
)

RELATIONSHIP_THEMES: Dict[str, Tuple[str, ...]] = {
    "multi_timeframe_alignment": (
        FeatureGroup.PRICE.value,
        FeatureGroup.RETURNS.value,
        FeatureGroup.MOMENTUM.value,
        FeatureGroup.MOVING_AVERAGES.value,
    ),
    "volatility_regimes": (
        FeatureGroup.VOLATILITY.value,
        FeatureGroup.REGIME.value,
        FeatureGroup.RETURNS.value,
    ),
    "session_behaviour": (
        FeatureGroup.SESSION.value,
        FeatureGroup.VOLUME.value,
        FeatureGroup.VOLATILITY.value,
    ),
    "inter_market_correlations": (
        FeatureGroup.RETURNS.value,
        FeatureGroup.CORRELATION.value,
        FeatureGroup.REGIME.value,
    ),
    "feature_interactions": (
        FeatureGroup.MOMENTUM.value,
        FeatureGroup.VOLATILITY.value,
        FeatureGroup.CANDLE_STRUCTURE.value,
    ),
    "temporal_effects": (
        FeatureGroup.SESSION.value,
        FeatureGroup.STRUCTURE.value,
        FeatureGroup.RETURNS.value,
    ),
    "liquidity_conditions": (
        FeatureGroup.VOLUME.value,
        FeatureGroup.VOLATILITY.value,
        FeatureGroup.CHANNELS.value,
    ),
}


@dataclass(frozen=True)
class FeatureHypothesis:
    """A candidate feature configuration discovered by search."""

    hypothesis_id: str
    theme: str
    enabled_groups: List[str]
    score: float
    kept: bool
    metrics: Dict[str, float] = field(default_factory=dict)
    rationale: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.hypothesis_id,
            "kind": "feature_discovery",
            "theme": self.theme,
            "symbol": "",
            "priority": float(self.score),
            "rationale": self.rationale,
            "actions": ["use_feature_groups", "retrain", "strict_validate"],
            "metadata": {
                "enabled_groups": list(self.enabled_groups),
                "kept": self.kept,
                "metrics": self.metrics,
            },
        }


@dataclass
class DiscoveryResult:
    candidates: List[FeatureHypothesis] = field(default_factory=list)
    selected_groups: List[str] = field(default_factory=list)
    best_score: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "n_candidates": len(self.candidates),
            "selected_groups": list(self.selected_groups),
            "best_score": self.best_score,
            "kept": [c.to_dict() for c in self.candidates if c.kept],
            "discarded": [c.to_dict() for c in self.candidates if not c.kept],
        }


class HypothesisDiscoveryEngine:
    """
    Generate and score feature-group hypotheses from data.

    Does not invent new indicators — searches combinations of existing groups
    and theme packs aligned with institutional research questions.
    """

    def __init__(
        self,
        config: AIConfig,
        *,
        min_score: float = 0.52,
        max_candidates: int = 12,
        model_type: str = "random_forest",
    ):
        self.config = config
        self.min_score = min_score
        self.max_candidates = max_candidates
        self.model_type = model_type

    def discover(
        self,
        candles: Sequence[CandleDict],
        *,
        correlation_candles: Dict[str, Sequence[CandleDict]] | None = None,
    ) -> DiscoveryResult:
        if len(candles) < 120:
            return DiscoveryResult(selected_groups=list(self.config.features.enabled_groups))

        candidates: List[FeatureHypothesis] = []
        # Theme packs
        for theme, groups in RELATIONSHIP_THEMES.items():
            usable = [g for g in groups if g in DISCOVERY_GROUPS or g == FeatureGroup.CORRELATION.value]
            if FeatureGroup.CORRELATION.value in usable and not correlation_candles:
                usable = [g for g in usable if g != FeatureGroup.CORRELATION.value]
            if len(usable) < 2:
                continue
            hyp = self._evaluate_groups(
                candles,
                theme=theme,
                groups=usable,
                correlation_candles=correlation_candles,
            )
            if hyp is not None:
                candidates.append(hyp)

        # Small combinatorial search over core groups (bounded)
        core = [
            FeatureGroup.RETURNS.value,
            FeatureGroup.MOMENTUM.value,
            FeatureGroup.VOLATILITY.value,
            FeatureGroup.SESSION.value,
            FeatureGroup.REGIME.value,
            FeatureGroup.VOLUME.value,
        ]
        for combo in itertools.combinations(core, 3):
            if len(candidates) >= self.max_candidates:
                break
            theme = "combo_" + "_".join(combo[:2])
            hyp = self._evaluate_groups(
                candles,
                theme=theme,
                groups=list(combo),
                correlation_candles=correlation_candles,
            )
            if hyp is not None:
                candidates.append(hyp)

        candidates.sort(key=lambda c: c.score, reverse=True)
        kept = [c for c in candidates if c.kept]
        selected = list(kept[0].enabled_groups) if kept else list(self.config.features.enabled_groups)
        best = kept[0].score if kept else None
        return DiscoveryResult(
            candidates=candidates[: self.max_candidates],
            selected_groups=selected,
            best_score=best,
        )

    def _evaluate_groups(
        self,
        candles: Sequence[CandleDict],
        *,
        theme: str,
        groups: Sequence[str],
        correlation_candles: Dict[str, Sequence[CandleDict]] | None,
    ) -> FeatureHypothesis | None:
        cfg = self.config.copy()
        cfg.features.enabled_groups = list(dict.fromkeys(groups))
        if correlation_candles:
            cfg.features.correlation_symbols = list(correlation_candles.keys())
        cfg.model.model_type = self.model_type
        cfg.data.auto_download = False

        try:
            engine = FeatureEngine(config=cfg)
            frame = engine.transform(list(candles), cfg)
            labels = LabelGenerator(config=cfg).generate(list(candles), cfg)
            label = next(iter(labels.values()))
            X, y = _align(frame.matrix, label.values)
            if len(X) < 80:
                return None
            split = int(len(X) * 0.7)
            model = create_model(self.model_type, cfg)
            trainer = Trainer(config=cfg)
            # Minimal fit via trainer if DatasetBundle available, else direct fit
            from ai.datasets.schema import DatasetBundle

            bundle = DatasetBundle(
                X_train=X[:split],
                y_train=y[:split],
                X_val=X[split:],
                y_val=y[split:],
                X_test=X[split:],
                y_test=y[split:],
                feature_names=list(frame.feature_names),
                timestamps=[],
                metadata={"theme": theme},
            )
            result = trainer.fit(model, bundle, cfg)
            preds = np.asarray(result.model.predict(X[split:])).reshape(-1)
            score = float(_score(y[split:], preds))
            metrics = {"val_score": score, **{k: float(v) for k, v in result.metrics.items() if _is_num(v)}}
            kept = score >= self.min_score
            return FeatureHypothesis(
                hypothesis_id=f"feat_{theme}_{abs(hash(tuple(groups))) % 10_000_000:x}",
                theme=theme,
                enabled_groups=list(groups),
                score=score,
                kept=kept,
                metrics=metrics,
                rationale=(
                    f"Theme '{theme}' with groups {list(groups)} "
                    f"scored {score:.4f} ({'kept' if kept else 'discarded'})."
                ),
            )
        except Exception as exc:
            logger.debug("discovery theme %s failed: %s", theme, exc)
            return None


def _align(matrix: np.ndarray, labels: Sequence[Any]) -> Tuple[np.ndarray, np.ndarray]:
    y = np.asarray(labels).reshape(-1)
    n = min(len(matrix), len(y))
    X = np.asarray(matrix[:n], dtype=float)
    y = y[:n]
    mask = np.isfinite(X).all(axis=1) & np.isfinite(y.astype(float, copy=False))
    return X[mask], y[mask]


def _score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    yt = np.asarray(y_true).reshape(-1)
    yp = np.asarray(y_pred).reshape(-1)
    if yt.size == 0:
        return 0.0
    # Classification-friendly accuracy; falls back to sign agreement for continuous
    if np.unique(yt).size <= 10:
        return float(np.mean(yt == yp))
    return float(np.mean(np.sign(yt) == np.sign(yp)))


def _is_num(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)
