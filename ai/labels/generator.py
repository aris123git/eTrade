"""
ai/labels/generator.py - Label generation orchestration

RESPONSIBILITY:
Build all configured supervised learning labels with consistent horizon alignment.

VERSION: 1.0.0
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, Iterable, Mapping

from ai.config.settings import AIConfig
from ai.labels.barriers import RiskReward, StopLossDistance, TakeProfitDistance, TripleBarrierMethod
from ai.labels.base import BaseLabeler, LabelResult
from ai.labels.binary import BinaryDirectionLabeler
from ai.labels.meta import MetaLabels
from ai.labels.multiclass import MultiClassDirectionLabeler
from ai.labels.regression import FutureHigh, FutureLow, FutureReturn, FutureVolatility


# ==============================================================================
# REGISTRY
# ==============================================================================


LabelerFactory = Callable[[AIConfig], BaseLabeler]


def _default_registry() -> Dict[str, LabelerFactory]:
    return {
        "binary_direction": BinaryDirectionLabeler,
        "binary": BinaryDirectionLabeler,
        "multiclass_direction": MultiClassDirectionLabeler,
        "multiclass": MultiClassDirectionLabeler,
        "future_return": FutureReturn,
        "return": FutureReturn,
        "future_high": FutureHigh,
        "future_low": FutureLow,
        "future_volatility": FutureVolatility,
        "volatility": FutureVolatility,
        "triple_barrier": TripleBarrierMethod,
        "triple_barrier_method": TripleBarrierMethod,
        "risk_reward": RiskReward,
        "rr": RiskReward,
        "stop_loss_distance": StopLossDistance,
        "sl_distance": StopLossDistance,
        "take_profit_distance": TakeProfitDistance,
        "tp_distance": TakeProfitDistance,
        "meta_labels": MetaLabels,
        "meta": MetaLabels,
    }


# ==============================================================================
# GENERATOR
# ==============================================================================


@dataclass
class LabelGenerator:
    """Config-driven orchestrator for all production label methods."""

    config: AIConfig = field(default_factory=AIConfig)
    registry: Mapping[str, LabelerFactory] = field(default_factory=_default_registry)

    def generate(self, candles: object, config: AIConfig | None = None) -> Dict[str, LabelResult]:
        """
        Generate every configured label for every configured horizon.

        Label values remain aligned to the input candle row at time t. Future
        information is only used inside the target value, and invalid trailing
        rows are marked through each result's valid_mask.
        """
        active_config = config or self.config
        horizons = self._horizons(active_config.labels.horizons, active_config.labels.horizon)
        results: Dict[str, LabelResult] = {}

        for raw_method in active_config.labels.methods:
            method = raw_method.strip().lower()
            if method not in self.registry:
                raise ValueError(f"Unsupported label method: {raw_method}")
            labeler = self.registry[method](active_config)
            canonical_method = labeler.method
            for horizon in horizons:
                key = f"{canonical_method}_{horizon}"
                results[key] = labeler.label(candles, horizon=horizon, name=key)
        return results

    @staticmethod
    def _horizons(configured: Iterable[int], fallback: int) -> list[int]:
        seen: set[int] = set()
        horizons: list[int] = []
        for item in configured:
            horizon = int(item)
            if horizon <= 0:
                raise ValueError("label horizons must be > 0")
            if horizon not in seen:
                seen.add(horizon)
                horizons.append(horizon)
        if not horizons:
            horizons.append(int(fallback))
        return horizons


def create_label_generator(config: AIConfig | None = None) -> LabelGenerator:
    """Factory for dependency-injected label generators."""
    return LabelGenerator(config=config or AIConfig())
