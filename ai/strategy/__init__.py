"""
ai.strategy - Trading strategies and intent factories.

RESPONSIBILITY:
Expose strategy composition primitives for prediction, signal, and risk layers.

VERSION: 1.0.0
"""

from ai.strategy.base import (
    ConfidenceThresholdStrategy,
    SignalStrategy,
    Strategy,
    TradeIntent,
    create_confidence_threshold_strategy,
    create_signal_strategy,
)

__all__ = [
    "ConfidenceThresholdStrategy",
    "SignalStrategy",
    "Strategy",
    "TradeIntent",
    "create_confidence_threshold_strategy",
    "create_signal_strategy",
]
