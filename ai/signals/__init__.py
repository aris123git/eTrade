"""
ai.signals - Prediction-to-trade signal generation.

RESPONSIBILITY:
Expose signal dataclasses, filters, and factories.

VERSION: 1.0.0
"""

from ai.signals.engine import RiskHook, SignalEngine, SignalFilterConfig, TradeSignal, create_signal_engine

__all__ = [
    "RiskHook",
    "SignalEngine",
    "SignalFilterConfig",
    "TradeSignal",
    "create_signal_engine",
]
