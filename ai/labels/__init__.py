"""
ai/labels - Production supervised label generation

RESPONSIBILITY:
Expose labeler contracts, factories, and concrete label methods.

VERSION: 1.0.0
"""

from ai.labels.barriers import RiskReward, StopLossDistance, TakeProfitDistance, TripleBarrierMethod
from ai.labels.base import BaseLabeler, LabelResult
from ai.labels.binary import BinaryDirectionLabeler
from ai.labels.generator import LabelGenerator, create_label_generator
from ai.labels.meta import MetaLabels
from ai.labels.multiclass import MultiClassDirectionLabeler
from ai.labels.regression import FutureHigh, FutureLow, FutureReturn, FutureVolatility

__all__ = [
    "BaseLabeler",
    "LabelResult",
    "BinaryDirectionLabeler",
    "MultiClassDirectionLabeler",
    "FutureReturn",
    "FutureHigh",
    "FutureLow",
    "FutureVolatility",
    "TripleBarrierMethod",
    "RiskReward",
    "StopLossDistance",
    "TakeProfitDistance",
    "MetaLabels",
    "LabelGenerator",
    "create_label_generator",
]
