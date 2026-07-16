"""
ai.prediction - Live prediction services.

RESPONSIBILITY:
Expose production model-serving entry points for eTrade AI.

VERSION: 1.0.0
"""

from ai.prediction.service import PredictionService, create_prediction_service, prediction_to_signal

__all__ = [
    "PredictionService",
    "create_prediction_service",
    "prediction_to_signal",
]
