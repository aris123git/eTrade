"""
ai/validation - Historical validation and walk-forward backtesting (Phase 5a+).

New modules only. Does not alter existing training/signal/execution code.
"""

from ai.validation.phase5a_validator import (
    HistoricalDataLoader,
    OverfitDetector,
    Phase5aValidator,
    StatisticalTester,
    TrainTestSplit,
    WalkForwardBacktester,
)

__all__ = [
    "HistoricalDataLoader",
    "OverfitDetector",
    "Phase5aValidator",
    "StatisticalTester",
    "TrainTestSplit",
    "WalkForwardBacktester",
]
