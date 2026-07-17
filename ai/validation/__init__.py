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
from ai.validation.phase5b_validator import (
    Phase5bDecisionEngine,
    Phase5bDeltaCalculator,
    Phase5bValidator,
)

__all__ = [
    "HistoricalDataLoader",
    "OverfitDetector",
    "Phase5aValidator",
    "Phase5bDecisionEngine",
    "Phase5bDeltaCalculator",
    "Phase5bValidator",
    "StatisticalTester",
    "TrainTestSplit",
    "WalkForwardBacktester",
]
