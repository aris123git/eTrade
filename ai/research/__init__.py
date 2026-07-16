"""
ai/research - Autonomous Quantitative Research Engine (Phase 4).
"""

from ai.research.autonomous_scheduler import AutonomousScheduler, SchedulePlan
from ai.research.config import ResearchConfig
from ai.research.discovery import FeatureHypothesis, HypothesisDiscoveryEngine
from ai.research.gate import GateDecision, decide_promotion
from ai.research.hypotheses import Hypothesis, generate_hypotheses
from ai.research.paper_journal import PaperTradingJournal
from ai.research.platform import AutonomousResearchPlatform, create_research_platform
from ai.research.production_gate import ProductionReadinessGate
from ai.research.report import CycleReport
from ai.research.self_improve import SelfImprovementController
from ai.research.validation_gate import StrictValidationGate

__all__ = [
    "AutonomousScheduler",
    "SchedulePlan",
    "ResearchConfig",
    "FeatureHypothesis",
    "HypothesisDiscoveryEngine",
    "GateDecision",
    "decide_promotion",
    "Hypothesis",
    "generate_hypotheses",
    "PaperTradingJournal",
    "AutonomousResearchPlatform",
    "create_research_platform",
    "ProductionReadinessGate",
    "CycleReport",
    "SelfImprovementController",
    "StrictValidationGate",
]
