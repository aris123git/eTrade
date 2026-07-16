"""
ai/research - Autonomous Quant Research Platform.

Automates the research process (collect → validate → learn → keep improvements).
Does not invent market history that brokers/vendors do not provide.
"""

from ai.research.config import ResearchConfig
from ai.research.gate import GateDecision, decide_promotion
from ai.research.hypotheses import Hypothesis, generate_hypotheses
from ai.research.platform import AutonomousResearchPlatform, create_research_platform
from ai.research.report import CycleReport

__all__ = [
    "ResearchConfig",
    "GateDecision",
    "decide_promotion",
    "Hypothesis",
    "generate_hypotheses",
    "AutonomousResearchPlatform",
    "create_research_platform",
    "CycleReport",
]
