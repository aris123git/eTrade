"""
ai.risk - Risk management for live and simulated trading.

RESPONSIBILITY:
Expose sizing, validation, stop management, and portfolio limit controls.

VERSION: 1.0.0
"""

from ai.risk.manager import RiskDecision, RiskManager, create_risk_manager

__all__ = [
    "RiskDecision",
    "RiskManager",
    "create_risk_manager",
]
