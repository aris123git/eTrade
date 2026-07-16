"""
ai.portfolio - Portfolio state, PnL, and allocation management.

RESPONSIBILITY:
Expose portfolio dataclasses, manager, and factory helpers.

VERSION: 1.0.0
"""

from ai.portfolio.manager import PortfolioManager, Position, Trade, create_portfolio_manager

__all__ = [
    "PortfolioManager",
    "Position",
    "Trade",
    "create_portfolio_manager",
]
