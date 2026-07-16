"""
ai/portfolio/manager.py - Portfolio state, PnL, and allocation controls.

RESPONSIBILITY:
Track multi-symbol and multi-broker positions, closed trades, exposure,
allocation, rebalancing hooks, and performance metrics.

VERSION: 1.0.0
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, Mapping
from uuid import uuid4

import numpy as np

from ai.config.settings import AIConfig
from ai.execution import Fill
from ai.utils.types import OrderSide, PositionSide


@dataclass
class Position:
    """Open portfolio position."""

    symbol: str
    side: PositionSide
    size: float
    entry_price: float
    current_price: float
    broker: str = "default"
    position_id: str = field(default_factory=lambda: uuid4().hex)
    opened_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    sl: float | None = None
    tp: float | None = None
    commission: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def exposure(self) -> float:
        """Return notional exposure."""

        return abs(self.size * self.current_price)

    @property
    def signed_exposure(self) -> float:
        """Return direction-aware notional exposure."""

        direction = 1.0 if self.side == PositionSide.LONG else -1.0
        return direction * self.size * self.current_price

    @property
    def unrealized_pnl(self) -> float:
        """Return current unrealized PnL net of entry commission."""

        if self.side == PositionSide.LONG:
            gross = (self.current_price - self.entry_price) * self.size
        elif self.side == PositionSide.SHORT:
            gross = (self.entry_price - self.current_price) * self.size
        else:
            gross = 0.0
        return gross - self.commission

    def update_price(self, price: float) -> None:
        """Update mark price."""

        self.current_price = float(price)


@dataclass
class Trade:
    """Closed trade record."""

    symbol: str
    side: PositionSide
    size: float
    entry_price: float
    exit_price: float
    pnl: float
    opened_at: datetime
    closed_at: datetime
    broker: str = "default"
    trade_id: str = field(default_factory=lambda: uuid4().hex)
    commission: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PortfolioManager:
    """Portfolio manager for live, paper, and research workflows."""

    config: AIConfig = field(default_factory=AIConfig)
    cash: float = 100_000.0
    base_currency: str | None = None
    open_positions: Dict[str, Position] = field(default_factory=dict)
    closed_trades: list[Trade] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.base_currency = self.base_currency or self.config.risk.account_currency

    def open_position(
        self,
        *,
        symbol: str,
        side: PositionSide | OrderSide | str,
        size: float,
        price: float,
        broker: str = "default",
        sl: float | None = None,
        tp: float | None = None,
        commission: float = 0.0,
        metadata: Mapping[str, Any] | None = None,
    ) -> Position:
        """Open and register a new position."""

        position = Position(
            symbol=symbol,
            side=_position_side(side),
            size=abs(float(size)),
            entry_price=float(price),
            current_price=float(price),
            broker=broker,
            sl=sl,
            tp=tp,
            commission=float(commission),
            metadata=dict(metadata or {}),
        )
        self.open_positions[position.position_id] = position
        self.cash -= float(commission)
        return position

    def close_position(
        self,
        position_id: str,
        *,
        price: float,
        closed_at: datetime | None = None,
        commission: float = 0.0,
        metadata: Mapping[str, Any] | None = None,
    ) -> Trade:
        """Close a position and move it into closed trades."""

        position = self.open_positions.pop(position_id)
        position.update_price(price)
        total_commission = position.commission + float(commission)
        pnl = _position_pnl(position.side, position.entry_price, float(price), position.size) - total_commission
        trade = Trade(
            symbol=position.symbol,
            side=position.side,
            size=position.size,
            entry_price=position.entry_price,
            exit_price=float(price),
            pnl=pnl,
            opened_at=position.opened_at,
            closed_at=_utc(closed_at or datetime.now(timezone.utc)),
            broker=position.broker,
            commission=total_commission,
            metadata={**position.metadata, **dict(metadata or {})},
        )
        self.closed_trades.append(trade)
        self.cash += pnl
        return trade

    def apply_fill(self, fill: Fill, *, broker: str = "default") -> Position | Trade:
        """Apply an execution fill as an opening or closing portfolio event."""

        opposite = self._find_opposite_position(fill.symbol, fill.side, broker)
        if opposite is not None:
            return self.close_position(
                opposite.position_id,
                price=fill.price,
                closed_at=fill.timestamp,
                commission=fill.commission,
                metadata={"fill": fill},
            )
        return self.open_position(
            symbol=fill.symbol,
            side=fill.side,
            size=fill.quantity,
            price=fill.price,
            broker=broker,
            commission=fill.commission,
            metadata={"fill": fill},
        )

    def update_prices(self, prices: Mapping[str, float] | Mapping[tuple[str, str], float]) -> None:
        """Update current marks from symbol or (broker, symbol) price maps."""

        for position in self.open_positions.values():
            broker_key = (position.broker, position.symbol)
            price = prices.get(broker_key, prices.get(position.symbol))  # type: ignore[arg-type]
            if price is not None:
                position.update_price(float(price))

    def positions(
        self,
        *,
        symbol: str | None = None,
        broker: str | None = None,
    ) -> list[Position]:
        """Return open positions filtered by symbol and broker."""

        return [
            position
            for position in self.open_positions.values()
            if (symbol is None or position.symbol == symbol) and (broker is None or position.broker == broker)
        ]

    def realized_pnl(self) -> float:
        """Return cumulative closed-trade PnL."""

        return sum(trade.pnl for trade in self.closed_trades)

    def unrealized_pnl(self) -> float:
        """Return current open-position PnL."""

        return sum(position.unrealized_pnl for position in self.open_positions.values())

    def total_equity(self) -> float:
        """Return cash plus unrealized PnL."""

        return self.cash + self.unrealized_pnl()

    def exposure(self, *, symbol: str | None = None, broker: str | None = None) -> float:
        """Return gross exposure for a scope."""

        return sum(position.exposure for position in self.positions(symbol=symbol, broker=broker))

    def net_exposure(self, *, symbol: str | None = None, broker: str | None = None) -> float:
        """Return direction-aware exposure for a scope."""

        return sum(position.signed_exposure for position in self.positions(symbol=symbol, broker=broker))

    def allocation(self) -> Dict[str, float]:
        """Return symbol allocation as gross exposure divided by total gross exposure."""

        totals: Dict[str, float] = {}
        for position in self.open_positions.values():
            totals[position.symbol] = totals.get(position.symbol, 0.0) + position.exposure
        gross = sum(totals.values())
        if gross <= 0:
            return {symbol: 0.0 for symbol in totals}
        return {symbol: value / gross for symbol, value in totals.items()}

    def rebalance_hooks(
        self,
        target_allocation: Mapping[str, float],
        *,
        tolerance: float = 0.01,
    ) -> list[Dict[str, Any]]:
        """Return suggested allocation adjustments without submitting orders."""

        current = self.allocation()
        suggestions: list[Dict[str, Any]] = []
        symbols = sorted(set(current) | set(target_allocation))
        equity = max(self.total_equity(), 1e-12)
        for symbol in symbols:
            target = float(target_allocation.get(symbol, 0.0))
            actual = float(current.get(symbol, 0.0))
            drift = target - actual
            if abs(drift) > tolerance:
                suggestions.append(
                    {
                        "symbol": symbol,
                        "action": "increase" if drift > 0 else "decrease",
                        "target_allocation": target,
                        "current_allocation": actual,
                        "notional_delta": drift * equity,
                    }
                )
        return suggestions

    def performance_metrics(self) -> Dict[str, float]:
        """Return key closed and open portfolio performance metrics."""

        pnls = np.asarray([trade.pnl for trade in self.closed_trades], dtype=float)
        wins = pnls[pnls > 0]
        losses = pnls[pnls < 0]
        total = float(np.sum(pnls)) if pnls.size else 0.0
        win_rate = float(wins.size / pnls.size) if pnls.size else 0.0
        profit_factor = float(np.sum(wins) / abs(np.sum(losses))) if losses.size and abs(np.sum(losses)) > 0 else 0.0
        expectancy = float(np.mean(pnls)) if pnls.size else 0.0
        return {
            "cash": float(self.cash),
            "equity": float(self.total_equity()),
            "realized_pnl": float(self.realized_pnl()),
            "unrealized_pnl": float(self.unrealized_pnl()),
            "total_closed_pnl": total,
            "open_positions": float(len(self.open_positions)),
            "closed_trades": float(len(self.closed_trades)),
            "win_rate": win_rate,
            "profit_factor": profit_factor,
            "expectancy": expectancy,
            "gross_exposure": float(self.exposure()),
            "net_exposure": float(self.net_exposure()),
        }

    def _find_opposite_position(self, symbol: str, fill_side: OrderSide, broker: str) -> Position | None:
        target_side = PositionSide.SHORT if fill_side == OrderSide.BUY else PositionSide.LONG
        for position in self.open_positions.values():
            if position.symbol == symbol and position.broker == broker and position.side == target_side:
                return position
        return None


def create_portfolio_manager(
    config: AIConfig | None = None,
    *,
    cash: float = 100_000.0,
    base_currency: str | None = None,
) -> PortfolioManager:
    """Factory for PortfolioManager."""

    return PortfolioManager(config=config or AIConfig(), cash=cash, base_currency=base_currency)


def _position_side(value: PositionSide | OrderSide | str) -> PositionSide:
    raw = value.value if isinstance(value, (PositionSide, OrderSide)) else str(value)
    normalized = raw.upper()
    if normalized in {"LONG", "BUY"}:
        return PositionSide.LONG
    if normalized in {"SHORT", "SELL"}:
        return PositionSide.SHORT
    return PositionSide.FLAT


def _position_pnl(side: PositionSide, entry: float, exit_price: float, size: float) -> float:
    if side == PositionSide.LONG:
        return (exit_price - entry) * size
    if side == PositionSide.SHORT:
        return (entry - exit_price) * size
    return 0.0


def _utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)
