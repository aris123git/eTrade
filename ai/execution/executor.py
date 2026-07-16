"""
ai/execution/executor.py - Paper and simulated order execution.

RESPONSIBILITY:
Turn strategy intents into order lifecycle reports with configurable slippage,
commission, latency metadata, and partial fill behavior.

VERSION: 1.0.0
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Any, Dict, Mapping
from uuid import uuid4

from ai.config.settings import AIConfig, ExecutionConfig
from ai.utils.types import OrderSide, OrderType, SignalType


class OrderStatus(str, Enum):
    """Order lifecycle statuses used by the paper executor."""

    NEW = "NEW"
    FILLED = "FILLED"
    PARTIALLY_FILLED = "PARTIALLY_FILLED"
    CANCELLED = "CANCELLED"
    REJECTED = "REJECTED"
    OPEN = "OPEN"


@dataclass
class Order:
    """Broker-neutral order request."""

    symbol: str
    side: OrderSide
    quantity: float
    order_type: OrderType = OrderType.MARKET
    price: float | None = None
    stop_price: float | None = None
    sl: float | None = None
    tp: float | None = None
    client_order_id: str = field(default_factory=lambda: uuid4().hex)
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Fill:
    """Executed quantity and price for an order."""

    order_id: str
    symbol: str
    side: OrderSide
    quantity: float
    price: float
    commission: float
    slippage: float
    timestamp: datetime
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ExecutionReport:
    """Result of an order submission."""

    order: Order
    status: OrderStatus
    fills: list[Fill] = field(default_factory=list)
    requested_price: float | None = None
    remaining_quantity: float = 0.0
    message: str = ""
    submitted_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    completed_at: datetime | None = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def filled_quantity(self) -> float:
        """Return total filled quantity."""

        return sum(fill.quantity for fill in self.fills)

    @property
    def average_price(self) -> float | None:
        """Return volume-weighted average fill price."""

        qty = self.filled_quantity
        if qty <= 0:
            return None
        return sum(fill.quantity * fill.price for fill in self.fills) / qty


@dataclass
class OrderExecutor:
    """Paper/sim executor with a broker-neutral method surface."""

    config: AIConfig = field(default_factory=AIConfig)
    mode: str = "paper"
    point_size: float = 0.0001
    open_orders: Dict[str, Order] = field(default_factory=dict)
    reports: Dict[str, ExecutionReport] = field(default_factory=dict)

    @property
    def execution_config(self) -> ExecutionConfig:
        """Return the active nested execution config."""

        return self.config.execution

    def create_order(
        self,
        *,
        symbol: str,
        side: OrderSide | SignalType | str,
        quantity: float,
        order_type: OrderType | str | None = None,
        price: float | None = None,
        stop_price: float | None = None,
        sl: float | None = None,
        tp: float | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> Order:
        """Create a broker-neutral Order object."""

        return Order(
            symbol=symbol,
            side=_order_side(side),
            quantity=max(float(quantity), 0.0),
            order_type=_order_type(order_type or self.execution_config.default_order_type),
            price=price,
            stop_price=stop_price,
            sl=sl,
            tp=tp,
            metadata=dict(metadata or {}),
        )

    def execute_intent(
        self,
        intent: Any,
        *,
        market_price: float,
        timestamp: datetime | None = None,
    ) -> ExecutionReport:
        """Create and submit an order from a strategy TradeIntent-like object."""

        order = self.create_order(
            symbol=str(getattr(intent, "symbol")),
            side=getattr(intent, "side"),
            quantity=float(getattr(intent, "size", 0.0)),
            order_type=getattr(intent, "order_type", self.execution_config.default_order_type),
            price=getattr(intent, "entry", None),
            sl=getattr(intent, "sl", None),
            tp=getattr(intent, "tp", None),
            metadata={"intent": intent, **dict(getattr(intent, "metadata", {}) or {})},
        )
        return self.submit_order(order, market_price=market_price, timestamp=timestamp)

    def submit_order(
        self,
        order: Order,
        *,
        market_price: float,
        timestamp: datetime | None = None,
    ) -> ExecutionReport:
        """Submit an order to the configured execution mode."""

        ts = _utc(timestamp or datetime.now(timezone.utc))
        if order.quantity <= 0:
            report = self._report(order, OrderStatus.REJECTED, "quantity_must_be_positive", ts, market_price)
            self.reports[order.client_order_id] = report
            return report

        trigger_price = self._trigger_price(order, market_price)
        if trigger_price is None:
            report = self._report(order, OrderStatus.OPEN, "order_waiting_for_trigger", ts, market_price)
            self.open_orders[order.client_order_id] = order
            self.reports[order.client_order_id] = report
            return report

        fill_quantity = self._fill_quantity(order.quantity)
        fill_price = self._apply_slippage(trigger_price, order.side)
        commission = self._commission(fill_quantity)
        fill = Fill(
            order_id=order.client_order_id,
            symbol=order.symbol,
            side=order.side,
            quantity=fill_quantity,
            price=fill_price,
            commission=commission,
            slippage=fill_price - trigger_price,
            timestamp=ts + timedelta(milliseconds=float(self.execution_config.latency_ms)),
            metadata={"mode": self.mode, "magic_number": self.execution_config.magic_number},
        )
        remaining = max(order.quantity - fill_quantity, 0.0)
        status = OrderStatus.FILLED if remaining <= 1e-12 else OrderStatus.PARTIALLY_FILLED
        report = ExecutionReport(
            order=order,
            status=status,
            fills=[fill],
            requested_price=market_price,
            remaining_quantity=remaining,
            message="filled" if status == OrderStatus.FILLED else "partially_filled",
            submitted_at=ts,
            completed_at=fill.timestamp,
            metadata={
                "mode": self.mode,
                "latency_ms": self.execution_config.latency_ms,
                "commission_per_lot": self.execution_config.commission_per_lot,
            },
        )
        if remaining > 0:
            self.open_orders[order.client_order_id] = order
        else:
            self.open_orders.pop(order.client_order_id, None)
        self.reports[order.client_order_id] = report
        return report

    def cancel_order(self, client_order_id: str) -> ExecutionReport:
        """Cancel an open order by client order id."""

        order = self.open_orders.pop(client_order_id, None)
        if order is None:
            existing = self.reports.get(client_order_id)
            if existing is not None:
                return existing
            placeholder = Order(symbol="", side=OrderSide.BUY, quantity=0.0)
            return self._report(placeholder, OrderStatus.REJECTED, "order_not_found", datetime.now(timezone.utc), None)
        report = self._report(order, OrderStatus.CANCELLED, "cancelled", datetime.now(timezone.utc), None)
        self.reports[client_order_id] = report
        return report

    def _trigger_price(self, order: Order, market_price: float) -> float | None:
        order_type = order.order_type
        if order_type == OrderType.MARKET:
            return float(market_price)
        if order_type == OrderType.LIMIT:
            limit = float(order.price if order.price is not None else market_price)
            if order.side == OrderSide.BUY and market_price <= limit:
                return limit
            if order.side == OrderSide.SELL and market_price >= limit:
                return limit
            return None
        if order_type in {OrderType.STOP, OrderType.STOP_LIMIT}:
            stop = float(order.stop_price if order.stop_price is not None else order.price if order.price is not None else market_price)
            if order.side == OrderSide.BUY and market_price >= stop:
                return stop
            if order.side == OrderSide.SELL and market_price <= stop:
                return stop
            return None
        return float(market_price)

    def _apply_slippage(self, price: float, side: OrderSide) -> float:
        slippage = float(self.execution_config.slippage_points) * float(self.point_size)
        if side == OrderSide.BUY:
            return float(price) + slippage
        return float(price) - slippage

    def _fill_quantity(self, quantity: float) -> float:
        if not self.execution_config.partial_fill_enabled:
            return float(quantity)
        ratio = min(max(float(self.execution_config.partial_fill_ratio), 0.0), 1.0)
        return float(quantity) * ratio

    def _commission(self, quantity: float) -> float:
        return abs(float(quantity)) * float(self.execution_config.commission_per_lot)

    def _report(
        self,
        order: Order,
        status: OrderStatus,
        message: str,
        timestamp: datetime,
        requested_price: float | None,
    ) -> ExecutionReport:
        return ExecutionReport(
            order=order,
            status=status,
            requested_price=requested_price,
            remaining_quantity=order.quantity,
            message=message,
            submitted_at=timestamp,
            completed_at=timestamp if status in {OrderStatus.CANCELLED, OrderStatus.REJECTED} else None,
            metadata={"mode": self.mode},
        )


def create_order_executor(
    config: AIConfig | None = None,
    *,
    mode: str = "paper",
    point_size: float = 0.0001,
) -> OrderExecutor:
    """Factory for OrderExecutor."""

    return OrderExecutor(config=config or AIConfig(), mode=mode, point_size=point_size)


def _order_side(value: OrderSide | SignalType | str) -> OrderSide:
    raw = value.value if isinstance(value, (OrderSide, SignalType)) else str(value)
    normalized = raw.upper()
    if normalized in {"BUY", "LONG"}:
        return OrderSide.BUY
    if normalized in {"SELL", "SHORT"}:
        return OrderSide.SELL
    raise ValueError(f"Unsupported order side: {value!r}")


def _order_type(value: OrderType | str) -> OrderType:
    if isinstance(value, OrderType):
        return value
    normalized = str(value).lower()
    for order_type in OrderType:
        if normalized == order_type.value:
            return order_type
    raise ValueError(f"Unsupported order type: {value!r}")


def _utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)
