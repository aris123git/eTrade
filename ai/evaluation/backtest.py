"""
ai/evaluation/backtest.py - Deterministic multi-asset backtest engine.

VERSION: 1.0.0
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Dict, Iterable, List, Sequence

import numpy as np

from ai.config.settings import AIConfig
from ai.evaluation.trading_metrics import trading_metrics


LONG_SIDES = {"buy", "long"}
SHORT_SIDES = {"sell", "short"}
CLOSE_SIDES = {"close", "exit", "flat"}


@dataclass(frozen=True)
class Candle:
    """Normalized OHLCV candle."""

    symbol: str
    timeframe: str
    timestamp: datetime | float
    open: float
    high: float
    low: float
    close: float
    volume: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class BacktestSignal:
    """Normalized trading signal consumed by BacktestEngine."""

    symbol: str
    timestamp: datetime | float
    side: str
    order_type: str = "market"
    quantity: float | None = None
    price: float | None = None
    stop_loss: float | None = None
    take_profit: float | None = None
    timeframe: str | None = None
    expires_at: datetime | float | None = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Trade:
    """Closed trade record."""

    trade_id: str
    symbol: str
    timeframe: str
    side: str
    quantity: float
    entry_time: datetime | float
    entry_price: float
    exit_time: datetime | float
    exit_price: float
    pnl: float
    return_pct: float
    commission: float
    slippage: float
    holding_seconds: float
    entry_signal_id: str
    exit_reason: str
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize the trade to JSON-compatible primitives."""
        return {
            "trade_id": self.trade_id,
            "symbol": self.symbol,
            "timeframe": self.timeframe,
            "side": self.side,
            "quantity": self.quantity,
            "entry_time": _serialize_time(self.entry_time),
            "entry_price": self.entry_price,
            "exit_time": _serialize_time(self.exit_time),
            "exit_price": self.exit_price,
            "pnl": self.pnl,
            "return_pct": self.return_pct,
            "commission": self.commission,
            "slippage": self.slippage,
            "holding_seconds": self.holding_seconds,
            "entry_signal_id": self.entry_signal_id,
            "exit_reason": self.exit_reason,
            "metadata": self.metadata,
        }


@dataclass(frozen=True)
class EquityPoint:
    """Equity snapshot at a candle timestamp."""

    timestamp: datetime | float
    equity: float
    balance: float

    def to_dict(self) -> Dict[str, Any]:
        """Serialize the equity point."""
        return {
            "timestamp": _serialize_time(self.timestamp),
            "equity": self.equity,
            "balance": self.balance,
        }


@dataclass(frozen=True)
class BacktestResult:
    """Backtest output with closed trades, equity snapshots, and metrics."""

    trades: List[Trade]
    equity: List[EquityPoint]
    metrics: Dict[str, Any]
    initial_equity: float
    final_equity: float
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize the result to JSON-compatible primitives."""
        return {
            "trades": [trade.to_dict() for trade in self.trades],
            "equity": [point.to_dict() for point in self.equity],
            "metrics": self.metrics,
            "initial_equity": self.initial_equity,
            "final_equity": self.final_equity,
            "metadata": self.metadata,
        }


@dataclass
class _Position:
    symbol: str
    timeframe: str
    side: str
    quantity: float
    entry_time: datetime | float
    entry_price: float
    stop_loss: float | None
    take_profit: float | None
    entry_commission: float
    entry_slippage: float
    entry_signal_id: str
    metadata: Dict[str, Any]


@dataclass
class _PendingOrder:
    signal_id: str
    signal: BacktestSignal
    activation_time: datetime | float


@dataclass
class BacktestEngine:
    """Backtest market, limit, and stop orders over OHLCV candles."""

    config: AIConfig = field(default_factory=AIConfig)
    initial_equity: float = 10_000.0
    spread_points: float = 0.0
    point_size: float = 0.0001
    periods: int = 252
    commission_per_lot: float | None = None
    slippage_points: float | None = None
    latency_ms: float | None = None
    partial_fill_enabled: bool | None = None
    partial_fill_ratio: float | None = None

    def __post_init__(self) -> None:
        execution = self.config.execution
        self.commission_per_lot = (
            execution.commission_per_lot if self.commission_per_lot is None else self.commission_per_lot
        )
        self.slippage_points = execution.slippage_points if self.slippage_points is None else self.slippage_points
        self.latency_ms = execution.latency_ms if self.latency_ms is None else self.latency_ms
        self.partial_fill_enabled = (
            execution.partial_fill_enabled if self.partial_fill_enabled is None else self.partial_fill_enabled
        )
        self.partial_fill_ratio = (
            execution.partial_fill_ratio if self.partial_fill_ratio is None else self.partial_fill_ratio
        )

    def run(
        self,
        signals: Sequence[BacktestSignal | Dict[str, Any] | Any],
        candles: Sequence[Candle | Dict[str, Any] | Any],
    ) -> BacktestResult:
        """Run the backtest and return closed trades, equity snapshots, and metrics."""
        normalized_candles = sorted(
            (_normalize_candle(candle, self.config.primary_timeframe) for candle in candles),
            key=lambda candle: (_time_key(candle.timestamp), candle.symbol, candle.timeframe),
        )
        if not normalized_candles:
            return BacktestResult(
                trades=[],
                equity=[EquityPoint(timestamp=0.0, equity=self.initial_equity, balance=self.initial_equity)],
                metrics=trading_metrics([], initial_equity=self.initial_equity, periods=self.periods),
                initial_equity=self.initial_equity,
                final_equity=self.initial_equity,
                metadata={"signals": len(signals), "candles": 0},
            )

        pending = self._pending_orders(signals)
        pending_index = 0
        active_orders: List[_PendingOrder] = []
        positions: List[_Position] = []
        trades: List[Trade] = []
        equity: List[EquityPoint] = []
        latest_close: Dict[tuple[str, str], float] = {}
        balance = float(self.initial_equity)

        for candle in normalized_candles:
            latest_close[(candle.symbol, candle.timeframe)] = candle.close
            balance = self._process_position_exits(candle, positions, trades, balance)

            while pending_index < len(pending) and _time_key(pending[pending_index].activation_time) <= _time_key(candle.timestamp):
                active_orders.append(pending[pending_index])
                pending_index += 1

            remaining_orders: List[_PendingOrder] = []
            for order in active_orders:
                if _is_expired(order.signal, candle.timestamp):
                    continue
                if not self._matches(order.signal, candle):
                    remaining_orders.append(order)
                    continue
                balance, filled = self._try_execute_order(order, candle, positions, trades, balance)
                if not filled:
                    remaining_orders.append(order)
            active_orders = remaining_orders

            mark = self._mark_to_market(positions, latest_close)
            equity.append(EquityPoint(timestamp=candle.timestamp, equity=balance + mark, balance=balance))

        last_candles = self._last_candles_by_market(normalized_candles)
        for position in list(positions):
            candle = last_candles[(position.symbol, position.timeframe)]
            balance = self._close_position(position, candle.close, candle.timestamp, "end_of_data", trades, balance)
            positions.remove(position)
        equity.append(
            EquityPoint(
                timestamp=normalized_candles[-1].timestamp,
                equity=balance,
                balance=balance,
            )
        )

        trade_pnl = [trade.pnl for trade in trades]
        holding = [trade.holding_seconds for trade in trades]
        metrics = trading_metrics(
            trade_pnl,
            holding_periods=holding,
            initial_equity=self.initial_equity,
            periods=self.periods,
        )
        metrics["equity_curve"] = [point.equity for point in equity]
        return BacktestResult(
            trades=trades,
            equity=equity,
            metrics=metrics,
            initial_equity=self.initial_equity,
            final_equity=balance,
            metadata={
                "signals": len(signals),
                "candles": len(normalized_candles),
                "spread_points": self.spread_points,
                "slippage_points": self.slippage_points,
                "latency_ms": self.latency_ms,
            },
        )

    def backtest(
        self,
        signals: Sequence[BacktestSignal | Dict[str, Any] | Any],
        candles: Sequence[Candle | Dict[str, Any] | Any],
    ) -> BacktestResult:
        """Alias for run."""
        return self.run(signals=signals, candles=candles)

    def _pending_orders(self, signals: Sequence[BacktestSignal | Dict[str, Any] | Any]) -> List[_PendingOrder]:
        orders: List[_PendingOrder] = []
        for idx, raw_signal in enumerate(signals):
            signal = _normalize_signal(raw_signal, self.config)
            orders.append(
                _PendingOrder(
                    signal_id=str(signal.metadata.get("id", f"signal_{idx}")),
                    signal=signal,
                    activation_time=_add_latency(signal.timestamp, float(self.latency_ms or 0.0)),
                )
            )
        return sorted(orders, key=lambda order: _time_key(order.activation_time))

    def _process_position_exits(
        self,
        candle: Candle,
        positions: List[_Position],
        trades: List[Trade],
        balance: float,
    ) -> float:
        for position in list(positions):
            if position.symbol != candle.symbol or position.timeframe != candle.timeframe:
                continue
            exit_price, reason = self._exit_trigger(position, candle)
            if exit_price is None:
                continue
            balance = self._close_position(position, exit_price, candle.timestamp, reason, trades, balance)
            positions.remove(position)
        return balance

    def _try_execute_order(
        self,
        order: _PendingOrder,
        candle: Candle,
        positions: List[_Position],
        trades: List[Trade],
        balance: float,
    ) -> tuple[float, bool]:
        signal = order.signal
        side = signal.side.lower()
        fill_price = self._fill_price(signal, candle)
        if fill_price is None:
            return balance, False

        if side in CLOSE_SIDES:
            for position in list(positions):
                if position.symbol == candle.symbol and position.timeframe == candle.timeframe:
                    balance = self._close_position(position, fill_price, candle.timestamp, "signal_close", trades, balance)
                    positions.remove(position)
            return balance, True

        direction = _direction(side)
        if direction == 0:
            return balance, True

        if not self.config.execution.allow_hedging or self.config.execution.close_on_opposite:
            for position in list(positions):
                if (
                    position.symbol == candle.symbol
                    and position.timeframe == candle.timeframe
                    and _direction(position.side) != direction
                ):
                    balance = self._close_position(position, fill_price, candle.timestamp, "opposite_signal", trades, balance)
                    positions.remove(position)

        if not self.config.execution.allow_hedging and any(
            position.symbol == candle.symbol and position.timeframe == candle.timeframe for position in positions
        ):
            return balance, True

        quantity = self._filled_quantity(signal, candle)
        if quantity <= 0.0:
            return balance, True
        adjusted_price, slippage_cost = self._execution_price(fill_price, side, entry=True, quantity=quantity)
        positions.append(
            _Position(
                symbol=candle.symbol,
                timeframe=candle.timeframe,
                side="long" if direction > 0 else "short",
                quantity=quantity,
                entry_time=candle.timestamp,
                entry_price=adjusted_price,
                stop_loss=signal.stop_loss,
                take_profit=signal.take_profit,
                entry_commission=self._commission(quantity),
                entry_slippage=slippage_cost,
                entry_signal_id=order.signal_id,
                metadata=dict(signal.metadata),
            )
        )
        return balance, True

    def _fill_price(self, signal: BacktestSignal, candle: Candle) -> float | None:
        order_type = signal.order_type.lower()
        side = signal.side.lower()
        if order_type == "market" or side in CLOSE_SIDES:
            return candle.open
        price = signal.price
        if price is None:
            raise ValueError(f"{order_type} order requires price")
        if order_type == "limit":
            if side in LONG_SIDES and candle.low <= price:
                return price
            if side in SHORT_SIDES and candle.high >= price:
                return price
            return None
        if order_type == "stop":
            if side in LONG_SIDES and candle.high >= price:
                return price
            if side in SHORT_SIDES and candle.low <= price:
                return price
            return None
        raise ValueError(f"Unsupported order type: {signal.order_type}")

    def _exit_trigger(self, position: _Position, candle: Candle) -> tuple[float | None, str]:
        if position.side == "long":
            if position.stop_loss is not None and candle.low <= position.stop_loss:
                return position.stop_loss, "stop_loss"
            if position.take_profit is not None and candle.high >= position.take_profit:
                return position.take_profit, "take_profit"
        else:
            if position.stop_loss is not None and candle.high >= position.stop_loss:
                return position.stop_loss, "stop_loss"
            if position.take_profit is not None and candle.low <= position.take_profit:
                return position.take_profit, "take_profit"
        return None, ""

    def _close_position(
        self,
        position: _Position,
        raw_price: float,
        exit_time: datetime | float,
        reason: str,
        trades: List[Trade],
        balance: float,
    ) -> float:
        exit_price, exit_slippage = self._execution_price(raw_price, position.side, entry=False, quantity=position.quantity)
        direction = _direction(position.side)
        gross = (exit_price - position.entry_price) * direction * position.quantity
        exit_commission = self._commission(position.quantity)
        commission = position.entry_commission + exit_commission
        pnl = gross - commission
        notional = abs(position.entry_price * position.quantity)
        trade = Trade(
            trade_id=f"trade_{len(trades)}",
            symbol=position.symbol,
            timeframe=position.timeframe,
            side=position.side,
            quantity=position.quantity,
            entry_time=position.entry_time,
            entry_price=position.entry_price,
            exit_time=exit_time,
            exit_price=exit_price,
            pnl=float(pnl),
            return_pct=float(pnl / notional) if notional else 0.0,
            commission=float(commission),
            slippage=float(position.entry_slippage + exit_slippage),
            holding_seconds=_duration_seconds(position.entry_time, exit_time),
            entry_signal_id=position.entry_signal_id,
            exit_reason=reason,
            metadata=position.metadata,
        )
        trades.append(trade)
        return float(balance + pnl)

    def _execution_price(self, raw_price: float, side: str, entry: bool, quantity: float) -> tuple[float, float]:
        direction = _direction(side)
        is_buy = (entry and direction > 0) or (not entry and direction < 0)
        spread = float(self.spread_points) * self.point_size / 2.0
        slippage = float(self.slippage_points or 0.0) * self.point_size
        adjustment = spread + slippage
        price = raw_price + adjustment if is_buy else raw_price - adjustment
        return float(price), float(slippage * abs(quantity))

    def _filled_quantity(self, signal: BacktestSignal, candle: Candle) -> float:
        requested = float(signal.quantity if signal.quantity is not None else self.config.risk.default_lot_size)
        if not self.partial_fill_enabled:
            return requested
        ratio = float(self.partial_fill_ratio if self.partial_fill_ratio is not None else 1.0)
        ratio = min(max(ratio, 0.0), 1.0)
        filled = requested * ratio
        if candle.volume > 0.0:
            filled = min(filled, candle.volume)
        return float(filled)

    def _commission(self, quantity: float) -> float:
        return float(abs(quantity) * float(self.commission_per_lot or 0.0))

    def _matches(self, signal: BacktestSignal, candle: Candle) -> bool:
        timeframe = signal.timeframe or self.config.primary_timeframe
        return signal.symbol == candle.symbol and timeframe == candle.timeframe

    def _mark_to_market(
        self,
        positions: Sequence[_Position],
        latest_close: Dict[tuple[str, str], float],
    ) -> float:
        total = 0.0
        for position in positions:
            close = latest_close.get((position.symbol, position.timeframe))
            if close is None:
                continue
            direction = _direction(position.side)
            total += (close - position.entry_price) * direction * position.quantity - position.entry_commission
        return float(total)

    @staticmethod
    def _last_candles_by_market(candles: Sequence[Candle]) -> Dict[tuple[str, str], Candle]:
        result: Dict[tuple[str, str], Candle] = {}
        for candle in candles:
            result[(candle.symbol, candle.timeframe)] = candle
        return result


def create_backtest_engine(config: AIConfig | None = None, **kwargs: Any) -> BacktestEngine:
    """Factory for BacktestEngine."""
    return BacktestEngine(config=config or AIConfig(), **kwargs)


def _normalize_candle(raw: Candle | Dict[str, Any] | Any, default_timeframe: str) -> Candle:
    if isinstance(raw, Candle):
        return raw
    getter = raw.get if isinstance(raw, dict) else lambda key, default=None: getattr(raw, key, default)
    timestamp = _parse_time(getter("timestamp", getter("time", getter("date", 0.0))))
    return Candle(
        symbol=str(getter("symbol", "")),
        timeframe=str(getter("timeframe", default_timeframe)),
        timestamp=timestamp,
        open=float(getter("open")),
        high=float(getter("high")),
        low=float(getter("low")),
        close=float(getter("close")),
        volume=float(getter("volume", 0.0) or 0.0),
        metadata=dict(getter("metadata", {}) or {}),
    )


def _normalize_signal(raw: BacktestSignal | Dict[str, Any] | Any, config: AIConfig) -> BacktestSignal:
    if isinstance(raw, BacktestSignal):
        return raw
    getter = raw.get if isinstance(raw, dict) else lambda key, default=None: getattr(raw, key, default)
    side = str(getter("side", getter("direction", ""))).lower()
    timestamp = _parse_time(getter("timestamp", getter("time", 0.0)))
    expires_at = getter("expires_at", getter("expiry", None))
    return BacktestSignal(
        symbol=str(getter("symbol", config.symbols[0] if config.symbols else "")),
        timestamp=timestamp,
        side=side,
        order_type=str(getter("order_type", config.execution.default_order_type)).lower(),
        quantity=_optional_float(getter("quantity", getter("size", getter("volume", None)))),
        price=_optional_float(getter("price", getter("limit_price", getter("stop_price", None)))),
        stop_loss=_optional_float(getter("stop_loss", getter("sl", None))),
        take_profit=_optional_float(getter("take_profit", getter("tp", None))),
        timeframe=getter("timeframe", config.primary_timeframe),
        expires_at=_parse_time(expires_at) if expires_at is not None else None,
        metadata=dict(getter("metadata", {}) or {}),
    )


def _direction(side: str) -> int:
    side = side.lower()
    if side in LONG_SIDES:
        return 1
    if side in SHORT_SIDES:
        return -1
    return 0


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    return float(value)


def _parse_time(value: Any) -> datetime | float:
    if isinstance(value, datetime):
        return value
    if isinstance(value, (int, float, np.integer, np.floating)):
        return float(value)
    if isinstance(value, str):
        text = value.replace("Z", "+00:00")
        try:
            return datetime.fromisoformat(text)
        except ValueError:
            return float(value)
    return float(value)


def _time_key(value: datetime | float) -> float:
    if isinstance(value, datetime):
        return value.timestamp()
    return float(value)


def _add_latency(timestamp: datetime | float, latency_ms: float) -> datetime | float:
    if isinstance(timestamp, datetime):
        return timestamp + timedelta(milliseconds=latency_ms)
    return float(timestamp) + latency_ms / 1000.0


def _is_expired(signal: BacktestSignal, timestamp: datetime | float) -> bool:
    return signal.expires_at is not None and _time_key(timestamp) > _time_key(signal.expires_at)


def _duration_seconds(start: datetime | float, end: datetime | float) -> float:
    if isinstance(start, datetime) and isinstance(end, datetime):
        return float((end - start).total_seconds())
    return float(_time_key(end) - _time_key(start))


def _serialize_time(value: datetime | float) -> str | float:
    if isinstance(value, datetime):
        return value.isoformat()
    return float(value)
