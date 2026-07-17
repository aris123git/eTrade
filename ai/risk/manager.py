"""
ai/risk/manager.py - Portfolio and trade risk controls.

RESPONSIBILITY:
Provide position sizing, stop management, portfolio limit validation, circuit
breaker enforcement, and per-symbol / asset-class limits using AIConfig.RiskConfig.

VERSION: 1.1.0
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from typing import Any, Dict, Iterable, Mapping, Sequence

from ai.config.settings import AIConfig, RiskConfig
from ai.signals import TradeSignal
from ai.utils.types import SignalType

logger = logging.getLogger(__name__)

# Lightweight default asset-class map for concentration checks.
DEFAULT_ASSET_CLASSES: Dict[str, str] = {
    "EURUSD": "fx_major",
    "GBPUSD": "fx_major",
    "USDJPY": "fx_major",
    "AUDUSD": "fx_major",
    "USDCAD": "fx_major",
    "USDCHF": "fx_major",
    "XAUUSD": "metal",
    "XAGUSD": "metal",
    "US30": "index",
    "DJ30": "index",
    "NAS100": "index",
    "SPX500": "index",
    "BTCUSD": "crypto",
    "ETHUSD": "crypto",
}


@dataclass(frozen=True)
class RiskDecision:
    """Structured result from a risk validation check."""

    approved: bool
    reason: str
    size: float = 0.0
    risk_amount: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize for hooks and logging."""

        return {
            "approved": self.approved,
            "valid": self.approved,
            "reason": self.reason,
            "size": self.size,
            "risk_amount": self.risk_amount,
            "metadata": self.metadata,
        }


@dataclass
class RiskManager:
    """Production risk manager driven by RiskConfig."""

    config: AIConfig = field(default_factory=AIConfig)
    equity: float = 100_000.0
    peak_equity: float | None = None
    day_start_equity: float | None = None
    current_day: date | None = None
    circuit_breaker_tripped: bool = False
    asset_class_map: Dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.peak_equity = self.equity if self.peak_equity is None else self.peak_equity
        self.day_start_equity = self.equity if self.day_start_equity is None else self.day_start_equity
        self.current_day = datetime.now(timezone.utc).date() if self.current_day is None else self.current_day
        if not self.asset_class_map:
            self.asset_class_map = dict(DEFAULT_ASSET_CLASSES)
        logger.info(
            "RiskManager ready equity=%.2f circuit_breaker=%.2f%% max_dd=%.2f%%",
            self.equity,
            self.risk_config.circuit_breaker_loss * 100.0,
            self.risk_config.max_drawdown * 100.0,
        )

    @property
    def risk_config(self) -> RiskConfig:
        """Return the active nested risk config."""

        return self.config.risk

    def update_equity(self, equity: float, *, timestamp: datetime | None = None) -> None:
        """Update account equity and rolling loss baselines."""

        ts = _utc(timestamp or datetime.now(timezone.utc))
        value = float(equity)
        self.equity = value
        self.peak_equity = max(float(self.peak_equity or value), value)
        if self.current_day != ts.date():
            self.current_day = ts.date()
            self.day_start_equity = value
        if self._drawdown(value) >= self.risk_config.circuit_breaker_loss:
            if not self.circuit_breaker_tripped:
                logger.error(
                    "CIRCUIT BREAKER tripped drawdown=%.2f%% threshold=%.2f%%",
                    self._drawdown(value) * 100.0,
                    self.risk_config.circuit_breaker_loss * 100.0,
                )
            self.circuit_breaker_tripped = True

    def reset_circuit_breaker(self) -> None:
        """Manual reset after operator review."""

        self.circuit_breaker_tripped = False
        logger.warning("circuit breaker manually reset")

    def asset_class(self, symbol: str) -> str:
        key = str(symbol).upper().split(".")[0]
        return self.asset_class_map.get(key, self.asset_class_map.get(symbol.upper(), "other"))

    def size_position(
        self,
        *,
        symbol: str,
        entry: float,
        stop_loss: float | None = None,
        equity: float | None = None,
        atr: float | None = None,
        confidence: float | None = None,
        win_rate: float | None = None,
        reward_risk: float | None = None,
        pip_value: float = 1.0,
        fixed_lot: float | None = None,
    ) -> float:
        """Size a position using fixed_risk, kelly, atr, volatility, or fixed_lot."""

        account_equity = float(self.equity if equity is None else equity)
        mode = str(self.risk_config.position_sizing).lower()
        if mode == "fixed_lot":
            size = self._clip_lot(fixed_lot or self.risk_config.default_lot_size)
            return self._apply_drawdown_scale(size, account_equity)

        stop_distance = self._stop_distance(entry=entry, stop_loss=stop_loss, atr=atr)
        if stop_distance <= 0:
            return 0.0
        risk_fraction = self._risk_fraction(
            mode, confidence=confidence, win_rate=win_rate, reward_risk=reward_risk, atr=atr, entry=entry
        )
        risk_amount = account_equity * min(risk_fraction, self.risk_config.max_risk_per_trade)
        units = risk_amount / max(stop_distance * max(float(pip_value), 1e-12), 1e-12)
        return self._apply_drawdown_scale(self._clip_lot(units), account_equity)

    def validate_signal(
        self,
        signal: TradeSignal,
        *,
        open_positions: Sequence[Any] | None = None,
        equity: float | None = None,
        correlations: Mapping[Any, float] | None = None,
        atr: float | None = None,
        pip_value: float = 1.0,
        margin_available: float | None = None,
        required_margin: float | None = None,
    ) -> RiskDecision:
        """Validate a TradeSignal against trade and portfolio risk rules."""

        return self.pre_trade_validate(
            signal,
            open_positions=open_positions,
            equity=equity,
            correlations=correlations,
            atr=atr,
            pip_value=pip_value,
            margin_available=margin_available,
            required_margin=required_margin,
        )

    def pre_trade_validate(
        self,
        signal: TradeSignal,
        *,
        open_positions: Sequence[Any] | None = None,
        equity: float | None = None,
        correlations: Mapping[Any, float] | None = None,
        atr: float | None = None,
        pip_value: float = 1.0,
        margin_available: float | None = None,
        required_margin: float | None = None,
    ) -> RiskDecision:
        """
        Pre-trade validation: position size, margin, correlation, circuit breaker,
        and per-symbol / asset-class limits.
        """

        if signal.side == SignalType.HOLD:
            return RiskDecision(True, "hold_signal", metadata={"side": signal.side.value})
        if self.circuit_breaker_tripped:
            return RiskDecision(False, "circuit_breaker_active", metadata={"drawdown": self._drawdown(float(equity or self.equity))})
        if signal.confidence < self.risk_config.min_confidence:
            return RiskDecision(False, "confidence_below_minimum", metadata={"confidence": signal.confidence})
        if signal.entry is None:
            return RiskDecision(False, "missing_entry_price")
        if signal.side in {SignalType.BUY, SignalType.SELL} and signal.sl is None:
            return RiskDecision(False, "missing_stop_loss")

        rr = self._reward_risk(signal)
        if rr is not None and rr < self.risk_config.min_expected_rr:
            return RiskDecision(False, "reward_risk_below_minimum", metadata={"reward_risk": rr})

        positions = list(open_positions or ())
        symbol_count = sum(1 for p in positions if str(_get(p, "symbol", "")).upper() == signal.symbol.upper())
        if symbol_count >= int(self.risk_config.max_positions_per_symbol):
            return RiskDecision(
                False,
                "max_positions_per_symbol_exceeded",
                metadata={"symbol": signal.symbol, "count": symbol_count},
            )
        asset_class = self.asset_class(signal.symbol)
        class_count = sum(1 for p in positions if self.asset_class(str(_get(p, "symbol", ""))) == asset_class)
        if class_count >= int(self.risk_config.max_positions_per_asset_class):
            return RiskDecision(
                False,
                "max_positions_per_asset_class_exceeded",
                metadata={"asset_class": asset_class, "count": class_count},
            )

        portfolio = self.check_portfolio_limits(
            open_positions=positions,
            equity=equity,
            new_signal=signal,
            correlations=correlations,
        )
        if not portfolio.approved:
            return portfolio

        size = signal.size_hint or self.size_position(
            symbol=signal.symbol,
            entry=signal.entry,
            stop_loss=signal.sl,
            equity=equity,
            atr=atr,
            confidence=signal.confidence,
            reward_risk=rr,
            pip_value=pip_value,
        )
        risk_amount = abs(signal.entry - float(signal.sl or signal.entry)) * size * max(float(pip_value), 1e-12)
        if size <= 0:
            return RiskDecision(False, "position_size_zero")
        max_trade_risk = float(equity or self.equity) * self.risk_config.max_risk_per_trade
        if risk_amount > max_trade_risk:
            return RiskDecision(False, "risk_per_trade_exceeded", size=size, risk_amount=risk_amount)

        if margin_available is not None and required_margin is not None:
            if float(required_margin) > float(margin_available):
                return RiskDecision(
                    False,
                    "insufficient_margin",
                    size=size,
                    risk_amount=risk_amount,
                    metadata={"margin_available": margin_available, "required_margin": required_margin},
                )

        logger.info(
            "pre-trade APPROVED %s size=%.4f risk=%.2f class=%s",
            signal.symbol,
            size,
            risk_amount,
            asset_class,
        )
        return RiskDecision(
            True,
            "approved",
            size=size,
            risk_amount=risk_amount,
            metadata={"reward_risk": rr, "asset_class": asset_class},
        )

    def update_stops(
        self,
        positions: Iterable[Any],
        market_data: Mapping[str, Mapping[str, float]] | Mapping[str, float],
    ) -> list[Dict[str, Any]]:
        """Compute ATR trailing and break-even stop updates for open positions."""

        updates: list[Dict[str, Any]] = []
        for position in positions:
            symbol = str(_get(position, "symbol", ""))
            data = _market_for_symbol(market_data, symbol)
            current = _float(data.get("price", data.get("close", _get(position, "current_price", 0.0))), 0.0)
            atr = _float(data.get("atr", _get(position, "atr", 0.0)), 0.0)
            if current <= 0 or atr <= 0:
                continue
            side = str(_get(position, "side", "")).upper()
            entry = _float(_get(position, "entry_price", _get(position, "entry", 0.0)), 0.0)
            current_sl = _float(_get(position, "sl", _get(position, "stop_loss", 0.0)), 0.0)
            proposed = self._updated_stop(side=side, entry=entry, current=current, current_sl=current_sl, atr=atr)
            if proposed is not None and not _same_price(proposed, current_sl):
                _set_if_possible(position, "sl", proposed)
                _set_if_possible(position, "stop_loss", proposed)
                updates.append({"symbol": symbol, "old_sl": current_sl, "new_sl": proposed, "side": side})
        return updates

    def check_portfolio_limits(
        self,
        *,
        open_positions: Sequence[Any],
        equity: float | None = None,
        new_signal: TradeSignal | None = None,
        correlations: Mapping[Any, float] | None = None,
    ) -> RiskDecision:
        """Validate drawdown, daily loss, exposure, correlation, and open trade limits."""

        account_equity = float(self.equity if equity is None else equity)
        dd = self._drawdown(account_equity)
        if dd >= self.risk_config.circuit_breaker_loss or self.circuit_breaker_tripped:
            self.circuit_breaker_tripped = True
            return RiskDecision(False, "circuit_breaker_active", metadata={"drawdown": dd})
        if dd >= self.risk_config.max_drawdown:
            return RiskDecision(False, "max_drawdown_exceeded", metadata={"drawdown": dd})
        if self._daily_loss(account_equity) >= self.risk_config.daily_loss_limit:
            return RiskDecision(False, "daily_loss_limit_exceeded", metadata={"daily_loss": self._daily_loss(account_equity)})

        positions = list(open_positions)
        if new_signal is not None and new_signal.side in {SignalType.BUY, SignalType.SELL}:
            if len(positions) >= self.risk_config.max_open_trades:
                return RiskDecision(False, "max_open_trades_exceeded", metadata={"open_trades": len(positions)})
            correlation = self._max_correlation(new_signal.symbol, positions, correlations or {})
            if correlation > self.risk_config.max_correlation:
                return RiskDecision(False, "correlation_limit_exceeded", metadata={"correlation": correlation})

        portfolio_risk = sum(self._position_risk(position) for position in positions)
        max_portfolio_risk = account_equity * self.risk_config.max_portfolio_risk
        if portfolio_risk > max_portfolio_risk:
            return RiskDecision(
                False,
                "max_portfolio_risk_exceeded",
                risk_amount=portfolio_risk,
                metadata={"max_portfolio_risk": max_portfolio_risk},
            )
        return RiskDecision(True, "portfolio_limits_ok", risk_amount=portfolio_risk)

    def _risk_fraction(
        self,
        mode: str,
        *,
        confidence: float | None,
        win_rate: float | None,
        reward_risk: float | None,
        atr: float | None = None,
        entry: float | None = None,
    ) -> float:
        if mode == "kelly":
            probability = _bounded(win_rate if win_rate is not None else confidence, 0.0, 1.0)
            payoff = max(float(reward_risk or self.risk_config.min_expected_rr), 1e-12)
            kelly = probability - (1.0 - probability) / payoff
            return max(kelly, 0.0) * self.risk_config.kelly_fraction
        if mode in {"atr", "volatility"}:
            base = self.risk_config.risk_per_trade
            if atr is not None and entry is not None and float(entry) > 0:
                vol = float(atr) / float(entry)
                # Higher realized vol → smaller risk fraction (volatility-adjusted).
                scale = 1.0 / max(1.0, vol / 0.01)
                return base * min(scale, 1.0)
            return base
        return self.risk_config.risk_per_trade

    def _apply_drawdown_scale(self, size: float, equity: float) -> float:
        if not self.risk_config.drawdown_size_scale or size <= 0:
            return size
        dd = self._drawdown(equity)
        max_dd = max(float(self.risk_config.max_drawdown), 1e-12)
        # Linearly reduce size toward zero as drawdown approaches max.
        scale = max(0.25, 1.0 - (dd / max_dd) * 0.75)
        return self._clip_lot(size * scale)

    def _stop_distance(self, *, entry: float, stop_loss: float | None, atr: float | None) -> float:
        if stop_loss is not None:
            return abs(float(entry) - float(stop_loss))
        if atr is not None:
            return abs(float(atr)) * self.risk_config.atr_stop_mult
        return 0.0

    def _updated_stop(
        self,
        *,
        side: str,
        entry: float,
        current: float,
        current_sl: float,
        atr: float,
    ) -> float | None:
        trail = atr * self.risk_config.trailing_stop_atr_mult
        break_even_trigger = atr * self.risk_config.break_even_atr_mult
        if side in {"LONG", "BUY"}:
            breakeven = entry if current - entry >= break_even_trigger else current_sl
            trailing = current - trail
            candidate = max(current_sl, breakeven, trailing)
            return candidate if candidate > current_sl else None
        if side in {"SHORT", "SELL"}:
            breakeven = entry if entry - current >= break_even_trigger else current_sl
            trailing = current + trail
            baseline = current_sl if current_sl > 0 else float("inf")
            candidate = min(baseline, breakeven if breakeven > 0 else baseline, trailing)
            return candidate if candidate < baseline else None
        return None

    def _reward_risk(self, signal: TradeSignal) -> float | None:
        if signal.entry is None or signal.sl is None or signal.tp is None:
            return None
        risk = abs(signal.entry - signal.sl)
        reward = abs(signal.tp - signal.entry)
        if risk <= 0:
            return None
        return reward / risk

    def _drawdown(self, equity: float) -> float:
        peak = max(float(self.peak_equity or equity), 1e-12)
        return max((peak - equity) / peak, 0.0)

    def _daily_loss(self, equity: float) -> float:
        start = max(float(self.day_start_equity or equity), 1e-12)
        return max((start - equity) / start, 0.0)

    def _position_risk(self, position: Any) -> float:
        entry = _float(_get(position, "entry_price", _get(position, "entry", 0.0)), 0.0)
        sl = _float(_get(position, "sl", _get(position, "stop_loss", entry)), entry)
        size = abs(_float(_get(position, "size", _get(position, "volume", 0.0)), 0.0))
        pip_value = _float(_get(position, "pip_value", 1.0), 1.0)
        return abs(entry - sl) * size * max(pip_value, 1e-12)

    def _max_correlation(self, symbol: str, positions: Sequence[Any], correlations: Mapping[Any, float]) -> float:
        values: list[float] = []
        for position in positions:
            other = str(_get(position, "symbol", ""))
            keys = ((symbol, other), (other, symbol), f"{symbol}:{other}", f"{other}:{symbol}", other)
            for key in keys:
                if key in correlations:
                    values.append(abs(float(correlations[key])))
                    break
        return max(values) if values else 0.0

    def _clip_lot(self, size: float) -> float:
        value = max(float(size), 0.0)
        if value <= 0:
            return 0.0
        value = max(value, self.risk_config.default_lot_size)
        return min(value, self.risk_config.max_lot_size)


def create_risk_manager(config: AIConfig | None = None, *, equity: float = 100_000.0) -> RiskManager:
    """Factory for RiskManager."""

    return RiskManager(config=config or AIConfig(), equity=equity)


def _get(value: Any, key: str, default: Any = None) -> Any:
    if isinstance(value, Mapping):
        return value.get(key, default)
    return getattr(value, key, default)


def _set_if_possible(value: Any, key: str, item: Any) -> None:
    if isinstance(value, dict):
        value[key] = item
    elif hasattr(value, key):
        setattr(value, key, item)


def _market_for_symbol(market_data: Mapping[str, Mapping[str, float]] | Mapping[str, float], symbol: str) -> Mapping[str, float]:
    data = market_data.get(symbol) if isinstance(market_data.get(symbol), Mapping) else None  # type: ignore[arg-type]
    if data is not None:
        return data
    return market_data  # type: ignore[return-value]


def _float(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _bounded(value: float | None, low: float, high: float) -> float:
    number = low if value is None else float(value)
    return min(max(number, low), high)


def _same_price(a: float, b: float) -> bool:
    return abs(float(a) - float(b)) <= 1e-12


def _utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)
