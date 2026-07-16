"""
ai/signals/engine.py - Trading signal generation.

RESPONSIBILITY:
Convert model predictions into filtered, risk-aware TradeSignal objects.

VERSION: 1.0.0
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, time, timezone
from typing import Any, Callable, Dict, Mapping, Sequence

from ai.config.settings import AIConfig
from ai.prediction import prediction_to_signal
from ai.utils.types import PredictionResult, SignalType


RiskHook = Callable[["TradeSignal"], bool | tuple[bool, str] | Mapping[str, Any]]


@dataclass
class TradeSignal:
    """Actionable trading signal produced by SignalEngine."""

    symbol: str
    side: SignalType
    strength: float
    confidence: float
    entry: float | None = None
    sl: float | None = None
    tp: float | None = None
    size_hint: float | None = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def is_actionable(self) -> bool:
        """Return True when the signal can open or adjust risk."""

        return self.side in {SignalType.BUY, SignalType.SELL, SignalType.CLOSE, SignalType.REDUCE}

    def with_side(self, side: SignalType, reason: str) -> "TradeSignal":
        """Return a copy with a different side and appended filter reason."""

        metadata = dict(self.metadata)
        reasons = list(metadata.get("filter_reasons", []))
        reasons.append(reason)
        metadata["filter_reasons"] = reasons
        return TradeSignal(
            symbol=self.symbol,
            side=side,
            strength=self.strength,
            confidence=self.confidence,
            entry=self.entry,
            sl=self.sl,
            tp=self.tp,
            size_hint=self.size_hint,
            metadata=metadata,
        )


@dataclass
class SignalFilterConfig:
    """Runtime controls for SignalEngine filters."""

    min_confidence: float | None = None
    session_filter: bool = True
    trend_filter: bool = True
    cooldown_seconds: float = 0.0
    trading_days: Sequence[int] = (0, 1, 2, 3, 4)
    trading_hours_utc: Sequence[tuple[int, int]] = ((0, 24),)


@dataclass
class SignalEngine:
    """Prediction -> probability -> confidence -> filters -> risk -> TradeSignal."""

    config: AIConfig = field(default_factory=AIConfig)
    filters: SignalFilterConfig = field(default_factory=SignalFilterConfig)
    risk_hook: RiskHook | None = None
    _last_signal_at: Dict[str, datetime] = field(default_factory=dict, init=False)

    def generate(
        self,
        prediction: PredictionResult,
        *,
        market_context: Mapping[str, Any] | None = None,
        risk_hook: RiskHook | None = None,
    ) -> TradeSignal:
        """Build a filtered TradeSignal from a model PredictionResult."""

        context = dict(market_context or {})
        side = prediction_to_signal(prediction)
        signal = TradeSignal(
            symbol=prediction.symbol,
            side=side,
            strength=self._strength(prediction),
            confidence=float(prediction.confidence),
            entry=_float_or_none(context.get("entry", context.get("price"))),
            sl=_float_or_none(context.get("sl", context.get("stop_loss"))),
            tp=_float_or_none(context.get("tp", context.get("take_profit"))),
            size_hint=_float_or_none(context.get("size_hint")),
            metadata={
                "timeframe": prediction.timeframe,
                "prediction": prediction.prediction,
                "probabilities": dict(prediction.probabilities or {}),
                "prediction_metadata": dict(prediction.metadata),
                "market_context": context,
                "generated_at": datetime.now(timezone.utc).isoformat(),
            },
        )
        signal = self._apply_confidence_filter(signal)
        signal = self._apply_session_filter(signal, prediction.timestamp)
        signal = self._apply_trend_filter(signal, prediction, context)
        signal = self._apply_cooldown_filter(signal, prediction.timestamp)
        signal = self._apply_risk_hook(signal, risk_hook or self.risk_hook)
        self._record_signal(signal, prediction.timestamp)
        return signal

    def generate_many(
        self,
        predictions: Sequence[PredictionResult],
        *,
        market_context: Mapping[str, Any] | None = None,
        risk_hook: RiskHook | None = None,
    ) -> list[TradeSignal]:
        """Generate one TradeSignal for each prediction."""

        return [
            self.generate(prediction, market_context=market_context, risk_hook=risk_hook)
            for prediction in predictions
        ]

    def reset_cooldown(self, symbol: str | None = None) -> None:
        """Clear cooldown state for one symbol or every symbol."""

        if symbol is None:
            self._last_signal_at.clear()
        else:
            self._last_signal_at.pop(symbol, None)

    def _apply_confidence_filter(self, signal: TradeSignal) -> TradeSignal:
        min_confidence = self.filters.min_confidence
        threshold = self.config.risk.min_confidence if min_confidence is None else min_confidence
        if signal.side != SignalType.HOLD and signal.confidence < threshold:
            return signal.with_side(SignalType.HOLD, f"confidence_below_{threshold:.4f}")
        return signal

    def _apply_session_filter(self, signal: TradeSignal, timestamp: datetime) -> TradeSignal:
        if not self.filters.session_filter or signal.side == SignalType.HOLD:
            return signal
        ts = _aware_utc(timestamp)
        if ts.weekday() not in set(self.filters.trading_days):
            return signal.with_side(SignalType.HOLD, "outside_trading_day")
        current = ts.time()
        if any(_time_in_hour_range(current, start, end) for start, end in self.filters.trading_hours_utc):
            return signal
        return signal.with_side(SignalType.HOLD, "outside_trading_session")

    def _apply_trend_filter(
        self,
        signal: TradeSignal,
        prediction: PredictionResult,
        context: Mapping[str, Any],
    ) -> TradeSignal:
        if not self.filters.trend_filter or signal.side == SignalType.HOLD:
            return signal
        trend = str(
            context.get("trend")
            or prediction.metadata.get("trend")
            or prediction.metadata.get("market_regime")
            or ""
        ).lower()
        if trend in {"up", "bull", "bullish", "long"} and signal.side == SignalType.SELL:
            return signal.with_side(SignalType.HOLD, "trend_filter_bullish")
        if trend in {"down", "bear", "bearish", "short"} and signal.side == SignalType.BUY:
            return signal.with_side(SignalType.HOLD, "trend_filter_bearish")
        return signal

    def _apply_cooldown_filter(self, signal: TradeSignal, timestamp: datetime) -> TradeSignal:
        cooldown = float(self.filters.cooldown_seconds)
        if cooldown <= 0 or signal.side == SignalType.HOLD:
            return signal
        last = self._last_signal_at.get(signal.symbol)
        if last is None:
            return signal
        elapsed = (_aware_utc(timestamp) - _aware_utc(last)).total_seconds()
        if elapsed < cooldown:
            return signal.with_side(SignalType.HOLD, f"cooldown_{cooldown:.0f}s")
        return signal

    def _apply_risk_hook(self, signal: TradeSignal, hook: RiskHook | None) -> TradeSignal:
        if hook is None or signal.side == SignalType.HOLD:
            return signal
        result = hook(signal)
        approved = True
        reason = "risk_rejected"
        extra: Dict[str, Any] = {}
        if isinstance(result, tuple):
            approved = bool(result[0])
            reason = str(result[1]) if len(result) > 1 else reason
        elif isinstance(result, Mapping):
            approved = bool(result.get("approved", result.get("valid", True)))
            reason = str(result.get("reason", reason))
            extra = dict(result)
        else:
            approved = bool(result)
        if approved:
            if extra:
                signal.metadata["risk"] = extra
            return signal
        blocked = signal.with_side(SignalType.HOLD, reason)
        if extra:
            blocked.metadata["risk"] = extra
        return blocked

    def _record_signal(self, signal: TradeSignal, timestamp: datetime) -> None:
        if signal.side != SignalType.HOLD:
            self._last_signal_at[signal.symbol] = _aware_utc(timestamp)

    def _strength(self, prediction: PredictionResult) -> float:
        probabilities = prediction.probabilities or {}
        buy = _prob(probabilities, SignalType.BUY.value)
        sell = _prob(probabilities, SignalType.SELL.value)
        if buy or sell:
            return float(abs(buy - sell))
        expected = prediction.expected_return
        if expected is not None:
            return float(min(abs(expected), 1.0))
        return float(min(max(prediction.confidence, 0.0), 1.0))


def create_signal_engine(
    config: AIConfig | None = None,
    *,
    filters: SignalFilterConfig | None = None,
    risk_hook: RiskHook | None = None,
) -> SignalEngine:
    """Factory for SignalEngine."""

    return SignalEngine(config=config or AIConfig(), filters=filters or SignalFilterConfig(), risk_hook=risk_hook)


def _aware_utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def _time_in_hour_range(current: time, start_hour: int, end_hour: int) -> bool:
    start = max(0, min(int(start_hour), 24))
    end = max(0, min(int(end_hour), 24))
    current_hour = current.hour + current.minute / 60.0 + current.second / 3600.0
    if start <= end:
        return float(start) <= current_hour < float(end)
    return current_hour >= float(start) or current_hour < float(end)


def _prob(probabilities: Mapping[str, float], label: str) -> float:
    target = label.upper()
    for key, value in probabilities.items():
        if str(key).upper() == target:
            return float(value)
    return 0.0


def _float_or_none(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
