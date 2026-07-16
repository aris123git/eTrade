"""
ai/strategy/base.py - Strategy contracts and model-backed strategies.

RESPONSIBILITY:
Combine prediction, signal, and risk services into executable trade intents.

VERSION: 1.0.0
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, Mapping, Sequence

from ai.config.settings import AIConfig
from ai.prediction import PredictionService, create_prediction_service
from ai.risk import RiskManager, create_risk_manager
from ai.signals import SignalEngine, TradeSignal, create_signal_engine
from ai.utils.types import CandleDict, OrderType, SignalType


@dataclass
class TradeIntent:
    """Execution-ready instruction produced by a Strategy."""

    symbol: str
    side: SignalType
    order_type: OrderType = OrderType.MARKET
    size: float = 0.0
    entry: float | None = None
    sl: float | None = None
    tp: float | None = None
    confidence: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def actionable(self) -> bool:
        """Return True when the intent can be sent to execution."""

        return self.side in {SignalType.BUY, SignalType.SELL, SignalType.CLOSE, SignalType.REDUCE} and self.size >= 0.0


@dataclass
class Strategy:
    """Base strategy with shared config and a stable evaluation surface."""

    config: AIConfig = field(default_factory=AIConfig)
    name: str = "base"

    def evaluate(
        self,
        candles: Sequence[CandleDict],
        *,
        market_context: Mapping[str, Any] | None = None,
        open_positions: Sequence[Any] | None = None,
        correlations: Mapping[Any, float] | None = None,
    ) -> list[TradeIntent]:
        """Return intents for the provided market state."""

        return []


@dataclass
class SignalStrategy(Strategy):
    """Strategy that turns model predictions into risk-approved intents."""

    prediction_service: PredictionService | None = None
    signal_engine: SignalEngine | None = None
    risk_manager: RiskManager | None = None
    name: str = "signal_strategy"

    def __post_init__(self) -> None:
        self.prediction_service = self.prediction_service or create_prediction_service(self.config)
        self.signal_engine = self.signal_engine or create_signal_engine(self.config)
        self.risk_manager = self.risk_manager or create_risk_manager(self.config)

    def evaluate(
        self,
        candles: Sequence[CandleDict],
        *,
        market_context: Mapping[str, Any] | None = None,
        open_positions: Sequence[Any] | None = None,
        correlations: Mapping[Any, float] | None = None,
    ) -> list[TradeIntent]:
        """Produce a single actionable intent from the latest candles."""

        context = dict(market_context or {})
        prediction = self.prediction_service.predict_proba(candles)  # type: ignore[union-attr]
        risk_manager = self.risk_manager  # type: ignore[assignment]

        def risk_hook(signal: TradeSignal) -> Mapping[str, Any]:
            return risk_manager.validate_signal(
                signal,
                open_positions=open_positions or (),
                equity=context.get("equity"),
                correlations=correlations,
                atr=context.get("atr"),
                pip_value=float(context.get("pip_value", 1.0)),
            ).to_dict()

        signal = self.signal_engine.generate(  # type: ignore[union-attr]
            prediction,
            market_context=context,
            risk_hook=risk_hook,
        )
        if signal.side == SignalType.HOLD:
            return []
        return [self._intent_from_signal(signal)]

    def _intent_from_signal(self, signal: TradeSignal) -> TradeIntent:
        risk_payload = signal.metadata.get("risk", {})
        size = float(risk_payload.get("size", signal.size_hint or 0.0))
        return TradeIntent(
            symbol=signal.symbol,
            side=signal.side,
            order_type=OrderType.MARKET,
            size=size,
            entry=signal.entry,
            sl=signal.sl,
            tp=signal.tp,
            confidence=signal.confidence,
            metadata={
                "strategy": self.name,
                "signal": signal,
                "risk": risk_payload,
                "created_at": datetime.now(timezone.utc).isoformat(),
            },
        )


@dataclass
class ConfidenceThresholdStrategy(Strategy):
    """Simple rule strategy that emits intents from provided TradeSignal values."""

    min_confidence: float | None = None
    name: str = "confidence_threshold"

    def from_signal(self, signal: TradeSignal) -> list[TradeIntent]:
        """Convert a signal into an intent when confidence is high enough."""

        threshold = self.config.risk.min_confidence if self.min_confidence is None else self.min_confidence
        if signal.side == SignalType.HOLD or signal.confidence < threshold:
            return []
        return [
            TradeIntent(
                symbol=signal.symbol,
                side=signal.side,
                size=float(signal.size_hint or 0.0),
                entry=signal.entry,
                sl=signal.sl,
                tp=signal.tp,
                confidence=signal.confidence,
                metadata={"strategy": self.name, "source_signal": signal},
            )
        ]


def create_signal_strategy(
    config: AIConfig | None = None,
    *,
    prediction_service: PredictionService | None = None,
    signal_engine: SignalEngine | None = None,
    risk_manager: RiskManager | None = None,
) -> SignalStrategy:
    """Factory for SignalStrategy."""

    active_config = config or AIConfig()
    return SignalStrategy(
        config=active_config,
        prediction_service=prediction_service,
        signal_engine=signal_engine,
        risk_manager=risk_manager,
    )


def create_confidence_threshold_strategy(
    config: AIConfig | None = None,
    *,
    min_confidence: float | None = None,
) -> ConfidenceThresholdStrategy:
    """Factory for ConfidenceThresholdStrategy."""

    return ConfidenceThresholdStrategy(config=config or AIConfig(), min_confidence=min_confidence)
