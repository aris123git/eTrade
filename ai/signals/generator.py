"""
ai/signals/generator.py - Signal generation with multi-timeframe confirmation.

Converts model predictions into buy/sell/hold signals with 0-100% confidence
and optional higher-timeframe confirmation.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, Mapping, Optional, Sequence

from ai.config.settings import AIConfig
from ai.signals.engine import SignalEngine, SignalFilterConfig, TradeSignal, create_signal_engine
from ai.utils.types import PredictionResult, SignalType

logger = logging.getLogger(__name__)


@dataclass
class SignalGenerator:
    """
    Production signal generator for paper and live trading.

    Confidence is exposed as 0-100%. Multi-timeframe confirmation can veto
    or strengthen a primary-timeframe signal.
    """

    config: AIConfig = field(default_factory=AIConfig)
    engine: SignalEngine | None = None
    confirmation_timeframes: Sequence[str] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        self.engine = self.engine or create_signal_engine(self.config)
        if not self.confirmation_timeframes:
            self.confirmation_timeframes = tuple(
                tf
                for tf in self.config.features.multi_timeframes
                if str(tf).upper() != str(self.config.primary_timeframe).upper()
            )
        logger.info(
            "SignalGenerator ready primary=%s confirm=%s",
            self.config.primary_timeframe,
            list(self.confirmation_timeframes),
        )

    def generate(
        self,
        prediction: PredictionResult,
        *,
        market_context: Mapping[str, Any] | None = None,
        higher_tf_predictions: Sequence[PredictionResult] | None = None,
    ) -> TradeSignal:
        """Convert a prediction into a filtered TradeSignal."""
        assert self.engine is not None
        context = dict(market_context or {})
        signal = self.engine.generate(prediction, market_context=context)
        signal = self._attach_confidence_pct(signal)
        if higher_tf_predictions:
            signal = self._apply_mtf_confirmation(signal, higher_tf_predictions)
        logger.info(
            "signal %s side=%s confidence=%.1f%% mtf=%s",
            signal.symbol,
            signal.side.value,
            float(signal.metadata.get("confidence_pct", signal.confidence * 100.0)),
            signal.metadata.get("mtf_confirmation"),
        )
        return signal

    def generate_from_value(
        self,
        *,
        symbol: str,
        timeframe: str,
        prediction: float,
        confidence: float,
        price: float,
        atr: float | None = None,
        higher_tf_bias: Sequence[float] | None = None,
    ) -> TradeSignal:
        """Convenience builder when only scalar prediction/confidence are available."""
        from datetime import datetime, timezone

        result = PredictionResult(
            symbol=symbol,
            timeframe=timeframe,
            timestamp=datetime.now(timezone.utc),
            prediction=prediction,
            confidence=float(confidence),
            probabilities=None,
            expected_return=float(prediction) if self.config.model.task == "regression" else None,
            model_version=None,
            metadata={},
        )
        sl = tp = None
        if atr is not None and atr > 0:
            if prediction > 0:
                sl = price - atr * self.config.risk.atr_stop_mult
                tp = price + atr * self.config.risk.atr_tp_mult
            elif prediction < 0:
                sl = price + atr * self.config.risk.atr_stop_mult
                tp = price - atr * self.config.risk.atr_tp_mult
        context = {"price": price, "entry": price, "atr": atr, "sl": sl, "tp": tp}
        higher = None
        if higher_tf_bias:
            higher = [
                PredictionResult(
                    symbol=symbol,
                    timeframe=tf,
                    timestamp=result.timestamp,
                    prediction=bias,
                    confidence=max(0.5, float(confidence)),
                    probabilities=None,
                    expected_return=None,
                    model_version=None,
                    metadata={},
                )
                for tf, bias in zip(self.confirmation_timeframes, higher_tf_bias)
            ]
        return self.generate(result, market_context=context, higher_tf_predictions=higher)

    def _attach_confidence_pct(self, signal: TradeSignal) -> TradeSignal:
        metadata = dict(signal.metadata)
        pct = max(0.0, min(100.0, float(signal.confidence) * 100.0))
        metadata["confidence_pct"] = pct
        return TradeSignal(
            symbol=signal.symbol,
            side=signal.side,
            strength=signal.strength,
            confidence=signal.confidence,
            entry=signal.entry,
            sl=signal.sl,
            tp=signal.tp,
            size_hint=signal.size_hint,
            metadata=metadata,
        )

    def _apply_mtf_confirmation(
        self,
        signal: TradeSignal,
        higher: Sequence[PredictionResult],
    ) -> TradeSignal:
        if signal.side not in {SignalType.BUY, SignalType.SELL}:
            return signal
        votes = []
        for pred in higher:
            value = float(pred.prediction) if _is_number(pred.prediction) else 0.0
            if value > 0:
                votes.append(SignalType.BUY)
            elif value < 0:
                votes.append(SignalType.SELL)
            else:
                votes.append(SignalType.HOLD)
        agreeing = sum(1 for v in votes if v == signal.side)
        opposing = sum(1 for v in votes if v in {SignalType.BUY, SignalType.SELL} and v != signal.side)
        metadata = dict(signal.metadata)
        metadata["mtf_confirmation"] = {
            "votes": [v.value for v in votes],
            "agreeing": agreeing,
            "opposing": opposing,
            "timeframes": [p.timeframe for p in higher],
        }
        updated = TradeSignal(
            symbol=signal.symbol,
            side=signal.side,
            strength=signal.strength,
            confidence=signal.confidence,
            entry=signal.entry,
            sl=signal.sl,
            tp=signal.tp,
            size_hint=signal.size_hint,
            metadata=metadata,
        )
        # Veto when majority of higher TFs oppose
        if opposing > agreeing and opposing > 0:
            logger.info("mtf veto %s primary=%s opposing=%s", signal.symbol, signal.side.value, opposing)
            return updated.with_side(SignalType.HOLD, "mtf_confirmation_veto")
        # Strengthen confidence slightly when confirmed
        if agreeing > 0 and opposing == 0:
            boosted = min(1.0, float(signal.confidence) + 0.05 * agreeing)
            metadata["confidence_pct"] = boosted * 100.0
            return TradeSignal(
                symbol=signal.symbol,
                side=signal.side,
                strength=min(1.0, signal.strength + 0.05 * agreeing),
                confidence=boosted,
                entry=signal.entry,
                sl=signal.sl,
                tp=signal.tp,
                size_hint=signal.size_hint,
                metadata=metadata,
            )
        return updated


def create_signal_generator(config: AIConfig | None = None) -> SignalGenerator:
    return SignalGenerator(config=config or AIConfig())


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)
