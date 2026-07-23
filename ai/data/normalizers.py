"""
ai/data/normalizers.py - Candle/tick normalization to AI contracts

VERSION: 1.0.0
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, Mapping

from ai.utils.types import CandleDict


REQUIRED_OHLC = ("open", "high", "low", "close")


def _to_datetime(value: Any) -> datetime:
    if isinstance(value, datetime):
        return value
    if isinstance(value, (int, float)):
        ts = float(value)
        if ts > 1e12:
            ts /= 1000.0
        return datetime.utcfromtimestamp(ts)
    if isinstance(value, str):
        return datetime.fromisoformat(value.replace("Z", "+00:00")).replace(tzinfo=None)
    raise TypeError(f"Cannot convert {type(value)} to datetime")


def candle_entity_to_dict(entity: Any) -> CandleDict:
    """Convert repository Candle entities / dataclasses / mappings to CandleDict."""
    if isinstance(entity, Mapping):
        return normalize_candle_dict(dict(entity))

    def get(name: str, *aliases: str, default: Any = None) -> Any:
        for key in (name, *aliases):
            if hasattr(entity, key):
                value = getattr(entity, key)
                if value is not None:
                    return value
        return default

    payload: Dict[str, Any] = {
        "symbol": get("symbol", default=""),
        "timeframe": get("timeframe", default=""),
        "timestamp": get("timestamp", "time"),
        "open": get("open", "open_price"),
        "high": get("high"),
        "low": get("low"),
        "close": get("close"),
        "volume": get("volume", "tick_volume", "real_volume", default=0.0),
        "tick_volume": get("tick_volume", default=0.0),
        "real_volume": get("real_volume", default=0.0),
        "spread": get("spread", default=0.0),
        "market_id": get("market_id"),
        "broker_id": get("broker_id"),
    }
    return normalize_candle_dict(payload)


def normalize_candle_dict(raw: Mapping[str, Any]) -> CandleDict:
    """Normalize heterogeneous candle dicts into the AI canonical schema."""
    data = dict(raw)
    timestamp = data.get("timestamp", data.get("time"))
    if timestamp is None:
        raise ValueError("Candle missing timestamp/time")

    candle: CandleDict = {
        "symbol": str(data.get("symbol", "")).upper(),
        "timeframe": str(data.get("timeframe", "")).upper(),
        "timestamp": _to_datetime(timestamp),
        "open": float(data["open"] if "open" in data else data["open_price"]),
        "high": float(data["high"]),
        "low": float(data["low"]),
        "close": float(data["close"]),
        "volume": float(
            data.get("volume", data.get("tick_volume", data.get("real_volume", 0.0))) or 0.0
        ),
        "tick_volume": float(data.get("tick_volume", 0.0) or 0.0),
        "real_volume": float(data.get("real_volume", 0.0) or 0.0),
        "spread": float(data.get("spread", 0.0) or 0.0),
    }
    if data.get("market_id") is not None:
        candle["market_id"] = int(data["market_id"])
    if data.get("broker_id") is not None:
        candle["broker_id"] = int(data["broker_id"])

    for col in REQUIRED_OHLC:
        if candle[col] is None:  # type: ignore[literal-required]
            raise ValueError(f"Candle missing {col}")
    return candle
