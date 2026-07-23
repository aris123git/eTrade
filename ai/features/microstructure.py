"""
ai/features/microstructure.py - Order-flow imbalance and spread dynamics

RESPONSIBILITY:
Approximate market microstructure signals from OHLCV (+ optional bid/ask volumes
and spread). When true L2 data is absent, use candle-geometry proxies that are
standard in FX research (CLV imbalance, tick-rule pressure, spread z-scores).

VERSION: 1.0.0
"""

from __future__ import annotations

from enum import Enum
from typing import Dict, Sequence

import numpy as np
from numpy.typing import NDArray

from ai.utils.math_ops import rolling_mean, rolling_std, safe_div


FeatureMap = Dict[str, NDArray[np.floating]]


class MicrostructureSignal(str, Enum):
    """Supported microstructure feature families."""

    ORDER_FLOW = "order_flow"
    SPREAD = "spread"
    PRESSURE = "pressure"


def compute_microstructure_features(
    open_: NDArray[np.floating],
    high: NDArray[np.floating],
    low: NDArray[np.floating],
    close: NDArray[np.floating],
    volume: NDArray[np.floating],
    spread: NDArray[np.floating] | None = None,
    *,
    bid_volume: NDArray[np.floating] | None = None,
    ask_volume: NDArray[np.floating] | None = None,
    rolling_windows: Sequence[int] = (5, 10, 20, 50),
) -> FeatureMap:
    """
    Compute order-flow imbalance and spread-dynamics features.

    Order flow:
      - If bid/ask volumes exist → true bid/ask ratio and imbalance
      - Else proxy from close location in range (CLV) × volume (tick-rule style)

    Spread dynamics:
      - Level, % of price, change, rolling z-score, widening flag
    """

    open_arr = np.asarray(open_, dtype=float)
    high_arr = np.asarray(high, dtype=float)
    low_arr = np.asarray(low, dtype=float)
    close_arr = np.asarray(close, dtype=float)
    volume_arr = np.asarray(volume, dtype=float)
    n = len(close_arr)
    spread_arr = np.zeros(n, dtype=float) if spread is None else np.asarray(spread, dtype=float)

    # Close location value in [0, 1]: near 1 ⇒ closes at highs (buy pressure)
    range_ = np.maximum(high_arr - low_arr, 1e-12)
    clv = safe_div(close_arr - low_arr, range_, default=0.5)
    clv = np.clip(clv, 0.0, 1.0)

    buy_proxy = clv * volume_arr
    sell_proxy = (1.0 - clv) * volume_arr
    total_proxy = np.maximum(buy_proxy + sell_proxy, 1e-12)

    bid_arr = None if bid_volume is None else np.asarray(bid_volume, dtype=float)
    ask_arr = None if ask_volume is None else np.asarray(ask_volume, dtype=float)
    have_book = (
        bid_arr is not None
        and ask_arr is not None
        and len(bid_arr) == n
        and len(ask_arr) == n
        and np.isfinite(bid_arr).any()
        and np.isfinite(ask_arr).any()
    )

    if have_book:
        bid_safe = np.nan_to_num(bid_arr, nan=0.0)  # type: ignore[arg-type]
        ask_safe = np.nan_to_num(ask_arr, nan=0.0)  # type: ignore[arg-type]
        book_total = np.maximum(bid_safe + ask_safe, 1e-12)
        bid_ask_ratio = safe_div(bid_safe, ask_safe, default=1.0)
        order_flow_imbalance = safe_div(bid_safe - ask_safe, book_total, default=0.0)
        buy_volume = bid_safe
        sell_volume = ask_safe
        source = 1.0  # real book
    else:
        bid_ask_ratio = safe_div(buy_proxy, sell_proxy, default=1.0)
        order_flow_imbalance = safe_div(buy_proxy - sell_proxy, total_proxy, default=0.0)
        buy_volume = buy_proxy
        sell_volume = sell_proxy
        source = 0.0  # proxy

    # Tick-rule signed volume (+1 uptick, -1 downtick)
    tick_sign = np.zeros(n, dtype=float)
    if n > 1:
        tick_sign[1:] = np.sign(close_arr[1:] - close_arr[:-1])
        # Zero-tick inherits previous non-zero sign (Lee-Ready style)
        for i in range(1, n):
            if tick_sign[i] == 0.0:
                tick_sign[i] = tick_sign[i - 1]
    signed_volume = tick_sign * volume_arr

    # Relative spread (prefer broker spread; else high-low as instantaneous quote width proxy)
    quoted_spread = spread_arr.copy()
    missing_spread = ~np.isfinite(quoted_spread) | (quoted_spread <= 0.0)
    quoted_spread[missing_spread] = high_arr[missing_spread] - low_arr[missing_spread]
    relative_spread = safe_div(quoted_spread, close_arr, default=np.nan)
    spread_change = np.full(n, np.nan, dtype=float)
    if n > 1:
        spread_change[1:] = quoted_spread[1:] - quoted_spread[:-1]

    features: FeatureMap = {
        "micro_clv": clv,
        "micro_bid_ask_ratio": bid_ask_ratio,
        "micro_order_flow_imbalance": order_flow_imbalance,
        "micro_buy_volume": buy_volume,
        "micro_sell_volume": sell_volume,
        "micro_signed_volume": signed_volume,
        "micro_tick_sign": tick_sign,
        "micro_spread": quoted_spread,
        "micro_relative_spread": relative_spread,
        "micro_spread_change": spread_change,
        "micro_has_l2_book": np.full(n, source, dtype=float),
    }

    for window in _valid_windows(rolling_windows):
        imb_mean = rolling_mean(order_flow_imbalance, window)
        imb_std = rolling_std(order_flow_imbalance, window)
        spread_mean = rolling_mean(quoted_spread, window)
        spread_std = rolling_std(quoted_spread, window)
        buy_sum = _rolling_sum(buy_volume, window)
        sell_sum = _rolling_sum(sell_volume, window)
        signed_sum = _rolling_sum(signed_volume, window)
        vol_sum = _rolling_sum(volume_arr, window)

        features[f"micro_ofi_sma_{window}"] = imb_mean
        features[f"micro_ofi_zscore_{window}"] = safe_div(
            order_flow_imbalance - imb_mean, imb_std, default=np.nan
        )
        features[f"micro_bid_ask_ratio_sma_{window}"] = rolling_mean(bid_ask_ratio, window)
        features[f"micro_buy_sell_ratio_{window}"] = safe_div(buy_sum, sell_sum, default=1.0)
        features[f"micro_signed_volume_sum_{window}"] = signed_sum
        features[f"micro_signed_volume_ratio_{window}"] = safe_div(signed_sum, vol_sum, default=0.0)
        features[f"micro_spread_sma_{window}"] = spread_mean
        features[f"micro_spread_zscore_{window}"] = safe_div(
            quoted_spread - spread_mean, spread_std, default=np.nan
        )
        features[f"micro_spread_widening_{window}"] = (
            (quoted_spread > spread_mean) & np.isfinite(spread_mean)
        ).astype(float)
        features[f"micro_spread_volatility_{window}"] = rolling_std(quoted_spread, window)

    return features


def _valid_windows(windows: Sequence[int]) -> list[int]:
    return sorted({int(w) for w in windows if int(w) > 0})


def _rolling_sum(values: NDArray[np.floating], window: int) -> NDArray[np.floating]:
    arr = np.asarray(values, dtype=float)
    out = np.full(len(arr), np.nan, dtype=float)
    if window <= 0 or len(arr) < window:
        return out
    cs = np.cumsum(np.nan_to_num(arr, nan=0.0))
    out[window - 1] = cs[window - 1]
    if len(arr) > window:
        out[window:] = cs[window:] - cs[:-window]
    return out
