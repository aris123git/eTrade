"""
ai/evaluation/trading_metrics.py - Trading performance metrics.

VERSION: 1.0.0
"""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any, Dict, Sequence

import numpy as np
from numpy.typing import NDArray

from ai.utils.math_ops import max_drawdown as _max_drawdown
from ai.utils.math_ops import sharpe_ratio as _sharpe_ratio
from ai.utils.math_ops import sortino_ratio as _sortino_ratio


def profit_factor(trade_returns: Sequence[float]) -> float:
    """Gross profit divided by gross loss."""
    trades = _finite_array(trade_returns)
    gains = float(np.sum(trades[trades > 0.0]))
    losses = float(np.abs(np.sum(trades[trades < 0.0])))
    if losses == 0.0:
        return float("inf") if gains > 0.0 else 0.0
    return float(gains / losses)


def expectancy(trade_returns: Sequence[float]) -> float:
    """Average expected profit or return per trade."""
    trades = _finite_array(trade_returns)
    return float(np.mean(trades)) if len(trades) else 0.0


def sharpe(trade_returns: Sequence[float], risk_free: float = 0.0, periods: int = 252) -> float:
    """Annualized Sharpe ratio."""
    return _safe_ratio(_sharpe_ratio(_finite_array(trade_returns), risk_free=risk_free, periods=periods))


def sortino(trade_returns: Sequence[float], risk_free: float = 0.0, periods: int = 252) -> float:
    """Annualized Sortino ratio."""
    returns = _finite_array(trade_returns)
    if len(returns) < 2:
        return 0.0
    excess = returns - risk_free / periods
    downside = excess[excess < 0.0]
    if len(downside) == 1:
        return float("inf") if np.mean(excess) > 0.0 else 0.0
    return _safe_ratio(_sortino_ratio(returns, risk_free=risk_free, periods=periods))


def max_drawdown(equity: Sequence[float]) -> float:
    """Maximum drawdown as a negative decimal from the equity peak."""
    drawdown, _, _ = _max_drawdown(np.asarray(equity, dtype=float))
    return _safe_ratio(drawdown)


def max_drawdown_details(equity: Sequence[float]) -> Dict[str, float | int]:
    """Return drawdown value plus peak and trough indexes."""
    drawdown, peak, trough = _max_drawdown(np.asarray(equity, dtype=float))
    return {"drawdown": _safe_ratio(drawdown), "peak_index": int(peak), "trough_index": int(trough)}


def win_rate(trade_returns: Sequence[float]) -> float:
    """Fraction of trades with positive result."""
    trades = _finite_array(trade_returns)
    if len(trades) == 0:
        return 0.0
    return float(np.mean(trades > 0.0))


def avg_trade(trade_returns: Sequence[float]) -> float:
    """Average trade result."""
    return expectancy(trade_returns)


def avg_holding(holding_periods: Sequence[Any] | None) -> float:
    """Average holding period in seconds."""
    if not holding_periods:
        return 0.0
    seconds = np.asarray([_holding_seconds(value) for value in holding_periods], dtype=float)
    seconds = seconds[np.isfinite(seconds)]
    return float(np.mean(seconds)) if len(seconds) else 0.0


def risk_reward(trade_returns: Sequence[float]) -> float:
    """Average winning trade divided by the absolute average losing trade."""
    trades = _finite_array(trade_returns)
    winners = trades[trades > 0.0]
    losers = trades[trades < 0.0]
    if len(losers) == 0:
        return float("inf") if len(winners) else 0.0
    average_loss = float(np.abs(np.mean(losers)))
    if average_loss == 0.0:
        return float("inf") if len(winners) else 0.0
    return float(np.mean(winners) / average_loss) if len(winners) else 0.0


def equity_curve(
    trade_returns: Sequence[float],
    initial_equity: float = 1.0,
    compounded: bool = False,
) -> NDArray[np.floating]:
    """Build an equity curve from trade PnL values or compounded returns."""
    trades = _finite_array(trade_returns)
    curve = np.empty(len(trades) + 1, dtype=float)
    curve[0] = float(initial_equity)
    if compounded:
        for idx, value in enumerate(trades, start=1):
            curve[idx] = curve[idx - 1] * (1.0 + value)
    else:
        curve[1:] = float(initial_equity) + np.cumsum(trades)
    return curve


def trading_metrics(
    trade_returns: Sequence[float],
    holding_periods: Sequence[Any] | None = None,
    initial_equity: float = 1.0,
    periods: int = 252,
    compounded: bool = False,
) -> Dict[str, Any]:
    """Aggregate trade, risk, and equity metrics into a serializable dictionary."""
    trades = _finite_array(trade_returns)
    equity = equity_curve(trades, initial_equity=initial_equity, compounded=compounded)
    ratio_returns = trades if compounded else _equity_returns(equity)
    return {
        "total_trades": int(len(trades)),
        "profit_factor": profit_factor(trades),
        "expectancy": expectancy(trades),
        "sharpe": sharpe(ratio_returns, periods=periods),
        "sortino": sortino(ratio_returns, periods=periods),
        "max_drawdown": max_drawdown(equity),
        "max_drawdown_details": max_drawdown_details(equity),
        "win_rate": win_rate(trades),
        "avg_trade": avg_trade(trades),
        "avg_holding": avg_holding(holding_periods),
        "risk_reward": risk_reward(trades),
        "equity_curve": equity.tolist(),
        "final_equity": float(equity[-1]) if len(equity) else float(initial_equity),
        "total_return": float((equity[-1] - initial_equity) / initial_equity) if initial_equity else 0.0,
    }


def _finite_array(values: Sequence[float]) -> NDArray[np.floating]:
    arr = np.asarray(values, dtype=float).reshape(-1)
    return arr[np.isfinite(arr)]


def _equity_returns(equity: NDArray[np.floating]) -> NDArray[np.floating]:
    if len(equity) < 2:
        return np.empty(0, dtype=float)
    previous = equity[:-1]
    diff = np.diff(equity)
    out = np.zeros_like(diff, dtype=float)
    mask = previous != 0.0
    np.divide(diff, previous, out=out, where=mask)
    return out


def _holding_seconds(value: Any) -> float:
    if isinstance(value, timedelta):
        return float(value.total_seconds())
    if isinstance(value, tuple) and len(value) == 2:
        start, end = value
        if isinstance(start, datetime) and isinstance(end, datetime):
            return float((end - start).total_seconds())
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def _safe_ratio(value: float) -> float:
    if np.isnan(value):
        return 0.0
    return float(value)
