"""
ai.execution - Paper and simulated order execution.

RESPONSIBILITY:
Expose broker-neutral order execution dataclasses and factories.

VERSION: 1.0.0
"""

from ai.execution.executor import (
    ExecutionReport,
    Fill,
    LiveBrokerClient,
    Order,
    OrderExecutor,
    OrderStatus,
    create_order_executor,
)

__all__ = [
    "ExecutionReport",
    "Fill",
    "LiveBrokerClient",
    "Order",
    "OrderExecutor",
    "OrderStatus",
    "create_order_executor",
]
