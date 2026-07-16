"""
ai/monitoring/metrics.py - Runtime and resource metrics.

VERSION: 1.0.0
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field
from time import perf_counter
from typing import Dict, Iterator, List
import tracemalloc


@dataclass(frozen=True)
class ResourceMetrics:
    """CPU, memory, and GPU metrics."""

    memory_mb: float
    cpu_percent: float
    gpu_memory_mb: float = 0.0
    gpu_utilization_percent: float = 0.0


@dataclass
class MetricRecorder:
    """Record latency, prediction, and training durations."""

    latencies_ms: List[float] = field(default_factory=list)
    prediction_times_ms: List[float] = field(default_factory=list)
    training_times_ms: List[float] = field(default_factory=list)

    def record_latency(self, milliseconds: float) -> None:
        self.latencies_ms.append(float(milliseconds))

    def record_prediction_time(self, milliseconds: float) -> None:
        self.prediction_times_ms.append(float(milliseconds))

    def record_training_time(self, milliseconds: float) -> None:
        self.training_times_ms.append(float(milliseconds))

    @contextmanager
    def time_latency(self) -> Iterator[None]:
        start = perf_counter()
        try:
            yield
        finally:
            self.record_latency((perf_counter() - start) * 1000.0)

    @contextmanager
    def time_prediction(self) -> Iterator[None]:
        start = perf_counter()
        try:
            yield
        finally:
            self.record_prediction_time((perf_counter() - start) * 1000.0)

    @contextmanager
    def time_training(self) -> Iterator[None]:
        start = perf_counter()
        try:
            yield
        finally:
            self.record_training_time((perf_counter() - start) * 1000.0)

    def summary(self) -> Dict[str, float]:
        """Return aggregate timing metrics."""

        return {
            "latency_ms_mean": _mean(self.latencies_ms),
            "latency_ms_p95": _percentile(self.latencies_ms, 95.0),
            "prediction_ms_mean": _mean(self.prediction_times_ms),
            "prediction_ms_p95": _percentile(self.prediction_times_ms, 95.0),
            "training_ms_mean": _mean(self.training_times_ms),
            "training_ms_total": float(sum(self.training_times_ms)),
        }


def collect_resource_metrics() -> ResourceMetrics:
    """Collect memory, CPU, and optional GPU metrics."""

    memory_mb = 0.0
    cpu_percent = 0.0
    try:
        import psutil

        process = psutil.Process()
        memory_mb = float(process.memory_info().rss / (1024.0 * 1024.0))
        cpu_percent = float(process.cpu_percent(interval=None))
    except Exception:
        if not tracemalloc.is_tracing():
            tracemalloc.start()
        current, _ = tracemalloc.get_traced_memory()
        memory_mb = float(current / (1024.0 * 1024.0))

    gpu_memory_mb, gpu_util = _gpu_metrics()
    return ResourceMetrics(
        memory_mb=memory_mb,
        cpu_percent=cpu_percent,
        gpu_memory_mb=gpu_memory_mb,
        gpu_utilization_percent=gpu_util,
    )


def _gpu_metrics() -> tuple[float, float]:
    try:
        import torch

        if torch.cuda.is_available():
            memory = float(torch.cuda.memory_allocated() / (1024.0 * 1024.0))
            return memory, 0.0
    except Exception:
        pass
    return 0.0, 0.0


def _mean(values: List[float]) -> float:
    return float(sum(values) / len(values)) if values else 0.0


def _percentile(values: List[float], percentile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = min(len(ordered) - 1, max(0, int(round((percentile / 100.0) * (len(ordered) - 1)))))
    return float(ordered[idx])
