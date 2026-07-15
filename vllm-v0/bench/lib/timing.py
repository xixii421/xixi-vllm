"""Precise GPU timing utilities."""

from __future__ import annotations

import statistics
import time
from dataclasses import dataclass

import torch


def sync() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


@dataclass(frozen=True)
class TimingStats:
    median_ms: float
    mean_ms: float
    std_ms: float
    p90_ms: float
    min_ms: float
    max_ms: float
    n: int

    @classmethod
    def from_samples(cls, samples_ms: list[float]) -> TimingStats:
        if not samples_ms:
            raise ValueError("samples_ms is empty")
        ordered = sorted(samples_ms)
        n = len(ordered)
        p90_idx = min(int(0.9 * n) - 1, n - 1)
        return cls(
            median_ms=statistics.median(ordered),
            mean_ms=statistics.mean(ordered),
            std_ms=statistics.pstdev(ordered) if n > 1 else 0.0,
            p90_ms=ordered[p90_idx],
            min_ms=ordered[0],
            max_ms=ordered[-1],
            n=n,
        )


class CudaTimer:
    """Measure GPU elapsed time with CUDA events."""

    def __init__(self) -> None:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required for CudaTimer")
        self._start = torch.cuda.Event(enable_timing=True)
        self._end = torch.cuda.Event(enable_timing=True)

    def __enter__(self) -> CudaTimer:
        sync()
        self._start.record()
        return self

    def __exit__(self, *_args) -> None:
        self._end.record()
        sync()

    @property
    def ms(self) -> float:
        return self._start.elapsed_time(self._end)


def cuda_time_ms(fn) -> float:
    with CudaTimer() as timer:
        fn()
    return timer.ms


def wall_time_ms(fn) -> float:
    sync()
    t0 = time.perf_counter()
    fn()
    sync()
    return (time.perf_counter() - t0) * 1000.0


def benchmark_cuda(fn, *, warmup: int, iters: int) -> TimingStats:
    for _ in range(warmup):
        fn()
    sync()
    samples = [cuda_time_ms(fn) for _ in range(iters)]
    return TimingStats.from_samples(samples)


def benchmark_cuda_safe(fn, *, warmup: int, iters: int) -> tuple[TimingStats | None, str | None]:
    """Return (stats, error). On OOM returns (None, 'oom')."""
    try:
        return benchmark_cuda(fn, warmup=warmup, iters=iters), None
    except torch.cuda.OutOfMemoryError:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        sync()
        return None, "oom"


def benchmark_wall(fn, *, warmup: int, iters: int) -> TimingStats:
    for _ in range(warmup):
        fn()
    sync()
    samples = [wall_time_ms(fn) for _ in range(iters)]
    return TimingStats.from_samples(samples)
