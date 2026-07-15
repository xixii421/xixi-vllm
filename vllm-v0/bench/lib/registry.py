"""Benchmark suite registry."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from bench.lib.config import BenchConfig
from bench.lib.report import ResultWriter
from bench.lib.session import BenchSession
from bench.suites import decode, e2e, latency_seqlen, prefill, ttft, vram

SuiteFn = Callable[
    [BenchConfig, ResultWriter | None, BenchSession | None],
    dict[str, Any],
]

SUITE_REGISTRY: dict[str, SuiteFn] = {
    "ttft": ttft.run,
    "prefill": prefill.run,
    "decode": decode.run,
    "e2e": e2e.run,
    "vram": vram.run,
    "latency_seqlen": latency_seqlen.run,
}

ALL_SUITES = list(SUITE_REGISTRY.keys())
