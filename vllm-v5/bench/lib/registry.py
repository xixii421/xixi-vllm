"""Benchmark suite registry (uses ``bench/lib`` + ``bench/suites``)."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from bench.lib.config import BenchConfig
from bench.lib.report import ResultWriter
from bench.lib.session import BenchSession
from bench.suites import decode, e2e, kv_vram, prefill, ttft

SuiteFn = Callable[
    [BenchConfig, ResultWriter | None, BenchSession | None],
    dict[str, Any],
]

SUITE_REGISTRY: dict[str, SuiteFn] = {
    "e2e": e2e.run,
    "ttft": ttft.run,
    "prefill": prefill.run,
    "decode": decode.run,
    "kv_vram": kv_vram.run,
}

DEFAULT_SUITES = ["e2e", "ttft", "decode"]
ALL_SUITES = list(SUITE_REGISTRY.keys())
