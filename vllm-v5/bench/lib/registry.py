"""基准套件注册表。

主推（PagedAttention）::

    kv_vram       — 多请求真实 Prefill，分页 vs 连续对照
    paged_overhead — Decode gather 扫长计时

通用（继承自 v1）::

    ttft / prefill / decode / e2e / vram / latency_seqlen
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from bench.lib.config import BenchConfig
from bench.lib.report import ResultWriter
from bench.lib.session import BenchSession
from bench.suites import decode, e2e, kv_vram, latency_seqlen, paged_overhead, prefill, ttft, vram

SuiteFn = Callable[
    [BenchConfig, ResultWriter | None, BenchSession | None],
    dict[str, Any],
]

SUITE_REGISTRY: dict[str, SuiteFn] = {
    "kv_vram": kv_vram.run,
    "paged_overhead": paged_overhead.run,
    "ttft": ttft.run,
    "prefill": prefill.run,
    "decode": decode.run,
    "e2e": e2e.run,
    "vram": vram.run,
    "latency_seqlen": latency_seqlen.run,
}

# 教学默认：只跑与 PagedAttention 直接相关的套件
PAGED_SUITES = ["kv_vram", "paged_overhead"]

ALL_SUITES = list(SUITE_REGISTRY.keys())
