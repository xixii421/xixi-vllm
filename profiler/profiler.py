"""Profiler: torch.profiler 薄封装。

用法:
  pf = Profiler(trace_dir="./traces/v0", label="v0")
  with pf.annotate("prefill"):
      model(...)
  pf.print_full_table()      # 所有算子的 CPU+CUDA 时间
  pf.print_category_summary() # 按模块分组的分类汇总
"""

from __future__ import annotations

import os
from contextlib import nullcontext
from typing import Any

import torch
from torch.profiler import (
    ProfilerActivity,
    profile as _torch_profile,
    record_function,
    tensorboard_trace_handler,
)


class Profiler:
    def __init__(
        self,
        *,
        trace_dir: str = "./profiler_traces",
        label: str = "profile",
        with_stack: bool = False,
    ) -> None:
        self.trace_dir = trace_dir
        self.label = label
        self._with_stack = with_stack
        self._running = False
        self._prof: _torch_profile | None = None
        os.makedirs(trace_dir, exist_ok=True)

    def _build(self) -> _torch_profile:
        return _torch_profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            record_shapes=True,
            with_stack=self._with_stack,
            on_trace_ready=tensorboard_trace_handler(
                self.trace_dir, worker_name=self.label, use_gzip=False
            ),
        )

    def start(self) -> None:
        if self._running:
            return
        self._prof = self._build()
        self._prof.__enter__()
        self._running = True

    def stop(self) -> None:
        if self._prof is None:
            return
        self._prof.__exit__(None, None, None)
        self._running = False

    def annotate(self, name: str):
        if self._running:
            return record_function(name)
        return nullcontext()

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *args):
        self.stop()
        return False

    def print_full_table(self, sort_by: str = "cuda_time_total", top: int = 40) -> None:
        """打印所有 operator 的 CPU 和 CUDA 时间明细，精确到每个 kernel。"""
        if self._prof is None:
            return
        try:
            tbl = self._prof.key_averages().table(
                sort_by=sort_by, row_limit=top,
                max_src_column_width=80,
            )
            print(tbl)
        except Exception:
            print("[profiler] no data")

    def print_category_summary(self, top: int = 20) -> None:
        """按 aten/cudaMemcpy/cudaLaunch 等大类汇总 CUDA 时间。"""
        if self._prof is None:
            return
        groups: dict[str, dict[str, float]] = {}
        try:
            for evt in self._prof.key_averages():
                name = evt.key
                if "::" in name:
                    parts = name.split("::")
                    cat = parts[0]
                    # aten 内部细分: aten::linear, aten::matmul, aten::addmm 等
                    if cat == "aten" and len(parts) >= 2:
                        sub = parts[1]
                        if any(k in sub for k in ("linear", "addmm", "matmul", "bmm", "mm")):
                            cat = "aten::GEMM"
                        elif any(k in sub for k in ("scaled_dot_product", "flash_attn", "sdpa")):
                            cat = "aten::SDPA"
                        elif "norm" in sub:
                            cat = "aten::norm"
                        elif "mul" in sub or "add" in sub or "div" in sub:
                            cat = "aten::elementwise"
                        elif "copy" in sub or "clone" in sub or "to" in sub:
                            cat = "aten::copy"
                        elif "view" in sub or "reshape" in sub or "transpose" in sub:
                            cat = "aten::view"
                        elif "cat" in sub:
                            cat = "aten::cat"
                        else:
                            cat = "aten::other"
                elif name.startswith("cuda"):
                    if "Launch" in name:
                        cat = "cudaLaunch"
                    elif "Memcpy" in name or "memcpy" in name:
                        cat = "cudaMemcpy"
                    elif "Sync" in name:
                        cat = "cudaSync"
                    else:
                        cat = "cuda_runtime"
                elif name.startswith("void "):
                    # CUDA kernel: void cutlass::Kernel2<...>, void pytorch_flash::...
                    cat = "GPU_KERNEL"
                else:
                    cat = f"other::{name[:20]}"
                if cat not in groups:
                    groups[cat] = {"cuda_us": 0.0, "cpu_us": 0.0, "count": 0}
                # key_averages() 返回 FunctionEventAvg: 用 self_device_time_total
                cuda_us = (getattr(evt, "self_device_time_total", 0)
                           or getattr(evt, "self_cuda_time_total", 0) or 0)
                cpu_us = (getattr(evt, "self_cpu_time_total", 0)
                          or getattr(evt, "cpu_time_total", 0) or 0)
                groups[cat]["cuda_us"] += cuda_us
                groups[cat]["cpu_us"] += cpu_us
                groups[cat]["count"] += evt.count
        except Exception:
            return

        cuda_total = sum(v["cuda_us"] for v in groups.values()) or 1.0
        cpu_total = sum(v["cpu_us"] for v in groups.values()) or 1.0
        items = sorted(groups.items(), key=lambda x: -x[1]["cuda_us"])

        print(f"\n{'Category':<28s} {'CUDA(ms)':>10s} {'  %':>6s} "
              f"{'CPU(ms)':>10s} {'  %':>6s} {'Calls':>8s}")
        print("-" * 76)
        for cat, v in items[:top]:
            print(f"{cat:<28s} {v['cuda_us']/1000:>10.2f} {v['cuda_us']/cuda_total*100:>5.1f}% "
                  f"{v['cpu_us']/1000:>10.2f} {v['cpu_us']/cpu_total*100:>5.1f}% "
                  f"{v['count']:>8.0f}")

        print(f"\n  GPU 总时间: {cuda_total/1000:.1f}ms  "
              f"CPU 总时间: {cpu_total/1000:.1f}ms  "
              f"CPU/GPU={cpu_total/max(cuda_total,1):.1f}x")
        print(f"  若 CPU/GPU>3: CPU 瓶颈（Python 开销/数据搬运/同步等），"
              f"nsys 可精确定位")
