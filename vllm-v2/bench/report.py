"""Benchmark console reporting."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from .lib import BatchMetrics, RunMetrics, speedup


def _f(x: float, nd: int = 3) -> str:
    if x != x:
        return "n/a"
    return f"{x:.{nd}f}"


def _tps(x: float) -> str:
    if x != x:
        return "n/a"
    return f"{x:.1f}"


@dataclass
class TripleResult:
    key: str
    v0: RunMetrics
    v1: RunMetrics
    v2: RunMetrics


class Reporter:
    def __init__(self, *, width: int = 88) -> None:
        self.width = width

    def banner(self, title: str) -> None:
        line = "═" * self.width
        print(f"\n{line}")
        print(f" {title}")
        print(line)

    def progress(self, msg: str) -> None:
        print(f"  … {msg}", flush=True)

    def triple_line(self, t: TripleResult, *, focus: str = "e2e") -> None:
        if focus == "ttft":
            print(
                f"  [{t.key}]  "
                f"v0 {_f(t.v0.ttft_s)}s  v1 {_f(t.v1.ttft_s)}s  v2 {_f(t.v2.ttft_s)}s  |  "
                f"v1× {speedup(t.v0.ttft_s, t.v1.ttft_s)}  "
                f"v2× {speedup(t.v0.ttft_s, t.v2.ttft_s)}"
            )
        else:
            print(
                f"  [{t.key}]  "
                f"v0 {_f(t.v0.e2e_s)}s/{_tps(t.v0.decode_tps)}  "
                f"v1 {_f(t.v1.e2e_s)}s/{_tps(t.v1.decode_tps)}  "
                f"v2 {_f(t.v2.e2e_s)}s/{_tps(t.v2.decode_tps)}  |  "
                f"v1× {speedup(t.v0.e2e_s, t.v1.e2e_s)}  "
                f"v2× {speedup(t.v0.e2e_s, t.v2.e2e_s)}"
            )

    def triple_table(
        self, title: str, rows: Sequence[TripleResult], *, focus: str = "e2e"
    ) -> None:
        print(f"\n  {title}")
        if focus == "ttft":
            hdr = (
                f"  {'case':<10}{'v0':>8}{'v1':>8}{'v2':>8}"
                f"{'v1/v0':>8}{'v2/v0':>8}{'v2/v1':>8}"
            )
            print(hdr)
            print("  " + "─" * (len(hdr) - 2))
            for t in rows:
                print(
                    f"  {t.key:<10}"
                    f"{_f(t.v0.ttft_s):>8}{_f(t.v1.ttft_s):>8}{_f(t.v2.ttft_s):>8}"
                    f"{speedup(t.v0.ttft_s, t.v1.ttft_s):>8}"
                    f"{speedup(t.v0.ttft_s, t.v2.ttft_s):>8}"
                    f"{speedup(t.v1.ttft_s, t.v2.ttft_s):>8}"
                )
        else:
            hdr = (
                f"  {'case':<10}{'v0_E2E':>9}{'v1_E2E':>9}{'v2_E2E':>9}"
                f"{'v1/v0':>8}{'v2/v0':>8}{'v2/v1':>8}"
                f"{'v0_Dec':>9}{'v1_Dec':>9}{'v2_Dec':>9}"
            )
            print(hdr)
            print("  " + "─" * (len(hdr) - 2))
            for t in rows:
                print(
                    f"  {t.key:<10}"
                    f"{_f(t.v0.e2e_s):>9}{_f(t.v1.e2e_s):>9}{_f(t.v2.e2e_s):>9}"
                    f"{speedup(t.v0.e2e_s, t.v1.e2e_s):>8}"
                    f"{speedup(t.v0.e2e_s, t.v2.e2e_s):>8}"
                    f"{speedup(t.v1.e2e_s, t.v2.e2e_s):>8}"
                    f"{_tps(t.v0.decode_tps):>9}{_tps(t.v1.decode_tps):>9}"
                    f"{_tps(t.v2.decode_tps):>9}"
                )
        print()

    def batch_table(self, title: str, rows: Sequence[BatchMetrics], *, v0_wall: float) -> None:
        print(f"\n  {title}")
        hdr = (
            f"  {'label':<14}{'reqs':>6}{'new':>6}{'wall_s':>10}"
            f"{'tok/s':>10}{'vs_v0':>8}"
        )
        print(hdr)
        print("  " + "─" * (len(hdr) - 2))
        for r in rows:
            print(
                f"  {r.label:<14}{r.num_reqs:>6}{r.new_tokens:>6}{_f(r.wall_s):>10}"
                f"{_tps(r.tok_per_s):>10}{speedup(v0_wall, r.wall_s):>8}"
            )
        print()

    def single_table(self, title: str, rows: Sequence[RunMetrics]) -> None:
        print(f"\n  {title}")
        hdr = (
            f"  {'label':<14}{'prompt':>8}{'new':>6}{'cached':>8}"
            f"{'TTFT':>10}{'E2E':>10}{'Dec_tps':>10}"
        )
        print(hdr)
        print("  " + "─" * (len(hdr) - 2))
        for r in rows:
            print(
                f"  {r.label:<14}{r.prompt_tokens:>8}{r.new_tokens:>6}{r.cached_tokens:>8}"
                f"{_f(r.ttft_s):>10}{_f(r.e2e_s):>10}{_tps(r.decode_tps):>10}"
            )
        print()
