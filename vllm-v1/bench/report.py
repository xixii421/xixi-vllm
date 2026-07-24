"""Benchmark console reporting (presentation only; no measurement logic)."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from .lib import RunMetrics, speedup


def _f(x: float, nd: int = 3) -> str:
    if x != x:  # NaN
        return "n/a"
    return f"{x:.{nd}f}"


def _tps(x: float) -> str:
    if x != x:
        return "n/a"
    return f"{x:.1f}"


@dataclass
class PairResult:
    """One sweep point: baseline (v0) vs candidate (v1)."""

    key: str
    baseline: RunMetrics
    candidate: RunMetrics

    @property
    def e2e_speedup(self) -> str:
        return speedup(self.baseline, self.candidate)

    @property
    def ttft_speedup(self) -> str:
        if self.baseline.ttft_s <= 0:
            return "n/a"
        return f"{self.baseline.ttft_s / self.candidate.ttft_s:.2f}x"


class Reporter:
    """Stdout tables with fixed-width columns; scenarios only feed data."""

    def __init__(self, *, width: int = 78) -> None:
        self.width = width

    def banner(self, title: str) -> None:
        line = "═" * self.width
        print(f"\n{line}")
        print(f" {title}")
        print(line)

    def note(self, text: str) -> None:
        for raw in text.strip().splitlines():
            print(f"  · {raw.strip()}")
        print()

    def progress(self, msg: str) -> None:
        print(f"  … {msg}", flush=True)

    def pair_line(self, pair: PairResult, *, focus: str = "e2e") -> None:
        b, c = pair.baseline, pair.candidate
        if focus == "ttft":
            print(
                f"  [{pair.key}]  "
                f"{b.label} TTFT {_f(b.ttft_s)}s  |  "
                f"{c.label} TTFT {_f(c.ttft_s)}s  |  "
                f"TTFT× {pair.ttft_speedup}  E2E× {pair.e2e_speedup}"
            )
        else:
            print(
                f"  [{pair.key}]  "
                f"{b.label} E2E {_f(b.e2e_s)}s Dec {_tps(b.decode_tps)} t/s  |  "
                f"{c.label} E2E {_f(c.e2e_s)}s Dec {_tps(c.decode_tps)} t/s  |  "
                f"E2E× {pair.e2e_speedup}"
            )

    def pair_table(self, title: str, pairs: Sequence[PairResult], *, focus: str = "e2e") -> None:
        print(f"\n  {title}")
        if focus == "ttft":
            hdr = (
                f"  {'case':<12}{'v0_TTFT':>10}{'v1_TTFT':>10}{'TTFT×':>8}"
                f"{'v0_E2E':>10}{'v1_E2E':>10}{'E2E×':>8}{'v1_cached':>10}"
            )
            print(hdr)
            print("  " + "─" * (len(hdr) - 2))
            for p in pairs:
                b, c = p.baseline, p.candidate
                print(
                    f"  {p.key:<12}{_f(b.ttft_s):>10}{_f(c.ttft_s):>10}{p.ttft_speedup:>8}"
                    f"{_f(b.e2e_s):>10}{_f(c.e2e_s):>10}{p.e2e_speedup:>8}{c.cached_tokens:>10}"
                )
        else:
            hdr = (
                f"  {'case':<12}{'v0_E2E':>10}{'v1_E2E':>10}{'E2E×':>8}"
                f"{'v0_Dec':>10}{'v1_Dec':>10}{'v0_tok':>8}{'v1_tok':>8}"
            )
            print(hdr)
            print("  " + "─" * (len(hdr) - 2))
            for p in pairs:
                b, c = p.baseline, p.candidate
                print(
                    f"  {p.key:<12}{_f(b.e2e_s):>10}{_f(c.e2e_s):>10}{p.e2e_speedup:>8}"
                    f"{_tps(b.decode_tps):>10}{_tps(c.decode_tps):>10}"
                    f"{b.new_tokens:>8}{c.new_tokens:>8}"
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
