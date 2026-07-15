"""TTFT benchmark suite."""

from __future__ import annotations

import time

from bench.lib.config import BenchConfig
from bench.lib.report import ResultWriter, print_markdown_table
from bench.lib.session import BenchSession, collect_meta
from bench.lib.timing import sync


def run(cfg: BenchConfig, writer: ResultWriter | None = None, session: BenchSession | None = None) -> dict:
    session = session or BenchSession(cfg)
    meta = collect_meta(cfg)
    writer = writer or ResultWriter(cfg, meta)
    session.global_warmup()

    for _ in range(cfg.warmup_iters):
        for _ in session.engine.generate(cfg.prompt, max_new_tokens=1, temperature=0.0):
            pass
        sync()

    sync()
    t0 = time.perf_counter()
    for _ in session.engine.generate(cfg.prompt, max_new_tokens=1, temperature=0.0):
        ttft_ms = (time.perf_counter() - t0) * 1000.0
        break
    sync()

    row = {"prompt": cfg.prompt, "ttft_ms": round(ttft_ms, 3)}
    writer.write_csv("ttft", [row], ["prompt", "ttft_ms"])
    writer.append_report(
        "## TTFT\n\n"
        f"| Prompt | TTFT (ms) |\n| --- | --- |\n"
        f"| {cfg.prompt} | {row['ttft_ms']:.3f} |\n"
    )
    print_markdown_table("TTFT", ["Prompt", "TTFT (ms)"], [[cfg.prompt, f"{row['ttft_ms']:.3f}"]])
    print(f"Saved to {writer.latest_symlink_hint}")
    return row
