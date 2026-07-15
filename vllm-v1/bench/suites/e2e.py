"""End-to-end throughput benchmark suite."""

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
        for _ in session.engine.generate(
            cfg.prompt, max_new_tokens=cfg.max_new_tokens, temperature=0.0
        ):
            pass
        sync()

    sync()
    t0 = time.perf_counter()
    n = 0
    for _ in session.engine.generate(
        cfg.prompt, max_new_tokens=cfg.max_new_tokens, temperature=0.0
    ):
        n += 1
    sync()
    total_s = time.perf_counter() - t0

    row = {
        "prompt": cfg.prompt,
        "max_new_tokens": cfg.max_new_tokens,
        "generated_tokens": n,
        "total_time_s": round(total_s, 4),
        "e2e_tok_s": round(n / total_s, 2) if total_s > 0 else 0.0,
    }
    writer.write_csv(
        "e2e",
        [row],
        ["prompt", "max_new_tokens", "generated_tokens", "total_time_s", "e2e_tok_s"],
    )
    writer.append_report(
        "## E2E Throughput\n\n"
        f"| generated | e2e_tok/s | total_time_s |\n"
        f"| --- | --- | --- |\n"
        f"| {n} | {row['e2e_tok_s']:.2f} | {row['total_time_s']:.4f} |\n"
    )
    print_markdown_table(
        "E2E Throughput",
        ["generated", "e2e_tok/s", "total_time_s"],
        [[n, f"{row['e2e_tok_s']:.2f}", f"{row['total_time_s']:.4f}"]],
    )
    print(f"Saved to {writer.latest_symlink_hint}")
    return row
