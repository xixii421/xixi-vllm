"""Decode throughput benchmark suite."""

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
    ttft_s = None
    n = 0
    for _ in session.engine.generate(
        cfg.prompt, max_new_tokens=cfg.max_new_tokens, temperature=0.0
    ):
        if ttft_s is None:
            ttft_s = time.perf_counter() - t0
        n += 1
    sync()
    total_s = time.perf_counter() - t0

    decode_tokens = max(n - 1, 0)
    decode_time_s = max(total_s - (ttft_s or 0.0), 1e-9)
    row = {
        "prompt": cfg.prompt,
        "max_new_tokens": cfg.max_new_tokens,
        "generated_tokens": n,
        "ttft_ms": round((ttft_s or 0.0) * 1000.0, 3),
        "decode_tokens": decode_tokens,
        "decode_time_s": round(decode_time_s, 4),
        "decode_tok_s": round(decode_tokens / decode_time_s, 2),
    }
    writer.write_csv(
        "decode",
        [row],
        ["prompt", "max_new_tokens", "generated_tokens", "ttft_ms",
         "decode_tokens", "decode_time_s", "decode_tok_s"],
    )
    writer.append_report(
        "## Decode Throughput\n\n"
        f"| generated | decode_tok/s | decode_time_s |\n"
        f"| --- | --- | --- |\n"
        f"| {n} | {row['decode_tok_s']:.2f} | {row['decode_time_s']:.4f} |\n"
    )
    print_markdown_table(
        "Decode Throughput",
        ["generated", "decode_tok/s", "decode_time_s", "ttft_ms"],
        [[n, f"{row['decode_tok_s']:.2f}", f"{row['decode_time_s']:.4f}", f"{row['ttft_ms']:.2f}"]],
    )
    print(f"Saved to {writer.latest_symlink_hint}")
    return row
