"""Prefill latency benchmark suite."""

from __future__ import annotations

import random

import torch

from bench.lib.config import BenchConfig
from bench.lib.report import ResultWriter, format_stats_row, print_markdown_table
from bench.lib.session import BenchSession, collect_meta
from bench.lib.timing import benchmark_cuda_safe


def run(cfg: BenchConfig, writer: ResultWriter | None = None, session: BenchSession | None = None) -> list[dict]:
    session = session or BenchSession(cfg)
    meta = collect_meta(cfg)
    writer = writer or ResultWriter(cfg, meta)
    session.global_warmup()

    lengths = [n for n in cfg.prefill_lengths if n <= cfg.context_len]
    random.shuffle(lengths)

    rows: list[dict] = []
    table_rows: list[list] = []
    for seq_len in lengths:
        token_ids = session.make_prompt_tokens(seq_len)

        def _forward() -> None:
            session.forward_only(token_ids)

        stats, err = benchmark_cuda_safe(
            _forward,
            warmup=cfg.warmup_for_seq_len(seq_len),
            iters=cfg.iters_for_seq_len(seq_len),
        )
        if err == "oom":
            row = {
                "seq_len": seq_len,
                "metric": "prefill_forward_cuda",
                "status": "oom",
                "median_ms": "",
                "mean_ms": "",
                "std_ms": "",
                "p90_ms": "",
                "min_ms": "",
                "max_ms": "",
                "ms_per_token": "",
                "samples": 0,
            }
            rows.append(row)
            table_rows.append([seq_len, "OOM", "-", "-"])
            print(f"  prompt_len={seq_len:4d}  OOM (skipped)")
            continue

        row = format_stats_row(
            {"seq_len": seq_len, "metric": "prefill_forward_cuda", "status": "ok"},
            stats,
        )
        row["ms_per_token"] = round(row["median_ms"] / seq_len, 4)
        rows.append(row)
        table_rows.append([
            seq_len,
            f"{row['median_ms']:.2f}",
            f"{row['std_ms']:.2f}",
            f"{row['ms_per_token']:.4f}",
        ])
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    rows.sort(key=lambda r: r["seq_len"])
    table_rows.sort(key=lambda r: r[0])
    fields = [
        "seq_len", "metric", "status", "median_ms", "mean_ms", "std_ms", "p90_ms",
        "min_ms", "max_ms", "ms_per_token", "samples",
    ]
    writer.write_csv("prefill", rows, fields)
    md_lines = ["## Prefill Forward (CUDA)\n", "| seq_len | median_ms | std_ms | ms_per_token |", "| --- | --- | --- | --- |"]
    for r in rows:
        md_lines.append(
            f"| {r['seq_len']} | {r['median_ms']:.3f} | {r['std_ms']:.3f} | {r['ms_per_token']:.4f} |"
        )
    writer.append_report("\n".join(md_lines))
    print_markdown_table(
        "Prefill Forward (CUDA events, shuffled order)",
        ["seq_len", "median_ms", "std_ms", "ms_per_token"],
        table_rows,
    )
    print(f"Saved to {writer.latest_symlink_hint}")
    return rows
