"""Latency vs sequence length benchmark suite."""

from __future__ import annotations

import random

from bench.lib.config import BenchConfig
from bench.lib.report import ResultWriter, format_stats_row, print_markdown_table
from bench.lib.session import BenchSession, collect_meta
from bench.lib.timing import benchmark_cuda, benchmark_cuda_safe, benchmark_wall, sync


def measure_fixed_forward(session: BenchSession, cfg: BenchConfig, seq_len: int) -> dict:
    token_ids = session.make_prompt_tokens(seq_len)

    def _forward() -> None:
        session.forward_only(token_ids)

    stats, err = benchmark_cuda_safe(
        _forward,
        warmup=cfg.warmup_for_seq_len(seq_len),
        iters=cfg.iters_for_seq_len(seq_len),
    )
    if err == "oom":
        return {
            "mode": "fixed_forward",
            "seq_len": seq_len,
            "status": "oom",
            "median_ms": "",
            "std_ms": "",
            "ms_per_token_cuda": "",
            "samples": 0,
        }

    wall_stats = benchmark_wall(_forward, warmup=2, iters=min(cfg.measure_iters, 10))

    row = format_stats_row(
        {"mode": "fixed_forward", "seq_len": seq_len, "status": "ok"},
        stats,
    )
    row["wall_median_ms"] = round(wall_stats.median_ms, 3)
    row["ms_per_token_cuda"] = round(row["median_ms"] / seq_len, 4)
    return row


def measure_growing_decode(session: BenchSession, cfg: BenchConfig) -> list[dict]:
    """Each row is an independent measurement at a target seq_len.

    v0 recomputes the full sequence every step. We isolate each length instead of
    chaining steps in one run, which mixes cold-start and cumulative state effects.
    """
    prompt_len = cfg.latency_prompt_len
    max_seq = min(cfg.context_len, prompt_len + cfg.latency_decode_steps)
    target_lens: list[int] = []
    for step in range(cfg.latency_sample_every, cfg.latency_decode_steps + 1, cfg.latency_sample_every):
        seq_len = prompt_len + step
        if seq_len <= max_seq:
            target_lens.append(seq_len)
    if max_seq not in target_lens:
        target_lens.append(max_seq)
    target_lens = sorted(set(target_lens))

    rows: list[dict] = []
    for seq_len in target_lens:
        token_ids = session.make_prompt_tokens(seq_len)

        def _forward() -> None:
            session.forward_only(token_ids)

        stats, err = benchmark_cuda_safe(
            _forward,
            warmup=cfg.warmup_for_seq_len(seq_len),
            iters=cfg.iters_for_seq_len(seq_len),
        )
        if err == "oom":
            rows.append({
                "mode": "growing_decode_isolated",
                "decode_step": seq_len - prompt_len,
                "seq_len": seq_len,
                "status": "oom",
                "median_ms": "",
                "ms_per_token_cuda": "",
                "samples": 0,
            })
            continue

        decode_step = seq_len - prompt_len
        row = format_stats_row(
            {
                "mode": "growing_decode_isolated",
                "decode_step": decode_step,
                "seq_len": seq_len,
                "status": "ok",
            },
            stats,
        )
        row["ms_per_token_cuda"] = round(row["median_ms"] / seq_len, 4)
        rows.append(row)
    return rows


def run(cfg: BenchConfig, writer: ResultWriter | None = None, session: BenchSession | None = None) -> dict:
    session = session or BenchSession(cfg)
    meta = collect_meta(cfg)
    writer = writer or ResultWriter(cfg, meta)
    session.global_warmup()

    lengths = [n for n in cfg.latency_lengths if n <= cfg.context_len]
    random.shuffle(lengths)
    fixed_rows = [measure_fixed_forward(session, cfg, seq_len) for seq_len in lengths]
    fixed_rows.sort(key=lambda r: r["seq_len"])

    growing_rows = measure_growing_decode(session, cfg)

    fields = [
        "mode", "seq_len", "decode_step", "status", "median_ms", "mean_ms", "std_ms", "p90_ms",
        "min_ms", "max_ms", "wall_median_ms", "ms_per_token_cuda", "samples",
    ]
    writer.write_csv("latency_fixed_forward", fixed_rows, fields)
    writer.write_csv("latency_growing_decode", growing_rows, fields)

    md = ["## Latency vs SeqLen\n", "### Fixed Forward (CUDA)\n",
          "| seq_len | median_ms | std_ms | ms_per_token |", "| --- | --- | --- | --- |"]
    for r in fixed_rows:
        med = r.get("median_ms", "")
        std = r.get("std_ms", "")
        mpt = r.get("ms_per_token_cuda", "")
        med_s = f"{med:.3f}" if isinstance(med, (int, float)) else str(med)
        std_s = f"{std:.3f}" if isinstance(std, (int, float)) else str(std)
        mpt_s = f"{mpt:.4f}" if isinstance(mpt, (int, float)) else str(mpt)
        md.append(f"| {r['seq_len']} | {med_s} | {std_s} | {mpt_s} |")
    md += ["\n### Growing Decode (isolated per length, CUDA)\n",
           "| decode_step | seq_len | median_ms | ms_per_token |", "| --- | --- | --- | --- |"]
    for r in growing_rows:
        med = r.get("median_ms", "")
        mpt = r.get("ms_per_token_cuda", "")
        med_s = f"{med:.3f}" if isinstance(med, (int, float)) else str(med)
        mpt_s = f"{mpt:.4f}" if isinstance(mpt, (int, float)) else str(mpt)
        md.append(f"| {r['decode_step']} | {r['seq_len']} | {med_s} | {mpt_s} |")
    writer.append_report("\n".join(md))

    print_markdown_table(
        "Fixed Forward CUDA",
        ["seq_len", "median_ms", "std_ms", "ms_per_token"],
        [
            [r["seq_len"], r.get("median_ms", "OOM"), r.get("std_ms", "-"), r.get("ms_per_token_cuda", "-")]
            for r in fixed_rows
        ],
    )
    print_markdown_table(
        "Growing Decode Isolated CUDA",
        ["decode_step", "seq_len", "median_ms", "ms_per_token"],
        [[r["decode_step"], r["seq_len"], f"{r['median_ms']:.2f}", f"{r['ms_per_token_cuda']:.4f}"] for r in growing_rows],
    )
    print(f"Saved to {writer.latest_symlink_hint}")
    return {"fixed_forward": fixed_rows, "growing_decode": growing_rows}
