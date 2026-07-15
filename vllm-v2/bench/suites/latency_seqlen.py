"""Latency vs sequence length benchmark suite (v1 KV Cache)."""

from __future__ import annotations

import random

import torch

from bench.lib.config import BenchConfig
from bench.lib.report import ResultWriter, format_stats_row, print_markdown_table
from bench.lib.session import BenchSession, collect_meta
from bench.lib.timing import benchmark_cuda_safe, benchmark_wall, sync


def measure_fixed_prefill(session: BenchSession, cfg: BenchConfig, seq_len: int) -> dict:
    token_ids = session.make_prompt_tokens(seq_len)

    def _forward() -> None:
        session.prefill_forward_only(token_ids)

    stats, err = benchmark_cuda_safe(
        _forward,
        warmup=cfg.warmup_for_seq_len(seq_len),
        iters=cfg.iters_for_seq_len(seq_len),
    )
    if err == "oom":
        return {
            "mode": "prefill",
            "seq_len": seq_len,
            "status": "oom",
            "median_ms": "",
            "std_ms": "",
            "ms_per_token_cuda": "",
            "samples": 0,
        }

    wall_stats = benchmark_wall(_forward, warmup=2, iters=min(cfg.measure_iters, 10))

    row = format_stats_row(
        {"mode": "prefill", "seq_len": seq_len, "status": "ok"},
        stats,
    )
    row["wall_median_ms"] = round(wall_stats.median_ms, 3)
    row["ms_per_token_cuda"] = round(row["median_ms"] / seq_len, 4)
    session.clear_kv_state()
    return row


def measure_decode_at_cache_len(session: BenchSession, cfg: BenchConfig) -> list[dict]:
    """Single-token decode latency at various filled cache lengths."""
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
        prompt_ids = session.make_prompt_tokens(seq_len - 1)
        cache_len = len(prompt_ids)

        try:
            session.fill_kv_cache(prompt_ids)
        except torch.cuda.OutOfMemoryError:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            rows.append({
                "mode": "decode_step",
                "decode_step": seq_len - prompt_len,
                "seq_len": seq_len,
                "cache_len": cache_len,
                "status": "oom",
                "median_ms": "",
                "ms_per_token_cuda": "",
                "samples": 0,
            })
            continue
        sync()

        def _decode() -> None:
            session.decode_forward_only(cache_len)

        stats, err = benchmark_cuda_safe(
            _decode,
            warmup=min(cfg.warmup_for_seq_len(seq_len), 3),
            iters=cfg.iters_for_seq_len(seq_len),
        )
        if err == "oom":
            rows.append({
                "mode": "decode_step",
                "decode_step": seq_len - prompt_len,
                "seq_len": seq_len,
                "cache_len": cache_len,
                "status": "oom",
                "median_ms": "",
                "ms_per_token_cuda": "",
                "samples": 0,
            })
            continue

        decode_step = seq_len - prompt_len
        row = format_stats_row(
            {
                "mode": "decode_step",
                "decode_step": decode_step,
                "seq_len": seq_len,
                "cache_len": cache_len,
                "status": "ok",
            },
            stats,
        )
        row["ms_per_token_cuda"] = round(row["median_ms"], 4)
        rows.append(row)
        session.clear_kv_state()
    return rows


def run(cfg: BenchConfig, writer: ResultWriter | None = None, session: BenchSession | None = None) -> dict:
    session = session or BenchSession(cfg)
    meta = collect_meta(cfg)
    writer = writer or ResultWriter(cfg, meta)
    session.global_warmup()

    lengths = [n for n in cfg.latency_lengths if n <= cfg.context_len]

    # Decode sweep first: needs a clean allocator when run_all shares one session.
    decode_rows = measure_decode_at_cache_len(session, cfg)
    session.clear_kv_state()

    random.shuffle(lengths)
    prefill_rows = [measure_fixed_prefill(session, cfg, seq_len) for seq_len in lengths]
    prefill_rows.sort(key=lambda r: r["seq_len"])

    fields = [
        "mode", "seq_len", "decode_step", "cache_len", "status", "median_ms", "mean_ms", "std_ms", "p90_ms",
        "min_ms", "max_ms", "wall_median_ms", "ms_per_token_cuda", "samples",
    ]
    writer.write_csv("latency_prefill", prefill_rows, fields)
    writer.write_csv("latency_decode_step", decode_rows, fields)

    md = ["## Latency vs SeqLen (v2 PagedAttention)\n", "### Prefill (CUDA)\n",
          "| seq_len | median_ms | std_ms | ms_per_token |", "| --- | --- | --- | --- |"]
    for r in prefill_rows:
        med = r.get("median_ms", "")
        std = r.get("std_ms", "")
        mpt = r.get("ms_per_token_cuda", "")
        med_s = f"{med:.3f}" if isinstance(med, (int, float)) else str(med)
        std_s = f"{std:.3f}" if isinstance(std, (int, float)) else str(std)
        mpt_s = f"{mpt:.4f}" if isinstance(mpt, (int, float)) else str(mpt)
        md.append(f"| {r['seq_len']} | {med_s} | {std_s} | {mpt_s} |")
    md += ["\n### Decode Step (1 token, CUDA)\n",
           "| decode_step | cache_len | median_ms |", "| --- | --- | --- |"]
    for r in decode_rows:
        med = r.get("median_ms", "")
        med_s = f"{med:.3f}" if isinstance(med, (int, float)) else str(med)
        md.append(
            f"| {r['decode_step']} | {r.get('cache_len', '-')} | {med_s} |"
        )
    writer.append_report("\n".join(md))

    print_markdown_table(
        "Prefill CUDA",
        ["seq_len", "median_ms", "std_ms", "ms_per_token"],
        [
            [r["seq_len"], r.get("median_ms", "OOM"), r.get("std_ms", "-"), r.get("ms_per_token_cuda", "-")]
            for r in prefill_rows
        ],
    )
    print_markdown_table(
        "Decode Step CUDA",
        ["decode_step", "cache_len", "median_ms"],
        [[r["decode_step"], r.get("cache_len", "-"), f"{r['median_ms']:.2f}" if isinstance(r.get('median_ms'), (int, float)) else r.get('median_ms', 'OOM')] for r in decode_rows],
    )
    print(f"Saved to {writer.latest_symlink_hint}")
    return {"prefill": prefill_rows, "decode_step": decode_rows}
