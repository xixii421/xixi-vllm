"""Decode microbench: paged gather cost vs growing history length.

Measures one decode forward (write 1 slot + gather seq_len + SDPA) after
filling a paged cache to ``cache_len``. Teaching goal: show that teaching
v2 decode still gathers the full prefix each step (SDPA backend), while
preserving Prefill/Decode semantics of v1.
"""

from __future__ import annotations

from bench.lib.config import BenchConfig
from bench.lib.report import ResultWriter, print_markdown_table
from bench.lib.session import BenchSession, collect_meta
from bench.lib.timing import benchmark_cuda_safe


def run(cfg: BenchConfig, writer: ResultWriter | None = None, session: BenchSession | None = None) -> dict:
    session = session or BenchSession(cfg)
    meta = collect_meta(cfg)
    writer = writer or ResultWriter(cfg, meta)

    lengths = sorted({n for n in cfg.latency_lengths if 0 < n < cfg.context_len})
    if not lengths:
        lengths = [128, 256, 512]

    rows: list[dict] = []
    for cache_len in lengths:
        prompt_ids = session.make_prompt_tokens(cache_len)
        session.fill_kv_cache(prompt_ids)

        def _one() -> None:
            session.decode_forward_only(cache_len)

        stats, err = benchmark_cuda_safe(
            _one,
            warmup=cfg.warmup_for_seq_len(cache_len),
            iters=cfg.iters_for_seq_len(cache_len),
        )
        pages = (cache_len + session.engine.block_size - 1) // session.engine.block_size
        if err == "oom" or stats is None:
            rows.append(
                {
                    "cache_len": cache_len,
                    "pages": pages,
                    "block_size": session.engine.block_size,
                    "status": "oom",
                    "median_ms": "",
                    "mean_ms": "",
                    "std_ms": "",
                }
            )
        else:
            rows.append(
                {
                    "cache_len": cache_len,
                    "pages": pages,
                    "block_size": session.engine.block_size,
                    "status": "ok",
                    "median_ms": round(stats.median_ms, 4),
                    "mean_ms": round(stats.mean_ms, 4),
                    "std_ms": round(stats.std_ms, 4),
                }
            )
        session.clear_kv_state()

    writer.write_csv(
        "paged_overhead",
        rows,
        ["cache_len", "pages", "block_size", "status", "median_ms", "mean_ms", "std_ms"],
    )
    md = [
        "## PagedAttention Decode Overhead (gather + SDPA)\n",
        "| cache_len | pages | status | median_ms | std_ms |",
        "| --- | --- | --- | --- | --- |",
    ]
    table_rows = []
    for r in rows:
        med = r["median_ms"] if r["status"] == "ok" else "oom"
        std = r["std_ms"] if r["status"] == "ok" else ""
        md.append(f"| {r['cache_len']} | {r['pages']} | {r['status']} | {med} | {std} |")
        table_rows.append([r["cache_len"], r["pages"], r["status"], med, std])
    writer.append_report("\n".join(md) + "\n")
    print_markdown_table(
        "Paged decode gather overhead",
        ["cache_len", "pages", "status", "median_ms", "std_ms"],
        table_rows,
    )
    print(f"Saved to {writer.latest_symlink_hint}")
    return {"rows": rows}
