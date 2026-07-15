"""Peak VRAM benchmark suite."""

from __future__ import annotations

from bench.lib.config import BenchConfig
from bench.lib.report import ResultWriter, print_markdown_table
from bench.lib.session import BenchSession, collect_meta
from bench.lib.timing import sync


def run(cfg: BenchConfig, writer: ResultWriter | None = None, session: BenchSession | None = None) -> dict:
    session = session or BenchSession(cfg)
    meta = collect_meta(cfg)
    writer = writer or ResultWriter(cfg, meta)

    session.reset_peak_memory()
    sync()
    mem_after_load_gb = session.allocated_memory_gb()

    session.reset_peak_memory()
    n = 0
    for _ in session.engine.generate(
        cfg.prompt, max_new_tokens=cfg.max_new_tokens, temperature=0.0
    ):
        n += 1
    sync()
    peak_gen_gb = session.peak_memory_gb()

    row = {
        "prompt": cfg.prompt,
        "max_new_tokens": cfg.max_new_tokens,
        "generated_tokens": n,
        "mem_after_load_gb": round(mem_after_load_gb, 4),
        "peak_during_gen_gb": round(peak_gen_gb, 4),
    }
    writer.write_csv(
        "vram",
        [row],
        ["prompt", "max_new_tokens", "generated_tokens",
         "mem_after_load_gb", "peak_during_gen_gb"],
    )
    writer.append_report(
        "## Peak VRAM\n\n"
        f"| after_load_gb | peak_gen_gb | generated |\n"
        f"| --- | --- | --- |\n"
        f"| {row['mem_after_load_gb']:.4f} | {row['peak_during_gen_gb']:.4f} | {n} |\n"
    )
    print_markdown_table(
        "Peak VRAM",
        ["after_load_gb", "peak_gen_gb", "generated"],
        [[f"{row['mem_after_load_gb']:.4f}", f"{row['peak_during_gen_gb']:.4f}", n]],
    )
    print(f"Saved to {writer.latest_symlink_hint}")
    return row
