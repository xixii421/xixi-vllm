"""Human-readable and machine-readable benchmark reporting."""

from __future__ import annotations

import csv
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any

from .config import RESULTS_ROOT, BenchConfig
from .session import BenchMeta


class ResultWriter:
    """Write CSV tables and Markdown summaries into ``bench/results/``.

    All runs share one output directory (no timestamp subfolders).
    Each ``run_all`` invocation overwrites ``meta.csv`` and, by default,
    starts a fresh ``report.md``; individual suite CSVs are overwritten
    when that suite runs.
    """

    def __init__(self, cfg: BenchConfig, meta: BenchMeta, *, fresh_report: bool = True) -> None:
        self.run_dir = RESULTS_ROOT
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.cfg = cfg
        self.meta = meta
        self._report_started = False

        if fresh_report:
            report_path = self.run_dir / "report.md"
            if report_path.exists():
                report_path.unlink()

        self._write_meta()

    def _write_meta(self) -> None:
        meta_path = self.run_dir / "meta.csv"
        with open(meta_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["key", "value"])
            writer.writerow(["run_timestamp", self.run_timestamp])
            for key, value in asdict(self.meta).items():
                writer.writerow([key, value])
            writer.writerow(["prompt", self.cfg.prompt])
            writer.writerow(["max_new_tokens", self.cfg.max_new_tokens])
            writer.writerow(["context_len", self.cfg.context_len])
            writer.writerow(["prefill_lengths", ",".join(str(x) for x in self.cfg.prefill_lengths)])

    def write_csv(self, name: str, rows: list[dict[str, Any]], fieldnames: list[str]) -> Path:
        path = self.run_dir / f"{name}.csv"
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)
        return path

    def write_markdown(self, name: str, content: str) -> Path:
        path = self.run_dir / f"{name}.md"
        path.write_text(content, encoding="utf-8")
        return path

    def append_report(self, section_md: str) -> None:
        report_path = self.run_dir / "report.md"
        if not self._report_started:
            header = (
                f"# Benchmark Report\n\n"
                f"| Key | Value |\n| --- | --- |\n"
                f"| Run time | {self.run_timestamp} |\n"
                f"| Model | {self.meta.model} |\n"
                f"| GPU | {self.meta.gpu} |\n"
                f"| VRAM (GB) | {self.meta.vram_gb} |\n"
                f"| PyTorch | {self.meta.torch_version} |\n"
                f"| CUDA | {self.meta.cuda_version} |\n"
                f"| Warmup | {self.meta.warmup_iters} |\n"
                f"| Measure | {self.meta.measure_iters} |\n"
            )
            report_path.write_text(header + "\n\n" + section_md, encoding="utf-8")
            self._report_started = True
        else:
            existing = report_path.read_text(encoding="utf-8")
            report_path.write_text(existing + "\n\n" + section_md, encoding="utf-8")

    @property
    def latest_symlink_hint(self) -> str:
        return str(self.run_dir)


def format_stats_row(prefix: dict[str, Any], stats, *, unit: str = "ms") -> dict[str, Any]:
    return {
        **prefix,
        f"median_{unit}": round(stats.median_ms, 3),
        f"mean_{unit}": round(stats.mean_ms, 3),
        f"std_{unit}": round(stats.std_ms, 3),
        f"p90_{unit}": round(stats.p90_ms, 3),
        f"min_{unit}": round(stats.min_ms, 3),
        f"max_{unit}": round(stats.max_ms, 3),
        "samples": stats.n,
    }


def print_markdown_table(title: str, headers: list[str], rows: list[list[Any]]) -> None:
    print_table(title, headers, rows)


def print_table(
    title: str,
    headers: list[str],
    rows: list[list[Any]],
    *,
    footnote: str = "",
) -> None:
    print(f"\n{'=' * 72}")
    print(title)
    print(f"{'=' * 72}")

    def _w(s: str) -> int:
        n = 0
        for ch in str(s):
            n += 2 if ord(ch) > 127 else 1
        return n

    widths = [_w(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], _w(cell))

    def _pad(s: Any, width: int) -> str:
        s = str(s)
        pad = width - _w(s)
        return s + " " * max(0, pad)

    print("  ".join(_pad(h, widths[i]) for i, h in enumerate(headers)))
    print("  ".join("-" * widths[i] for i in range(len(headers))))
    for row in rows:
        print("  ".join(_pad(c, widths[i]) for i, c in enumerate(row)))
    if footnote:
        print()
        print(footnote)
