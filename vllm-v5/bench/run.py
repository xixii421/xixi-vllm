#!/usr/bin/env python3
"""Single entry for micro-benchmarks built on ``bench/lib``.

Examples::

    PYTHONPATH=. python -m bench.run --suite e2e,ttft,decode
    PYTHONPATH=. python -m bench.run --suite all
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bench.lib.cli import add_common_args, config_from_args
from bench.lib.registry import ALL_SUITES, DEFAULT_SUITES, SUITE_REGISTRY
from bench.lib.report import ResultWriter
from bench.lib.session import BenchSession, collect_meta


def main() -> None:
    ap = argparse.ArgumentParser(description="vllm-v7 micro-bench (bench/lib)")
    add_common_args(ap)
    ap.add_argument(
        "--suite",
        default=",".join(DEFAULT_SUITES),
        help=f"comma list or 'all'. available: {','.join(ALL_SUITES)}",
    )
    args = ap.parse_args()
    cfg = config_from_args(args)
    names = (
        list(ALL_SUITES)
        if args.suite.strip().lower() == "all"
        else [s.strip() for s in args.suite.split(",") if s.strip()]
    )
    unknown = [n for n in names if n not in SUITE_REGISTRY]
    if unknown:
        raise SystemExit(f"unknown suite(s): {unknown}; choose from {ALL_SUITES}")

    meta = collect_meta(cfg)
    writer = ResultWriter(cfg, meta)
    session = BenchSession(cfg)
    try:
        for name in names:
            print(f"\n=== suite: {name} ===", flush=True)
            SUITE_REGISTRY[name](cfg, writer, session)
    finally:
        try:
            session.engine.destroy()
        except Exception:
            pass
    print(f"\nResults under {writer.run_dir}", flush=True)


if __name__ == "__main__":
    main()
