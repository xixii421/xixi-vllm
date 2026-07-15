"""Unified benchmark entry point for vllm-v1 (KV Cache)."""

from __future__ import annotations

import argparse

from bench.lib.cli import add_common_args, config_from_args
from bench.lib.config import BenchConfig, DEFAULT_LENGTH_SWEEP
from bench.lib.registry import ALL_SUITES, SUITE_REGISTRY
from bench.lib.report import ResultWriter
from bench.lib.session import BenchSession, collect_meta


def run_suites(cfg: BenchConfig, suite_names: list[str]) -> ResultWriter:
    print(f"Loading {cfg.model} ...")
    meta = collect_meta(cfg)
    writer = ResultWriter(cfg, meta)

    shared_names = [n for n in suite_names if n != "latency_seqlen"]
    session: BenchSession | None = None
    if shared_names:
        session = BenchSession(cfg)
        session.global_warmup()

    for name in suite_names:
        print(f"\n>>> Running {name}")
        if name == "latency_seqlen":
            SUITE_REGISTRY[name](cfg, writer, session=None)
        else:
            assert session is not None
            SUITE_REGISTRY[name](cfg, writer, session=session)

    print(f"\nBenchmarks done.\nReport: {writer.run_dir / 'report.md'}")
    return writer


def main() -> None:
    parser = argparse.ArgumentParser(description="vllm-v1 benchmarks (KV Cache)")
    add_common_args(parser)
    parser.add_argument(
        "--suite",
        choices=["all", *ALL_SUITES],
        default="all",
        help="run one suite or all (default: all)",
    )
    parser.add_argument(
        "--lengths",
        type=int,
        nargs="+",
        default=DEFAULT_LENGTH_SWEEP,
        help="seq lengths for prefill / latency suites",
    )
    parser.add_argument("--prompt-len", type=int, default=BenchConfig().latency_prompt_len)
    parser.add_argument("--decode-steps", type=int, default=None)
    parser.add_argument("--sample-every", type=int, default=BenchConfig().latency_sample_every)
    parser.set_defaults(max_new_tokens=64)
    args = parser.parse_args()

    cfg = config_from_args(args)
    cfg.prefill_lengths = args.lengths
    cfg.latency_lengths = args.lengths
    cfg.latency_prompt_len = args.prompt_len
    if args.decode_steps is not None:
        cfg.latency_decode_steps = args.decode_steps
    else:
        cfg.latency_decode_steps = cfg.context_len - args.prompt_len

    suite_names = ALL_SUITES if args.suite == "all" else [args.suite]
    run_suites(cfg, suite_names)


if __name__ == "__main__":
    main()
