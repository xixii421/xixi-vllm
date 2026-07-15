"""Shared CLI argument helpers for benchmark entry points."""

from __future__ import annotations

import argparse

from bench.lib.config import BenchConfig, DEFAULT_MODEL, DEFAULT_PROMPT


def add_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--context-len", type=int, default=BenchConfig().context_len)
    parser.add_argument("--warmup", type=int, default=10, help="warmup iterations per measurement")
    parser.add_argument("--iters", type=int, default=25, help="timed iterations per measurement")


def config_from_args(args: argparse.Namespace) -> BenchConfig:
    max_seq = args.context_len
    prompt_len = BenchConfig().latency_prompt_len
    return BenchConfig(
        model=args.model,
        prompt=args.prompt,
        max_new_tokens=args.max_new_tokens,
        context_len=max_seq,
        warmup_iters=args.warmup,
        measure_iters=args.iters,
        latency_decode_steps=max_seq - prompt_len,
    )
