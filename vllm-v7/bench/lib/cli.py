"""Shared CLI argument helpers for benchmark entry points."""

from __future__ import annotations

import argparse

from bench.lib.config import BenchConfig, DEFAULT_MODEL, DEFAULT_PROMPT


def add_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--model", default=DEFAULT_MODEL, help="模型名或路径")
    parser.add_argument("--prompt", default=DEFAULT_PROMPT, help="e2e/ttft/vram 等使用的提示词")
    parser.add_argument("--max-new-tokens", type=int, default=128, help="generate 最多新 token 数")
    parser.add_argument(
        "--context-len",
        type=int,
        default=BenchConfig().context_len,
        help="最大上下文 M（连续方案按 R×M 预约）",
    )
    parser.add_argument(
        "--block-size",
        type=int,
        default=256,
        help="页大小 B（token/页，FlashInfer 需 256 倍数）",
    )
    parser.add_argument(
        "--max-num-seqs",
        type=int,
        default=4,
        help="Engine 共享 KV 池按 ceil(M/B)×该值 预留上限",
    )
    parser.add_argument("--warmup", type=int, default=10, help="计时前热身次数")
    parser.add_argument("--iters", type=int, default=25, help="正式计时次数")


def config_from_args(args: argparse.Namespace) -> BenchConfig:
    max_seq = args.context_len
    prompt_len = BenchConfig().latency_prompt_len
    return BenchConfig(
        model=args.model,
        prompt=args.prompt,
        max_new_tokens=args.max_new_tokens,
        context_len=max_seq,
        block_size=args.block_size,
        max_num_seqs=args.max_num_seqs,
        warmup_iters=args.warmup,
        measure_iters=args.iters,
        latency_decode_steps=max_seq - prompt_len,
    )
