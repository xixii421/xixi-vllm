"""v0 vs v1 core benchmark: decode scaling, prefill scaling, prefix second hit.

Usage (from vllm-v1)::

    python -m bench.compare_v0 --model Qwen3-0.6B
    python -m bench.compare_v0 --model Qwen3-0.6B --max-new 32,128,256 --repeats 3
"""

from __future__ import annotations

import argparse
import gc

import torch

from .lib import (
    BenchEngine,
    RunMetrics,
    load_engine,
    make_prompt,
    measure,
    resolve_model,
    sync_device,
    timed_run,
)
from .report import PairResult, Reporter


def _parse_int_list(s: str) -> list[int]:
    return [int(x.strip()) for x in s.split(",") if x.strip()]


def _free(*engines: BenchEngine | None) -> None:
    for eng in engines:
        if eng is None:
            continue
        del eng._engine
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    sync_device()


def bench_decode_scaling(
    v0: BenchEngine,
    v1: BenchEngine,
    *,
    prompt: str,
    max_new_list: list[int],
    warmup: int,
    repeats: int,
    rep: Reporter,
) -> None:
    rep.banner("A) Decode scaling")
    pairs: list[PairResult] = []
    for n in max_new_list:
        rep.progress(f"max_new={n}")
        m0 = measure(v0, prompt, n, warmup=warmup, repeats=repeats)
        m1 = measure(v1, prompt, n, warmup=warmup, repeats=repeats)
        pair = PairResult(key=f"new={n}", baseline=m0, candidate=m1)
        pairs.append(pair)
        rep.pair_line(pair, focus="e2e")
    rep.pair_table("Summary: Decode scaling (v0 vs v1/none)", pairs, focus="e2e")


def bench_prefill_scaling(
    v0: BenchEngine,
    v1: BenchEngine,
    *,
    prompt_lens: list[int],
    max_new: int,
    warmup: int,
    repeats: int,
    rep: Reporter,
) -> None:
    rep.banner("B) Prefill scaling")
    encode = v1.encode
    pairs: list[PairResult] = []
    for plen in prompt_lens:
        prompt = make_prompt(target_tokens=plen, tokenizer_encode=encode)
        actual = len(encode(prompt))
        rep.progress(f"prompt≈{actual} tok, max_new={max_new}")
        m0 = measure(v0, prompt, max_new, warmup=warmup, repeats=repeats)
        m1 = measure(v1, prompt, max_new, warmup=warmup, repeats=repeats)
        pair = PairResult(key=f"p≈{actual}", baseline=m0, candidate=m1)
        pairs.append(pair)
        rep.pair_line(pair, focus="ttft")
    rep.pair_table("Summary: Prefill scaling (v0 vs v1/none)", pairs, focus="ttft")


def bench_prefix_second_hit(
    v1_hash: BenchEngine,
    *,
    prompt: str,
    max_new: int,
    warmup: int,
    repeats: int,
    rep: Reporter,
) -> None:
    rep.banner("C) Prefix second hit")
    first_runs: list[RunMetrics] = []
    second_runs: list[RunMetrics] = []
    for i in range(max(1, repeats)):
        rep.progress(f"trial {i + 1}/{repeats}")

        v1_hash.reset_kv("hash")
        for _ in range(max(0, warmup)):
            for _tok in v1_hash.generate(prompt, max_new_tokens=min(8, max_new)):
                pass
            sync_device()
            v1_hash.reset_kv("hash")

        first_runs.append(timed_run(v1_hash, prompt, max_new))
        second_runs.append(timed_run(v1_hash, prompt, max_new))

    first_runs.sort(key=lambda r: r.e2e_s)
    second_runs.sort(key=lambda r: r.e2e_s)
    r1 = first_runs[len(first_runs) // 2]
    r2 = second_runs[len(second_runs) // 2]
    m1 = RunMetrics(
        "v1/hash-1st", r1.prompt_tokens, r1.new_tokens, r1.e2e_s, r1.ttft_s, r1.cached_tokens
    )
    m2 = RunMetrics(
        "v1/hash-2nd", r2.prompt_tokens, r2.new_tokens, r2.e2e_s, r2.ttft_s, r2.cached_tokens
    )
    rep.single_table("Summary: Prefix first vs second", [m1, m2])
    ttft_x = m1.ttft_s / m2.ttft_s if m2.ttft_s > 0 else float("nan")
    print(
        f"  TTFT speedup (1st→2nd): {ttft_x:.2f}x  |  "
        f"cached {m1.cached_tokens} → {m2.cached_tokens}"
    )


def main() -> None:
    ap = argparse.ArgumentParser(description="Benchmark vllm-v1 (KV) against vllm-v0")
    ap.add_argument("--model", default="Qwen3-0.6B")
    ap.add_argument("--context-len", type=int, default=2048)
    ap.add_argument("--block-size", type=int, default=16)
    ap.add_argument("--warmup", type=int, default=1)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--max-new", default="32,64,128")
    ap.add_argument("--prompt-lens", default="64,256")
    ap.add_argument("--prefill-max-new", type=int, default=8)
    ap.add_argument("--skip-prefill", action="store_true")
    ap.add_argument("--skip-prefix", action="store_true")
    ap.add_argument("--compile", action="store_true")
    args = ap.parse_args()

    rep = Reporter()
    model = resolve_model(args.model)
    eager = not args.compile
    max_new_list = _parse_int_list(args.max_new)
    prompt_lens = _parse_int_list(args.prompt_lens)

    rep.banner("vllm-v0 vs vllm-v1 benchmark")
    print(f"  model     : {model}")
    print(f"  eager     : {eager}")
    print(f"  warmup    : {args.warmup}    repeats: {args.repeats}")
    print(f"  max_new   : {max_new_list}")
    print(f"  prompt≈   : {prompt_lens}")
    print()

    print("  loading v0 …", flush=True)
    v0 = load_engine("v0", model, enforce_eager=eager, context_len=args.context_len)
    print("  loading v1 (prefix=none) …", flush=True)
    v1 = load_engine(
        "v1",
        model,
        enforce_eager=eager,
        context_len=args.context_len,
        prefix_backend="none",
        block_size=args.block_size,
    )

    short_prompt = make_prompt(target_tokens=32, tokenizer_encode=v1.encode)
    print(f"  short_prompt_tokens={len(v1.encode(short_prompt))}")

    bench_decode_scaling(
        v0,
        v1,
        prompt=short_prompt,
        max_new_list=max_new_list,
        warmup=args.warmup,
        repeats=args.repeats,
        rep=rep,
    )

    if not args.skip_prefill:
        bench_prefill_scaling(
            v0,
            v1,
            prompt_lens=prompt_lens,
            max_new=args.prefill_max_new,
            warmup=args.warmup,
            repeats=args.repeats,
            rep=rep,
        )

    if not args.skip_prefix:
        _free(v0)
        print("\n  switch v1 → prefix=hash (no weight reload) …", flush=True)
        v1.reset_kv("hash")
        v1.label = "v1/hash"
        long_prompt = make_prompt(target_tokens=128, tokenizer_encode=v1.encode)
        bench_prefix_second_hit(
            v1,
            prompt=long_prompt,
            max_new=max(16, args.prefill_max_new),
            warmup=args.warmup,
            repeats=args.repeats,
            rep=rep,
        )

    rep.banner("How to read these numbers")

    print("BENCH_DONE")


if __name__ == "__main__":
    main()
