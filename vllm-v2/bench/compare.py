"""v0 / v1 / v2 three-way benchmark.

Scenarios:
  A) Single-stream decode scaling — fair apples-to-apples (v2≈v1 when batch=1)
  B) Prefill scaling — TTFT focus
  C) Continuous batching — v0/v1 sequential vs v2 generate_batch (v2's win)
  D) Prefix second hit — v1 vs v2 hash

Usage (from vllm-v2)::

    python -m bench.compare --model Qwen3-0.6B
    python -m bench.compare --model Qwen3-0.6B --batch-size 4 --max-new 32,64
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
    measure_batch,
    resolve_model,
    sync_device,
    timed_run,
)
from .report import Reporter, TripleResult


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


def bench_decode(
    v0: BenchEngine,
    v1: BenchEngine,
    v2: BenchEngine,
    *,
    prompt: str,
    max_new_list: list[int],
    warmup: int,
    repeats: int,
    rep: Reporter,
) -> None:
    rep.banner("A) Single-stream decode (v0 / v1 / v2)")
    rows: list[TripleResult] = []
    for n in max_new_list:
        rep.progress(f"max_new={n}")
        t = TripleResult(
            key=f"new={n}",
            v0=measure(v0, prompt, n, warmup=warmup, repeats=repeats),
            v1=measure(v1, prompt, n, warmup=warmup, repeats=repeats),
            v2=measure(v2, prompt, n, warmup=warmup, repeats=repeats),
        )
        rows.append(t)
        rep.triple_line(t, focus="e2e")
    rep.triple_table("Summary: decode E2E (× vs v0; v2/v1)", rows, focus="e2e")


def bench_prefill(
    v0: BenchEngine,
    v1: BenchEngine,
    v2: BenchEngine,
    *,
    prompt_lens: list[int],
    max_new: int,
    warmup: int,
    repeats: int,
    rep: Reporter,
) -> None:
    rep.banner("B) Prefill scaling (TTFT)")
    encode = v2.encode
    rows: list[TripleResult] = []
    for plen in prompt_lens:
        prompt = make_prompt(target_tokens=plen, tokenizer_encode=encode)
        actual = len(encode(prompt))
        rep.progress(f"prompt≈{actual}, max_new={max_new}")
        t = TripleResult(
            key=f"p≈{actual}",
            v0=measure(v0, prompt, max_new, warmup=warmup, repeats=repeats),
            v1=measure(v1, prompt, max_new, warmup=warmup, repeats=repeats),
            v2=measure(v2, prompt, max_new, warmup=warmup, repeats=repeats),
        )
        rows.append(t)
        rep.triple_line(t, focus="ttft")
    rep.triple_table("Summary: prefill TTFT", rows, focus="ttft")


def bench_cb(
    v0: BenchEngine,
    v1: BenchEngine,
    v2: BenchEngine,
    *,
    prompt: str,
    batch_size: int,
    max_new: int,
    warmup: int,
    repeats: int,
    rep: Reporter,
) -> None:
    rep.banner("C) Continuous batching (v2 win vs sequential v0/v1)")
    # Slightly vary suffixes so requests are independent (no accidental prefix share).
    prompts = [f"{prompt} [{i}]" for i in range(batch_size)]
    rep.progress(f"batch={batch_size}, max_new={max_new}")
    m0 = measure_batch(v0, prompts, max_new, warmup=warmup, repeats=repeats)
    m1 = measure_batch(v1, prompts, max_new, warmup=warmup, repeats=repeats)
    m2 = measure_batch(v2, prompts, max_new, warmup=warmup, repeats=repeats)
    rep.batch_table(
        "Summary: multi-request wall time (v0/v1=sequential, v2=CB)",
        [m0, m1, m2],
        v0_wall=m0.wall_s,
    )
    print(
        f"  CB gain vs v1 sequential: {m1.wall_s / m2.wall_s:.2f}x  "
        f"(wall {m1.wall_s:.3f}s → {m2.wall_s:.3f}s)"
        if m2.wall_s > 0
        else "  CB gain: n/a"
    )


def bench_prefix(
    v1: BenchEngine,
    v2: BenchEngine,
    *,
    prompt: str,
    max_new: int,
    warmup: int,
    repeats: int,
    rep: Reporter,
) -> None:
    rep.banner("D) Prefix second hit (hash)")

    def _pair(eng: BenchEngine, tag: str) -> tuple[RunMetrics, RunMetrics]:
        first: list[RunMetrics] = []
        second: list[RunMetrics] = []
        for _ in range(max(1, repeats)):
            eng.reset_kv("hash")
            for _w in range(max(0, warmup)):
                for _tok in eng.generate(prompt, max_new_tokens=min(8, max_new)):
                    pass
                sync_device()
                eng.reset_kv("hash")
            first.append(timed_run(eng, prompt, max_new))
            second.append(timed_run(eng, prompt, max_new))
        first.sort(key=lambda r: r.e2e_s)
        second.sort(key=lambda r: r.e2e_s)
        a, b = first[len(first) // 2], second[len(second) // 2]
        return (
            RunMetrics(f"{tag}-1st", a.prompt_tokens, a.new_tokens, a.e2e_s, a.ttft_s, a.cached_tokens),
            RunMetrics(f"{tag}-2nd", b.prompt_tokens, b.new_tokens, b.e2e_s, b.ttft_s, b.cached_tokens),
        )

    rep.progress("v1 hash")
    v1_1, v1_2 = _pair(v1, "v1/hash")
    rep.progress("v2 hash")
    v2_1, v2_2 = _pair(v2, "v2/hash")
    rep.single_table("Summary: prefix 1st vs 2nd", [v1_1, v1_2, v2_1, v2_2])


def main() -> None:
    ap = argparse.ArgumentParser(description="Benchmark v0 / v1 / v2")
    ap.add_argument("--model", default="Qwen3-0.6B")
    ap.add_argument("--context-len", type=int, default=2048)
    ap.add_argument("--block-size", type=int, default=16)
    ap.add_argument("--max-num-seqs", type=int, default=8)
    ap.add_argument("--warmup", type=int, default=1)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--max-new", default="32,64,128")
    ap.add_argument("--prompt-lens", default="64,256")
    ap.add_argument("--prefill-max-new", type=int, default=8)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--batch-max-new", type=int, default=32)
    ap.add_argument("--skip-prefill", action="store_true")
    ap.add_argument("--skip-batch", action="store_true")
    ap.add_argument("--skip-prefix", action="store_true")
    ap.add_argument("--compile", action="store_true")
    args = ap.parse_args()

    rep = Reporter()
    model = resolve_model(args.model)
    eager = not args.compile
    max_new_list = _parse_int_list(args.max_new)
    prompt_lens = _parse_int_list(args.prompt_lens)

    rep.banner("vllm-v0 / v1 / v2 benchmark")
    print(f"  model       : {model}")
    print(f"  eager       : {eager}")
    print(f"  warmup/rep  : {args.warmup}/{args.repeats}")
    print(f"  max_new     : {max_new_list}")
    print(f"  batch_size  : {args.batch_size}")
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
    print("  loading v2 (prefix=none) …", flush=True)
    v2 = load_engine(
        "v2",
        model,
        enforce_eager=eager,
        context_len=args.context_len,
        prefix_backend="none",
        block_size=args.block_size,
        max_num_seqs=max(args.max_num_seqs, args.batch_size),
    )

    short = make_prompt(target_tokens=32, tokenizer_encode=v2.encode)
    print(f"  short_prompt_tokens={len(v2.encode(short))}")

    bench_decode(
        v0,
        v1,
        v2,
        prompt=short,
        max_new_list=max_new_list,
        warmup=args.warmup,
        repeats=args.repeats,
        rep=rep,
    )

    if not args.skip_prefill:
        bench_prefill(
            v0,
            v1,
            v2,
            prompt_lens=prompt_lens,
            max_new=args.prefill_max_new,
            warmup=args.warmup,
            repeats=args.repeats,
            rep=rep,
        )

    if not args.skip_batch:
        bench_cb(
            v0,
            v1,
            v2,
            prompt=short,
            batch_size=args.batch_size,
            max_new=args.batch_max_new,
            warmup=args.warmup,
            repeats=args.repeats,
            rep=rep,
        )

    if not args.skip_prefix:
        print("\n  switch v1/v2 → prefix=hash …", flush=True)
        v1.reset_kv("hash")
        v1.label = "v1/hash"
        v2.reset_kv("hash")
        v2.label = "v2/hash"
        long_p = make_prompt(target_tokens=128, tokenizer_encode=v2.encode)
        bench_prefix(
            v1,
            v2,
            prompt=long_p,
            max_new=max(16, args.prefill_max_new),
            warmup=args.warmup,
            repeats=args.repeats,
            rep=rep,
        )

    rep.banner("How to read")
    print("  · A/B: single stream — v2 ≈ v1 (CB idle); both usually slower than v0 (teaching PA).")
    print("  · C: multi-request — v2 CB should beat sequential v0/v1 wall time.")
    print("  · D: 2nd hit TTFT drop + cached tokens > 0.")
    print("BENCH_DONE")


if __name__ == "__main__":
    main()
