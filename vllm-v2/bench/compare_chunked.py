"""PD-interleave (chunked mix) concurrency bench.

Core metric for Prefill/Decode interleave
-----------------------------------------
NOT total wall throughput. PD mix exists so **in-flight decode requests are not
starved** when a long prefill arrives.

Primary metrics (section B, v2 only):

* ``decode_during_P`` — new tokens produced by short reqs **while** the long
  prompt is still prefilling. Prefill-first ≈ 0; mix should be ≫ 0.
* ``short_from_P`` — wall time from long-admit → all shorts finished.
  This is the fairness / decode-tail metric mix should improve.
* ``wall`` — full batch end (secondary; often does **not** improve).

Usage::

    python -m bench.compare_chunked --model Qwen3-0.6B
    python -m bench.compare_chunked --model Qwen3-0.6B --budget 32 --long-prompt 512
"""

from __future__ import annotations

import argparse
import gc
import time
from dataclasses import dataclass

import torch

from .lib import (
    BatchMetrics,
    BenchEngine,
    load_engine,
    make_prompt,
    measure_batch,
    resolve_model,
    speedup,
    sync_device,
)
from .report import Reporter


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


def _f(x: float, nd: int = 3) -> str:
    if x != x:
        return "n/a"
    return f"{x:.{nd}f}"


@dataclass(frozen=True)
class StaggerMetrics:
    label: str
    num_reqs: int
    # Primary (PD fairness)
    decode_during_prefill: int  # short completion toks while long still prefilling
    short_from_prefill_s: float  # long-admit → shorts done
    prefill_window_s: float  # long-admit → long prefill done
    # Secondary
    wall_s: float  # t0 → all done
    new_tokens: int


def timed_staggered_v2(
    engine: BenchEngine,
    *,
    short_prompt: str,
    long_prompt: str,
    n_short: int,
    short_new: int,
    long_new: int,
    warm_decode_steps: int,
) -> StaggerMetrics:
    eng = engine._engine
    from src.sampling_params import SamplingParams  # type: ignore

    sp_s = SamplingParams(temperature=0.0, max_tokens=short_new, ignore_eos=True)
    sp_l = SamplingParams(temperature=0.0, max_tokens=long_new, ignore_eos=True)

    sync_device()
    t0 = time.perf_counter()
    shorts = [
        eng.add_request(f"{short_prompt} [{i}]", sp_s) for i in range(n_short)
    ]
    while any(s.status.name == "WAITING" for s in shorts):
        eng.step()
    for _ in range(max(0, warm_decode_steps)):
        if all(s.is_finished for s in shorts):
            break
        eng.step()

    long = eng.add_request(long_prompt, sp_l)
    sync_device()
    t_admit = time.perf_counter()
    comp_at_admit = sum(s.num_completion_tokens for s in shorts)

    decode_during = 0
    prefill_window_s = float("nan")
    short_from_prefill_s: float | None = None
    prefill_marked = False

    while not eng.scheduler.is_finished():
        long_still_p = long.num_cached_tokens < long.num_prompt_tokens
        eng.step()
        if long_still_p and not prefill_marked:
            if long.num_cached_tokens >= long.num_prompt_tokens:
                sync_device()
                prefill_window_s = time.perf_counter() - t_admit
                decode_during = (
                    sum(s.num_completion_tokens for s in shorts) - comp_at_admit
                )
                prefill_marked = True
        if short_from_prefill_s is None and all(s.is_finished for s in shorts):
            sync_device()
            short_from_prefill_s = time.perf_counter() - t_admit

    sync_device()
    wall = time.perf_counter() - t0
    if not prefill_marked:
        # Long finished in the same step it was admitted (tiny prompt).
        prefill_window_s = 0.0
        decode_during = sum(s.num_completion_tokens for s in shorts) - comp_at_admit
    if short_from_prefill_s is None:
        short_from_prefill_s = time.perf_counter() - t_admit

    return StaggerMetrics(
        label=engine.label,
        num_reqs=n_short + 1,
        decode_during_prefill=max(0, decode_during),
        short_from_prefill_s=short_from_prefill_s,
        prefill_window_s=prefill_window_s,
        wall_s=wall,
        new_tokens=n_short * short_new + long_new,
    )


def timed_staggered_seq(
    engine: BenchEngine,
    *,
    short_prompt: str,
    long_prompt: str,
    n_short: int,
    short_new: int,
    long_new: int,
    warm_decode_steps: int = 0,
) -> StaggerMetrics:
    """v0/v1: no shared scheduler — sequential baseline (cannot overlap)."""
    del warm_decode_steps
    sync_device()
    t0 = time.perf_counter()
    for i in range(n_short):
        for _ in engine.generate(
            f"{short_prompt} [{i}]", max_new_tokens=short_new, temperature=0.0
        ):
            pass
    t_shorts = time.perf_counter() - t0
    for _ in engine.generate(long_prompt, max_new_tokens=long_new, temperature=0.0):
        pass
    sync_device()
    wall = time.perf_counter() - t0
    return StaggerMetrics(
        label=engine.label,
        num_reqs=n_short + 1,
        decode_during_prefill=0,  # no overlap possible
        short_from_prefill_s=t_shorts,  # not comparable; shown for reference
        prefill_window_s=float("nan"),
        wall_s=wall,
        new_tokens=n_short * short_new + long_new,
    )


def timed_staggered(engine: BenchEngine, **kw) -> StaggerMetrics:
    if engine.version == "v2":
        return timed_staggered_v2(engine, **kw)
    return timed_staggered_seq(engine, **kw)


def measure_staggered(engine: BenchEngine, **kw) -> StaggerMetrics:
    warmup = kw.pop("warmup", 1)
    repeats = kw.pop("repeats", 3)
    for _ in range(max(0, warmup)):
        timed_staggered(engine, **kw)
        sync_device()
        if engine.version != "v0":
            engine.reset_kv()
    runs = [timed_staggered(engine, **kw) for _ in range(max(1, repeats))]
    if engine.version != "v0":
        engine.reset_kv()
    # Prefer more decode-during-prefill, then lower short latency.
    runs.sort(key=lambda r: (-r.decode_during_prefill, r.short_from_prefill_s, r.wall_s))
    return runs[len(runs) // 2]


def _print_batch_rows(title: str, rows: list[BatchMetrics], baseline: BatchMetrics) -> None:
    print(f"\n  {title}")
    hdr = f"  {'label':<14}{'reqs':>6}{'wall_s':>10}{'tok/s':>10}{'vs_base':>8}"
    print(hdr)
    print("  " + "─" * (len(hdr) - 2))
    for r in rows:
        tps = r.new_tokens / r.wall_s if r.wall_s > 0 else float("nan")
        print(
            f"  {r.label:<14}{r.num_reqs:>6}{_f(r.wall_s):>10}"
            f"{tps:>10.1f}{speedup(baseline.wall_s, r.wall_s):>8}"
        )
    print()


def _print_stagger_rows(title: str, rows: list[StaggerMetrics]) -> None:
    print(f"\n  {title}")
    print(
        "  (PD core: decode_during_P ≫ 0 and lower short_from_P under mix; "
        "wall is secondary)"
    )
    hdr = (
        f"  {'label':<14}{'dec@P':>7}{'P_win':>8}{'short_P':>9}"
        f"{'wall':>9}{'short×':>8}"
    )
    print(hdr)
    print("  " + "─" * (len(hdr) - 2))
    # Compare short_from_P against v2/pfirst when present.
    pfirst = next((r for r in rows if "pfirst" in r.label), rows[0])
    for r in rows:
        print(
            f"  {r.label:<14}{r.decode_during_prefill:>7}"
            f"{_f(r.prefill_window_s):>8}{_f(r.short_from_prefill_s):>9}"
            f"{_f(r.wall_s):>9}"
            f"{speedup(pfirst.short_from_prefill_s, r.short_from_prefill_s):>8}"
        )
    print()


def main() -> None:
    ap = argparse.ArgumentParser(
        description="PD interleave bench (fairness metric, not throughput)"
    )
    ap.add_argument("--model", default="Qwen3-0.6B")
    ap.add_argument("--context-len", type=int, default=2048)
    ap.add_argument("--warmup", type=int, default=0)
    ap.add_argument("--repeats", type=int, default=2)
    ap.add_argument("--conc", default="2,4", help="short concurrent request counts")
    # Small budget + long prompt → long starve window under prefill-first.
    ap.add_argument("--budget", type=int, default=32, help="max_num_batched_tokens")
    ap.add_argument("--short-prompt", type=int, default=16)
    ap.add_argument("--long-prompt", type=int, default=512)
    ap.add_argument("--short-new", type=int, default=16)
    ap.add_argument("--long-new", type=int, default=4)
    ap.add_argument("--batch-new", type=int, default=16)
    ap.add_argument("--warm-decode", type=int, default=3)
    ap.add_argument("--skip-uniform", action="store_true")
    args = ap.parse_args()

    rep = Reporter()
    model = resolve_model(args.model)
    conc_list = _parse_int_list(args.conc)
    max_seqs = max(conc_list) + 2

    rep.banner("PD interleave — what we optimize")
    print("  Target : in-flight decode latency / fairness when long prefill arrives")
    print("  NOT    : total batch wall throughput (same token work either way)")
    print(f"  model  : {model}")
    print(f"  budget : {args.budget}  long_prompt≈{args.long_prompt}  short_new={args.short_new}")
    print(f"  starve window ≈ ceil(long/budget) prefill-only steps under pfirst")
    print()

    print("  loading engines …", flush=True)
    v0 = load_engine("v0", model, context_len=args.context_len)
    v1 = load_engine("v1", model, context_len=args.context_len, prefix_backend="none")
    v2_legacy = load_engine(
        "v2",
        model,
        context_len=args.context_len,
        prefix_backend="none",
        max_num_seqs=max_seqs,
        max_num_batched_tokens=args.budget,
        mix_prefill_decode=False,
    )
    v2_mix = load_engine(
        "v2",
        model,
        context_len=args.context_len,
        prefix_backend="none",
        max_num_seqs=max_seqs,
        max_num_batched_tokens=args.budget,
        mix_prefill_decode=True,
        reuse_src=True,
    )

    enc = v2_mix.encode
    short_p = make_prompt(target_tokens=args.short_prompt, tokenizer_encode=enc)
    long_p = make_prompt(target_tokens=args.long_prompt, tokenizer_encode=enc)
    long_tok = len(enc(long_p))
    print(
        f"  actual short/long tokens: {len(enc(short_p))}/{long_tok}  "
        f"→ pfirst starve steps ≈ {(long_tok + args.budget - 1) // args.budget}",
        flush=True,
    )

    if not args.skip_uniform:
        rep.banner("A) Uniform batch (throughput — CB vs sequential; mix≈pfirst)")
        for n in conc_list:
            prompts = [f"{short_p} [{i}]" for i in range(n)]
            rep.progress(f"n={n}")
            rows = [
                measure_batch(
                    v0, prompts, args.batch_new, warmup=args.warmup, repeats=args.repeats
                ),
                measure_batch(
                    v1, prompts, args.batch_new, warmup=args.warmup, repeats=args.repeats
                ),
                measure_batch(
                    v2_legacy,
                    prompts,
                    args.batch_new,
                    warmup=args.warmup,
                    repeats=args.repeats,
                ),
                measure_batch(
                    v2_mix,
                    prompts,
                    args.batch_new,
                    warmup=args.warmup,
                    repeats=args.repeats,
                ),
            ]
            _print_batch_rows(f"n={n} uniform", rows, rows[0])

    rep.banner("B) Staggered PD (fairness — mix should win on dec@P / short_P)")
    for n in conc_list:
        rep.progress(f"n_short={n}")
        kw = dict(
            short_prompt=short_p,
            long_prompt=long_p,
            n_short=n,
            short_new=args.short_new,
            long_new=args.long_new,
            warm_decode_steps=args.warm_decode,
            warmup=args.warmup,
            repeats=args.repeats,
        )
        rows = [
            measure_staggered(v0, **kw),
            measure_staggered(v1, **kw),
            measure_staggered(v2_legacy, **kw),
            measure_staggered(v2_mix, **kw),
        ]
        _print_stagger_rows(f"n_short={n}", rows)
        pfirst, mix = rows[2], rows[3]
        print(
            f"  mix vs pfirst:  dec@P {pfirst.decode_during_prefill}→{mix.decode_during_prefill}"
            f"  short_P {speedup(pfirst.short_from_prefill_s, mix.short_from_prefill_s)}"
            f"  wall {speedup(pfirst.wall_s, mix.wall_s)}"
        )

    rep.banner("How to read")
    print("  · dec@P  : tokens short reqs decoded during long's prefill (mix ≫ pfirst≈0)")
    print("  · short_P: time from long-admit to shorts done (fairness / decode tail)")
    print("  · wall   : total batch end — may be flat or worse (extra forward cost)")
    print("  · If dec@P mix>0 but short_P not faster: dual-forward teaching tax dominates")
    print("BENCH_DONE")
    _free(v0, v1, v2_legacy, v2_mix)


if __name__ == "__main__":
    main()
