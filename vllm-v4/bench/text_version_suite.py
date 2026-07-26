#!/usr/bin/env python3
"""Text-only TTFT / decode / E2E / concurrency: v0–v5 + official vLLM.

Expects PYTHONPATH=<repo>/vllm-<ver> (package root with ``src``) or the
official venv. Metrics use the same wall-clock method for every backend:

  TTFT         = time to first generated token
  decode_tok/s = (gen-1) / (e2e - TTFT)   [prefix cache OFF]
  e2e_tok/s    = gen / e2e_wall
  e2e_latency  = full generate wall time
  conc n       = continuous batch when available; else serial×n (flagged)
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import torch


def sync() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def mean(xs: list[float]) -> float:
    return float(statistics.mean(xs)) if xs else 0.0


def peak_gb() -> float:
    if not torch.cuda.is_available():
        return 0.0
    return torch.cuda.max_memory_allocated() / (1024**3)


PROMPT = (
    "Explain continuous batching for LLM inference in clear prose. "
    "Cover scheduling, KV cache paging, and why decode throughput scales with "
    "concurrency. Keep the answer factual and structured. "
) * 4


def pack_metrics(ttft: float, total: float, n: int, prompt_tokens: int) -> dict:
    if total > ttft * 1.02 and n > 1:
        decode_s = total - ttft
        decode_tok_s = (n - 1) / decode_s
    else:
        decode_s = total
        decode_tok_s = n / total if total > 0 else 0.0
    return {
        "ttft_s": ttft,
        "prefill_s": ttft,
        "decode_s": decode_s,
        "total_s": total,
        "decode_tok_s": decode_tok_s,
        "e2e_tok_s": n / total if total > 0 else 0.0,
        "e2e_latency_s": total,
        "gen_tokens": float(n),
        "prompt_tokens": float(prompt_tokens),
    }


# ── v0 / v1: generate() iterator ─────────────────────────────────────────────


def once_generate(eng, prompt: str, max_new: int) -> dict:
    sync()
    t0 = time.perf_counter()
    first = None
    n = 0
    for _ in eng.generate(prompt, max_new_tokens=max_new, temperature=0.0):
        n += 1
        if first is None:
            sync()
            first = time.perf_counter() - t0
    sync()
    total = time.perf_counter() - t0
    pt = len(eng.tokenizer.encode(prompt))
    return pack_metrics(first if first is not None else total, total, n, pt)


def serial_conc(eng, prompt: str, max_new: int, n: int) -> dict:
    sync()
    t = time.perf_counter()
    gens = 0
    for _ in range(n):
        r = once_generate(eng, prompt, max_new)
        gens += int(r["gen_tokens"])
    sync()
    wall = time.perf_counter() - t
    return {
        "wall_s": wall,
        "batch_tok_s": gens / wall if wall > 0 else 0.0,
        "e2e_latency_s": wall,
        "note": "serial (no continuous batching)",
    }


# ── v2–v5: continuous batching ───────────────────────────────────────────────


def _sp(SamplingParams, max_new: int):
    try:
        return SamplingParams(max_tokens=max_new, temperature=0.0, ignore_eos=True)
    except TypeError:
        return SamplingParams(max_tokens=max_new, temperature=0.0)


def once_cb(eng, SamplingParams, prompt: str, max_new: int) -> dict:
    sp = _sp(SamplingParams, max_new)
    seq = eng.add_request(prompt, sp)
    first = None
    sync()
    t0 = time.perf_counter()
    while not seq.is_finished:
        eng.step()
        if first is None and seq.num_completion_tokens > 0:
            sync()
            first = time.perf_counter() - t0
    sync()
    total = time.perf_counter() - t0
    return pack_metrics(
        first if first is not None else total,
        total,
        seq.num_completion_tokens,
        seq.num_prompt_tokens,
    )


def _scheduler_finished(eng) -> bool:
    if hasattr(eng, "is_finished"):
        return eng.is_finished()
    return eng.scheduler.is_finished()


def conc_cb(eng, SamplingParams, prompt: str, max_new: int, n: int) -> dict:
    sp = _sp(SamplingParams, max_new)
    seqs = [eng.add_request(prompt, sp) for _ in range(n)]
    sync()
    t = time.perf_counter()
    while not _scheduler_finished(eng):
        eng.step()
    sync()
    wall = time.perf_counter() - t
    gens = sum(s.num_completion_tokens for s in seqs)
    return {
        "wall_s": wall,
        "batch_tok_s": gens / wall if wall > 0 else 0.0,
        "e2e_latency_s": wall,
        "gen_tokens": float(gens),
        "per_seq_tokens": [s.num_completion_tokens for s in seqs],
    }


def make_lean_engine(version: str, args):
    # v0–v2: torch.compile fights paged KV mutations → keep eager.
    # v3–v5: Decode CUDA Graph / FlashInfer is the real fast path.
    use_eager = version in ("v0", "v1", "v2") or args.enforce_eager
    kwargs = dict(
        model=args.model,
        context_len=args.context_len,
        enforce_eager=use_eager,
        dtype="float16",
    )

    if version == "v0":
        from src.config import EngineConfig  # type: ignore
        from src.engine import Engine  # type: ignore

        eng = Engine(EngineConfig(**kwargs))
        return eng, None, False, (not use_eager)

    if version == "v1":
        from src.config import EngineConfig  # type: ignore
        from src.engine import Engine  # type: ignore

        eng = Engine(EngineConfig(**kwargs, prefix_backend="none"))
        return eng, None, False, (not use_eager)

    if version == "v2":
        from src.config import EngineConfig  # type: ignore
        from src.engine.engine import Engine  # type: ignore
        from src.sampling_params import SamplingParams  # type: ignore

        cfg = EngineConfig(
            **kwargs,
            prefix_backend="none",
            max_num_seqs=max(8, max(args.conc_list)),
            max_num_batched_tokens=8192,
            mix_prefill_decode=True,
            num_kv_blocks=max(512, args.context_len // 4),
        )
        return Engine(cfg), SamplingParams, True, (not use_eager)

    if version in ("v3", "v4"):
        from src.config import EngineConfig  # type: ignore
        from src.engine.engine import Engine  # type: ignore
        from src.sampling_params import SamplingParams  # type: ignore

        cfg = EngineConfig(
            **kwargs,
            prefix_backend="none",
            max_num_seqs=max(8, max(args.conc_list)),
            max_num_batched_tokens=8192,
            mix_prefill_decode=True,
            gpu_memory_utilization=args.gpu_util,
        )
        return Engine(cfg), SamplingParams, True, (not use_eager)

    if version == "v5":
        # v5 layout: src.runtime.Engine / enable_radix (no prefix_backend=).
        from src import Engine, EngineConfig, SamplingParams  # type: ignore

        cfg = EngineConfig(
            model=args.model,
            context_len=args.context_len,
            enforce_eager=use_eager,
            dtype="float16",
            max_num_seqs=max(8, max(args.conc_list)),
            max_num_batched_tokens=8192,
            gpu_memory_utilization=args.gpu_util,
            enable_radix=False,  # fair vs v3/v4 prefix_backend=none
            enable_vit_cuda_graph=False,  # text-only suite
            tp_size=1,
            pp_size=1,
        )
        return Engine(cfg), SamplingParams, True, (not use_eager)

    raise ValueError(f"unsupported lean version: {version}")


def bench_lean(version: str, args) -> dict:
    torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    eng, SamplingParams, has_cb, cg = make_lean_engine(version, args)
    sync()
    load_s = time.perf_counter() - t0
    print(
        f"[{version}] load={load_s:.2f}s cb={has_cb} cg_or_compile={cg} "
        f"device={torch.cuda.get_device_name(0)}"
    )

    def once() -> dict:
        if has_cb:
            return once_cb(eng, SamplingParams, args.prompt, args.max_new)
        return once_generate(eng, args.prompt, args.max_new)

    for _ in range(args.warmup):
        once()

    rows = [once() for _ in range(args.iters)]
    single = {k: mean([float(r[k]) for r in rows]) for k in rows[0]}
    print(
        f"  single: ttft={single['ttft_s']*1000:.1f}ms "
        f"decode={single['decode_tok_s']:.1f} tok/s "
        f"e2e={single['e2e_tok_s']:.1f} tok/s "
        f"lat={single['e2e_latency_s']*1000:.1f}ms "
        f"prompt_tok={single['prompt_tokens']:.0f}"
    )

    conc = {}
    for n in args.conc_list:
        if has_cb:
            _ = conc_cb(eng, SamplingParams, args.prompt, args.max_new, n)
            crs = [
                conc_cb(eng, SamplingParams, args.prompt, args.max_new, n)
                for _ in range(max(1, args.iters - 1))
            ]
        else:
            _ = serial_conc(eng, args.prompt, args.max_new, n)
            crs = [
                serial_conc(eng, args.prompt, args.max_new, n)
                for _ in range(max(1, args.iters - 1))
            ]
        entry = {
            "wall_s": mean([r["wall_s"] for r in crs]),
            "batch_tok_s": mean([r["batch_tok_s"] for r in crs]),
            "e2e_latency_s": mean([r["e2e_latency_s"] for r in crs]),
        }
        if "gen_tokens" in crs[0]:
            entry["gen_tokens"] = mean([float(r["gen_tokens"]) for r in crs])
        if not has_cb:
            entry["note"] = "serial (no continuous batching)"
        conc[f"n{n}"] = entry
        print(f"  conc n={n}: batch={conc[f'n{n}']['batch_tok_s']:.1f} tok/s")

    destroy = getattr(eng, "destroy", None)
    if callable(destroy):
        try:
            destroy()
        except Exception:
            pass

    return {
        "backend": version,
        "model": args.model,
        "device": torch.cuda.get_device_name(0),
        "cg_or_compile": cg,
        "prefix_cache": False,
        "continuous_batching": has_cb,
        "load_s": load_s,
        "peak_mem_gb": peak_gb(),
        "max_new": args.max_new,
        "prompt_chars": len(args.prompt),
        "single": single,
        "concurrency": conc,
    }


def bench_official(args) -> dict:
    import vllm.platforms as vplat

    def _cuda_plugin():
        return "vllm.platforms.cuda.CudaPlatform" if torch.cuda.is_available() else None

    vplat.cuda_platform_plugin = _cuda_plugin
    vplat.builtin_platform_plugins["cuda"] = _cuda_plugin
    vplat._current_platform = None

    from vllm import LLM, SamplingParams

    torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    llm = LLM(
        model=args.model,
        dtype="float16",
        trust_remote_code=True,
        gpu_memory_utilization=args.gpu_util,
        max_model_len=args.context_len,
        enforce_eager=False,
        enable_prefix_caching=False,
        max_num_seqs=max(8, max(args.conc_list)),
    )
    sync()
    load_s = time.perf_counter() - t0
    print(f"[official] load={load_s:.2f}s vllm={__import__('vllm').__version__}")

    def once() -> dict:
        req = {"prompt": args.prompt}
        sync()
        t = time.perf_counter()
        llm.generate(
            [req], SamplingParams(temperature=0.0, max_tokens=1, ignore_eos=True)
        )
        sync()
        ttft = time.perf_counter() - t

        sync()
        t = time.perf_counter()
        outs = llm.generate(
            [req],
            SamplingParams(temperature=0.0, max_tokens=args.max_new, ignore_eos=True),
        )
        sync()
        total = time.perf_counter() - t
        n = len(outs[0].outputs[0].token_ids)
        pids = getattr(outs[0], "prompt_token_ids", None)
        return pack_metrics(ttft, total, n, len(pids) if pids is not None else -1)

    def concurrent(n: int) -> dict:
        reqs = [{"prompt": args.prompt} for _ in range(n)]
        sync()
        t = time.perf_counter()
        outs = llm.generate(
            reqs,
            SamplingParams(temperature=0.0, max_tokens=args.max_new, ignore_eos=True),
        )
        sync()
        wall = time.perf_counter() - t
        gens = sum(len(o.outputs[0].token_ids) for o in outs)
        return {
            "wall_s": wall,
            "batch_tok_s": gens / wall if wall > 0 else 0.0,
            "e2e_latency_s": wall,
            "gen_tokens": float(gens),
        }

    for _ in range(args.warmup):
        once()
    rows = [once() for _ in range(args.iters)]
    single = {k: mean([float(r[k]) for r in rows]) for k in rows[0]}
    print(
        f"  single: ttft={single['ttft_s']*1000:.1f}ms "
        f"decode={single['decode_tok_s']:.1f} tok/s "
        f"e2e={single['e2e_tok_s']:.1f} tok/s "
        f"lat={single['e2e_latency_s']*1000:.1f}ms"
    )

    conc = {}
    for n in args.conc_list:
        _ = concurrent(n)
        crs = [concurrent(n) for _ in range(max(1, args.iters - 1))]
        conc[f"n{n}"] = {
            "wall_s": mean([r["wall_s"] for r in crs]),
            "batch_tok_s": mean([r["batch_tok_s"] for r in crs]),
            "e2e_latency_s": mean([r["e2e_latency_s"] for r in crs]),
            "gen_tokens": mean([float(r["gen_tokens"]) for r in crs]),
        }
        print(f"  conc n={n}: batch={conc[f'n{n}']['batch_tok_s']:.1f} tok/s")

    return {
        "backend": "official",
        "vllm_version": __import__("vllm").__version__,
        "model": args.model,
        "device": torch.cuda.get_device_name(0),
        "cg_or_compile": True,
        "prefix_cache": False,
        "continuous_batching": True,
        "load_s": load_s,
        "peak_mem_gb": peak_gb(),
        "max_new": args.max_new,
        "prompt_chars": len(args.prompt),
        "single": single,
        "concurrency": conc,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--version",
        required=True,
        choices=["v0", "v1", "v2", "v3", "v4", "v5", "official"],
    )
    ap.add_argument("--model", required=True)
    ap.add_argument("--max-new", type=int, default=64)
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--iters", type=int, default=4)
    ap.add_argument("--gpu-util", type=float, default=0.85)
    ap.add_argument("--context-len", type=int, default=4096)
    ap.add_argument("--conc", default="1,2,4,8")
    ap.add_argument("--enforce-eager", action="store_true")
    ap.add_argument("--prompt", default=PROMPT)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    args.conc_list = [int(x) for x in str(args.conc).split(",") if x.strip()]

    report = (
        bench_official(args)
        if args.version == "official"
        else bench_lean(args.version, args)
    )
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
