#!/usr/bin/env python3
"""vllm-v4 multimodal CG suite: load / TTFT / decode / E2E / concurrency / profile.

Both decode CG and ViT encoder CG enabled by default.

  source ../.venv/bin/activate && source ../.env.cuda.sh
  export CUDA_VISIBLE_DEVICES=<gpu>
  PYTHONPATH=. python bench/mm_cg_suite_v4.py \\
    --model /mnt/nfs/zhangli/huggingface/Qwen3-VL-2B-Instruct \\
    --image cat.jpg --out bench/results/cg_suite_v4.json
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from PIL import Image


def sync() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def peak_gb() -> float:
    return torch.cuda.max_memory_allocated() / (1024**3) if torch.cuda.is_available() else 0.0


def load_image(path: str, max_side: int) -> Image.Image:
    im = Image.open(path).convert("RGB")
    if max_side > 0 and max(im.size) > max_side:
        im = im.copy()
        im.thumbnail((max_side, max_side))
    return im


def mean(xs: list[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def bench_single(eng, prompt, images, max_new: int) -> dict:
    from src.sampling_params import SamplingParams

    sp = SamplingParams(max_tokens=max_new, temperature=0.0)
    seq = eng.add_request(prompt, sp, images=images)

    vit_s = lm_prefill_s = decode_s = prefill_s = 0.0
    prefill_tok = decode_steps = 0
    first_token_s = None
    t_start = time.perf_counter()

    orig_encode = None
    if images and hasattr(eng.model, "encode_and_merge"):
        orig_encode = eng.model.encode_and_merge

        def _timed_encode(*a, **kw):
            nonlocal vit_s
            sync()
            t0 = time.perf_counter()
            out = orig_encode(*a, **kw)
            sync()
            vit_s += time.perf_counter() - t0
            return out

        eng.model.encode_and_merge = _timed_encode

    try:
        while not seq.is_finished:
            prefills, decodes = eng.scheduler.schedule()
            if not prefills and not decodes:
                break
            if decodes:
                sync()
                t0 = time.perf_counter()
                d_ids = eng.model_runner._run_batch(decodes, is_prefill=False)
                sync()
                decode_s += time.perf_counter() - t0
                decode_steps += len(decodes)
                eng.scheduler.postprocess(decodes, d_ids, is_prefill=False)
            if prefills:
                vit_before = vit_s
                sync()
                t0 = time.perf_counter()
                p_ids = eng.model_runner._run_batch(prefills, is_prefill=True)
                sync()
                dt = time.perf_counter() - t0
                prefill_s += dt
                prefill_tok += sum(s.num_scheduled_tokens for s in prefills)
                lm_prefill_s += max(0.0, dt - (vit_s - vit_before))
                eng.scheduler.postprocess(prefills, p_ids, is_prefill=True)
            if first_token_s is None and seq.num_completion_tokens > 0:
                first_token_s = time.perf_counter() - t_start
    finally:
        if orig_encode is not None:
            eng.model.encode_and_merge = orig_encode

    total = time.perf_counter() - t_start
    return {
        "prompt_tokens": seq.num_prompt_tokens,
        "gen_tokens": seq.num_completion_tokens,
        "ttft_s": first_token_s if first_token_s is not None else prefill_s,
        "vit_encode_s": vit_s,
        "lm_prefill_s": lm_prefill_s,
        "prefill_s": prefill_s,
        "decode_s": decode_s,
        "total_s": total,
        "prefill_tok_s": (prefill_tok / prefill_s) if prefill_s > 0 else 0.0,
        "decode_tok_s": (decode_steps / decode_s) if decode_s > 0 else 0.0,
        "e2e_tok_s": (seq.num_completion_tokens / total) if total > 0 else 0.0,
        "peak_mem_gb": peak_gb(),
        "vit_cg_ready": bool(getattr(eng.model, "_encoder_cg_ready", False)),
        "vit_cg_hits": _vit_cg_hits(eng),
    }


def _vit_cg_hits(eng) -> int:
    vt = getattr(eng.model, "vision_tower", None)
    cg = getattr(vt, "cudagraph", None) if vt is not None else None
    return int(getattr(cg, "hits", 0) or 0)


def bench_concurrent(eng, jobs: list[tuple[str, list | None]], max_new: int) -> dict:
    """Fire N requests then drain; report batch wall + aggregate tok/s."""
    from src.sampling_params import SamplingParams

    sp = SamplingParams(max_tokens=max_new, temperature=0.0)
    seqs = [eng.add_request(p, sp, images=imgs) for p, imgs in jobs]
    sync()
    t0 = time.perf_counter()
    while not eng.scheduler.is_finished():
        eng.step()
    sync()
    wall = time.perf_counter() - t0
    gens = [s.num_completion_tokens for s in seqs]
    prompts = [s.num_prompt_tokens for s in seqs]
    return {
        "num_reqs": len(seqs),
        "wall_s": wall,
        "gen_tokens_sum": sum(gens),
        "prompt_tokens_mean": mean([float(x) for x in prompts]),
        "gen_tokens_mean": mean([float(x) for x in gens]),
        "batch_tok_s": (sum(gens) / wall) if wall > 0 else 0.0,
        "peak_mem_gb": peak_gb(),
    }


def summarize(rows: list[dict]) -> dict:
    keys = [
        "ttft_s",
        "vit_encode_s",
        "lm_prefill_s",
        "prefill_s",
        "decode_s",
        "total_s",
        "prefill_tok_s",
        "decode_tok_s",
        "e2e_tok_s",
        "prompt_tokens",
        "gen_tokens",
    ]
    out = {k: mean([float(r[k]) for r in rows]) for k in keys}
    out["peak_mem_gb"] = max(float(r["peak_mem_gb"]) for r in rows)
    out["vit_cg_ready"] = bool(rows[-1].get("vit_cg_ready"))
    out["vit_cg_hits"] = int(rows[-1].get("vit_cg_hits", 0))
    out["iters"] = len(rows)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--image", default="cat.jpg")
    ap.add_argument("--max-new", type=int, default=32)
    ap.add_argument("--warmup", type=int, default=1)
    ap.add_argument("--iters", type=int, default=3)
    ap.add_argument("--gpu-util", type=float, default=0.85)
    ap.add_argument("--context-len", type=int, default=8192)
    ap.add_argument("--max-num-seqs", type=int, default=16)
    ap.add_argument("--out", default="bench/results/cg_suite_v4.json")
    ap.add_argument("--scales", default="256,512,768", help="image max_side list")
    ap.add_argument("--conc", default="1,2,4,8", help="concurrency levels")
    ap.add_argument("--no-vit-cg", action="store_true")
    ap.add_argument("--eager", action="store_true", help="disable decode CG")
    args = ap.parse_args()

    from src.config import EngineConfig
    from src.engine.engine import Engine

    img_path = Path(args.image)
    if not img_path.is_file():
        raise SystemExit(f"image not found: {img_path}")
    scales = [int(x) for x in args.scales.split(",") if x.strip()]
    conc_levels = [int(x) for x in args.conc.split(",") if x.strip()]
    prompt = "Describe this image in one short sentence."

    torch.cuda.reset_peak_memory_stats()
    t_load0 = time.perf_counter()
    eng = Engine(
        EngineConfig(
            model=args.model,
            dtype="float16",
            enforce_eager=args.eager,
            enable_vit_cuda_graph=not args.no_vit_cg and not args.eager,
            gpu_memory_utilization=args.gpu_util,
            max_num_seqs=args.max_num_seqs,
            max_num_batched_tokens=8192,
            context_len=args.context_len,
            mix_prefill_decode=True,
            prefix_backend="hash",
        )
    )
    sync()
    load_s = time.perf_counter() - t_load0
    print(
        f"[v4] loaded in {load_s:.2f}s mm={eng.config.is_multimodal} "
        f"decode_cg={not eng.config.enforce_eager} "
        f"vit_cg={eng.config.enable_vit_cuda_graph} "
        f"ready={getattr(eng.model, '_encoder_cg_ready', False)}"
    )

    report: dict = {
        "backend": "vllm-v4",
        "model": args.model,
        "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        "decode_cg": not eng.config.enforce_eager,
        "vit_cg": bool(eng.config.enable_vit_cuda_graph),
        "vit_cg_ready": bool(getattr(eng.model, "_encoder_cg_ready", False)),
        "load_s": load_s,
        "load_peak_mem_gb": peak_gb(),
        "max_new": args.max_new,
        "cases": {},
        "concurrency": {},
        "profile": {},
    }

    # Warmup
    warm_img = load_image(str(img_path), scales[0])
    for _ in range(args.warmup):
        bench_single(eng, prompt, [warm_img], args.max_new)
        bench_single(eng, "The capital of France is", None, args.max_new)

    # Text-only baseline
    rows = [bench_single(eng, "The capital of France is", None, args.max_new) for _ in range(args.iters)]
    report["cases"]["text_only"] = summarize(rows)
    print(
        f"  text_only ttft={report['cases']['text_only']['ttft_s']*1000:.1f}ms "
        f"decode={report['cases']['text_only']['decode_tok_s']:.1f} tok/s"
    )

    # Image scale sweep
    for side in scales:
        img = load_image(str(img_path), side)
        name = f"mm_side_{side}"
        rows = [bench_single(eng, prompt, [img], args.max_new) for _ in range(args.iters)]
        report["cases"][name] = summarize(rows)
        s = report["cases"][name]
        print(
            f"  {name} ttft={s['ttft_s']*1000:.1f}ms vit={s['vit_encode_s']*1000:.1f}ms "
            f"lm_prefill={s['lm_prefill_s']*1000:.1f}ms decode={s['decode_tok_s']:.1f} "
            f"e2e={s['e2e_tok_s']:.1f} prompt_tok={s['prompt_tokens']:.0f}"
        )
        if side == scales[min(1, len(scales) - 1)]:
            report["profile"]["mm_mid"] = {
                "vit_ms": s["vit_encode_s"] * 1000,
                "lm_prefill_ms": s["lm_prefill_s"] * 1000,
                "decode_ms": s["decode_s"] * 1000,
                "ttft_ms": s["ttft_s"] * 1000,
                "share_vit": (s["vit_encode_s"] / s["ttft_s"]) if s["ttft_s"] > 0 else 0,
                "share_lm_prefill": (s["lm_prefill_s"] / s["ttft_s"]) if s["ttft_s"] > 0 else 0,
            }

    # Concurrency sweep (fixed mid scale)
    mid = scales[min(1, len(scales) - 1)]
    img = load_image(str(img_path), mid)
    for n in conc_levels:
        # Mix: half mm, half text when n>=2
        jobs = []
        for i in range(n):
            if i % 2 == 0:
                jobs.append((prompt, [img]))
            else:
                jobs.append(("The capital of France is", None))
        # warmup one concurrent round
        if n <= 4:
            _ = bench_concurrent(eng, jobs, args.max_new)
        rows = [bench_concurrent(eng, jobs, args.max_new) for _ in range(max(1, args.iters - 1))]
        report["concurrency"][f"n{n}"] = {
            "num_reqs": n,
            "wall_s": mean([r["wall_s"] for r in rows]),
            "batch_tok_s": mean([r["batch_tok_s"] for r in rows]),
            "gen_tokens_mean": mean([r["gen_tokens_mean"] for r in rows]),
            "prompt_tokens_mean": mean([r["prompt_tokens_mean"] for r in rows]),
            "peak_mem_gb": max(r["peak_mem_gb"] for r in rows),
            "mix": "mm+text_alt",
        }
        c = report["concurrency"][f"n{n}"]
        print(
            f"  conc n={n} wall={c['wall_s']*1000:.0f}ms batch={c['batch_tok_s']:.1f} tok/s "
            f"mem={c['peak_mem_gb']:.1f}GiB"
        )

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))
    print(f"[v4] wrote {out}")


if __name__ == "__main__":
    main()
