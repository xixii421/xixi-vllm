"""Parallel performance matrix: text/VL × single|TP|PP × eager|CG.

Measures TTFT / prefill / decode / e2e / VRAM. Uses Engine.step() so
multi-GPU Executor paths are covered (unlike model_runner-only benches).

Examples:

  # Text: single vs TP=2 vs PP=2, each with eager + CG
  CUDA_DEVICE_ORDER=PCI_BUS_ID PYTHONPATH=. python bench/bench_parallel_matrix.py \\
    --domain text --model ~/huggingface/Qwen3-0.6B \\
    --modes single,tp2,pp2 --cg-modes eager,cg

  # VL separately (put first PP / heavy work on 4090 via --device-ids)
  CUDA_DEVICE_ORDER=PCI_BUS_ID PYTHONPATH=. python bench/bench_parallel_matrix.py \\
    --domain vl --model ~/huggingface/Qwen3-VL-2B-Instruct \\
    --modes single,tp2,pp2 --cg-modes eager,cg --device-ids 1,0,1
"""

from __future__ import annotations

import argparse
import gc
import json
import time
from pathlib import Path

import torch
from PIL import Image, ImageDraw


def _sync() -> None:
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            torch.cuda.synchronize(i)


def _free() -> None:
    gc.collect()
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            with torch.cuda.device(i):
                torch.cuda.empty_cache()
                torch.cuda.synchronize()


def _mem_gb(device_ids: list[int]) -> dict[str, float]:
    out: dict[str, float] = {}
    for d in device_ids:
        free, total = torch.cuda.mem_get_info(d)
        out[f"gpu{d}_used_gb"] = round((total - free) / 1e9, 2)
        out[f"gpu{d}_total_gb"] = round(total / 1e9, 2)
    out["peak_used_gb"] = max(
        (v for k, v in out.items() if k.endswith("_used_gb")), default=0.0
    )
    return out


def _img(size: int = 128) -> Image.Image:
    im = Image.new("RGB", (size, size), (40, 100, 200))
    d = ImageDraw.Draw(im)
    d.rectangle([16, 16, size - 16, size - 16], fill=(220, 60, 60))
    return im


def _bench_once(eng, prompt: str, *, max_new: int, images=None) -> dict:
    from src.sampling_params import SamplingParams

    sp = SamplingParams(max_tokens=max_new, temperature=0.0, ignore_eos=True)
    seq = eng.add_request(prompt, sp, images=images)

    prefill_s = decode_s = 0.0
    prefill_tok = decode_tok = 0
    first_token_s = None
    t_start = time.perf_counter()

    while not seq.is_finished:
        _sync()
        t0 = time.perf_counter()
        outs, metric = eng.step()
        _sync()
        dt = time.perf_counter() - t0
        if metric > 0:
            prefill_s += dt
            prefill_tok += int(metric)
            if first_token_s is None and seq.num_completion_tokens > 0:
                first_token_s = time.perf_counter() - t_start
        elif metric < 0:
            decode_s += dt
            decode_tok += -int(metric)
        _ = outs

    total_s = time.perf_counter() - t_start
    gen = seq.num_completion_tokens
    return {
        "prompt_tokens": seq.num_prompt_tokens,
        "gen_tokens": gen,
        "ttft_s": first_token_s if first_token_s is not None else prefill_s,
        "prefill_s": prefill_s,
        "decode_s": decode_s,
        "total_s": total_s,
        "prefill_tok_s": (prefill_tok / prefill_s) if prefill_s > 0 else 0.0,
        "decode_tok_s": (decode_tok / decode_s) if decode_s > 0 else 0.0,
        "e2e_tok_s": (gen / total_s) if total_s > 0 else 0.0,
    }


def _mode_spec(name: str, all_ids: list[int]) -> tuple[int, int, list[int]] | None:
    table = {
        "single": (1, 1, all_ids[:1] or [0]),
        "single0": (1, 1, [all_ids[0]]),
        "single1": (1, 1, [all_ids[1]] if len(all_ids) > 1 else [all_ids[0]]),
        "tp2": (2, 1, all_ids[:2]),
        "pp2": (1, 2, all_ids[:2]),
    }
    return table.get(name)


def _run_cell(
    *,
    model: str,
    domain: str,
    tp_size: int,
    pp_size: int,
    device_ids: list[int],
    enforce_eager: bool,
    prompt: str,
    images,
    max_new: int,
    warmup: int,
    iters: int,
    gpu_util: float,
    mm_encoder_tp_mode: str,
) -> dict:
    from src.config import EngineConfig
    from src.runtime.engine import Engine

    _free()
    cfg = EngineConfig(
        model=model,
        dtype="float16",
        enforce_eager=enforce_eager,
        enable_vit_cuda_graph=False,
        tp_size=tp_size,
        pp_size=pp_size,
        device_ids=list(device_ids),
        gpu_memory_utilization=gpu_util,
        max_num_seqs=2,
        max_num_batched_tokens=8192,
        mm_encoder_tp_mode=mm_encoder_tp_mode,
    )
    eng = Engine(cfg)
    try:
        for _ in range(warmup):
            _bench_once(eng, prompt, max_new=min(8, max_new), images=images)
        rows = [
            _bench_once(eng, prompt, max_new=max_new, images=images)
            for _ in range(iters)
        ]
    finally:
        eng.destroy()
        _free()

    keys = [
        "prompt_tokens",
        "gen_tokens",
        "ttft_s",
        "prefill_s",
        "decode_s",
        "total_s",
        "prefill_tok_s",
        "decode_tok_s",
        "e2e_tok_s",
    ]
    avg = {k: sum(r[k] for r in rows) / len(rows) for k in keys}
    avg.update(_mem_gb(device_ids))
    avg["tp_size"] = tp_size
    avg["pp_size"] = pp_size
    avg["device_ids"] = list(device_ids)
    avg["enforce_eager"] = enforce_eager
    avg["cg"] = not enforce_eager
    avg["domain"] = domain
    avg["mm_encoder_tp_mode"] = mm_encoder_tp_mode if domain == "vl" else None
    return avg


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--domain", choices=["text", "vl"], required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--modes", default="single,tp2,pp2")
    ap.add_argument("--cg-modes", default="eager,cg", help="eager and/or cg (decode CUDA Graph)")
    ap.add_argument("--device-ids", default="0,1")
    ap.add_argument("--max-new", type=int, default=64)
    ap.add_argument("--warmup", type=int, default=1)
    ap.add_argument("--iters", type=int, default=3)
    ap.add_argument("--gpu-util", type=float, default=0.75)
    ap.add_argument("--image-size", type=int, default=128)
    ap.add_argument("--mm-encoder-tp-mode", choices=["data", "weights"], default="data")
    ap.add_argument(
        "--prompt",
        default=None,
        help="override prompt (text default / VL default if unset)",
    )
    ap.add_argument(
        "--out",
        default=None,
        help="json output path (default: bench/results/parallel_{domain}.json)",
    )
    args = ap.parse_args()

    all_ids = [int(x) for x in args.device_ids.split(",") if x.strip() != ""]
    prompt = args.prompt
    images = None
    if args.domain == "text":
        prompt = prompt or "Explain tensor parallelism in two short sentences."
    else:
        prompt = prompt or "Describe the image briefly in one sentence."
        images = [_img(args.image_size)]

    cg_flags = []
    for name in args.cg_modes.split(","):
        name = name.strip().lower()
        if name == "eager":
            cg_flags.append((name, True))
        elif name in ("cg", "compile"):
            cg_flags.append(("cg", False))
        else:
            print(f"skip unknown cg mode {name}")

    report = {
        "domain": args.domain,
        "model": args.model,
        "max_new": args.max_new,
        "warmup": args.warmup,
        "iters": args.iters,
        "mm_encoder_tp_mode": args.mm_encoder_tp_mode if args.domain == "vl" else None,
        "cells": {},
    }

    for mode_name in args.modes.split(","):
        mode_name = mode_name.strip()
        spec = _mode_spec(mode_name, all_ids)
        if spec is None:
            print(f"skip unknown mode {mode_name}")
            continue
        tp, pp, ids = spec
        if len(ids) < tp * pp:
            print(f"skip {mode_name}: need {tp * pp} devices, got {ids}")
            continue
        for cg_name, eager in cg_flags:
            key = f"{mode_name}/{cg_name}"
            print(f"\n======== {key} tp={tp} pp={pp} devices={ids} ========")
            try:
                row = _run_cell(
                    model=args.model,
                    domain=args.domain,
                    tp_size=tp,
                    pp_size=pp,
                    device_ids=ids,
                    enforce_eager=eager,
                    prompt=prompt,
                    images=images,
                    max_new=args.max_new,
                    warmup=args.warmup,
                    iters=args.iters,
                    gpu_util=args.gpu_util,
                    mm_encoder_tp_mode=args.mm_encoder_tp_mode,
                )
            except Exception as e:
                print(f"FAILED {key}: {type(e).__name__}: {e}")
                report["cells"][key] = {"error": f"{type(e).__name__}: {e}"}
                _free()
                continue
            report["cells"][key] = row
            print(
                f"  TTFT={row['ttft_s']*1000:.1f}ms  "
                f"prefill={row['prefill_tok_s']:.0f} tok/s  "
                f"decode={row['decode_tok_s']:.1f} tok/s  "
                f"e2e={row['e2e_tok_s']:.1f} tok/s  "
                f"peak_vram={row['peak_used_gb']:.2f}GB"
            )

    out = Path(
        args.out
        or Path(__file__).resolve().parent / f"results/parallel_{args.domain}.json"
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False))

    md = out.with_suffix(".md")
    lines = [
        f"# Parallel matrix — {args.domain}",
        "",
        f"- model: `{args.model}`",
        f"- max_new={args.max_new}, iters={args.iters}, warmup={args.warmup}",
        "",
        "| cell | tp | pp | CG | TTFT_ms | prefill_tok/s | decode_tok/s | e2e_tok/s | peak_GB |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for key, m in report["cells"].items():
        if "error" in m:
            lines.append(f"| {key} | — | — | — | FAIL | — | — | — | `{m['error'][:60]}` |")
            continue
        lines.append(
            f"| {key} | {m['tp_size']} | {m['pp_size']} | {str(m['cg']).lower()} | "
            f"{m['ttft_s']*1000:.1f} | {m['prefill_tok_s']:.0f} | "
            f"{m['decode_tok_s']:.1f} | {m['e2e_tok_s']:.1f} | {m['peak_used_gb']:.2f} |"
        )
    lines += [
        "",
        "## How to read",
        "",
        "- **TTFT**: wall time to first generated token (includes multimodal encode + first prefill).",
        "- **prefill_tok/s**: scheduled prefill tokens / prefill wall time.",
        "- **decode_tok/s**: decode steps (≈ tokens for batch=1) / decode wall time.",
        "- **e2e_tok/s**: gen_tokens / total wall time.",
        "- Compare **same domain** cells only; text vs VL prompt lengths differ.",
        "- On hetero GPUs, PP often loses to TP on decode; CG helps decode, not prefill/ViT.",
        "",
    ]
    md.write_text("\n".join(lines))
    print(f"\nWrote {out} and {md}")


if __name__ == "__main__":
    main()
