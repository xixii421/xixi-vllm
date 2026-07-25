#!/usr/bin/env python3
"""Image-text TTFT profile for vllm-v4 (processor / encode / LM prefill split).

  source ../.venv/bin/activate && source ../.env.cuda.sh
  export CUDA_VISIBLE_DEVICES=<free_gpu>
  PYTHONPATH=. python bench/mm_ttft_profile.py \\
    --model /mnt/nfs/zhangli/huggingface/Qwen3-VL-8B-Instruct --image cat.jpg
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from PIL import Image


def _sync() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--image", default="cat.jpg")
    ap.add_argument("--max-side", type=int, default=512)
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--iters", type=int, default=5)
    ap.add_argument("--out", default="bench/results/v4_ttft_profile.json")
    args = ap.parse_args()

    from src.config import EngineConfig
    from src.engine.engine import Engine
    from src.sampling_params import SamplingParams

    im = Image.open(args.image).convert("RGB")
    if args.max_side > 0 and max(im.size) > args.max_side:
        im = im.copy()
        im.thumbnail((args.max_side, args.max_side))

    eng = Engine(
        EngineConfig(
            model=args.model,
            dtype="float16",
            enforce_eager=False,
            max_num_seqs=1,
            max_num_batched_tokens=8192,
            context_len=8192,
            gpu_memory_utilization=0.85,
        )
    )
    prompt = "Describe this image in one short sentence."
    sp1 = SamplingParams(max_tokens=1, temperature=0.0)

    times: dict[str, float] = {}

    def stamp(k: str, t0: float) -> None:
        times[k] = times.get(k, 0.0) + (time.perf_counter() - t0) * 1000.0

    orig_proc = eng.mm_processor.process

    def timed_proc(*a, **k):
        t0 = time.perf_counter()
        out = orig_proc(*a, **k)
        stamp("processor_ms", t0)
        return out

    eng.mm_processor.process = timed_proc  # type: ignore[method-assign]

    orig_enc = eng.model.encode_and_merge

    def timed_enc(*a, **k):
        _sync()
        t0 = time.perf_counter()
        out = orig_enc(*a, **k)
        _sync()
        stamp("encode_and_merge_ms", t0)
        return out

    eng.model.encode_and_merge = timed_enc  # type: ignore[method-assign]

    orig_vt = eng.model.vision_tower.forward

    def timed_vt(*a, **k):
        _sync()
        t0 = time.perf_counter()
        out = orig_vt(*a, **k)
        _sync()
        stamp("vit_forward_ms", t0)
        return out

    eng.model.vision_tower.forward = timed_vt  # type: ignore[method-assign]

    orig_fwd = eng.model.forward

    def timed_fwd(*a, **k):
        _sync()
        t0 = time.perf_counter()
        out = orig_fwd(*a, **k)
        _sync()
        stamp("lm_forward_ms", t0)
        return out

    eng.model.forward = timed_fwd  # type: ignore[method-assign]

    def one(label: str) -> dict:
        times.clear()
        _sync()
        t0 = time.perf_counter()
        seq = eng.add_request(prompt, sp1, images=[im])
        add_ms = (time.perf_counter() - t0) * 1000.0
        _sync()
        t1 = time.perf_counter()
        while not seq.is_finished:
            eng.step()
        _sync()
        ttft_ms = (time.perf_counter() - t0) * 1000.0
        step_ms = (time.perf_counter() - t1) * 1000.0
        row = {
            "label": label,
            "ttft_ms": round(ttft_ms, 2),
            "add_request_ms": round(add_ms, 2),
            "gpu_steps_ms": round(step_ms, 2),
            "prompt_tokens": seq.num_prompt_tokens,
            **{k: round(v, 2) for k, v in times.items()},
        }
        print(json.dumps(row))
        return row

    for _ in range(args.warmup):
        one("warmup")
    # Force a ViT miss after warmups (clear encoder cache only).
    eng.model._encoder_cache.clear()
    miss = one("cache_miss")
    hits = [one("cache_hit") for _ in range(args.iters)]

    # Text-only baseline (same engine).
    text_rows = []
    for _ in range(args.iters):
        times.clear()
        _sync()
        t0 = time.perf_counter()
        seq = eng.add_request("The capital of France is", sp1, images=None)
        while not seq.is_finished:
            eng.step()
        _sync()
        text_rows.append(
            {
                "ttft_ms": round((time.perf_counter() - t0) * 1000.0, 2),
                "prompt_tokens": seq.num_prompt_tokens,
                **{k: round(v, 2) for k, v in times.items()},
            }
        )
    print("text_only_mean_ttft_ms", round(sum(r["ttft_ms"] for r in text_rows) / len(text_rows), 2))

    report = {
        "device": torch.cuda.get_device_name(0),
        "cache_miss": miss,
        "cache_hit_mean_ttft_ms": round(sum(r["ttft_ms"] for r in hits) / len(hits), 2),
        "cache_hit_mean_processor_ms": round(
            sum(r.get("processor_ms", 0.0) for r in hits) / len(hits), 2
        ),
        "cache_hit_mean_lm_forward_ms": round(
            sum(r.get("lm_forward_ms", 0.0) for r in hits) / len(hits), 2
        ),
        "cache_hits": hits,
        "text_only": text_rows,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
