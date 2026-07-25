#!/usr/bin/env python3
"""Module-level MM profile for vllm-v4 (flash-attn ViT + FlashInfer LM).

Requires env prepared outside the process (no code↔env coupling)::

  source ../.venv/bin/activate && source ../.env.cuda.sh
  export CUDA_VISIBLE_DEVICES=1
  PYTHONPATH=. python bench/mm_profile_modules.py ...

Profiles:
  1. vit_fa_attn     — flash-attn varlen (head_dim=72)
  2. vit_sdpa_attn   — torch SDPA reference
  3. e2e_modules     — encode / prefill / decode split (optional --model)
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch


def _sync() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _bench(fn, warmup: int = 3, iters: int = 20) -> float:
    for _ in range(warmup):
        fn()
    _sync()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    _sync()
    return (time.perf_counter() - t0) / iters * 1000.0


def profile_vit_attn(iters: int) -> dict:
    from flash_attn import flash_attn_varlen_func
    import torch.nn.functional as F
    from src.layers.vision.vit_attn import ViTAttentionBackend

    device = torch.device("cuda")
    dtype = torch.bfloat16
    h, d, n = 16, 72, 1296
    q = torch.randn(n, h, d, device=device, dtype=dtype)
    k = torch.randn(n, h, d, device=device, dtype=dtype)
    v = torch.randn(n, h, d, device=device, dtype=dtype)
    scale = d**-0.5
    cu = torch.tensor([0, n], dtype=torch.int32)

    be = ViTAttentionBackend(h, d, device, dtype)
    be.plan(cu, seqlens=[n])

    def fa():
        return be.run(q, k, v)

    def sdpa():
        return (
            F.scaled_dot_product_attention(
                q.transpose(0, 1).unsqueeze(0),
                k.transpose(0, 1).unsqueeze(0),
                v.transpose(0, 1).unsqueeze(0),
                scale=scale,
            )
            .squeeze(0)
            .transpose(0, 1)
        )

    t_fa = _bench(fa, iters=iters)
    t_sd = _bench(sdpa, iters=iters)
    diff = (fa().float() - sdpa().float()).abs()
    return {
        "shape": f"N={n} H={h} D={d}",
        "backend": "flash_attn_varlen",
        "flash_attn_ms": round(t_fa, 4),
        "sdpa_ms": round(t_sd, 4),
        "speedup_vs_sdpa": round(t_sd / max(t_fa, 1e-6), 3),
        "max_abs_diff": float(diff.max()),
        "flash_attn_ok": True,
        "flash_attn_version": getattr(
            __import__("flash_attn", fromlist=["__version__"]), "__version__", "?"
        ),
    }


def profile_e2e_modules(model: str, image: str, max_side: int, max_new: int) -> dict:
    from PIL import Image

    from src.config import EngineConfig
    from src.engine.engine import Engine
    from src.sampling_params import SamplingParams

    im = Image.open(image).convert("RGB")
    if max_side > 0 and max(im.size) > max_side:
        im = im.copy()
        im.thumbnail((max_side, max_side))

    eng = Engine(
        EngineConfig(
            model=model,
            dtype="float16",
            enforce_eager=False,
            max_num_seqs=1,
            max_num_batched_tokens=8192,
            context_len=8192,
            gpu_memory_utilization=0.85,
        )
    )
    sp = SamplingParams(max_tokens=max_new, temperature=0.0)
    vit_ms = 0.0
    orig = eng.model.encode_and_merge

    def timed_encode(*a, **kw):
        nonlocal vit_ms
        _sync()
        t0 = time.perf_counter()
        out = orig(*a, **kw)
        _sync()
        vit_ms += (time.perf_counter() - t0) * 1000
        return out

    eng.model.encode_and_merge = timed_encode  # type: ignore[method-assign]
    seq = eng.add_request("Describe briefly.", sp, images=[im])
    decode_ms = 0.0
    prefill_ms = 0.0
    _sync()
    t_all = time.perf_counter()
    while not eng.scheduler.is_finished():
        prefills, decodes = eng.scheduler.schedule()
        if prefills:
            _sync()
            t0 = time.perf_counter()
            toks = eng.model_runner.run(prefills, is_prefill=True)
            _sync()
            prefill_ms += (time.perf_counter() - t0) * 1000
            eng.scheduler.postprocess(prefills, toks, is_prefill=True)
        if decodes:
            _sync()
            t0 = time.perf_counter()
            toks = eng.model_runner.run(decodes, is_prefill=False)
            _sync()
            decode_ms += (time.perf_counter() - t0) * 1000
            eng.scheduler.postprocess(decodes, toks, is_prefill=False)
    total_ms = (time.perf_counter() - t_all) * 1000
    text = eng.tokenizer.decode(seq.token_ids[seq.num_prompt_tokens :], skip_special_tokens=True)
    return {
        "vit_encode_ms": round(vit_ms, 2),
        "prefill_step_ms": round(prefill_ms, 2),
        "decode_ms": round(decode_ms, 2),
        "total_ms": round(total_ms, 2),
        "output": text[:200],
        "peak_mem_gb": round(torch.cuda.max_memory_allocated() / 1024**3, 3),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="")
    ap.add_argument("--image", default="")
    ap.add_argument("--max-side", type=int, default=512)
    ap.add_argument("--max-new", type=int, default=16)
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--out", default="bench/results/v4_mm_profile.json")
    args = ap.parse_args()

    report: dict = {"vit_attn": profile_vit_attn(args.iters)}
    print(json.dumps(report["vit_attn"], indent=2))
    if args.model and args.image:
        print("[e2e] loading model …")
        report["e2e_modules"] = profile_e2e_modules(
            args.model, args.image, args.max_side, args.max_new
        )
        print(json.dumps(report["e2e_modules"], indent=2))
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
