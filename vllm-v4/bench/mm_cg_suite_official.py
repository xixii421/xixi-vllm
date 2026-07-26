#!/usr/bin/env python3
"""Official vLLM multimodal CG suite (decode CG + encoder MM CG).

  source ../.venv-vllm/bin/activate && source ../.env.vllm.sh
  python bench/mm_cg_suite_official.py \\
    --model /mnt/nfs/zhangli/huggingface/Qwen3-VL-2B-Instruct \\
    --image cat.jpg --out bench/results/cg_suite_official.json
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", os.environ.get("CUDA_VISIBLE_DEVICES", "0"))

import torch

torch.cuda.init()
assert torch.cuda.is_available()


def _force_cuda_platform_when_nvml_broken() -> None:
    import vllm.platforms as vplat

    _orig = vplat.cuda_platform_plugin

    def _cuda_plugin():
        r = _orig()
        if r is not None:
            return r
        if torch.cuda.is_available():
            return "vllm.platforms.cuda.CudaPlatform"
        return None

    vplat.cuda_platform_plugin = _cuda_plugin
    vplat.builtin_platform_plugins["cuda"] = _cuda_plugin
    vplat._current_platform = None


_force_cuda_platform_when_nvml_broken()

from PIL import Image


def sync() -> None:
    torch.cuda.synchronize()


def peak_gb() -> float:
    return torch.cuda.max_memory_allocated() / (1024**3)


def load_image(path: str, max_side: int) -> Image.Image:
    im = Image.open(path).convert("RGB")
    if max_side > 0 and max(im.size) > max_side:
        im = im.copy()
        im.thumbnail((max_side, max_side))
    return im


def mean(xs: list[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def make_req(tokenizer, prompt: str, image):
    if image is None:
        return {"prompt": prompt}
    messages = [
        {
            "role": "user",
            "content": [{"type": "image"}, {"type": "text", "text": prompt}],
        }
    ]
    try:
        rendered = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    except Exception:
        rendered = (
            f"<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>"
            f"{prompt}<|im_end|>\n<|im_start|>assistant\n"
        )
    return {"prompt": rendered, "multi_modal_data": {"image": image}}


def bench_once(llm, tokenizer, prompt: str, image, max_new: int) -> dict:
    from vllm import SamplingParams

    req = make_req(tokenizer, prompt, image)
    # TTFT ≈ 1-token generate
    sync()
    t0 = time.perf_counter()
    llm.generate([req], SamplingParams(temperature=0.0, max_tokens=1))
    sync()
    ttft = time.perf_counter() - t0

    sync()
    t1 = time.perf_counter()
    outs = llm.generate([req], SamplingParams(temperature=0.0, max_tokens=max_new))
    sync()
    total = time.perf_counter() - t1
    out = outs[0].outputs[0]
    n = len(out.token_ids)
    # When APC makes full generate faster than the 1-token probe, decode_s
    # would go negative / explode tok/s — fall back to total as decode proxy.
    if total > ttft * 1.05 and n > 1:
        decode_s = total - ttft
        decode_tok_s = (n - 1) / decode_s
    else:
        decode_s = total
        decode_tok_s = n / total if total > 0 else 0.0
    prompt_ids = getattr(outs[0], "prompt_token_ids", None)


def bench_batch(llm, tokenizer, jobs, max_new: int) -> dict:
    from vllm import SamplingParams

    reqs = [make_req(tokenizer, p, img) for p, img in jobs]
    sync()
    t0 = time.perf_counter()
    outs = llm.generate(reqs, SamplingParams(temperature=0.0, max_tokens=max_new))
    sync()
    wall = time.perf_counter() - t0
    gens = [len(o.outputs[0].token_ids) for o in outs]
    return {
        "num_reqs": len(reqs),
        "wall_s": wall,
        "gen_tokens_sum": sum(gens),
        "gen_tokens_mean": mean([float(x) for x in gens]),
        "batch_tok_s": (sum(gens) / wall) if wall > 0 else 0.0,
        "peak_mem_gb": peak_gb(),
    }


def summarize(rows: list[dict]) -> dict:
    keys = [
        "ttft_s",
        "prefill_s",
        "decode_s",
        "total_s",
        "decode_tok_s",
        "e2e_tok_s",
        "prompt_tokens",
        "gen_tokens",
    ]
    out = {}
    for k in keys:
        vals = [float(r[k]) for r in rows if r.get(k) is not None]
        out[k] = mean(vals) if vals else None
    out["vit_encode_s"] = None
    out["lm_prefill_s"] = None
    out["prefill_tok_s"] = None
    out["peak_mem_gb"] = max(float(r["peak_mem_gb"]) for r in rows)
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
    ap.add_argument("--max-model-len", type=int, default=8192)
    ap.add_argument("--out", default="bench/results/cg_suite_official.json")
    ap.add_argument("--scales", default="256,512,768")
    ap.add_argument("--conc", default="1,2,4,8")
    ap.add_argument("--no-encoder-cg", action="store_true")
    args = ap.parse_args()

    from vllm import LLM
    from vllm.config import CompilationConfig

    img_path = Path(args.image)
    if not img_path.is_file():
        raise SystemExit(f"image not found: {img_path}")
    scales = [int(x) for x in args.scales.split(",") if x.strip()]
    conc_levels = [int(x) for x in args.conc.split(",") if x.strip()]
    prompt = "Describe this image in one short sentence."

    cc = CompilationConfig(
        cudagraph_mm_encoder=not args.no_encoder_cg,
        encoder_cudagraph_token_budgets=[64, 128, 256, 512, 1024],
        encoder_cudagraph_max_vision_items_per_batch=1,
    )

    torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    llm = LLM(
        model=args.model,
        dtype="float16",
        trust_remote_code=True,
        gpu_memory_utilization=args.gpu_util,
        max_model_len=args.max_model_len,
        limit_mm_per_prompt={"image": 4},
        enforce_eager=False,
        compilation_config=cc,
    )
    sync()
    load_s = time.perf_counter() - t0
    tokenizer = llm.get_tokenizer()
    print(f"[official] loaded in {load_s:.2f}s encoder_cg={not args.no_encoder_cg}")

    report = {
        "backend": "vllm-official",
        "vllm_version": __import__("vllm").__version__,
        "model": args.model,
        "device": torch.cuda.get_device_name(0),
        "decode_cg": True,
        "vit_cg": not args.no_encoder_cg,
        "load_s": load_s,
        "load_peak_mem_gb": peak_gb(),
        "max_new": args.max_new,
        "cases": {},
        "concurrency": {},
        "profile": {},
    }

    warm = load_image(str(img_path), scales[0])
    for _ in range(args.warmup):
        bench_once(llm, tokenizer, prompt, warm, args.max_new)
        bench_once(llm, tokenizer, "The capital of France is", None, args.max_new)

    rows = [
        bench_once(llm, tokenizer, "The capital of France is", None, args.max_new)
        for _ in range(args.iters)
    ]
    report["cases"]["text_only"] = summarize(rows)
    print(
        f"  text_only ttft={report['cases']['text_only']['ttft_s']*1000:.1f}ms "
        f"decode={report['cases']['text_only']['decode_tok_s']:.1f}"
    )

    for side in scales:
        img = load_image(str(img_path), side)
        name = f"mm_side_{side}"
        rows = [bench_once(llm, tokenizer, prompt, img, args.max_new) for _ in range(args.iters)]
        report["cases"][name] = summarize(rows)
        s = report["cases"][name]
        print(
            f"  {name} ttft={s['ttft_s']*1000:.1f}ms decode={s['decode_tok_s']:.1f} "
            f"e2e={s['e2e_tok_s']:.1f} prompt_tok={s['prompt_tokens']}"
        )
        if side == scales[min(1, len(scales) - 1)]:
            report["profile"]["mm_mid"] = {
                "vit_ms": None,
                "lm_prefill_ms": None,
                "decode_ms": (s["decode_s"] or 0) * 1000,
                "ttft_ms": (s["ttft_s"] or 0) * 1000,
            }

    mid = scales[min(1, len(scales) - 1)]
    img = load_image(str(img_path), mid)
    for n in conc_levels:
        jobs = []
        for i in range(n):
            if i % 2 == 0:
                jobs.append((prompt, img))
            else:
                jobs.append(("The capital of France is", None))
        if n <= 4:
            _ = bench_batch(llm, tokenizer, jobs, args.max_new)
        rows = [bench_batch(llm, tokenizer, jobs, args.max_new) for _ in range(max(1, args.iters - 1))]
        report["concurrency"][f"n{n}"] = {
            "num_reqs": n,
            "wall_s": mean([r["wall_s"] for r in rows]),
            "batch_tok_s": mean([r["batch_tok_s"] for r in rows]),
            "gen_tokens_mean": mean([r["gen_tokens_mean"] for r in rows]),
            "peak_mem_gb": max(r["peak_mem_gb"] for r in rows),
            "mix": "mm+text_alt",
        }
        c = report["concurrency"][f"n{n}"]
        print(f"  conc n={n} wall={c['wall_s']*1000:.0f}ms batch={c['batch_tok_s']:.1f} tok/s")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))
    print(f"[official] wrote {out}")


if __name__ == "__main__":
    main()
