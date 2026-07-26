#!/usr/bin/env python3
"""Multimodal head-to-head: v4 / v5 / official.

Cases (longer prompts + longer decode):
  1. single_image  — one image + describe prompt
  2. mixed         — long text preamble + image + question (图文混合)
  3. pure_image    — image-dominant, minimal text instruction (纯图)
  4. text_only     — text baseline (no image)

Metrics: TTFT / vit_encode (lean only) / lm_prefill (lean only) /
         decode_tok/s / e2e_tok/s / e2e_latency / prompt_tokens.

Usage:
  # lean (v4 or v5)
  PYTHONPATH=vllm-v4 python bench/mm_version_suite.py --version v4 \\
    --model ~/huggingface/Qwen3-VL-2B-Instruct --image cat.jpg --out out/v4.json

  # official
  python bench/mm_version_suite.py --version official \\
    --model ~/huggingface/Qwen3-VL-2B-Instruct --image cat.jpg --out out/official.json
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import time
from pathlib import Path

import torch
from PIL import Image


def sync() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def mean(xs: list[float]) -> float:
    return float(statistics.mean(xs)) if xs else 0.0


def peak_gb() -> float:
    if not torch.cuda.is_available():
        return 0.0
    return torch.cuda.max_memory_allocated() / (1024**3)


def load_image(path: str, max_side: int) -> Image.Image:
    im = Image.open(path).convert("RGB")
    if max_side > 0 and max(im.size) > max_side:
        im = im.copy()
        im.thumbnail((max_side, max_side))
    return im


# Longer text for mixed case (~200+ tokens of prose before the image question).
MIXED_PREAMBLE = (
    "You are an expert visual analyst. Carefully inspect the accompanying photograph "
    "and relate it to the surrounding discussion. Consider composition, lighting, "
    "subject matter, and any text or objects that appear. Then answer the question "
    "with a structured, factual response of several sentences. "
) * 4

CASES = {
    "single_image": {
        "prompt": (
            "Describe this image in detail: main subject, setting, colors, "
            "and any notable objects. Write at least three sentences."
        ),
        "with_image": True,
    },
    "mixed": {
        "prompt": MIXED_PREAMBLE
        + "Question: What is the primary subject of the image, and what mood does it convey?",
        "with_image": True,
    },
    "pure_image": {
        # Minimal instruction → vision tokens dominate the prompt.
        "prompt": "Describe.",
        "with_image": True,
    },
    "text_only": {
        "prompt": (
            "Explain continuous batching for LLM inference in clear prose. "
            "Cover scheduling, KV cache paging, and why decode throughput scales "
            "dramatically with concurrency. Keep the answer factual, structured, "
            "and several paragraphs long. "
        )
        * 10,
        "with_image": False,
    },
}


def pack(
    *,
    ttft: float,
    total: float,
    gen_n: int,
    prompt_n: int,
    vit_s: float | None = None,
    lm_prefill_s: float | None = None,
    prefill_s: float | None = None,
    decode_s: float | None = None,
    text: str = "",
) -> dict:
    if decode_s is None:
        if total > ttft * 1.02 and gen_n > 1:
            decode_s = total - ttft
        else:
            decode_s = total
    if gen_n > 1 and decode_s > 0:
        decode_tok_s = (gen_n - 1) / decode_s
    else:
        decode_tok_s = gen_n / total if total > 0 else 0.0
    return {
        "ttft_s": ttft,
        "vit_encode_s": vit_s,
        "lm_prefill_s": lm_prefill_s,
        "prefill_s": prefill_s if prefill_s is not None else ttft,
        "decode_s": decode_s,
        "total_s": total,
        "decode_tok_s": decode_tok_s,
        "e2e_tok_s": gen_n / total if total > 0 else 0.0,
        "e2e_latency_s": total,
        "gen_tokens": float(gen_n),
        "prompt_tokens": float(prompt_n),
        "text": text[:200],
        "peak_mem_gb": peak_gb(),
    }


# ── v4 instrumented path ─────────────────────────────────────────────────────


def bench_once_v4(eng, prompt: str, images, max_new: int) -> dict:
    from src.sampling_params import SamplingParams

    sp = SamplingParams(max_tokens=max_new, temperature=0.0, ignore_eos=True)
    seq = eng.add_request(prompt, sp, images=images)
    vit_s = lm_prefill_s = prefill_s = decode_s = 0.0
    first = None
    t_start = time.perf_counter()

    orig = None
    if images and hasattr(eng.model, "encode_and_merge"):
        orig = eng.model.encode_and_merge

        def _timed(*a, **kw):
            nonlocal vit_s
            sync()
            t0 = time.perf_counter()
            out = orig(*a, **kw)
            sync()
            vit_s += time.perf_counter() - t0
            return out

        eng.model.encode_and_merge = _timed

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
                eng.scheduler.postprocess(decodes, d_ids, is_prefill=False)
            if prefills:
                vit_before = vit_s
                sync()
                t0 = time.perf_counter()
                p_ids = eng.model_runner._run_batch(prefills, is_prefill=True)
                sync()
                dt = time.perf_counter() - t0
                prefill_s += dt
                lm_prefill_s += max(0.0, dt - (vit_s - vit_before))
                eng.scheduler.postprocess(prefills, p_ids, is_prefill=True)
            if first is None and seq.num_completion_tokens > 0:
                first = time.perf_counter() - t_start
    finally:
        if orig is not None:
            eng.model.encode_and_merge = orig

    sync()
    total = time.perf_counter() - t_start
    text = eng.tokenizer.decode(seq.token_ids[seq.num_prompt_tokens :])
    return pack(
        ttft=first if first is not None else prefill_s,
        total=total,
        gen_n=seq.num_completion_tokens,
        prompt_n=seq.num_prompt_tokens,
        vit_s=vit_s,
        lm_prefill_s=lm_prefill_s,
        prefill_s=prefill_s,
        decode_s=decode_s if decode_s > 0 else None,
        text=text,
    )


def run_v4(args, image: Image.Image) -> dict:
    from src.config import EngineConfig
    from src.engine.engine import Engine

    torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    eng = Engine(
        EngineConfig(
            model=args.model,
            dtype="float16",
            enforce_eager=False,
            gpu_memory_utilization=args.gpu_util,
            max_num_seqs=4,
            max_num_batched_tokens=8192,
            context_len=args.context_len,
            mix_prefill_decode=True,
            prefix_backend="none",
            enable_vit_cuda_graph=True,
            compile_prefill=True,
        )
    )
    load_s = time.perf_counter() - t0
    print(f"[v4] load={load_s:.1f}s is_mm={eng.config.is_multimodal} device={torch.cuda.get_device_name(0)}")

    cases = {}
    for name, spec in CASES.items():
        imgs = [image] if spec["with_image"] else None
        prompt = spec["prompt"]
        print(f"[v4] case={name} ...")
        for _ in range(args.warmup):
            _ = bench_once_v4(eng, prompt, imgs, args.max_new)
        rows = [bench_once_v4(eng, prompt, imgs, args.max_new) for _ in range(args.iters)]
        summary = {k: mean([float(r[k]) for r in rows if r.get(k) is not None]) for k in rows[0] if isinstance(rows[0].get(k), (int, float))}
        summary["text_sample"] = rows[-1]["text"]
        cases[name] = summary
        print(
            f"  ttft={summary['ttft_s']*1000:.1f}ms vit={summary.get('vit_encode_s',0)*1000:.1f}ms "
            f"decode={summary['decode_tok_s']:.1f} e2e={summary['e2e_tok_s']:.1f} "
            f"prompt_tok={summary['prompt_tokens']:.0f}"
        )

    eng.destroy()
    return {
        "backend": "v4",
        "model": args.model,
        "device": torch.cuda.get_device_name(0),
        "load_s": load_s,
        "max_new": args.max_new,
        "max_side": args.max_side,
        "cases": cases,
    }


# ── v5 path (RPC engine: wall-clock TTFT/decode; optional encode hook) ────────


def bench_once_v5(eng, prompt: str, images, max_new: int) -> dict:
    from src import SamplingParams

    sp = SamplingParams(max_tokens=max_new, temperature=0.0, ignore_eos=True)
    vit_s = 0.0
    orig = None
    # Best-effort: hook encode if reachable on local model (tp=1 single proc).
    model = getattr(getattr(eng, "executor", None), "driver_worker", None)
    # Fall back: try eng.model if present
    target = getattr(eng, "model", None)
    if target is None:
        runner = getattr(eng, "model_runner", None)
        target = getattr(runner, "model", None) if runner else None

    if images and target is not None and hasattr(target, "encode_and_merge"):
        orig = target.encode_and_merge

        def _timed(*a, **kw):
            nonlocal vit_s
            sync()
            t0 = time.perf_counter()
            out = orig(*a, **kw)
            sync()
            vit_s += time.perf_counter() - t0
            return out

        target.encode_and_merge = _timed

    try:
        sync()
        t0 = time.perf_counter()
        seq = eng.add_request(prompt, sp, images=images)
        first = None
        while not seq.is_finished:
            eng.step()
            if first is None and seq.num_completion_tokens > 0:
                sync()
                first = time.perf_counter() - t0
        sync()
        total = time.perf_counter() - t0
    finally:
        if orig is not None and target is not None:
            target.encode_and_merge = orig

    text = eng.decode_tokens(seq.token_ids[seq.num_prompt_tokens :])
    return pack(
        ttft=first if first is not None else total,
        total=total,
        gen_n=seq.num_completion_tokens,
        prompt_n=seq.num_prompt_tokens,
        vit_s=vit_s if vit_s > 0 else None,
        text=text,
    )


def run_v5(args, image: Image.Image) -> dict:
    from src import Engine, EngineConfig

    torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    eng = Engine(
        EngineConfig(
            model=args.model,
            dtype="float16",
            enforce_eager=False,
            gpu_memory_utilization=args.gpu_util,
            max_num_seqs=4,
            max_num_batched_tokens=8192,
            context_len=args.context_len,
            enable_vit_cuda_graph=True,
            enable_radix=False,
            tp_size=1,
            pp_size=1,
        )
    )
    load_s = time.perf_counter() - t0
    print(f"[v5] load={load_s:.1f}s is_mm={eng.config.is_multimodal} device={torch.cuda.get_device_name(0)}")

    cases = {}
    for name, spec in CASES.items():
        imgs = [image] if spec["with_image"] else None
        prompt = spec["prompt"]
        print(f"[v5] case={name} ...")
        for _ in range(args.warmup):
            _ = bench_once_v5(eng, prompt, imgs, args.max_new)
        rows = [bench_once_v5(eng, prompt, imgs, args.max_new) for _ in range(args.iters)]
        summary = {
            k: mean([float(r[k]) for r in rows if r.get(k) is not None])
            for k in rows[0]
            if isinstance(rows[0].get(k), (int, float))
        }
        summary["text_sample"] = rows[-1]["text"]
        cases[name] = summary
        print(
            f"  ttft={summary['ttft_s']*1000:.1f}ms "
            f"decode={summary['decode_tok_s']:.1f} e2e={summary['e2e_tok_s']:.1f} "
            f"prompt_tok={summary['prompt_tokens']:.0f}"
        )

    eng.destroy()
    return {
        "backend": "v5",
        "model": args.model,
        "device": torch.cuda.get_device_name(0),
        "load_s": load_s,
        "max_new": args.max_new,
        "max_side": args.max_side,
        "cases": cases,
    }


# ── official ─────────────────────────────────────────────────────────────────


def bench_once_official(llm, tokenizer, prompt: str, image, max_new: int) -> dict:
    from vllm import SamplingParams

    sp = SamplingParams(temperature=0.0, max_tokens=max_new, ignore_eos=True)
    if image is not None:
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": prompt},
                ],
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
        req = {"prompt": rendered, "multi_modal_data": {"image": image}}
    else:
        req = {"prompt": prompt}

    sync()
    t = time.perf_counter()
    llm.generate([req], SamplingParams(temperature=0.0, max_tokens=1, ignore_eos=True))
    sync()
    ttft = time.perf_counter() - t

    sync()
    t = time.perf_counter()
    outs = llm.generate([req], sp)
    sync()
    total = time.perf_counter() - t
    out = outs[0].outputs[0]
    n = len(out.token_ids)
    pids = getattr(outs[0], "prompt_token_ids", None)
    return pack(
        ttft=ttft,
        total=total,
        gen_n=n,
        prompt_n=len(pids) if pids is not None else -1,
        text=out.text,
    )


def run_official(args, image: Image.Image) -> dict:
    import vllm.platforms as vplat

    def _cuda_plugin():
        return "vllm.platforms.cuda.CudaPlatform" if torch.cuda.is_available() else None

    vplat.cuda_platform_plugin = _cuda_plugin
    vplat.builtin_platform_plugins["cuda"] = _cuda_plugin
    vplat._current_platform = None

    from vllm import LLM

    torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    llm_kwargs = dict(
        model=args.model,
        dtype="float16",
        trust_remote_code=True,
        gpu_memory_utilization=args.gpu_util,
        max_model_len=args.context_len,
        enforce_eager=False,
        enable_prefix_caching=False,
        limit_mm_per_prompt={"image": 1},
    )
    try:
        llm = LLM(**llm_kwargs)
    except ValueError as e:
        if "limit_mm_per_prompt" not in str(e):
            raise
        llm_kwargs.pop("limit_mm_per_prompt", None)
        llm = LLM(**llm_kwargs)
    load_s = time.perf_counter() - t0
    tokenizer = llm.get_tokenizer()
    print(f"[official] load={load_s:.1f}s vllm={__import__('vllm').__version__}")

    cases = {}
    for name, spec in CASES.items():
        img = image if spec["with_image"] else None
        prompt = spec["prompt"]
        print(f"[official] case={name} ...")
        for _ in range(args.warmup):
            _ = bench_once_official(llm, tokenizer, prompt, img, args.max_new)
        rows = [
            bench_once_official(llm, tokenizer, prompt, img, args.max_new)
            for _ in range(args.iters)
        ]
        summary = {
            k: mean([float(r[k]) for r in rows if r.get(k) is not None])
            for k in rows[0]
            if isinstance(rows[0].get(k), (int, float))
        }
        summary["text_sample"] = rows[-1]["text"]
        cases[name] = summary
        print(
            f"  ttft={summary['ttft_s']*1000:.1f}ms "
            f"decode={summary['decode_tok_s']:.1f} e2e={summary['e2e_tok_s']:.1f} "
            f"prompt_tok={summary['prompt_tokens']:.0f}"
        )

    return {
        "backend": "official",
        "vllm_version": __import__("vllm").__version__,
        "model": args.model,
        "device": torch.cuda.get_device_name(0),
        "load_s": load_s,
        "max_new": args.max_new,
        "max_side": args.max_side,
        "cases": cases,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--version", required=True, choices=["v4", "v5", "official"])
    ap.add_argument("--model", required=True)
    ap.add_argument("--image", default="cat.jpg")
    ap.add_argument("--max-side", type=int, default=768)
    ap.add_argument("--max-new", type=int, default=128)
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--iters", type=int, default=3)
    ap.add_argument("--gpu-util", type=float, default=0.85)
    ap.add_argument("--context-len", type=int, default=8192)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    img_path = Path(args.image)
    if not img_path.is_file():
        raise SystemExit(f"image not found: {img_path}")
    image = load_image(str(img_path), args.max_side)
    print(f"image={img_path} size={image.size} max_side={args.max_side}")

    if args.version == "v4":
        report = run_v4(args, image)
    elif args.version == "v5":
        report = run_v5(args, image)
    else:
        report = run_official(args, image)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
