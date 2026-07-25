#!/usr/bin/env python3
"""Comprehensive benchmark: vllm-v4 (compiled) vs official vLLM.

Scenarios:
  1.  Text-only short (~50 tok)
  2.  Text-only medium (~200 tok)
  3.  Text-only long (~1000 tok)
  4.  Single image (768px)
  5.  Single image (1024px)
  6.  Two images
  7.  2x2 grid images
  8.  Mixed text+image
  9.  Concurrent mixed-length (8 seqs: 4 short + 4 long)
  10. Concurrent mixed (multimodal + text)

Measures: TTFT_ms, prefill_tok_s, decode_tok_s, output_hash.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import torch
from PIL import Image


def sync() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def load_image(path: str, max_side: int) -> torch.Tensor:
    img = Image.open(path).convert("RGB")
    w, h = img.size
    if max(w, h) > max_side:
        scale = max_side / max(w, h)
        img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
    return img


def text_output_hash(tokens: list[int]) -> str:
    return hashlib.sha1(bytes(tokens)).hexdigest()[:12]


# ── Text prompts ──────────────────────────────────────────────────────────
TEXT_SHORT = "What is the capital of France?"
TEXT_MEDIUM = "Explain the process of photosynthesis in detail, including the role of chlorophyll, light reactions, and the Calvin cycle. " * 2
TEXT_LONG = "Discuss the history and development of artificial intelligence from the 1950s to the present day, covering key breakthroughs, important researchers, and the societal impact of AI technologies. " * 8
IMAGE_PROMPT = "Describe this image in one short sentence."
TWO_IMAGE_PROMPT = "Compare these two images in one sentence."


def bench_v4(args):
    """V4 benchmark: compile + run all scenarios."""
    from src.config import EngineConfig
    from src.engine.engine import Engine
    from src.sampling_params import SamplingParams

    # Load images
    img0 = load_image(args.image, args.max_side)
    img1 = load_image(args.image2 or args.image, args.max_side)

    print("=== vllm-v4: initializing engine (compile enabled) ===")
    cfg = EngineConfig(
        model=args.model,
        dtype="float16",
        enforce_eager=not args.use_decode_cg,
        context_len=args.context_len,
        gpu_memory_utilization=args.gpu_mem,
        max_num_seqs=args.max_num_seqs,
        max_num_batched_tokens=args.max_batched_tokens,
        enable_vit_cuda_graph=args.use_vit_cg,
    )
    eng = Engine(cfg)
    sp = SamplingParams(max_tokens=args.max_tokens, temperature=0.0)

    # Warmup (also triggers compile)
    print("Warmup...")
    for _ in range(3):
        s = eng.add_request("Hello world", sp, images=None)
        while not s.is_finished:
            eng.step()

    results: list[dict] = []

    def measure(name: str, make_prompt, images=None, n_repeat=1, concurrent=False):
        nonlocal results
        for trial in range(n_repeat):
            prompt = make_prompt() if callable(make_prompt) else make_prompt
            s = eng.add_request(prompt, sp, images=images)
            toks = []
            sync()
            t0 = time.perf_counter()
            while not s.is_finished:
                steps = eng.step()
                for t in steps.get(s.seq_id, []):
                    if t is not None and t >= 0:
                        toks.append(t)
            sync()
            wall = time.perf_counter() - t0
            results.append({
                "engine": "v4",
                "scenario": name,
                "trial": trial,
                "ttft_ms": round(wall * 1000, 1),
                "num_tokens": len(toks),
                "decode_tok_s": round(len(toks) / wall, 1) if wall > 0 else 0,
                "output_hash": text_output_hash(toks),
            })
            print(f"  {name} [{trial}]: TTFT={wall*1000:.0f}ms  tok={len(toks)}  "
                  f"hash={text_output_hash(toks)}")

    # ── Scenarios ─────────────────────────────────────────────────────────
    print("\n=== Scenario 1-3: Text-only ===")
    measure("text_short", TEXT_SHORT)
    measure("text_medium", TEXT_MEDIUM)
    measure("text_long", TEXT_LONG)

    print("\n=== Scenario 4-5: Single image ===")
    measure("img_768", IMAGE_PROMPT, images=[img0])
    img_big = load_image(args.image, 1024)
    measure("img_1024", IMAGE_PROMPT, images=[img_big])

    print("\n=== Scenario 6-7: Multiple images ===")
    measure("img_x2", TWO_IMAGE_PROMPT, images=[img0, img1])
    imgs4 = [load_image(args.image, 512)] * 4
    measure("img_x4", "Describe all four quadrants of this scene.", images=imgs4)

    print("\n=== Scenario 8: Mixed text+image ===")
    measure("mixed", "The image shows: " + IMAGE_PROMPT, images=[img0])

    print("\n=== Scenario 9: Concurrent mixed-length text ===")
    # Submit 8 requests simultaneously, measure batch-level performance
    texts = [TEXT_SHORT] * 4 + [TEXT_MEDIUM] * 4
    seqs = []
    sync()
    t0 = time.perf_counter()
    for t in texts:
        seqs.append(eng.add_request(t, sp, images=None))
    all_toks: dict[int, list] = {s.seq_id: [] for s in seqs}
    while any(not s.is_finished for s in seqs):
        steps = eng.step()
        for sid, toks in steps.items():
            all_toks[sid].extend(t for t in toks if t is not None and t >= 0)
    sync()
    wall = time.perf_counter() - t0
    n_tok = sum(len(v) for v in all_toks.values())
    results.append({
        "engine": "v4", "scenario": "concurrent_8_mixed_len", "trial": 0,
        "ttft_ms": round(wall * 1000, 1),
        "num_tokens": n_tok,
        "decode_tok_s": round(n_tok / wall, 1) if wall > 0 else 0,
        "output_hash": text_output_hash(
            [t for v in all_toks.values() for t in v]
        ),
    })
    print(f"  concurrent_8_mixed_len: wall={wall*1000:.0f}ms  tok={n_tok}")

    print("\n=== Scenario 10: Concurrent multimodal + text ===")
    sync(); t0 = time.perf_counter()
    s_mm = eng.add_request(IMAGE_PROMPT + " Be concise.", sp, images=[img0])
    s_txt = eng.add_request(TEXT_MEDIUM, sp, images=None)
    all_toks2 = {s_mm.seq_id: [], s_txt.seq_id: []}
    while not s_mm.is_finished or not s_txt.is_finished:
        steps = eng.step()
        for sid, toks in steps.items():
            all_toks2[sid].extend(t for t in toks if t is not None and t >= 0)
    sync(); wall2 = time.perf_counter() - t0
    n_tok2 = sum(len(v) for v in all_toks2.values())
    results.append({
        "engine": "v4", "scenario": "concurrent_mm_text", "trial": 0,
        "ttft_ms": round(wall2 * 1000, 1),
        "num_tokens": n_tok2,
        "decode_tok_s": round(n_tok2 / wall2, 1) if wall2 > 0 else 0,
        "output_hash": text_output_hash([t for v in all_toks2.values() for t in v]),
    })
    print(f"  concurrent_mm_text: wall={wall2*1000:.0f}ms  tok={n_tok2}")

    eng.destroy()
    return results


def bench_official(args):
    """Official vLLM benchmark."""
    import os
    os.environ["VLLM_DISABLE_NVML"] = "1"

    # Monkey patch for NVML issue (from existing mm_compare_official.py)
    def _force_cuda():
        try:
            from vllm.platforms import current_platform
        except Exception:
            import vllm.platforms
            current_platform = vllm.platforms.current_platform()
        if not current_platform.is_cuda():
            import vllm.platforms.cuda
            vllm.platforms.cuda._ = None
            old = vllm.platforms.resolve_obj_by_qualname
            def _patched(qn):
                if qn == "vllm.platforms.cuda.CudaPlatform":
                    return vllm.platforms.cuda.CudaPlatform()
                return old(qn)
            vllm.platforms.resolve_obj_by_qualname = _patched
    _force_cuda()

    from vllm import LLM, SamplingParams

    img0 = load_image(args.image, args.max_side)
    img1 = load_image(args.image2 or args.image, args.max_side)
    img_big = load_image(args.image, 1024)
    imgs4 = [load_image(args.image, 512)] * 4

    print("=== Official vLLM: loading model ===")
    llm = LLM(
        model=args.model, dtype="float16",
        enforce_eager=not args.use_decode_cg,
        max_model_len=args.context_len,
        gpu_memory_utilization=args.gpu_mem,
        max_num_seqs=args.max_num_seqs,
        trust_remote_code=True,
    )
    sp = SamplingParams(max_tokens=args.max_tokens, temperature=0.0)

    # Warmup
    llm.generate(["Hello world"], sampling_params=sp)

    results: list[dict] = []

    def measure(name, prompts, images=None):
        if images is None:
            messages = [[{"role": "user", "content": [{"type": "text", "text": p}]}]
                        for p in prompts]
        else:
            messages = []
            for i, p in enumerate(prompts):
                imgs = images if isinstance(images[0], list) else [images[i]]
                content = [{"type": "image", "image": img} for img in imgs]
                content.insert(0, {"type": "text", "text": p})
                messages.append([{"role": "user", "content": content}])

        sync(); t0 = time.perf_counter()
        outputs = llm.generate(messages, sampling_params=sp)
        sync(); wall = time.perf_counter() - t0

        for i, out in enumerate(outputs):
            toks = list(out.outputs[0].token_ids)
            results.append({
                "engine": "official",
                "scenario": f"{name}_{i}" if len(outputs) > 1 else name,
                "trial": 0,
                "ttft_ms": round(wall * 1000 / len(outputs), 1),
                "num_tokens": len(toks),
                "decode_tok_s": round(len(toks) / wall * len(outputs), 1) if wall > 0 else 0,
                "output_hash": text_output_hash(toks),
                "wall_ms": round(wall * 1000, 1),
            })
        print(f"  {name}: wall={wall*1000:.0f}ms  "
              f"out={[text_output_hash(list(o.outputs[0].token_ids)) for o in outputs]}")

    print("\n=== Scenario 1-3: Text-only ===")
    measure("text_short", [TEXT_SHORT])
    measure("text_medium", [TEXT_MEDIUM])
    measure("text_long", [TEXT_LONG])

    print("\n=== Scenario 4-5: Single image ===")
    measure("img_768", [IMAGE_PROMPT], images=[img0])
    measure("img_1024", [IMAGE_PROMPT], images=[img_big])

    print("\n=== Scenario 6-7: Multiple images ===")
    measure("img_x2", [TWO_IMAGE_PROMPT], images=[[img0, img1]])
    measure("img_x4", ["Describe all four quadrants."], images=[imgs4])

    print("\n=== Scenario 8: Mixed text+image ===")
    measure("mixed", ["The image shows: " + IMAGE_PROMPT], images=[img0])

    print("\n=== Scenario 9: Concurrent mixed-length text ===")
    prompts = [TEXT_SHORT] * 4 + [TEXT_MEDIUM] * 4
    measure("concurrent_8_mixed_len", prompts)

    print("\n=== Scenario 10: Concurrent multimodal + text ===")
    measure("concurrent_mm_text", [IMAGE_PROMPT, TEXT_MEDIUM],
            images=[img0, None])

    del llm
    return results


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="/mnt/nfs/zhangli/huggingface/Qwen3-VL-8B-Instruct")
    p.add_argument("--image", default="cat.jpg")
    p.add_argument("--image2", default=None)
    p.add_argument("--max-side", type=int, default=768)
    p.add_argument("--context-len", type=int, default=4096)
    p.add_argument("--max-tokens", type=int, default=32)
    p.add_argument("--gpu-mem", type=float, default=0.55)
    p.add_argument("--max-num-seqs", type=int, default=256)
    p.add_argument("--max-batched-tokens", type=int, default=4096)
    p.add_argument("--use-decode-cg", action="store_true", default=False,
                   help="Enable Decode CUDA Graph (v4)")
    p.add_argument("--use-vit-cg", action="store_true", default=False,
                   help="Enable ViT CUDA Graph (v4)")
    p.add_argument("--out", default="bench/results/mm_compare_full.json")
    p.add_argument("--engines", default="v4,official",
                   help="Comma-separated: v4, official, or both")
    args = p.parse_args()

    all_results = []

    if "v4" in args.engines:
        v4_results = bench_v4(args)
        all_results.extend(v4_results)

    if "official" in args.engines:
        off_results = bench_official(args)
        all_results.extend(off_results)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {args.out}")
    print(f"Total results: {len(all_results)}")


if __name__ == "__main__":
    main()
