"""Official vLLM multimodal bench (run under .venv-vllm).

Official vLLM does not expose ViT vs LM-prefill splits in the public API, so we
report the closest apples-to-apples metrics:
  - ttft_s      : wall time from generate() start → first token (approx via
                  one-token then continue, or full e2e first-token estimate)
  - decode_tok_s: (gen_tokens-1) / (total - ttft) for multi-token runs
  - total_s / e2e_tok_s / peak mem via torch

Usage:
  source ../.venv-vllm/bin/activate && source ../.env.vllm.sh
  python bench/mm_compare_official.py \\
    --model /mnt/nfs/zhangli/huggingface/Qwen3-VL-8B-Instruct \\
    --image cat.jpg --out bench/results/official_mm.json
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

# NVML driver/library mismatch on this host breaks vLLM's CUDA platform
# autodetect (returns UnspecifiedPlatform → empty device_type). Force CUDA
# when torch can see a GPU — no sudo / reboot required.
os.environ.setdefault("CUDA_VISIBLE_DEVICES", os.environ.get("CUDA_VISIBLE_DEVICES", "0"))

import torch

torch.cuda.init()
assert torch.cuda.is_available(), "CUDA required for official vLLM bench"


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


def _sync() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _load_image(path: str, max_side: int) -> Image.Image:
    im = Image.open(path).convert("RGB")
    if max_side > 0 and max(im.size) > max_side:
        im = im.copy()
        im.thumbnail((max_side, max_side))
    return im


def _peak_mem_gb() -> float:
    if not torch.cuda.is_available():
        return 0.0
    return torch.cuda.max_memory_allocated() / (1024**3)


def _bench_once(llm, tokenizer, prompt: str, image, max_new: int) -> dict:
    from vllm import SamplingParams

    sp = SamplingParams(temperature=0.0, max_tokens=max_new, ignore_eos=False)

    if image is not None:
        # Chat-style multimodal prompt for Qwen3-VL.
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        # Prefer processor chat template when available.
        try:
            rendered = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        except Exception:
            rendered = f"<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>{prompt}<|im_end|>\n<|im_start|>assistant\n"
        req = {
            "prompt": rendered,
            "multi_modal_data": {"image": image},
        }
        n_images = 1
    else:
        req = {"prompt": prompt}
        n_images = 0

    # Two-phase timing to approximate TTFT without internal hooks:
    #   1) max_tokens=1  → TTFT ≈ encode + prefill + 1 decode
    #   2) remaining tokens with prefix cache (if enabled) / fresh run
    # For fairness we also measure a single full generate for e2e numbers.
    _sync()
    t0 = time.perf_counter()
    outs = llm.generate([req], SamplingParams(temperature=0.0, max_tokens=1))
    _sync()
    ttft = time.perf_counter() - t0
    first_text = outs[0].outputs[0].text
    first_ids = list(outs[0].outputs[0].token_ids)

    # Full generate for decode throughput (independent run, same prompt).
    _sync()
    t1 = time.perf_counter()
    outs2 = llm.generate([req], sp)
    _sync()
    total = time.perf_counter() - t1
    out = outs2[0].outputs[0]
    gen_ids = list(out.token_ids)
    n = len(gen_ids)
    # Approximate decode-only: total - ttft_of_1tok (same encode+prefill cost).
    decode_s = max(1e-6, total - ttft)
    decode_tok_s = max(0, n - 1) / decode_s if n > 1 else (1.0 / total if total > 0 else 0.0)

    prompt_lens = getattr(outs2[0], "prompt_token_ids", None)
    prompt_tokens = len(prompt_lens) if prompt_lens is not None else -1

    return {
        "prompt_tokens": prompt_tokens,
        "gen_tokens": n,
        "gen_ids": gen_ids,
        "total_s": total,
        "ttft_s": ttft,
        "vit_encode_s": None,  # not exposed by official API
        "lm_prefill_s": None,
        "prefill_s": ttft,  # encode+prefill+1tok upper bound
        "decode_s": decode_s,
        "prefill_tok_s": None,
        "decode_tok_s": decode_tok_s,
        "e2e_tok_s": (n / total) if total > 0 else 0.0,
        "text": out.text,
        "first_token_text": first_text,
        "num_images": n_images,
        "peak_mem_gb": _peak_mem_gb(),
    }


def _mean(rows: list[dict], key: str) -> float | None:
    vals = [r[key] for r in rows if r.get(key) is not None]
    if not vals:
        return None
    return sum(float(v) for v in vals) / len(vals)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--image", default="cat.jpg")
    ap.add_argument(
        "--max-side",
        type=int,
        default=768,
        help="resize image so max(H,W)<=this (0=keep original).",
    )
    ap.add_argument("--max-new", type=int, default=32)
    ap.add_argument("--warmup", type=int, default=1)
    ap.add_argument("--iters", type=int, default=3)
    ap.add_argument("--gpu-util", type=float, default=0.70)
    ap.add_argument("--max-model-len", type=int, default=16384)
    ap.add_argument("--out", default="bench/results/official_mm.json")
    ap.add_argument("--prompt", default="Describe this image in one short sentence.")
    args = ap.parse_args()

    from vllm import LLM

    img_path = Path(args.image)
    if not img_path.is_file():
        raise SystemExit(f"image not found: {img_path}")
    image = _load_image(str(img_path), args.max_side)
    print(f"[official] image {img_path} -> size={image.size} max_side={args.max_side}")

    print(f"[official] loading {args.model} ...")
    torch.cuda.reset_peak_memory_stats()
    llm = LLM(
        model=args.model,
        dtype="float16",
        trust_remote_code=True,
        gpu_memory_utilization=args.gpu_util,
        max_model_len=args.max_model_len,
        limit_mm_per_prompt={"image": 1},
        enforce_eager=False,
    )
    tokenizer = llm.get_tokenizer()
    print("[official] ready")

    cases = {
        "mm_image": image,
        "text_only": None,
    }
    text_prompt = "The capital of France is"

    report = {
        "backend": "vllm-official",
        "vllm_version": __import__("vllm").__version__,
        "model": args.model,
        "image": str(img_path),
        "max_new": args.max_new,
        "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        "cases": {},
    }

    for name, img in cases.items():
        prompt = args.prompt if img is not None else text_prompt
        print(f"[official] case={name} warmup={args.warmup} iters={args.iters}")
        for _ in range(args.warmup):
            _bench_once(llm, tokenizer, prompt, img, args.max_new)
        rows = [
            _bench_once(llm, tokenizer, prompt, img, args.max_new)
            for _ in range(args.iters)
        ]
        summary = {
            "prompt_tokens_mean": _mean(rows, "prompt_tokens"),
            "gen_tokens_mean": _mean(rows, "gen_tokens"),
            "ttft_s": _mean(rows, "ttft_s"),
            "vit_encode_s": None,
            "lm_prefill_s": None,
            "prefill_s": _mean(rows, "prefill_s"),
            "decode_s": _mean(rows, "decode_s"),
            "decode_tok_s": _mean(rows, "decode_tok_s"),
            "prefill_tok_s": None,
            "e2e_tok_s": _mean(rows, "e2e_tok_s"),
            "peak_mem_gb": max(r["peak_mem_gb"] for r in rows),
            "text_sample": rows[-1]["text"],
            "gen_ids_sample": rows[-1]["gen_ids"],
            "iters": rows,
        }
        report["cases"][name] = summary
        print(
            f"  ttft={summary['ttft_s']*1000:.1f}ms  "
            f"decode={summary['decode_tok_s']:.1f} tok/s  "
            f"text={summary['text_sample'][:80]!r}"
        )

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"[official] wrote {out}")


if __name__ == "__main__":
    main()
