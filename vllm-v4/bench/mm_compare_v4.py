"""vllm-v4 multimodal stage bench (no nvidia-smi required).

Splits wall time with torch.cuda.synchronize():
  - vit_encode_s   : vision tower (+ EncoderCache miss)
  - lm_prefill_s   : LM prefill after embeds are ready
  - prefill_s      : vit + lm (all prefill steps; chunked-prefill aware)
  - decode_s       : decode steps
  - ttft_s         : time to first completion token

Also runs a text-only case on the same Engine for baseline.

Usage:
  source ../.venv/bin/activate && source ../.env.cuda.sh
  PYTHONPATH=. python bench/mm_compare_v4.py \\
    --model /mnt/nfs/zhangli/huggingface/Qwen3-VL-8B-Instruct \\
    --image cat.jpg --out bench/results/v4_mm.json
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


def _peak_mem_gb() -> float:
    if not torch.cuda.is_available():
        return 0.0
    return torch.cuda.max_memory_allocated() / (1024**3)


def _bench_once(eng, prompt: str, images, max_new: int) -> dict:
    """Instrument Engine path to split ViT / LM-prefill / decode."""
    from src.sampling_params import SamplingParams

    sp = SamplingParams(max_tokens=max_new, temperature=0.0)
    seq = eng.add_request(prompt, sp, images=images)

    vit_s = lm_prefill_s = decode_s = 0.0
    prefill_tok = decode_steps = 0
    first_token_s = None
    t_start = time.perf_counter()
    is_mm = bool(images)

    # Patch encode_and_merge to time ViT separately when multimodal.
    orig_encode = None
    if is_mm and hasattr(eng.model, "encode_and_merge"):
        orig_encode = eng.model.encode_and_merge

        def _timed_encode(*a, **kw):
            nonlocal vit_s
            _sync()
            t0 = time.perf_counter()
            out = orig_encode(*a, **kw)
            _sync()
            vit_s += time.perf_counter() - t0
            return out

        eng.model.encode_and_merge = _timed_encode

    prefill_s = 0.0
    try:
        while not seq.is_finished:
            prefills, decodes = eng.scheduler.schedule()
            if not prefills and not decodes:
                break

            # Time decode / prefill separately (model_runner.run does decode then
            # prefill as two forwards; mixed PD must not attribute all to decode).
            d_ids: list[int] = []
            p_ids: list[int] = []
            if decodes:
                _sync()
                t0 = time.perf_counter()
                d_ids = eng.model_runner._run_batch(decodes, is_prefill=False)
                _sync()
                decode_s += time.perf_counter() - t0
                decode_steps += len(decodes)
                eng.scheduler.postprocess(decodes, d_ids, is_prefill=False)
            if prefills:
                vit_before = vit_s
                _sync()
                t0 = time.perf_counter()
                p_ids = eng.model_runner._run_batch(prefills, is_prefill=True)
                _sync()
                dt = time.perf_counter() - t0
                prefill_s += dt
                prefill_tok += sum(s.num_scheduled_tokens for s in prefills)
                # This step's LM prefill ≈ wall − ViT work that happened inside.
                lm_prefill_s += max(0.0, dt - (vit_s - vit_before))
                eng.scheduler.postprocess(prefills, p_ids, is_prefill=True)

            if first_token_s is None and seq.num_completion_tokens > 0:
                first_token_s = time.perf_counter() - t_start
    finally:
        if orig_encode is not None:
            eng.model.encode_and_merge = orig_encode

    total_s = time.perf_counter() - t_start

    text = eng.tokenizer.decode(seq.token_ids[seq.num_prompt_tokens :])
    return {
        "prompt_tokens": seq.num_prompt_tokens,
        "gen_tokens": seq.num_completion_tokens,
        "gen_ids": list(seq.token_ids[seq.num_prompt_tokens :]),
        "total_s": total_s,
        "ttft_s": first_token_s if first_token_s is not None else prefill_s,
        "vit_encode_s": vit_s,
        "lm_prefill_s": lm_prefill_s,
        "prefill_s": prefill_s,
        "decode_s": decode_s,
        "prefill_tok_s": (prefill_tok / prefill_s) if prefill_s > 0 else 0.0,
        "decode_tok_s": (decode_steps / decode_s) if decode_s > 0 else 0.0,
        "e2e_tok_s": (seq.num_completion_tokens / total_s) if total_s > 0 else 0.0,
        "text": text,
        "num_images": len(images) if images else 0,
        "peak_mem_gb": _peak_mem_gb(),
    }


def _mean(rows: list[dict], key: str) -> float:
    vals = [float(r[key]) for r in rows if r.get(key) is not None]
    return sum(vals) / len(vals) if vals else 0.0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--image", default="cat.jpg")
    ap.add_argument("--max-new", type=int, default=32)
    ap.add_argument("--warmup", type=int, default=1)
    ap.add_argument("--iters", type=int, default=3)
    ap.add_argument("--gpu-util", type=float, default=0.70)
    ap.add_argument("--compile", action="store_true", help="enable Decode CUDA Graph")
    ap.add_argument("--vit-cg", action="store_true", help="enable ViT CUDA Graph")
    ap.add_argument("--out", default="bench/results/v4_mm.json")
    ap.add_argument("--prompt", default="Describe this image in one short sentence.")
    args = ap.parse_args()

    from src.config import EngineConfig
    from src.engine.engine import Engine

    img_path = Path(args.image)
    if not img_path.is_file():
        raise SystemExit(f"image not found: {img_path}")
    image = Image.open(img_path).convert("RGB")

    print(f"[v4] loading {args.model} ...")
    torch.cuda.reset_peak_memory_stats()
    eng = Engine(
        EngineConfig(
            model=args.model,
            dtype="float16",
            enforce_eager=not args.compile,
            enable_vit_cuda_graph=args.vit_cg,
            gpu_memory_utilization=args.gpu_util,
            max_num_seqs=4,
            max_num_batched_tokens=16384,
            context_len=8192,
            mix_prefill_decode=True,
            prefix_backend="hash",
        )
    )
    print(
        f"[v4] ready is_mm={eng.config.is_multimodal} "
        f"eager={eng.config.enforce_eager} vit_cg={args.vit_cg}"
    )

    cases = {
        "mm_image": [image],
        "text_only": None,
    }
    text_prompt = "The capital of France is"

    report = {
        "backend": "vllm-v4",
        "model": args.model,
        "image": str(img_path),
        "max_new": args.max_new,
        "compile": args.compile,
        "vit_cg": args.vit_cg,
        "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        "cases": {},
    }

    for name, images in cases.items():
        prompt = args.prompt if images else text_prompt
        print(f"[v4] case={name} warmup={args.warmup} iters={args.iters}")
        for _ in range(args.warmup):
            _bench_once(eng, prompt, images, args.max_new)
        rows = [_bench_once(eng, prompt, images, args.max_new) for _ in range(args.iters)]
        summary = {
            "prompt_tokens_mean": _mean(rows, "prompt_tokens"),
            "gen_tokens_mean": _mean(rows, "gen_tokens"),
            "ttft_s": _mean(rows, "ttft_s"),
            "vit_encode_s": _mean(rows, "vit_encode_s"),
            "lm_prefill_s": _mean(rows, "lm_prefill_s"),
            "prefill_s": _mean(rows, "prefill_s"),
            "decode_s": _mean(rows, "decode_s"),
            "decode_tok_s": _mean(rows, "decode_tok_s"),
            "prefill_tok_s": _mean(rows, "prefill_tok_s"),
            "e2e_tok_s": _mean(rows, "e2e_tok_s"),
            "peak_mem_gb": max(r["peak_mem_gb"] for r in rows),
            "text_sample": rows[-1]["text"],
            "gen_ids_sample": rows[-1]["gen_ids"],
            "iters": rows,
        }
        report["cases"][name] = summary
        print(
            f"  ttft={summary['ttft_s']*1000:.1f}ms  "
            f"vit={summary['vit_encode_s']*1000:.1f}ms  "
            f"lm_prefill={summary['lm_prefill_s']*1000:.1f}ms  "
            f"decode={summary['decode_tok_s']:.1f} tok/s  "
            f"text={summary['text_sample'][:80]!r}"
        )

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"[v4] wrote {out}")
    eng.destroy()


if __name__ == "__main__":
    main()
