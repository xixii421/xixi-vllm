"""Multimodal end-to-end performance: TTFT / prefill / decode / e2e.

Usage:
  CUDA_VISIBLE_DEVICES=1 PYTHONPATH=. python bench/bench_vl_e2e.py \\
    --model ~/huggingface/Qwen3-VL-2B-Instruct
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from PIL import Image, ImageDraw


def _img(size: int = 256) -> Image.Image:
    im = Image.new("RGB", (size, size), (40, 100, 200))
    d = ImageDraw.Draw(im)
    d.rectangle([30, 30, size - 30, size - 30], fill=(220, 60, 60))
    d.ellipse([size // 4, size // 4, 3 * size // 4, 3 * size // 4], fill=(240, 220, 40))
    return im


def _sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _bench_once(eng, prompt: str, images, max_new: int) -> dict:
    """Instrument schedule/run to split prefill vs decode wall time."""
    from src.sampling_params import SamplingParams

    sp = SamplingParams(max_tokens=max_new, temperature=0.0)
    seq = eng.add_request(prompt, sp, images=images)

    prefill_s = decode_s = 0.0
    prefill_tok = decode_steps = 0
    n_out = 0
    first_token_s = None
    t_start = time.perf_counter()

    while not seq.is_finished:
        _sync()
        t0 = time.perf_counter()
        seqs, is_prefill = eng.scheduler.schedule()
        if not seqs:
            break
        scheduled_tok = sum(s.num_scheduled_tokens for s in seqs)
        tokens = eng.model_runner.run(seqs, is_prefill)
        eng.scheduler.postprocess(seqs, tokens, is_prefill)
        _sync()
        dt = time.perf_counter() - t0
        if is_prefill:
            prefill_s += dt
            prefill_tok += scheduled_tok
            if first_token_s is None and seq.num_completion_tokens > 0:
                first_token_s = time.perf_counter() - t_start
        else:
            decode_s += dt
            decode_steps += len(seqs)
        n_out = seq.num_completion_tokens

    total_s = time.perf_counter() - t_start
    text = eng.tokenizer.decode(seq.token_ids[seq.num_prompt_tokens :])
    return {
        "prompt_tokens": seq.num_prompt_tokens,
        "gen_tokens": n_out,
        "total_s": total_s,
        "ttft_s": first_token_s if first_token_s is not None else prefill_s,
        "prefill_s": prefill_s,
        "decode_s": decode_s,
        "prefill_tok_s": (prefill_tok / prefill_s) if prefill_s > 0 else 0.0,
        "decode_tok_s": (decode_steps / decode_s) if decode_s > 0 else 0.0,
        "e2e_tok_s": (n_out / total_s) if total_s > 0 else 0.0,
        "text": text,
        "num_images": len(images) if images else 0,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=str(Path.home() / "huggingface/Qwen3-VL-2B-Instruct"))
    ap.add_argument("--max-new", type=int, default=32)
    ap.add_argument("--image-size", type=int, default=256)
    ap.add_argument("--warmup", type=int, default=1)
    ap.add_argument("--iters", type=int, default=3)
    ap.add_argument("--gpu-util", type=float, default=0.55)
    ap.add_argument(
        "--out",
        default=str(Path(__file__).resolve().parent / "results/vl_e2e_perf.json"),
    )
    args = ap.parse_args()

    from src.config import EngineConfig
    from src.engine.engine import Engine

    prompt = "What colors are in this image? Answer briefly."
    cases = {
        "single_image": [_img(args.image_size)],
        "multi_image_2": [_img(args.image_size), _img(args.image_size // 2)],
        "text_only": None,
    }

    eng = Engine(
        EngineConfig(
            model=args.model,
            dtype="float16",
            enforce_eager=True,
            enable_vit_cuda_graph=False,
            gpu_memory_utilization=args.gpu_util,
            max_num_seqs=4,
            max_num_batched_tokens=16384,
        )
    )

    report = {"model": args.model, "max_new": args.max_new, "cases": {}}
    for name, images in cases.items():
        p = prompt if images else "Say hello in one short sentence."
        for _ in range(args.warmup):
            _bench_once(eng, p, images, max_new=8)
        rows = [_bench_once(eng, p, images, max_new=args.max_new) for _ in range(args.iters)]
        keys = [
            "prompt_tokens",
            "gen_tokens",
            "total_s",
            "ttft_s",
            "prefill_s",
            "decode_s",
            "prefill_tok_s",
            "decode_tok_s",
            "e2e_tok_s",
        ]
        avg = {k: sum(r[k] for r in rows) / len(rows) for k in keys}
        avg["sample_text"] = rows[-1]["text"]
        avg["num_images"] = rows[-1]["num_images"]
        report["cases"][name] = avg
        print(
            f"[{name}] prompt={avg['prompt_tokens']:.0f} gen={avg['gen_tokens']:.0f} "
            f"TTFT={avg['ttft_s']*1000:.1f}ms prefill={avg['prefill_tok_s']:.0f} tok/s "
            f"decode={avg['decode_tok_s']:.1f} tok/s e2e={avg['e2e_tok_s']:.1f} tok/s "
            f"text={avg['sample_text']!r}"
        )

    # Concurrent batch throughput (2 single-image reqs)
    from src.sampling_params import SamplingParams

    imgs_batch = [[_img(args.image_size)], [_img(args.image_size)]]
    prompts = [prompt, prompt]
    for _ in range(args.warmup):
        eng.generate_batch(prompts, SamplingParams(max_tokens=8, temperature=0.0), images=imgs_batch)
    _sync()
    t0 = time.perf_counter()
    outs = eng.generate_batch(
        prompts, SamplingParams(max_tokens=args.max_new, temperature=0.0), images=imgs_batch
    )
    _sync()
    batch_s = time.perf_counter() - t0
    # Approximate generated tokens from decoded lengths (greedy EOS often < max_new).
    gen_est = sum(max(1, len(eng.tokenizer.encode(t))) for t in outs)
    report["cases"]["concurrent_2x1"] = {
        "batch_s": batch_s,
        "reqs": 2,
        "max_new": args.max_new,
        "gen_tokens_est": gen_est,
        "agg_tok_s": gen_est / batch_s if batch_s > 0 else 0.0,
        "sample_texts": outs,
    }
    print(
        f"[concurrent_2x1] batch={batch_s:.3f}s gen≈{gen_est} "
        f"agg≈{report['cases']['concurrent_2x1']['agg_tok_s']:.1f} tok/s "
        f"texts={outs!r}"
    )

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False))

    md = out.with_suffix(".md")
    lines = [
        "# Qwen3-VL e2e performance",
        "",
        f"- model: `{args.model}`",
        f"- max_new={args.max_new}, image_size={args.image_size}, iters={args.iters}",
        "",
        "| case | images | prompt_tok | TTFT_ms | prefill_tok/s | decode_tok/s | e2e_tok/s | sample |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for name, c in report["cases"].items():
        if name == "concurrent_2x1":
            continue
        lines.append(
            f"| {name} | {c['num_images']} | {c['prompt_tokens']:.0f} | "
            f"{c['ttft_s']*1000:.1f} | {c['prefill_tok_s']:.0f} | "
            f"{c['decode_tok_s']:.1f} | {c['e2e_tok_s']:.1f} | `{c['sample_text'][:48]}` |"
        )
    c = report["cases"]["concurrent_2x1"]
    lines += [
        "",
        "## Concurrent batch (2×1 image)",
        f"- wall={c['batch_s']:.3f}s, aggregate gen≈{c['agg_tok_s']:.1f} tok/s",
        f"- texts: {c['sample_texts']!r}",
        "",
    ]
    md.write_text("\n".join(lines))
    print(f"\nWrote {out} and {md}")


if __name__ == "__main__":
    main()
