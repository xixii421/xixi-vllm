"""Workload compare: text-only / single-image / image+text mixed.

Reports TTFT, prefill, decode, e2e for eager and both_cg.

Usage:
  CUDA_VISIBLE_DEVICES=1 PYTHONPATH=. python bench/bench_vl_workloads.py \\
    --model ~/huggingface/Qwen3-VL-2B-Instruct
"""

from __future__ import annotations

import argparse
import gc
import json
import time
from pathlib import Path

import torch
from PIL import Image, ImageDraw


def _img(size: int = 128) -> Image.Image:
    im = Image.new("RGB", (size, size), (40, 100, 200))
    d = ImageDraw.Draw(im)
    d.rectangle([16, 16, size - 16, size - 16], fill=(220, 60, 60))
    d.ellipse([size // 4, size // 4, 3 * size // 4, 3 * size // 4], fill=(240, 220, 40))
    return im


def _sync() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _pick_image(model_path: str) -> Image.Image:
    """Prefer grid [1,16,16] so ViT CG can fire under both_cg."""
    from src.multimodal import MultiModalProcessor

    proc = MultiModalProcessor(model_path)
    for size in (128, 112, 140, 168, 224, 256):
        _, _, info = proc.process("x", [_img(size)])
        g = info[0]["grid_thw"].view(-1).tolist()
        if g == [1, 16, 16]:
            print(f"  image size={size} grid={g}")
            return _img(size)
    print("  WARN: no [1,16,16] grid; using 128")
    return _img(128)


MIXED_PROMPT = (
    "Look carefully at the image. "
    "1) List the main colors you see. "
    "2) Describe the relative positions of the shapes. "
    "3) Say whether the background is cool-toned or warm-toned. "
    "Answer in three short sentences."
)


def _workloads(image: Image.Image) -> dict[str, tuple[str, list | None]]:
    return {
        "text_only": (
            "Write a short paragraph about why efficient LLM serving matters. "
            "Keep it concrete and under 80 words.",
            None,
        ),
        "single_image": (
            "What colors are in this image? Answer briefly.",
            [image],
        ),
        "image_text_mixed": (
            MIXED_PROMPT,
            [image],
        ),
    }


def _bench_once(eng, prompt: str, images, max_new: int) -> dict:
    from src.sampling_params import SamplingParams

    sp = SamplingParams(max_tokens=max_new, temperature=0.0, ignore_eos=True)
    seq = eng.add_request(prompt, sp, images=images)

    prefill_s = decode_s = 0.0
    prefill_tok = decode_steps = 0
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

    total_s = time.perf_counter() - t_start
    return {
        "prompt_tokens": seq.num_prompt_tokens,
        "gen_tokens": seq.num_completion_tokens,
        "total_s": total_s,
        "ttft_s": first_token_s if first_token_s is not None else prefill_s,
        "prefill_s": prefill_s,
        "decode_s": decode_s,
        "prefill_tok_s": (prefill_tok / prefill_s) if prefill_s > 0 else 0.0,
        "decode_tok_s": (decode_steps / decode_s) if decode_s > 0 else 0.0,
        "e2e_tok_s": (seq.num_completion_tokens / total_s) if total_s > 0 else 0.0,
        "num_images": 0 if not images else len(images),
    }


def _avg(rows: list[dict], keys: list[str]) -> dict:
    out = {k: sum(r[k] for r in rows) / len(rows) for k in keys}
    out["num_images"] = rows[0]["num_images"]
    return out


def _make_engine(model: str, eager: bool, vit_cg: bool, gpu_util: float):
    from src.config import EngineConfig
    from src.engine.engine import Engine

    return Engine(
        EngineConfig(
            model=model,
            dtype="float16",
            enforce_eager=eager,
            enable_vit_cuda_graph=vit_cg,
            gpu_memory_utilization=gpu_util,
            max_num_seqs=4,
            max_num_batched_tokens=16384,
        )
    )


def _destroy(eng) -> None:
    try:
        eng.model_runner.destroy()
    except Exception:
        pass
    del eng
    gc.collect()
    torch.cuda.empty_cache()
    _sync()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=str(Path.home() / "huggingface/Qwen3-VL-2B-Instruct"))
    ap.add_argument("--max-new", type=int, default=64)
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--iters", type=int, default=5)
    ap.add_argument("--gpu-util", type=float, default=0.55)
    ap.add_argument(
        "--out",
        default=str(Path(__file__).resolve().parent / "results/vl_workloads_perf.json"),
    )
    args = ap.parse_args()

    print("Picking image …")
    image = _pick_image(args.model)
    workloads = _workloads(image)
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

    engine_modes = (
        ("eager", True, False),
        ("both_cg", False, True),
    )

    report = {
        "model": args.model,
        "max_new": args.max_new,
        "ignore_eos": True,
        "warmup": args.warmup,
        "iters": args.iters,
        "results": {},
    }

    for eng_name, eager, vit_cg in engine_modes:
        print(f"\n======== ENGINE {eng_name} ========")
        eng = _make_engine(args.model, eager, vit_cg, args.gpu_util)
        report["results"][eng_name] = {}
        for wl_name, (prompt, images) in workloads.items():
            for _ in range(args.warmup):
                _bench_once(eng, prompt, images, max_new=min(16, args.max_new))
            rows = [
                _bench_once(eng, prompt, images, max_new=args.max_new)
                for _ in range(args.iters)
            ]
            avg = _avg(rows, keys)
            report["results"][eng_name][wl_name] = avg
            print(
                f"  [{wl_name}] imgs={avg['num_images']} prompt={avg['prompt_tokens']:.0f} "
                f"TTFT={avg['ttft_s']*1000:.2f}ms "
                f"prefill={avg['prefill_tok_s']:.0f} tok/s ({avg['prefill_s']*1000:.1f}ms) "
                f"decode={avg['decode_tok_s']:.1f} tok/s "
                f"e2e={avg['e2e_tok_s']:.1f} tok/s"
            )
        _destroy(eng)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False))

    md = out.with_suffix(".md")
    lines = [
        "# VL workloads: text-only / single-image / image+text mixed",
        "",
        f"- model: `{args.model}`",
        f"- max_new={args.max_new}, ignore_eos=True, iters={args.iters}",
        "",
    ]
    for eng_name, block in report["results"].items():
        lines += [
            f"## Engine: `{eng_name}`",
            "",
            "| workload | images | prompt_tok | TTFT_ms | prefill_ms | prefill_tok/s | decode_tok/s | e2e_tok/s |",
            "| --- | --- | --- | --- | --- | --- | --- | --- |",
        ]
        for wl, m in block.items():
            lines.append(
                f"| {wl} | {m['num_images']} | {m['prompt_tokens']:.0f} | "
                f"{m['ttft_s']*1000:.2f} | {m['prefill_s']*1000:.1f} | "
                f"{m['prefill_tok_s']:.0f} | {m['decode_tok_s']:.1f} | {m['e2e_tok_s']:.1f} |"
            )
        lines.append("")
    md.write_text("\n".join(lines))
    print(f"\nWrote {out} and {md}")
    print("\n" + "\n".join(lines))


if __name__ == "__main__":
    main()
