"""Compare four CG modes: eager / ViT-CG / text-CG / both.

Metrics: TTFT, prefill tok/s, decode tok/s, e2e tok/s
(Uses ignore_eos so decode length is fixed and fair.)

Usage:
  CUDA_VISIBLE_DEVICES=1 PYTHONPATH=. python bench/bench_cg_modes.py \\
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


MODES = (
    ("eager", dict(enforce_eager=True, enable_vit_cuda_graph=False)),
    ("vit_cg", dict(enforce_eager=True, enable_vit_cuda_graph=True)),
    ("text_cg", dict(enforce_eager=False, enable_vit_cuda_graph=False)),
    ("both_cg", dict(enforce_eager=False, enable_vit_cuda_graph=True)),
)


def _img(size: int) -> Image.Image:
    im = Image.new("RGB", (size, size), (40, 100, 200))
    d = ImageDraw.Draw(im)
    d.rectangle([20, 20, size - 20, size - 20], fill=(220, 60, 60))
    d.ellipse([size // 4, size // 4, 3 * size // 4, 3 * size // 4], fill=(240, 220, 40))
    return im


def _sync() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _pick_image_for_vit_cg(model_path: str, preferred: int = 128) -> tuple[Image.Image, list[int]]:
    """Prefer a resolution whose ``image_grid_thw`` matches ViT CG base ``(1,16,16)``."""
    from src.multimodal import MultiModalProcessor

    proc = MultiModalProcessor(model_path)
    for size in (preferred, 112, 140, 168, 196, 224, 256):
        _, pv, info = proc.process("x", [_img(size)])
        assert info and info[0].get("grid_thw") is not None
        g = info[0]["grid_thw"].view(-1).tolist()
        print(f"  probe size={size} grid={g} tokens={info[0]['num_encoder_tokens']}")
        if g == [1, 16, 16]:
            return _img(size), g
    # Fallback: still run, but ViT CG will not fire.
    print("  WARNING: no size maps to [1,16,16]; ViT CG may fall back to eager")
    return _img(preferred), g


def _bench_once(eng, prompt: str, images, max_new: int) -> dict:
    from src.sampling_params import SamplingParams

    # Fixed-length decode for comparable decode tok/s across modes.
    sp = SamplingParams(max_tokens=max_new, temperature=0.0, ignore_eos=True)
    seq = eng.add_request(prompt, sp, images=images)

    prefill_s = decode_s = 0.0
    prefill_tok = decode_steps = 0
    first_token_s = None
    t_start = time.perf_counter()
    vit_cg_hits = 0

    while not seq.is_finished:
        _sync()
        t0 = time.perf_counter()
        seqs, is_prefill = eng.scheduler.schedule()
        if not seqs:
            break
        scheduled_tok = sum(s.num_scheduled_tokens for s in seqs)
        if is_prefill and getattr(eng.model, "_vit_cg", None) is not None:
            # Heuristic: will be used inside encode if can_use; count if runner exists + mm.
            if any(s.has_multimodal for s in seqs):
                cg = eng.model._vit_cg
                # Rough: check last request grids if available after schedule doesn't expose;
                # increment intent when CG captured (actual use logged by presence).
                if cg is not None and cg._captured:
                    vit_cg_hits += 1
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
        "vit_cg_prefills": vit_cg_hits,
    }


def _avg(rows: list[dict], keys: list[str]) -> dict:
    return {k: sum(r[k] for r in rows) / len(rows) for k in keys}


def _make_engine(model: str, flags: dict, gpu_util: float):
    from src.config import EngineConfig
    from src.engine.engine import Engine

    return Engine(
        EngineConfig(
            model=model,
            dtype="float16",
            gpu_memory_utilization=gpu_util,
            max_num_seqs=4,
            max_num_batched_tokens=16384,
            **flags,
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
    ap.add_argument("--warmup", type=int, default=1)
    ap.add_argument("--iters", type=int, default=5)
    ap.add_argument("--gpu-util", type=float, default=0.55)
    ap.add_argument(
        "--out",
        default=str(Path(__file__).resolve().parent / "results/vl_cg_modes_perf.json"),
    )
    args = ap.parse_args()

    print("Selecting image size for ViT CG base grid [1,16,16] …")
    image, grid = _pick_image_for_vit_cg(args.model)
    prompt = "What colors are in this image? Answer briefly."
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

    report = {
        "model": args.model,
        "grid": grid,
        "max_new": args.max_new,
        "ignore_eos": True,
        "warmup": args.warmup,
        "iters": args.iters,
        "modes": {},
    }

    for mode_name, flags in MODES:
        print(f"\n======== MODE {mode_name} flags={flags} ========")
        eng = _make_engine(args.model, flags, args.gpu_util)
        text_cg = eng.model_runner.cuda_graph_runner is not None
        vit_cg = getattr(eng.model, "_vit_cg", None) is not None
        print(f"  runners: text_cg={text_cg} vit_cg={vit_cg}")

        for _ in range(args.warmup):
            _bench_once(eng, prompt, [image], max_new=min(16, args.max_new))

        rows = [_bench_once(eng, prompt, [image], max_new=args.max_new) for _ in range(args.iters)]
        avg = _avg(rows, keys)
        avg["text_cg"] = text_cg
        avg["vit_cg"] = vit_cg
        # Verify ViT CG actually can_use this grid when enabled
        if vit_cg:
            g = torch.tensor([grid], device=eng.device)
            avg["vit_cg_can_use"] = bool(eng.model._vit_cg.can_use(g))
        else:
            avg["vit_cg_can_use"] = False
        report["modes"][mode_name] = avg
        print(
            f"  TTFT={avg['ttft_s']*1000:.2f}ms  "
            f"prefill={avg['prefill_tok_s']:.0f} tok/s ({avg['prefill_s']*1000:.2f}ms)  "
            f"decode={avg['decode_tok_s']:.1f} tok/s ({avg['decode_s']*1000:.1f}ms)  "
            f"e2e={avg['e2e_tok_s']:.1f} tok/s  "
            f"vit_can_use={avg['vit_cg_can_use']}"
        )
        _destroy(eng)

    # Relative speedups vs eager
    base = report["modes"]["eager"]
    for name, m in report["modes"].items():
        m["speedup_ttft"] = (base["ttft_s"] / m["ttft_s"]) if m["ttft_s"] > 0 else 0.0
        m["speedup_prefill"] = (
            m["prefill_tok_s"] / base["prefill_tok_s"] if base["prefill_tok_s"] > 0 else 0.0
        )
        m["speedup_decode"] = (
            m["decode_tok_s"] / base["decode_tok_s"] if base["decode_tok_s"] > 0 else 0.0
        )
        m["speedup_e2e"] = m["e2e_tok_s"] / base["e2e_tok_s"] if base["e2e_tok_s"] > 0 else 0.0

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False))

    md = out.with_suffix(".md")
    lines = [
        "# VL CG modes: eager / ViT-CG / text-CG / both",
        "",
        f"- model: `{args.model}`",
        f"- grid: `{grid}` (ViT CG base must be `[1,16,16]`)",
        f"- max_new={args.max_new}, ignore_eos=True, iters={args.iters}",
        "",
        "| mode | text_cg | vit_cg | TTFT_ms | ×ttft | prefill_tok/s | ×prefill | decode_tok/s | ×decode | e2e_tok/s | ×e2e |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for name, m in report["modes"].items():
        lines.append(
            f"| {name} | {m['text_cg']} | {m['vit_cg_can_use']} | "
            f"{m['ttft_s']*1000:.2f} | {m['speedup_ttft']:.2f}× | "
            f"{m['prefill_tok_s']:.0f} | {m['speedup_prefill']:.2f}× | "
            f"{m['decode_tok_s']:.1f} | {m['speedup_decode']:.2f}× | "
            f"{m['e2e_tok_s']:.1f} | {m['speedup_e2e']:.2f}× |"
        )
    lines.append("")
    md.write_text("\n".join(lines))
    print(f"\nWrote {out} and {md}")
    print("\n" + "\n".join(lines))


if __name__ == "__main__":
    main()
