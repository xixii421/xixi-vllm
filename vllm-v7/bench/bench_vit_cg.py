"""Bench eager ViT vs budget CUDA-Graph ViT (v5).

Measures:
  1) Pure ViT encode latency (encode only, EncoderCache off)
  2) End-to-end TTFT proxy: first new token after multimodal prefill

Usage:
  CUDA_VISIBLE_DEVICES=1 python bench/bench_vit_cg.py \\
    --model ~/huggingface/Qwen3-VL-2B-Instruct
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import torch
from PIL import Image, ImageDraw


def _make_image(size: int = 256) -> Image.Image:
    im = Image.new("RGB", (size, size), (30, 120, 200))
    d = ImageDraw.Draw(im)
    d.rectangle([40, 40, size - 40, size - 40], fill=(220, 60, 60))
    d.ellipse([size // 4, size // 4, 3 * size // 4, 3 * size // 4], fill=(240, 220, 40))
    return im


def _sync() -> None:
    torch.cuda.synchronize()


def _time_cuda(fn, warmup: int, iters: int) -> list[float]:
    for _ in range(warmup):
        fn()
    _sync()
    samples = []
    for _ in range(iters):
        _sync()
        t0 = time.perf_counter()
        fn()
        _sync()
        samples.append((time.perf_counter() - t0) * 1000.0)
    return samples


def _stats(ms: list[float]) -> dict:
    return {
        "mean_ms": round(statistics.mean(ms), 3),
        "median_ms": round(statistics.median(ms), 3),
        "p90_ms": round(sorted(ms)[max(0, int(len(ms) * 0.9) - 1)], 3),
        "min_ms": round(min(ms), 3),
        "max_ms": round(max(ms), 3),
        "iters": len(ms),
    }


def bench_vit_encode(model_path: str, image: Image.Image, warmup: int, iters: int) -> dict:
    from src.config import EngineConfig
    from src.models.qwen3_vl import Qwen3VLForCausalLM
    from src.multimodal import MultiModalProcessor
    from src.cuda_graph.vit import ViTCUDAGraphRunner

    device = torch.device("cuda:0")
    cfg = EngineConfig(model=model_path, enforce_eager=True, gpu_memory_utilization=0.35)
    model = Qwen3VLForCausalLM(cfg.hf_config)
    model.load_weights(model_path, device, torch.float16)
    model.eval()

    proc = MultiModalProcessor(model_path)
    _, pv, info = proc.process("describe", [image])
    assert pv is not None and info
    grid = info[0]["grid_thw"].to(device).view(-1, 3)
    pv = pv.to(device=device, dtype=torch.float16)
    print(f"  image grid={grid.tolist()} tokens={info[0]['num_encoder_tokens']}")

    def eager():
        with torch.inference_mode():
            model.vision_tower(pv, grid_thw=grid)

    eager_ms = _time_cuda(eager, warmup, iters)

    # Capture ViT CG on same model.
    vcfg = model.get_vit_cg_config()
    runner = ViTCUDAGraphRunner(
        vit_forward=model.vit_forward_cg,
        metadata_fn=model.vision_tower.prepare_metadata,
        config=vcfg,
        device=device,
        dtype=torch.float16,
    )
    runner.capture()
    model._vit_cg = runner
    assert runner.can_use(grid), (
        f"ViT CG cannot cover this grid {grid.tolist()}; "
        "bench image must match base_grid [1,16,16]"
    )

    def cg():
        with torch.inference_mode():
            runner.replay(pv, grid)

    cg_ms = _time_cuda(cg, warmup, iters)

    # Correctness: max |eager - cg|
    with torch.inference_mode():
        e = model.vision_tower(pv, grid_thw=grid)
        e_flat = torch.cat(list(e), 0)
        e_dps = [t.clone() for t in (model.vision_tower.deepstack_features or [])]
        c_flat, c_dps = runner.replay(pv, grid)
    diff = (e_flat.float() - c_flat.float()).abs().max().item()
    dps_diff = 0.0
    if e_dps and c_dps:
        dps_diff = max(
            (a.float() - b.float()).abs().max().item() for a, b in zip(e_dps, c_dps)
        )

    eager_s, cg_s = _stats(eager_ms), _stats(cg_ms)
    speedup = eager_s["mean_ms"] / cg_s["mean_ms"] if cg_s["mean_ms"] > 0 else float("inf")
    out = {
        "eager": eager_s,
        "vit_cg": cg_s,
        "speedup_x": round(speedup, 3),
        "max_abs_diff_merged": diff,
        "max_abs_diff_deepstack": dps_diff,
        "grid": grid.tolist(),
        "num_tokens": info[0]["num_encoder_tokens"],
    }
    # Drop heavy objects before returning so the next bench phase has headroom.
    runner.destroy()
    del runner, model
    import gc

    gc.collect()
    torch.cuda.empty_cache()
    return out


def bench_ttft(model_path: str, image: Image.Image, warmup: int, iters: int) -> dict:
    """Time multimodal prefill+first sample token (EncoderCache disabled each iter)."""
    from src.config import EngineConfig
    from src.runtime.engine import Engine
    from src.sampling_params import SamplingParams

    prompt = "What colors are in this image? Answer briefly."

    def run(enable_cg: bool) -> list[float]:
        eng = Engine(
            EngineConfig(
                model=model_path,
                enforce_eager=True,  # isolate ViT CG from decode CG
                enable_vit_cuda_graph=enable_cg,
                gpu_memory_utilization=0.45,
                max_num_seqs=1,
            )
        )
        sp = SamplingParams(temperature=0.0, max_tokens=1)

        def one():
            eng.model_runner.model._encoder_cache.clear()
            eng.add_request(prompt, sp, images=[image])
            while not eng.scheduler.is_finished():
                eng.step()

        samples = _time_cuda(one, warmup=max(1, warmup // 2), iters=iters)
        eng.model_runner.destroy()
        del eng
        import gc

        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        return samples

    eager_ms = run(False)
    cg_ms = run(True)
    es, cs = _stats(eager_ms), _stats(cg_ms)
    return {
        "eager_ttft_proxy": es,
        "vit_cg_ttft_proxy": cs,
        "speedup_x": round(es["mean_ms"] / cs["mean_ms"], 3) if cs["mean_ms"] else None,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=str(Path.home() / "huggingface/Qwen3-VL-2B-Instruct"))
    ap.add_argument("--image-size", type=int, default=256)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument(
        "--out",
        default=str(Path(__file__).resolve().parent / "results" / "vit_cg_bench.json"),
    )
    args = ap.parse_args()

    assert torch.cuda.is_available()
    img = _make_image(args.image_size)
    print("=== Pure ViT encode ===")
    vit = bench_vit_encode(args.model, img, args.warmup, args.iters)
    print(json.dumps(vit, indent=2))

    # Free ViT-only process state before loading a full Engine.
    import gc

    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()

    print("\n=== E2E first-token (TTFT proxy) ===")
    try:
        ttft = bench_ttft(args.model, img, args.warmup, max(5, args.iters // 2))
        print(json.dumps(ttft, indent=2))
    except torch.cuda.OutOfMemoryError as e:
        ttft = {"error": f"OOM during TTFT bench: {e}"}
        print(ttft["error"])

    report = {
        "model": args.model,
        "image_size": args.image_size,
        "device": torch.cuda.get_device_name(0),
        "vit_encode": vit,
        "ttft_proxy": ttft,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))
    md = out.with_suffix(".md")
    lines = [
        "# ViT CUDA Graph bench (vllm-v5)",
        "",
        f"- model: `{args.model}`",
        f"- device: {report['device']}",
        f"- image: {args.image_size}x{args.image_size} → grid {vit['grid']}",
        "",
        "## Pure ViT encode",
        "",
        "| mode | mean (ms) | median | p90 |",
        "| --- | ---: | ---: | ---: |",
        f"| eager | {vit['eager']['mean_ms']} | {vit['eager']['median_ms']} | {vit['eager']['p90_ms']} |",
        f"| ViT CG | {vit['vit_cg']['mean_ms']} | {vit['vit_cg']['median_ms']} | {vit['vit_cg']['p90_ms']} |",
        "",
        f"**Speedup (encode): {vit['speedup_x']}×**  ",
        f"max|Δ|merged={vit['max_abs_diff_merged']:.4f}, deepstack={vit['max_abs_diff_deepstack']:.4f}",
        "",
    ]
    if "error" not in ttft:
        lines += [
            "## TTFT proxy (prefill + 1 token, decode eager)",
            "",
            "| mode | mean (ms) | median |",
            "| --- | ---: | ---: |",
            f"| eager ViT | {ttft['eager_ttft_proxy']['mean_ms']} | {ttft['eager_ttft_proxy']['median_ms']} |",
            f"| ViT CG | {ttft['vit_cg_ttft_proxy']['mean_ms']} | {ttft['vit_cg_ttft_proxy']['median_ms']} |",
            "",
            f"**Speedup (TTFT proxy): {ttft['speedup_x']}×**",
            "",
        ]
    else:
        lines += ["## TTFT proxy", "", f"Skipped: {ttft['error']}", ""]
    md.write_text("\n".join(lines))
    print(f"\nWrote {out} and {md}")


if __name__ == "__main__":
    main()
