#!/usr/bin/env python3
"""Verify vision encode: eager vs CUDA Graph vs HF (bit-exact where expected)."""
from __future__ import annotations

import glob
import sys
import time
from pathlib import Path

import torch
from PIL import Image
from safetensors import safe_open
from transformers import AutoConfig, AutoProcessor
from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLVisionModel

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

MODEL = "/mnt/nfs/zhangli/huggingface/Qwen3-VL-8B-Instruct"
DEVICE = "cuda:0"
DTYPE = torch.bfloat16


def _load_sd():
    sd = {}
    for f in sorted(glob.glob(f"{MODEL}/model*.safetensors")):
        with safe_open(f, framework="pt") as s:
            for k in s.keys():
                if k.startswith("model.visual."):
                    sd[k[len("model.visual.") :]] = s.get_tensor(k)
    return sd


@torch.inference_mode()
def main():
    from src.multimodal.image import convert_image_mode, normalize_image
    from src.multimodal.vision import EncoderCache, Qwen3VisionTower

    torch.cuda.set_device(DEVICE)
    vc = AutoConfig.from_pretrained(MODEL, trust_remote_code=True).vision_config
    vc._attn_implementation = "flash_attention_2"
    proc = AutoProcessor.from_pretrained(MODEL, trust_remote_code=True)
    sd = _load_sd()

    # --- image prep sizes ---
    pils = [
        convert_image_mode(normalize_image(Image.new("RGB", (256, 192), (40, 80, 120))), "RGB"),
        convert_image_mode(normalize_image(Image.new("RGB", (384, 256), (10, 20, 30))), "RGB"),
    ]
    enc = proc.image_processor(images=pils, return_tensors="pt")
    pv = enc["pixel_values"].to(DEVICE, dtype=DTYPE)
    grid = enc["image_grid_thw"].to(DEVICE)
    print(f"batch pixel={tuple(pv.shape)} grid={grid.tolist()}")

    # --- HF reference ---
    hf = Qwen3VLVisionModel(vc).to(DEVICE, dtype=DTYPE).eval()
    hf.load_state_dict(sd, strict=False)
    dim = vc.hidden_size // vc.num_heads // 2
    hf.rotary_pos_emb.inv_freq = 1.0 / (
        10000 ** (torch.arange(0, dim, 2, dtype=torch.float, device=DEVICE) / dim)
    )
    hf_out = hf(pv, grid_thw=grid)
    hf_merged = hf_out.pooler_output
    hf_ds = list(hf_out.deepstack_features or [])

    # --- v4 tower ---
    tower = Qwen3VisionTower(vc)
    tower.load_weights({f"model.visual.{k}": t for k, t in sd.items()}, vc)
    tower.vit = tower.vit.to(device=DEVICE, dtype=DTYPE)

    eager = tower.encode(pv, grid, use_cudagraph=False)
    d_e = (eager.embeds.float() - hf_merged.float()).abs()
    print(f"[eager vs HF] merged max={d_e.max().item():.6g} mean={d_e.mean().item():.6g}")
    ok = torch.equal(eager.embeds, hf_merged)
    for i, (a, b) in enumerate(zip(hf_ds, eager.deepstack or [])):
        eq = torch.equal(a, b)
        print(f"[eager vs HF] deepstack[{i}] equal={eq}")
        ok = ok and eq
    print(f"[eager vs HF] merged equal={torch.equal(eager.embeds, hf_merged)}")

    # Free HF before CG capture (3090 / shared GPUs are tight).
    del hf, hf_out, hf_merged, hf_ds
    torch.cuda.empty_cache()

    # --- CUDA Graph (max_batch=1 keeps memory modest; multi-image packs per item) ---
    tower.warm_cudagraph(token_budgets=[128, 256, 512], max_batch_size=1)
    assert tower.cudagraph is not None and tower.cudagraph.enabled
    _ = tower.encode(pv, grid, use_cudagraph=True)
    torch.cuda.synchronize()
    cg = tower.encode(pv, grid, use_cudagraph=True)
    d_c = (cg.embeds.float() - eager.embeds.float()).abs()
    # Official CG bakes max_seqlen at capture → bf16-level diffs vs eager are expected.
    ok_cg = bool(torch.allclose(cg.embeds, eager.embeds, rtol=1e-2, atol=2e-2))
    print(
        f"[CG vs eager] merged max={d_c.max().item():.6g} mean={d_c.mean().item():.6g} "
        f"allclose={ok_cg} hits={tower.cudagraph.hits}"
    )
    for i, (a, b) in enumerate(zip(eager.deepstack or [], cg.deepstack or [])):
        ad = (a.float() - b.float()).abs()
        print(
            f"[CG vs eager] ds[{i}] max={ad.max().item():.6g} "
            f"allclose={torch.allclose(a,b,rtol=1e-2,atol=2e-2)}"
        )

    # --- EncoderCache token budget ---
    cache = EncoderCache(max_tokens=128)
    cache.put("a", eager.embeds[:30].clone())
    cache.retain("a", "req1")
    cache.put("b", eager.embeds[:80].clone())  # may over-commit if a retained
    print(f"[cache] entries={len(cache)} used={cache.used_tokens} has_a={'a' in cache}")
    cache.release("req1")
    cache.put("c", eager.embeds[:100].clone())
    print(f"[cache] after release+put c: entries={len(cache)} used={cache.used_tokens} keys={list(cache._cache)}")

    # --- microbench ---
    def bench(fn, n=10):
        for _ in range(3):
            fn()
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(n):
            fn()
        torch.cuda.synchronize()
        return (time.perf_counter() - t0) / n * 1e3

    t_eager = bench(lambda: tower.encode(pv, grid, use_cudagraph=False))
    t_cg = bench(lambda: tower.encode(pv, grid, use_cudagraph=True))
    print(f"[bench] eager={t_eager:.2f}ms  cudagraph={t_cg:.2f}ms  speedup={t_eager/max(t_cg,1e-6):.2f}x")

    print("=" * 60)
    status = "PASS" if ok and ok_cg else "FAIL"
    print(f"RESULT {status} eager_vs_hf={ok} cg_vs_eager={ok_cg}")
    sys.exit(0 if status == "PASS" else 1)


if __name__ == "__main__":
    main()
