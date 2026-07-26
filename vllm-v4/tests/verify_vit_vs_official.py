#!/usr/bin/env python3
"""Functional A/B: v4 ViT + image prep vs official vLLM image.py and HF VisionModel."""
from __future__ import annotations

import glob
import importlib.util
import sys
from pathlib import Path

import torch
from PIL import Image
from safetensors import safe_open
from transformers import AutoConfig, AutoProcessor
from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLVisionModel

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parent
sys.path.insert(0, str(ROOT))

MODEL = "/mnt/nfs/zhangli/huggingface/Qwen3-VL-8B-Instruct"
DEVICE = "cuda:0"  # pair with CUDA_VISIBLE_DEVICES
DTYPE = torch.bfloat16
ATOL, RTOL = 0.0, 0.0  # expect bit-exact vs HF with float32 RoPE + HF bilinear


def _load_official_image():
    path = REPO / "vllm" / "vllm" / "multimodal" / "image.py"
    spec = importlib.util.spec_from_file_location("official_image", path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def _stats(a: torch.Tensor, b: torch.Tensor, name: str) -> dict:
    a = a.float().reshape(-1)
    b = b.float().reshape(-1)
    diff = (a - b).abs()
    return {
        "name": name,
        "max": float(diff.max()),
        "mean": float(diff.mean()),
        "allclose": bool(torch.allclose(a, b, rtol=RTOL, atol=ATOL)),
    }


def verify_image_prep() -> bool:
    from src.multimodal import image as v4_img

    off = _load_official_image()
    cases = [
        Image.new("RGB", (64, 48), (10, 20, 30)),
        Image.new("RGBA", (40, 40), (255, 0, 0, 128)),
        Image.new("L", (32, 32), 90),
        Image.new("P", (24, 24)),
    ]
    cases[3].putpalette([i % 256 for i in range(768)])
    cases[3].info["transparency"] = 0

    ok = True
    for i, im in enumerate(cases):
        a = v4_img.convert_image_mode(v4_img.normalize_image(im.copy()), "RGB")
        b = off.convert_image_mode(off.normalize_image(im.copy()), "RGB")
        same = list(a.tobytes()) == list(b.tobytes()) and a.size == b.size and a.mode == b.mode
        print(f"[image] case{i} mode={im.mode} match={same}")
        ok = ok and same
    return ok


def _load_visual_sd() -> dict[str, torch.Tensor]:
    sd: dict[str, torch.Tensor] = {}
    for f in sorted(glob.glob(f"{MODEL}/model*.safetensors")):
        with safe_open(f, framework="pt") as s:
            for k in s.keys():
                if k.startswith("model.visual."):
                    sd[k[len("model.visual.") :]] = s.get_tensor(k)
    return sd


@torch.inference_mode()
def verify_vit() -> bool:
    from src.multimodal.image import convert_image_mode, normalize_image
    from src.multimodal.vit import Qwen3VisionTransformer

    cfg = AutoConfig.from_pretrained(MODEL, trust_remote_code=True)
    vc = cfg.vision_config
    vc._attn_implementation = "flash_attention_2"
    proc = AutoProcessor.from_pretrained(MODEL, trust_remote_code=True)

    pil = convert_image_mode(
        normalize_image(Image.new("RGB", (256, 192), (40, 80, 120))), "RGB"
    )
    enc = proc.image_processor(images=[pil], return_tensors="pt")
    pv = enc["pixel_values"].to(DEVICE, dtype=DTYPE)
    grid = enc["image_grid_thw"].to(DEVICE)
    print(f"[vit] pixel_values={tuple(pv.shape)} grid_thw={grid.tolist()}")

    sd = _load_visual_sd()
    hf = Qwen3VLVisionModel(vc).to(DEVICE, dtype=DTYPE).eval()
    hf.load_state_dict(sd, strict=False)
    # `.to(bf16)` corrupts rotary inv_freq; restore float32 (matches training / v4).
    dim = vc.hidden_size // vc.num_heads // 2
    hf.rotary_pos_emb.inv_freq = 1.0 / (
        10000 ** (torch.arange(0, dim, 2, dtype=torch.float, device=DEVICE) / dim)
    )

    v4 = Qwen3VisionTransformer(vc).to(DEVICE, dtype=DTYPE).eval()
    n = v4.load_weights({f"model.visual.{k}": t for k, t in sd.items()})
    print(f"[vit] v4 loaded params={n}")

    # 1D grid_thw must work
    _ = v4.prepare_metadata(grid.view(-1))

    hf_out = hf(pv, grid_thw=grid)
    hf_merged = hf_out.pooler_output
    hf_ds = list(hf_out.deepstack_features or [])

    v4_merged = torch.cat(v4(pv, grid_thw=grid), dim=0)
    v4_ds = list(v4.deepstack_features or [])

    reports = [_stats(hf_merged, v4_merged, "merged")]
    for i, (a, b) in enumerate(zip(hf_ds, v4_ds)):
        reports.append(_stats(a, b, f"deepstack[{i}]"))

    ok = True
    for r in reports:
        print(
            f"[vit] {r['name']}: max_abs={r['max']:.6g} mean_abs={r['mean']:.6g} "
            f"allclose={r['allclose']}"
        )
        ok = ok and r["allclose"]
    return ok


def main():
    torch.cuda.set_device(DEVICE)
    img_ok = verify_image_prep()
    vit_ok = verify_vit()
    print("=" * 60)
    print(f"RESULT image_prep={'PASS' if img_ok else 'FAIL'} vit={'PASS' if vit_ok else 'FAIL'}")
    sys.exit(0 if img_ok and vit_ok else 1)


if __name__ == "__main__":
    main()
