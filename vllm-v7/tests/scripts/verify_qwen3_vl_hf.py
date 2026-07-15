"""HF vs vllm-v5 correctness verifier for Qwen3-VL (any Instruct size).

Checks (single full prefill path):
  1. Weight fingerprints (assembled qkv / vision fc)
  2. ViT merger + DeepStack vs HF visual
  3. MRoPE 3D positions vs HF compute_3d_position_ids
  4. Prefill last-token logits cosine / top1
  5. Greedy generate string match (short)

Usage:
  CUDA_VISIBLE_DEVICES=1 PYTHONPATH=. python tests/verify_qwen3_vl_hf.py \\
    --model ~/huggingface/Qwen3-VL-2B-Instruct
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch
from PIL import Image, ImageDraw


def _image(size: int = 256) -> Image.Image:
    im = Image.new("RGB", (size, size), (30, 120, 200))
    d = ImageDraw.Draw(im)
    d.rectangle([40, 40, size - 40, size - 40], fill=(220, 60, 60))
    d.ellipse(
        [size // 4, size // 4, 3 * size // 4, 3 * size // 4], fill=(240, 220, 40)
    )
    return im


def _cos(a: torch.Tensor, b: torch.Tensor) -> float:
    a = a.float().flatten()
    b = b.float().flatten()
    return float(torch.nn.functional.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0)))


def _report(name: str, ok: bool, detail: dict) -> dict:
    status = "PASS" if ok else "FAIL"
    print(f"[{status}] {name}: {detail}")
    return {"name": name, "ok": ok, **detail}


@torch.inference_mode()
def verify_one(model_path: str, image_size: int = 256, max_new: int = 16) -> dict:
    from transformers import AutoProcessor, AutoModelForImageTextToText

    from src.config import EngineConfig
    from src.runtime.engine import Engine
    from src.runtime.model_inputs import build_model_inputs
    from src.layers.attention import attention_context
    from src.layers.attention_backend import compute_paged_kv_metadata
    from src.sampling_params import SamplingParams
    from src.context import forward_context

    device = torch.device("cuda:0")
    prompt = "What colors are in this image? Answer briefly."
    img = _image(image_size)
    results: list[dict] = []

    # ---------- HF ----------
    proc = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    msgs = [
        {
            "role": "user",
            "content": [{"type": "image", "image": img}, {"type": "text", "text": prompt}],
        }
    ]
    chat = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    batch = proc(text=chat, images=[img], return_tensors="pt")
    ids = batch["input_ids"].to(device)
    pv = batch["pixel_values"].to(device=device, dtype=torch.float16)
    grid = batch["image_grid_thw"].to(device)
    mm_types = batch.get("mm_token_type_ids")
    if mm_types is not None:
        mm_types = mm_types.to(device)

    hf = (
        AutoModelForImageTextToText.from_pretrained(
            model_path, dtype=torch.float16, trust_remote_code=True
        )
        .to(device)
        .eval()
    )
    cfg_snip = {
        "architectures": getattr(hf.config, "architectures", None),
        "text_hidden": hf.config.text_config.hidden_size,
        "text_layers": hf.config.text_config.num_hidden_layers,
        "vision_depth": getattr(hf.config.vision_config, "depth", None),
        "rope": getattr(hf.config.text_config, "rope_scaling", None)
        or getattr(hf.config.text_config, "rope_parameters", None),
    }

    # Vision + logits + positions: stash refs on CPU so we can free HF VRAM.
    hf_vis = hf.model.visual(pv, grid)
    hf_merged = hf_vis.pooler_output.detach().cpu()
    hf_dps = [t.detach().cpu() for t in (hf_vis.deepstack_features or [])]
    hq = hf.model.language_model.layers[0].self_attn.q_proj.weight.detach().cpu()
    hk = hf.model.language_model.layers[0].self_attn.k_proj.weight.detach().cpu()
    hv = hf.model.language_model.layers[0].self_attn.v_proj.weight.detach().cpu()
    hf_fc1 = hf.model.visual.blocks[0].mlp.linear_fc1.weight.detach().cpu()

    hf_kwargs = {k: v.to(device) if hasattr(v, "to") else v for k, v in batch.items()}
    hf_out = hf(**hf_kwargs, use_cache=False)
    hf_logits = hf_out.logits[0, -1].float().detach().cpu()

    pos_hf = hf.model.compute_3d_position_ids(
        input_ids=ids,
        image_grid_thw=grid,
        video_grid_thw=None,
        inputs_embeds=None,
        mm_token_type_ids=mm_types,
        attention_mask=None,
        past_key_values=None,
    )[:, 0, :].cpu()

    hf_gen = hf.generate(**hf_kwargs, max_new_tokens=max_new, do_sample=False)
    hf_text = proc.batch_decode(
        hf_gen[:, ids.shape[1] :], skip_special_tokens=False
    )[0]

    # Drop HF from GPU before Engine load (8B needs almost the full card).
    import gc

    hf.to("cpu")
    del hf, hf_vis, hf_out, hf_gen, hf_kwargs
    gc.collect()
    torch.cuda.synchronize()
    torch.cuda.empty_cache()

    # ---------- v5 ----------
    # Force fp16 so ViT/LM compare matches the HF side (config often says bf16).
    eng = Engine(
        EngineConfig(
            model=model_path,
            dtype="float16",
            enforce_eager=True,
            enable_vit_cuda_graph=False,
            gpu_memory_utilization=0.85,
            max_num_seqs=1,
            # Keep full prompt in one chunk so we test correctness, not chunk bugs.
            max_num_batched_tokens=16384,
        )
    )
    mr = eng.model_runner
    model = eng.model

    our_qkv = model.vl_model.layers[0].self_attn.qkv_proj.weight.detach().cpu()
    qkv_diff = (our_qkv - torch.cat([hq, hk, hv], 0)).abs().max().item()
    results.append(
        _report("weight_qkv_layer0", qkv_diff < 1e-5, {"max_abs": qkv_diff})
    )

    our_fc1 = model.vision_tower.vit.blocks[0].mlp.fc1.weight.detach().cpu()
    fc1_diff = (hf_fc1 - our_fc1).abs().max().item()
    results.append(_report("weight_vit_fc1", fc1_diff < 1e-5, {"max_abs": fc1_diff}))

    our_out = model.vision_tower(pv, grid_thw=grid)
    our_merged = torch.cat(list(our_out), 0).detach().cpu()
    our_dps = [t.detach().cpu() for t in (model.vision_tower.deepstack_features or [])]
    m_diff = (our_merged.float() - hf_merged.float()).abs().max().item()
    m_cos = _cos(our_merged, hf_merged)
    results.append(
        _report(
            "vit_merger",
            m_cos > 0.999 and m_diff < 2.0,
            {"max_abs": m_diff, "cos": m_cos},
        )
    )
    if hf_dps and our_dps:
        dps_ok = True
        dps_detail = []
        for i, (a, b) in enumerate(zip(hf_dps, our_dps)):
            d = (a.float() - b.float()).abs().max().item()
            c = _cos(a, b)
            dps_detail.append({"i": i, "max_abs": d, "cos": c})
            if c < 0.995:
                dps_ok = False
        results.append(_report("vit_deepstack", dps_ok, {"levels": dps_detail}))
    else:
        results.append(
            _report(
                "vit_deepstack",
                False,
                {"error": f"hf={len(hf_dps)} our={len(our_dps)}"},
            )
        )

    seq = eng.add_request(prompt, SamplingParams(max_tokens=1, temperature=0.0), images=[img])
    seqs, is_prefill = eng.scheduler.schedule()
    assert is_prefill and seqs
    inputs = build_model_inputs(seqs, True, mr.block_size, mr.device)
    ie, dps, pos = model.encode_and_merge(inputs.input_ids, seqs, mr.device)
    pos_cpu = pos.cpu()
    pos_diff = (pos_hf.long() - pos_cpu.long()).abs().max().item()
    results.append(
        _report(
            "mrope_positions",
            pos_diff == 0 and torch.equal(pos_hf.long(), pos_cpu.long()),
            {"max_abs": pos_diff, "rope_delta": seq.rope_delta},
        )
    )

    seq_lens = [s.num_cached_tokens + s.num_scheduled_tokens for s in seqs]
    indptr_cpu, idx_gpu, last_cpu = compute_paged_kv_metadata(
        seq_lens, inputs.block_tables, mr.block_size, mr.device
    )
    cu_cpu = torch.tensor([0, seqs[0].num_scheduled_tokens], dtype=torch.int32)
    meta = mr.backend.build_metadata(
        1, indptr_cpu, idx_gpu, last_cpu, inputs.slot_mapping, True, mr.kv_cache, cu_cpu
    )
    with forward_context(
        is_prefill=True,
        seqs=seqs,
        slot_mapping=inputs.slot_mapping,
        cu_seqlens_q=inputs.cu_seqlens_q,
        cu_seqlens_k=inputs.cu_seqlens_k,
        max_seqlen_q=inputs.max_seqlen_q,
        max_seqlen_k=inputs.max_seqlen_k,
        block_tables=inputs.block_tables,
    ):
        with attention_context(meta):
            h = model(
                input_ids=None,
                positions=pos,
                inputs_embeds=ie,
                deepstack_input_embeds=dps,
            )
    v5_logits = model.compute_logits(h[-1:]).float()[0].cpu()
    logit_cos = _cos(hf_logits, v5_logits)
    logit_max = (hf_logits - v5_logits).abs().max().item()
    hf_top = int(hf_logits.argmax())
    v5_top = int(v5_logits.argmax())
    results.append(
        _report(
            "prefill_logits",
            logit_cos > 0.99 and hf_top == v5_top,
            {
                "cos": logit_cos,
                "max_abs": logit_max,
                "hf_top": hf_top,
                "v5_top": v5_top,
                "hf_tok": eng.tokenizer.decode([hf_top]),
                "v5_tok": eng.tokenizer.decode([v5_top]),
            },
        )
    )

    eng.scheduler.postprocess(seqs, [v5_top], True)
    text = "".join(
        list(
            eng.generate(
                prompt,
                max_new_tokens=max_new,
                temperature=0.0,
                images=[img],
            )
        )
    )

    def _norm(s: str) -> str:
        return (
            s.replace("<|im_end|>", "")
            .replace("<|endoftext|>", "")
            .strip()
        )

    exact = text == hf_text
    soft = _norm(text) == _norm(hf_text)
    results.append(
        _report(
            "greedy_generate",
            soft,
            {"exact": exact, "v5": text, "hf": hf_text, "soft_match": soft},
        )
    )

    ok_all = all(r["ok"] for r in results)
    return {
        "model": model_path,
        "image_size": image_size,
        "ok": ok_all,
        "checks": results,
        "config_snip": cfg_snip,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--model",
        action="append",
        dest="models",
        help="Local model dir; can repeat",
    )
    ap.add_argument("--image-size", type=int, default=256)
    ap.add_argument("--max-new", type=int, default=16)
    ap.add_argument(
        "--out",
        default=str(Path(__file__).resolve().parent.parent / "bench/results/vl_hf_verify.json"),
    )
    args = ap.parse_args()
    models = args.models or [
        str(Path.home() / "huggingface/Qwen3-VL-2B-Instruct"),
    ]

    reports = []
    for m in models:
        print(f"\n======== VERIFY {m} ========")
        if not Path(m).expanduser().is_dir():
            reports.append({"model": m, "ok": False, "error": "missing"})
            print("MISSING")
            continue
        torch.cuda.empty_cache()
        rep = verify_one(str(Path(m).expanduser()), args.image_size, args.max_new)
        reports.append(rep)
        print("OVERALL:", "PASS" if rep["ok"] else "FAIL")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(reports, indent=2, ensure_ascii=False))

    md = out.with_suffix(".md")
    lines = ["# Qwen3-VL HF ↔ vllm-v5 verify", ""]
    for r in reports:
        lines.append(f"## `{r.get('model')}` — {'PASS' if r.get('ok') else 'FAIL'}")
        if "error" in r:
            lines.append(f"Error: {r['error']}")
            continue
        cfg = r.get("config_snip", {})
        lines.append(
            f"- hidden={cfg.get('text_hidden')} layers={cfg.get('text_layers')} "
            f"vision_depth={cfg.get('vision_depth')}"
        )
        lines.append("")
        lines.append("| check | ok | detail |")
        lines.append("| --- | --- | --- |")
        for c in r["checks"]:
            detail = {k: v for k, v in c.items() if k not in ("name", "ok")}
            lines.append(f"| {c['name']} | {c['ok']} | `{detail}` |")
        lines.append("")
    md.write_text("\n".join(lines))
    print(f"\nWrote {out} and {md}")
    if not all(r.get("ok") for r in reports):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
