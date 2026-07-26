#!/usr/bin/env python3
"""Max-CG head-to-head: vllm-v4 vs official (everything cranked).

Metrics
  - load_s
  - ttft_s          : time to first token (MM or text)
  - vit_encode_s    : v4 only (official API does not expose)
  - lm_prefill_s    : v4 only — LM forward on prompt after ViT embeds ready
  - decode_tok_s    : steady decode (prefix-cache OFF for fairness)
  - e2e_tok_s       : gen_tokens / total wall for full generate

Usage (pick ONE backend per process):
  # v4
  source ../.venv/bin/activate && source ../.env.cuda.sh
  PYTHONPATH=. python bench/mm_max_cg_head2head.py --backend v4 ...

  # official
  source ../.venv-vllm/bin/activate && source ../.env.vllm.sh
  python bench/mm_max_cg_head2head.py --backend official ...
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import torch
from PIL import Image


def sync() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def peak_gb() -> float:
    if not torch.cuda.is_available():
        return 0.0
    return torch.cuda.max_memory_allocated() / (1024**3)


def load_image(path: str, max_side: int) -> Image.Image:
    im = Image.open(path).convert("RGB")
    if max_side > 0 and max(im.size) > max_side:
        im = im.copy()
        im.thumbnail((max_side, max_side))
    return im


def mean(xs: list[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


# ─── v4 ──────────────────────────────────────────────────────────────────────


def run_v4(args) -> dict:
    from src.config import EngineConfig
    from src.engine.engine import Engine
    from src.sampling_params import SamplingParams

    img = load_image(args.image, args.max_side)
    prompt_mm = args.prompt
    prompt_tx = "The capital of France is"

    torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    eng = Engine(
        EngineConfig(
            model=args.model,
            dtype="float16",
            enforce_eager=False,  # decode CG ON
            enable_vit_cuda_graph=True,  # ViT CG ON
            vit_cg_token_budgets=tuple(
                int(x) for x in args.vit_budgets.split(",") if x.strip()
            ),
            vit_cg_max_batch_size=args.vit_max_batch,
            gpu_memory_utilization=args.gpu_util,
            max_num_seqs=max(
                8, max(int(x) for x in str(args.conc).split(",") if x.strip())
            ),
            max_num_batched_tokens=8192,
            context_len=args.context_len,
            mix_prefill_decode=True,
            prefix_backend="none",  # fair decode (no APC)
        )
    )
    sync()
    load_s = time.perf_counter() - t0
    print(
        f"[v4] load={load_s:.2f}s decode_cg=True vit_cg={getattr(eng.model, '_encoder_cg_ready', False)}"
    )

    def once(prompt: str, images) -> dict:
        sp = SamplingParams(max_tokens=args.max_new, temperature=0.0)
        seq = eng.add_request(prompt, sp, images=images)
        vit_s = lm_s = decode_s = prefill_s = 0.0
        decode_steps = 0
        first = None
        t_start = time.perf_counter()

        orig = None
        if images and hasattr(eng.model, "encode_and_merge"):
            orig = eng.model.encode_and_merge

            def timed(*a, **k):
                nonlocal vit_s
                sync()
                t = time.perf_counter()
                out = orig(*a, **k)
                sync()
                vit_s += time.perf_counter() - t
                return out

            eng.model.encode_and_merge = timed

        try:
            while not seq.is_finished:
                prefills, decodes = eng.scheduler.schedule()
                if not prefills and not decodes:
                    break
                if decodes:
                    sync()
                    t = time.perf_counter()
                    ids = eng.model_runner._run_batch(decodes, is_prefill=False)
                    sync()
                    decode_s += time.perf_counter() - t
                    decode_steps += len(decodes)
                    eng.scheduler.postprocess(decodes, ids, is_prefill=False)
                if prefills:
                    vb = vit_s
                    sync()
                    t = time.perf_counter()
                    ids = eng.model_runner._run_batch(prefills, is_prefill=True)
                    sync()
                    dt = time.perf_counter() - t
                    prefill_s += dt
                    lm_s += max(0.0, dt - (vit_s - vb))
                    eng.scheduler.postprocess(prefills, ids, is_prefill=True)
                if first is None and seq.num_completion_tokens > 0:
                    first = time.perf_counter() - t_start
        finally:
            if orig is not None:
                eng.model.encode_and_merge = orig

        total = time.perf_counter() - t_start
        return {
            "ttft_s": first if first is not None else prefill_s,
            "vit_encode_s": vit_s,
            "lm_prefill_s": lm_s,
            "prefill_s": prefill_s,
            "decode_s": decode_s,
            "total_s": total,
            "decode_tok_s": (decode_steps / decode_s) if decode_s > 0 else 0.0,
            "e2e_tok_s": (seq.num_completion_tokens / total) if total > 0 else 0.0,
            "prompt_tokens": seq.num_prompt_tokens,
            "gen_tokens": seq.num_completion_tokens,
        }

    def concurrent(n: int) -> dict:
        sp = SamplingParams(max_tokens=args.max_new, temperature=0.0)
        jobs = []
        for i in range(n):
            if i % 2 == 0:
                jobs.append((prompt_mm, [img]))
            else:
                jobs.append((prompt_tx, None))
        seqs = [eng.add_request(p, sp, images=im) for p, im in jobs]
        sync()
        t = time.perf_counter()
        while not eng.scheduler.is_finished():
            eng.step()
        sync()
        wall = time.perf_counter() - t
        gens = sum(s.num_completion_tokens for s in seqs)
        return {"wall_s": wall, "batch_tok_s": gens / wall if wall > 0 else 0.0, "n": n}

    # warmup
    for _ in range(args.warmup):
        once(prompt_mm, [img])
        once(prompt_tx, None)

    cases = {}
    for name, prompt, images in (
        ("text_only", prompt_tx, None),
        ("mm_image", prompt_mm, [img]),
    ):
        rows = [once(prompt, images) for _ in range(args.iters)]
        cases[name] = {k: mean([float(r[k]) for r in rows]) for k in rows[0]}
        print(
            f"  {name}: ttft={cases[name]['ttft_s']*1000:.1f}ms "
            f"vit={cases[name]['vit_encode_s']*1000:.1f}ms "
            f"lm_prefill={cases[name]['lm_prefill_s']*1000:.1f}ms "
            f"decode={cases[name]['decode_tok_s']:.1f} tok/s "
            f"e2e={cases[name]['e2e_tok_s']:.1f}"
        )

    conc = {}
    for n in [int(x) for x in args.conc.split(",") if x.strip()]:
        _ = concurrent(n)
        rows = [concurrent(n) for _ in range(max(1, args.iters - 1))]
        conc[f"n{n}"] = {
            "wall_s": mean([r["wall_s"] for r in rows]),
            "batch_tok_s": mean([r["batch_tok_s"] for r in rows]),
        }
        print(f"  conc n={n}: batch={conc[f'n{n}']['batch_tok_s']:.1f} tok/s")

    return {
        "backend": "vllm-v4",
        "model": args.model,
        "device": torch.cuda.get_device_name(0),
        "decode_cg": True,
        "vit_cg": True,
        "prefix_cache": False,
        "load_s": load_s,
        "peak_mem_gb": peak_gb(),
        "max_side": args.max_side,
        "max_new": args.max_new,
        "cases": cases,
        "concurrency": conc,
    }


# ─── official ───────────────────────────────────────────────────────────────


def run_official(args) -> dict:
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", os.environ.get("CUDA_VISIBLE_DEVICES", "0"))
    torch.cuda.init()

    import vllm.platforms as vplat

    _orig = vplat.cuda_platform_plugin

    def _cuda_plugin():
        r = _orig()
        if r is not None:
            return r
        return "vllm.platforms.cuda.CudaPlatform" if torch.cuda.is_available() else None

    vplat.cuda_platform_plugin = _cuda_plugin
    vplat.builtin_platform_plugins["cuda"] = _cuda_plugin
    vplat._current_platform = None

    from vllm import LLM, SamplingParams
    from vllm.config import CompilationConfig

    img = load_image(args.image, args.max_side)
    prompt_mm = args.prompt
    prompt_tx = "The capital of France is"
    tokenizer_holder: dict = {}

    def make_req(prompt: str, image):
        tok = tokenizer_holder["tok"]
        if image is None:
            return {"prompt": prompt}
        messages = [
            {
                "role": "user",
                "content": [{"type": "image"}, {"type": "text", "text": prompt}],
            }
        ]
        try:
            rendered = tok.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        except Exception:
            rendered = (
                f"<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>"
                f"{prompt}<|im_end|>\n<|im_start|>assistant\n"
            )
        return {"prompt": rendered, "multi_modal_data": {"image": image}}

    budgets = [int(x) for x in args.vit_budgets.split(",") if x.strip()]
    cc = CompilationConfig(
        cudagraph_mm_encoder=True,
        encoder_cudagraph_token_budgets=budgets,
        encoder_cudagraph_max_vision_items_per_batch=args.vit_max_batch,
    )

    torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    llm = LLM(
        model=args.model,
        dtype="float16",
        trust_remote_code=True,
        gpu_memory_utilization=args.gpu_util,
        max_model_len=args.context_len,
        limit_mm_per_prompt={"image": 4},
        enforce_eager=False,  # decode CG
        enable_prefix_caching=False,  # fair vs v4
        compilation_config=cc,
    )
    sync()
    load_s = time.perf_counter() - t0
    tokenizer_holder["tok"] = llm.get_tokenizer()
    print(f"[official] load={load_s:.2f}s decode_cg=True encoder_cg=True prefix=False")

    def once(prompt: str, image) -> dict:
        req = make_req(prompt, image)
        # TTFT: 1-token
        sync()
        t = time.perf_counter()
        llm.generate([req], SamplingParams(temperature=0.0, max_tokens=1))
        sync()
        ttft = time.perf_counter() - t

        # Full generate for e2e + decode estimate
        sync()
        t = time.perf_counter()
        outs = llm.generate(
            [req], SamplingParams(temperature=0.0, max_tokens=args.max_new)
        )
        sync()
        total = time.perf_counter() - t
        n = len(outs[0].outputs[0].token_ids)
        # With prefix cache OFF, total should exceed ttft for max_new>1.
        if total > ttft * 1.02 and n > 1:
            decode_s = total - ttft
            decode_tok_s = (n - 1) / decode_s
        else:
            decode_s = total
            decode_tok_s = n / total if total > 0 else 0.0
        pids = getattr(outs[0], "prompt_token_ids", None)
        return {
            "ttft_s": ttft,
            "vit_encode_s": None,
            "lm_prefill_s": None,
            "prefill_s": ttft,
            "decode_s": decode_s,
            "total_s": total,
            "decode_tok_s": decode_tok_s,
            "e2e_tok_s": n / total if total > 0 else 0.0,
            "prompt_tokens": len(pids) if pids is not None else -1,
            "gen_tokens": float(n),
        }

    def concurrent(n: int) -> dict:
        reqs = []
        for i in range(n):
            if i % 2 == 0:
                reqs.append(make_req(prompt_mm, img))
            else:
                reqs.append(make_req(prompt_tx, None))
        sync()
        t = time.perf_counter()
        outs = llm.generate(reqs, SamplingParams(temperature=0.0, max_tokens=args.max_new))
        sync()
        wall = time.perf_counter() - t
        gens = sum(len(o.outputs[0].token_ids) for o in outs)
        return {"wall_s": wall, "batch_tok_s": gens / wall if wall > 0 else 0.0, "n": n}

    for _ in range(args.warmup):
        once(prompt_mm, img)
        once(prompt_tx, None)

    cases = {}
    for name, prompt, image in (
        ("text_only", prompt_tx, None),
        ("mm_image", prompt_mm, img),
    ):
        rows = [once(prompt, image) for _ in range(args.iters)]
        keys = [k for k in rows[0] if rows[0][k] is not None]
        cases[name] = {k: mean([float(r[k]) for r in rows]) for k in keys}
        cases[name]["vit_encode_s"] = None
        cases[name]["lm_prefill_s"] = None
        print(
            f"  {name}: ttft={cases[name]['ttft_s']*1000:.1f}ms "
            f"decode={cases[name]['decode_tok_s']:.1f} tok/s "
            f"e2e={cases[name]['e2e_tok_s']:.1f}"
        )

    conc = {}
    for n in [int(x) for x in args.conc.split(",") if x.strip()]:
        _ = concurrent(n)
        rows = [concurrent(n) for _ in range(max(1, args.iters - 1))]
        conc[f"n{n}"] = {
            "wall_s": mean([r["wall_s"] for r in rows]),
            "batch_tok_s": mean([r["batch_tok_s"] for r in rows]),
        }
        print(f"  conc n={n}: batch={conc[f'n{n}']['batch_tok_s']:.1f} tok/s")

    return {
        "backend": "vllm-official",
        "vllm_version": __import__("vllm").__version__,
        "model": args.model,
        "device": torch.cuda.get_device_name(0),
        "decode_cg": True,
        "vit_cg": True,
        "prefix_cache": False,
        "load_s": load_s,
        "peak_mem_gb": peak_gb(),
        "max_side": args.max_side,
        "max_new": args.max_new,
        "cases": cases,
        "concurrency": conc,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", choices=["v4", "official"], required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--image", default="cat.jpg")
    ap.add_argument("--max-side", type=int, default=512)
    ap.add_argument("--max-new", type=int, default=64)
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--iters", type=int, default=5)
    ap.add_argument("--gpu-util", type=float, default=0.90)
    ap.add_argument("--context-len", type=int, default=8192)
    ap.add_argument("--vit-budgets", default="64,128,256,512,1024,2048")
    ap.add_argument("--vit-max-batch", type=int, default=8)
    ap.add_argument("--conc", default="1,2,4,8")
    ap.add_argument("--prompt", default="Describe this image in one short sentence.")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    report = run_v4(args) if args.backend == "v4" else run_official(args)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
