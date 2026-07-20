#!/usr/bin/env python3
"""MoE TTFT / prefill sweep vs prompt length (v7 Triton vs official)."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parent
MOE = Path(os.environ.get("MOE_MODEL", "/mnt/nfs/zhangli/huggingface/Qwen1.5-MoE-A2.7B"))
PHYS = int(os.environ.get("BENCH_GPU", "3"))
OUT = ROOT / "bench" / "results" / f"moe_ttft_sweep_{time.strftime('%Y%m%d_%H%M%S')}.json"

# prompt lens: short (old bench) → long prefill
LENGTHS = [8, 32, 128, 512, 1024]


def _make_prompt_ids(tok, n: int) -> list[int]:
    base = tok.encode("The capital of France is Paris. " * 64, add_special_tokens=False)
    if len(base) < n:
        base = (base * ((n // len(base)) + 1))[:n]
    else:
        base = base[:n]
    return base


def run_v7(lengths: list[int], warmup: int = 1, repeats: int = 2) -> list[dict]:
    code = f"""
import time, json, torch
from src.config import EngineConfig
from src import Engine, SamplingParams

cfg = EngineConfig(
    model={str(MOE)!r},
    tp_size=1, pp_size=1, device_ids=[0],
    enforce_eager=True,
    context_len=2048,
    num_kvcache_blocks=64,
    gpu_memory_utilization=0.45,
    moe_fused_backend="triton",
    max_num_seqs=4,
    nccl_port=0,
)
eng = Engine(cfg)
tok = eng.tokenizer
lengths = {lengths!r}

def ids_for(n):
    base = tok.encode("The capital of France is Paris. " * 64, add_special_tokens=False)
    return (base * ((n // max(len(base),1)) + 1))[:n]

# warmup
w = ids_for(64)
seq = eng.add_request(w, SamplingParams(temperature=0.0, max_tokens=4, ignore_eos=True))
while not seq.is_finished:
    eng.step()

rows = []
for n in lengths:
    ids = ids_for(n)
    # naive? M*topk*4 <= E  → n*4*4 <= 60 → n<=3
    naive = (n * 4 * 4) <= 60
    times = []
    for _ in range({repeats}):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        seq = eng.add_request(ids, SamplingParams(temperature=0.0, max_tokens=1, ignore_eos=True))
        while not seq.is_finished:
            eng.step()
        torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1000)
    rows.append({{
        "engine": "v7-triton",
        "prompt_tokens": n,
        "naive_path": naive,
        "ttft_ms": sum(times)/len(times),
        "ttft_ms_all": times,
    }})
    print(json.dumps(rows[-1]), flush=True)
print("RESULT:"+json.dumps(rows), flush=True)
eng.destroy()
"""
    env = os.environ.copy()
    env["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    env["CUDA_VISIBLE_DEVICES"] = str(PHYS)
    env["CUDA_HOME"] = "/usr/local/cuda-12.8"
    env["PATH"] = "/usr/local/cuda-12.8/bin:" + env.get("PATH", "")
    env["PYTHONPATH"] = str(ROOT)
    py = REPO / ".venv" / "bin" / "python"
    p = subprocess.run(
        [str(py), "-c", code],
        cwd=str(ROOT),
        env=env,
        capture_output=True,
        text=True,
        timeout=1800,
    )
    if p.returncode != 0:
        print(p.stdout[-2000:], p.stderr[-3000:])
        raise RuntimeError(f"v7 failed rc={p.returncode}")
    for line in p.stdout.splitlines():
        if line.startswith("RESULT:"):
            return json.loads(line[len("RESULT:") :])
    raise RuntimeError("no RESULT from v7\n" + p.stdout[-2000:])


def run_official(lengths: list[int], warmup: int = 1, repeats: int = 2) -> list[dict]:
    code = f"""
import time, json, os, torch
from vllm import LLM, SamplingParams

llm = LLM(
    model={str(MOE)!r},
    tensor_parallel_size=1,
    enforce_eager=True,
    max_model_len=2048,
    gpu_memory_utilization=0.45,
    trust_remote_code=True,
)
tok = llm.get_tokenizer()
lengths = {lengths!r}

def text_for(n):
    base = tok.encode("The capital of France is Paris. " * 64, add_special_tokens=False)
    ids = (base * ((n // max(len(base),1)) + 1))[:n]
    return tok.decode(ids)

# warmup
llm.generate([text_for(64)], SamplingParams(temperature=0.0, max_tokens=4, ignore_eos=True))

rows = []
sp1 = SamplingParams(temperature=0.0, max_tokens=1, ignore_eos=True)
for n in lengths:
    prompt = text_for(n)
    naive = (n * 4 * 4) <= 60
    times = []
    for _ in range({repeats}):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        llm.generate([prompt], sp1)
        torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1000)
    rows.append({{
        "engine": "official",
        "prompt_tokens": n,
        "naive_path": naive,
        "ttft_ms": sum(times)/len(times),
        "ttft_ms_all": times,
    }})
    print(json.dumps(rows[-1]), flush=True)
print("RESULT:"+json.dumps(rows), flush=True)
"""
    env = os.environ.copy()
    env["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    env["CUDA_VISIBLE_DEVICES"] = str(PHYS)
    # official needs cudart 13
    cu13 = REPO / ".venv-vllm" / "lib" / "python3.11" / "site-packages" / "nvidia" / "cu13" / "lib"
    env["LD_LIBRARY_PATH"] = str(cu13) + ":" + env.get("LD_LIBRARY_PATH", "")
    py = REPO / ".venv-vllm" / "bin" / "python"
    p = subprocess.run(
        [str(py), "-c", code],
        cwd=str(ROOT),
        env=env,
        capture_output=True,
        text=True,
        timeout=1800,
    )
    if p.returncode != 0:
        print(p.stdout[-2000:], p.stderr[-4000:])
        raise RuntimeError(f"official failed rc={p.returncode}")
    for line in p.stdout.splitlines():
        if line.startswith("RESULT:"):
            return json.loads(line[len("RESULT:") :])
    raise RuntimeError("no RESULT from official\n" + p.stdout[-2000:])


def main() -> None:
    print(f"GPU={PHYS} model={MOE} lengths={LENGTHS}", flush=True)
    print("=== v7 ===", flush=True)
    v7 = run_v7(LENGTHS)
    for r in v7:
        print(f"  v7 N={r['prompt_tokens']} ttft={r['ttft_ms']:.1f}ms naive={r['naive_path']}", flush=True)
    # Ensure A100 is free before official loads ~28GB weights.
    time.sleep(3)
    print("=== official ===", flush=True)
    off = run_official(LENGTHS)
    for r in off:
        print(f"  off N={r['prompt_tokens']} ttft={r['ttft_ms']:.1f}ms", flush=True)

    by_n = {r["prompt_tokens"]: r for r in off}
    table = []
    for r in v7:
        o = by_n[r["prompt_tokens"]]
        ratio = r["ttft_ms"] / o["ttft_ms"] if o["ttft_ms"] > 0 else float("nan")
        row = {
            "prompt_tokens": r["prompt_tokens"],
            "naive_path": r["naive_path"],
            "v7_ttft_ms": r["ttft_ms"],
            "off_ttft_ms": o["ttft_ms"],
            "v7_over_off": ratio,
            "tok_per_s_v7": r["prompt_tokens"] / (r["ttft_ms"] / 1000),
            "tok_per_s_off": o["prompt_tokens"] / (o["ttft_ms"] / 1000),
        }
        table.append(row)
        print(
            f"N={row['prompt_tokens']:<5} naive={row['naive_path']!s:<5} "
            f"v7={row['v7_ttft_ms']:.1f}ms off={row['off_ttft_ms']:.1f}ms "
            f"ratio={row['v7_over_off']:.2f}x "
            f"prefill_tok/s v7={row['tok_per_s_v7']:.1f} off={row['tok_per_s_off']:.1f}",
            flush=True,
        )

    payload = {"gpu": PHYS, "model": str(MOE), "v7": v7, "official": off, "summary": table}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2))
    md = OUT.with_suffix(".md")
    lines = [
        "# MoE TTFT / prefill sweep (v7 Triton vs official)",
        "",
        f"- GPU: `{PHYS}` (A100 preferred)",
        f"- Model: `{MOE}`",
        f"- Mode: eager, max_tokens=1 (wall TTFT ≈ prefill)",
        "",
        "| prompt_tokens | naive? | v7 TTFT ms | off TTFT ms | v7/off | v7 tok/s | off tok/s |",
        "|---:|:---:|---:|---:|---:|---:|---:|",
    ]
    for row in table:
        lines.append(
            f"| {row['prompt_tokens']} | {row['naive_path']} | "
            f"{row['v7_ttft_ms']:.1f} | {row['off_ttft_ms']:.1f} | "
            f"{row['v7_over_off']:.2f}x | {row['tok_per_s_v7']:.1f} | {row['tok_per_s_off']:.1f} |"
        )
    md.write_text("\n".join(lines) + "\n")
    print(f"Wrote {OUT}\nWrote {md}", flush=True)


if __name__ == "__main__":
    main()
