#!/usr/bin/env python3
"""Greedy token match: v7 Triton MoE vs official vLLM (same prompt token ids)."""

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
N = int(os.environ.get("N_TOKENS", "32"))
OUT = ROOT / "bench" / "results" / f"moe_greedy_match_{time.strftime('%Y%m%d_%H%M%S')}.json"

PROMPTS = [
    "The capital of France is",
    "1+1=",
    "Hello, my name is",
    "In mathematics, the derivative of x^2 is",
    "Write a haiku about autumn:\n",
]


def run_v7(prompts: list[str], n: int) -> dict:
    code = f"""
import json
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
sp = SamplingParams(temperature=0.0, max_tokens={n}, ignore_eos=True)
out = {{}}
for p in {prompts!r}:
    ids = eng.tokenizer.encode(p)
    seq = eng.add_request(ids, sp)
    while not seq.is_finished:
        eng.step()
    toks = list(seq.token_ids[seq.num_prompt_tokens :])
    text = eng.tokenizer.decode(toks)
    out[p] = {{"prompt_ids": ids, "out_ids": toks, "text": text}}
    print("V7", json.dumps({{"prompt": p, "out_ids": toks, "text": text}}, ensure_ascii=False), flush=True)
print("RESULT:"+json.dumps(out, ensure_ascii=False), flush=True)
eng.destroy()
"""
    env = os.environ.copy()
    env["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    env["CUDA_VISIBLE_DEVICES"] = str(PHYS)
    env["CUDA_HOME"] = "/usr/local/cuda-12.8"
    env["PATH"] = "/usr/local/cuda-12.8/bin:" + env.get("PATH", "")
    env["PYTHONPATH"] = str(ROOT)
    p = subprocess.run(
        [str(REPO / ".venv" / "bin" / "python"), "-c", code],
        cwd=str(ROOT),
        env=env,
        capture_output=True,
        text=True,
        timeout=1800,
    )
    if p.returncode != 0:
        print(p.stdout[-3000:], p.stderr[-4000:])
        raise RuntimeError(f"v7 failed rc={p.returncode}")
    for line in p.stdout.splitlines():
        if line.startswith("RESULT:"):
            return json.loads(line[len("RESULT:") :])
    raise RuntimeError("no RESULT from v7\n" + p.stdout[-3000:])


def run_official(prompts: list[str], n: int, prompt_ids_by_text: dict[str, list[int]]) -> dict:
    # Feed official the SAME prompt token ids (prompt_token_ids) to avoid decode/re-encode drift.
    code = f"""
import json
from vllm import LLM, SamplingParams

llm = LLM(
    model={str(MOE)!r},
    tensor_parallel_size=1,
    enforce_eager=True,
    max_model_len=2048,
    gpu_memory_utilization=0.45,
    trust_remote_code=True,
)
sp = SamplingParams(temperature=0.0, max_tokens={n}, ignore_eos=True)
prompt_ids = {prompt_ids_by_text!r}
out = {{}}
for text, ids in prompt_ids.items():
    outs = llm.generate(
        [{{"prompt_token_ids": ids}}],
        sp,
        use_tqdm=False,
    )
    oids = list(outs[0].outputs[0].token_ids)
    otext = outs[0].outputs[0].text
    out[text] = {{"prompt_ids": ids, "out_ids": oids, "text": otext}}
    print("OFF", json.dumps({{"prompt": text, "out_ids": oids, "text": otext}}, ensure_ascii=False), flush=True)
print("RESULT:"+json.dumps(out, ensure_ascii=False), flush=True)
"""
    env = os.environ.copy()
    env["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    env["CUDA_VISIBLE_DEVICES"] = str(PHYS)
    cu13 = (
        REPO
        / ".venv-vllm"
        / "lib"
        / "python3.11"
        / "site-packages"
        / "nvidia"
        / "cu13"
        / "lib"
    )
    env["LD_LIBRARY_PATH"] = str(cu13) + ":" + env.get("LD_LIBRARY_PATH", "")
    p = subprocess.run(
        [str(REPO / ".venv-vllm" / "bin" / "python"), "-c", code],
        cwd=str(ROOT),
        env=env,
        capture_output=True,
        text=True,
        timeout=1800,
    )
    if p.returncode != 0:
        print(p.stdout[-3000:], p.stderr[-5000:])
        raise RuntimeError(f"official failed rc={p.returncode}")
    for line in p.stdout.splitlines():
        if line.startswith("RESULT:"):
            return json.loads(line[len("RESULT:") :])
    raise RuntimeError("no RESULT from official\n" + p.stdout[-3000:])


def main() -> None:
    print(f"GPU={PHYS} model={MOE} n={N} prompts={len(PROMPTS)}", flush=True)
    print("=== v7 ===", flush=True)
    v7 = run_v7(PROMPTS, N)
    prompt_ids = {k: v["prompt_ids"] for k, v in v7.items()}
    time.sleep(2)
    print("=== official (same prompt_token_ids) ===", flush=True)
    off = run_official(PROMPTS, N, prompt_ids)

    rows = []
    n_match = 0
    for p in PROMPTS:
        a, b = v7[p]["out_ids"], off[p]["out_ids"]
        # Compare first min length; ideally equal length N
        L = min(len(a), len(b), N)
        match = a[:L] == b[:L] and len(a) >= L and len(b) >= L
        # Also report first mismatch index
        mism = None
        for i in range(L):
            if a[i] != b[i]:
                mism = i
                break
        if match and len(a) == len(b):
            n_match += 1
        rows.append(
            {
                "prompt": p,
                "match": match and len(a) == len(b),
                "first_mismatch": mism,
                "v7_ids": a,
                "off_ids": b,
                "v7_text": v7[p]["text"],
                "off_text": off[p]["text"],
            }
        )
        status = "MATCH" if rows[-1]["match"] else f"DIFF@{mism}"
        print(
            f"[{status}] {p!r}\n"
            f"  v7 : {v7[p]['text']!r}\n"
            f"  off: {off[p]['text']!r}",
            flush=True,
        )

    payload = {
        "gpu": PHYS,
        "model": str(MOE),
        "n": N,
        "n_match": n_match,
        "n_total": len(PROMPTS),
        "rows": rows,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    print(f"\nSummary: {n_match}/{len(PROMPTS)} exact greedy matches", flush=True)
    print(f"Wrote {OUT}", flush=True)
    if n_match != len(PROMPTS):
        sys.exit(2)


if __name__ == "__main__":
    main()
