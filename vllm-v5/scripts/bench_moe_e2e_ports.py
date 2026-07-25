#!/usr/bin/env python3
"""End-to-end MoE performance: vllm-v7 port vs official vLLM port.

Cases (eager + CUDA-graph attempt):
  - TP1 on A100 (Triton; CG usually N/A for MoE)
  - TP2 on homogeneous 3090s (Triton)
  - TP1 on PRO6000 / Blackwell (cutlass + real MoE CG when env OK)

Metrics (batch=1, greedy, ignore_eos):
  - TTFT ms  : wall time of max_tokens=1
  - e2e s    : wall time of max_tokens=N
  - e2e tok/s: N / e2e
  - decode tok/s: (N-1) / max(e2e - TTFT, eps)

Fairness:
  - Same prompt token ids on both engines (no decode→re-encode)
  - Both use wall-clock (no official metrics TTFT)
  - Triton MoE backend on v7; official auto (Triton on A100/3090)
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
V7 = ROOT / "vllm-v7"
V7_PY = ROOT / ".venv" / "bin" / "python"
OFF_PY = ROOT / ".venv-vllm" / "bin" / "python"
MOE = Path(
    os.environ.get("MOE_MODEL", "/mnt/nfs/zhangli/huggingface/Qwen1.5-MoE-A2.7B")
)
OUT = V7 / "bench" / "results" / f"moe_e2e_ports_{time.strftime('%Y%m%d_%H%M%S')}.json"

PROMPT = "The capital of France is"
MAX_NEW = 64
GPU_A100 = 3
GPU_3090 = [1, 5, 2, 4]
GPU_PRO6000 = 0


def _free_mib() -> dict[int, int]:
    out = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-gpu=index,memory.free",
            "--format=csv,noheader,nounits",
        ],
        text=True,
    )
    d = {}
    for line in out.strip().splitlines():
        i, m = line.split(",")
        d[int(i.strip())] = int(m.strip())
    return d


def _nvidia_libs() -> str:
    site = subprocess.check_output(
        [str(OFF_PY), "-c", "import site; print(site.getsitepackages()[0])"],
        text=True,
    ).strip()
    libs = []
    nvidia = Path(site) / "nvidia"
    if nvidia.is_dir():
        cu13 = nvidia / "cu13" / "lib"
        if cu13.is_dir():
            libs.append(str(cu13))
        for p in nvidia.rglob("lib"):
            if p.is_dir():
                libs.append(str(p))
    return ":".join(dict.fromkeys(libs))


def _base_env(gpus: list[int], *, blackwell: bool = False) -> dict:
    env = os.environ.copy()
    env["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    env["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, gpus))
    env.setdefault("NCCL_IB_DISABLE", "1")
    env.setdefault("NCCL_NET", "Socket")
    cuda_home = "/usr/local/cuda-12.8"
    if Path(cuda_home).is_dir():
        env["CUDA_HOME"] = cuda_home
        env["PATH"] = f"{cuda_home}/bin:" + env.get("PATH", "")
        env["LD_LIBRARY_PATH"] = f"{cuda_home}/lib64:" + env.get("LD_LIBRARY_PATH", "")
    if blackwell:
        # nvcc 12.8 + FlashInfer arch 12.0a → cutlass MoE JIT (enables MoE CG).
        # Do not pin TORCH_CUDA_ARCH_LIST (breaks PyTorch kernels on SM120).
        env["FLASHINFER_CUDA_ARCH_LIST"] = "12.0a"
        env.pop("TORCH_CUDA_ARCH_LIST", None)
    return env


def _off_env(gpus: list[int], *, blackwell: bool = False) -> dict:
    env = _base_env(gpus, blackwell=blackwell)
    libs = _nvidia_libs()
    parts = [libs, "/usr/local/cuda-12.8/lib64", env.get("LD_LIBRARY_PATH", "")]
    env["LD_LIBRARY_PATH"] = ":".join(p for p in parts if p)
    # Avoid FlashInfer sampler JIT pitfalls on mixed-toolkit boxes.
    env.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
    return env


def _run(py: Path, code: str, env: dict, timeout: int = 1800) -> dict:
    p = subprocess.run(
        [str(py), "-c", code],
        cwd=str(V7),
        env=env,
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    if p.returncode != 0:
        print(p.stdout[-2500:])
        print(p.stderr[-4000:])
        raise RuntimeError(f"failed rc={p.returncode}")
    for line in p.stdout.splitlines():
        if line.startswith("JSON:"):
            return json.loads(line[5:])
    raise RuntimeError("no JSON\n" + p.stdout[-2000:])


def run_v7(
    *,
    gpus: list[int],
    tp: int,
    eager: bool,
    ids: list[int],
    n: int,
    warmup: int,
    repeats: int,
    moe_backend: str = "triton",
    blackwell: bool = False,
) -> dict:
    code = f"""
import gc, json, time, torch
from src import Engine, EngineConfig, SamplingParams

cfg = EngineConfig(
    model={str(MOE)!r},
    tp_size={tp}, pp_size=1,
    device_ids=list(range({tp})),
    enforce_eager={eager},
    moe_cuda_graph={not eager},
    moe_fused_backend={moe_backend!r},
    max_num_seqs=4,
    context_len=2048,
    num_kvcache_blocks=64,
    gpu_memory_utilization=0.85,
    nccl_port=0,
)
eng = Engine(cfg)
ids = {ids!r}
n, warmup, repeats = {n}, {warmup}, {repeats}
sp_n = SamplingParams(temperature=0.0, max_tokens=n, ignore_eos=True)
sp1 = SamplingParams(temperature=0.0, max_tokens=1, ignore_eos=True)
spw = SamplingParams(temperature=0.0, max_tokens=4, ignore_eos=True)

def once(sp):
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    seq = eng.add_request(list(ids), sp)
    while not seq.is_finished:
        eng.step()
    torch.cuda.synchronize()
    return time.perf_counter() - t0

for _ in range(warmup):
    once(spw)

ttfts, e2es = [], []
for _ in range(repeats):
    ttfts.append(once(sp1) * 1000.0)
    e2es.append(once(sp_n))

ttft = sum(ttfts) / len(ttfts)
e2e = sum(e2es) / len(e2es)
decode_s = max(e2e - ttft / 1000.0, 1e-6)
out = {{
    "engine": "vllm-v7",
    "tp": {tp},
    "eager": {eager},
    "moe_backend": {moe_backend!r},
    "physical_gpus": {gpus!r},
    "prompt_tokens": len(ids),
    "max_tokens": n,
    "ttft_ms": ttft,
    "ttft_ms_all": ttfts,
    "e2e_s": e2e,
    "e2e_s_all": e2es,
    "e2e_tok_s": n / e2e,
    "decode_tok_s": (n - 1) / decode_s,
    "cg_requested": {not eager},
}}
print("JSON:" + json.dumps(out), flush=True)
eng.destroy()
gc.collect()
"""
    env = _base_env(gpus, blackwell=blackwell)
    env["PYTHONPATH"] = str(V7)
    return _run(V7_PY, code, env)


def run_off(
    *,
    gpus: list[int],
    tp: int,
    eager: bool,
    ids: list[int],
    n: int,
    warmup: int,
    repeats: int,
    blackwell: bool = False,
) -> dict:
    code = f"""
import gc, json, time, torch
from vllm import LLM, SamplingParams

llm = LLM(
    model={str(MOE)!r},
    tensor_parallel_size={tp},
    enforce_eager={eager},
    max_model_len=2048,
    max_num_seqs=4,
    gpu_memory_utilization=0.85,
    trust_remote_code=True,
    disable_log_stats=True,
)
ids = {ids!r}
n, warmup, repeats = {n}, {warmup}, {repeats}
sp_n = SamplingParams(temperature=0.0, max_tokens=n, ignore_eos=True)
sp1 = SamplingParams(temperature=0.0, max_tokens=1, ignore_eos=True)
spw = SamplingParams(temperature=0.0, max_tokens=4, ignore_eos=True)

def once(sp):
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    llm.generate([ids], sp, use_tqdm=False)
    torch.cuda.synchronize()
    return time.perf_counter() - t0

for _ in range(warmup):
    once(spw)

ttfts, e2es = [], []
for _ in range(repeats):
    ttfts.append(once(sp1) * 1000.0)
    e2es.append(once(sp_n))

ttft = sum(ttfts) / len(ttfts)
e2e = sum(e2es) / len(e2es)
decode_s = max(e2e - ttft / 1000.0, 1e-6)
out = {{
    "engine": "vllm-official",
    "tp": {tp},
    "eager": {eager},
    "physical_gpus": {gpus!r},
    "prompt_tokens": len(ids),
    "max_tokens": n,
    "ttft_ms": ttft,
    "ttft_ms_all": ttfts,
    "e2e_s": e2e,
    "e2e_s_all": e2es,
    "e2e_tok_s": n / e2e,
    "decode_tok_s": (n - 1) / decode_s,
    "cg_requested": {not eager},
}}
print("JSON:" + json.dumps(out), flush=True)
try:
    llm.llm_engine.engine_core.shutdown()
except Exception:
    pass
del llm
gc.collect()
"""
    return _run(OFF_PY, code, _off_env(gpus, blackwell=blackwell))


def encode_prompt() -> list[int]:
    code = f"""
import json
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained({str(MOE)!r}, trust_remote_code=True)
ids = tok.encode({PROMPT!r}, add_special_tokens=True)
print("JSON:"+json.dumps(ids))
"""
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = ""
    return _run(V7_PY, code, env)


def pick_3090_pair(free: dict[int, int], need: int = 20000) -> list[int] | None:
    cands = [g for g in GPU_3090 if free.get(g, 0) >= need]
    if len(cands) >= 2:
        return cands[:2]
    return None


def summarize(case: str, v7: dict, off: dict) -> dict:
    return {
        "case": case,
        "v7": v7,
        "official": off,
        "ratio_decode_v7_over_off": v7["decode_tok_s"] / max(off["decode_tok_s"], 1e-9),
        "ratio_ttft_v7_over_off": v7["ttft_ms"] / max(off["ttft_ms"], 1e-9),
        "ratio_e2e_tok_v7_over_off": v7["e2e_tok_s"] / max(off["e2e_tok_s"], 1e-9),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--max-tokens", type=int, default=MAX_NEW)
    ap.add_argument(
        "--modes",
        default="eager,cg",
        help="comma list: eager,cg",
    )
    ap.add_argument(
        "--suite",
        default="pro6000",
        choices=("all", "ampere", "pro6000"),
        help="all | ampere (A100+3090) | pro6000 (Blackwell cutlass CG)",
    )
    args = ap.parse_args()
    modes = [m.strip() for m in args.modes.split(",") if m.strip()]

    free = _free_mib()
    print(f"model={MOE} free={free} suite={args.suite}", flush=True)
    print("encoding shared prompt ids …", flush=True)
    ids = encode_prompt()
    if isinstance(ids, dict):
        raise RuntimeError("unexpected encode payload")
    print(f"prompt_tokens={len(ids)} ids[:8]={ids[:8]}", flush=True)

    results = []

    def do_case(
        name: str,
        gpus: list[int],
        tp: int,
        eager: bool,
        *,
        moe_backend: str = "triton",
        blackwell: bool = False,
    ) -> None:
        print(
            f"\n=== {name} gpus={gpus} eager={eager} backend={moe_backend} ===",
            flush=True,
        )
        print("  v7 …", flush=True)
        v7 = run_v7(
            gpus=gpus,
            tp=tp,
            eager=eager,
            ids=ids,
            n=args.max_tokens,
            warmup=args.warmup,
            repeats=args.repeats,
            moe_backend=moe_backend,
            blackwell=blackwell,
        )
        print(
            f"  v7  ttft={v7['ttft_ms']:.2f} decode={v7['decode_tok_s']:.2f} "
            f"e2e_tok/s={v7['e2e_tok_s']:.2f} e2e_s={v7['e2e_s']:.3f}",
            flush=True,
        )
        time.sleep(2)
        print("  official …", flush=True)
        off = run_off(
            gpus=gpus,
            tp=tp,
            eager=eager,
            ids=ids,
            n=args.max_tokens,
            warmup=args.warmup,
            repeats=args.repeats,
            blackwell=blackwell,
        )
        print(
            f"  off ttft={off['ttft_ms']:.2f} decode={off['decode_tok_s']:.2f} "
            f"e2e_tok/s={off['e2e_tok_s']:.2f} e2e_s={off['e2e_s']:.3f}",
            flush=True,
        )
        row = summarize(name, v7, off)
        results.append(row)
        time.sleep(2)

    if args.suite in ("all", "ampere"):
        if free.get(GPU_A100, 0) >= 35000:
            if "eager" in modes:
                do_case("moe_tp1_a100_eager", [GPU_A100], 1, True)
            if "cg" in modes:
                do_case("moe_tp1_a100_cg", [GPU_A100], 1, False)
        else:
            print(f"SKIP TP1 A100 free={free.get(GPU_A100)}", flush=True)

        pair = pick_3090_pair(free)
        if pair:
            if "eager" in modes:
                do_case("moe_tp2_3090_eager", pair, 2, True)
            if "cg" in modes:
                do_case("moe_tp2_3090_cg", pair, 2, False)
        else:
            print("SKIP TP2: need 2 free 3090s", flush=True)

    if args.suite in ("all", "pro6000"):
        if free.get(GPU_PRO6000, 0) >= 40000:
            # cutlass + MoE CG on both sides (fair CG compare)
            if "eager" in modes:
                do_case(
                    "moe_tp1_pro6000_cutlass_eager",
                    [GPU_PRO6000],
                    1,
                    True,
                    moe_backend="cutlass",
                    blackwell=True,
                )
            if "cg" in modes:
                do_case(
                    "moe_tp1_pro6000_cutlass_cg",
                    [GPU_PRO6000],
                    1,
                    False,
                    moe_backend="cutlass",
                    blackwell=True,
                )
        else:
            print(f"SKIP PRO6000 free={free.get(GPU_PRO6000)}", flush=True)

    payload = {
        "meta": {
            "model": str(MOE),
            "prompt": PROMPT,
            "prompt_tokens": len(ids),
            "max_tokens": args.max_tokens,
            "warmup": args.warmup,
            "repeats": args.repeats,
            "modes": modes,
            "suite": args.suite,
            "generated": time.strftime("%Y-%m-%d %H:%M:%S"),
            "gpu_free": free,
            "notes": [
                "TTFT = wall max_tokens=1 with shared prompt_token_ids",
                "PRO6000: FLASHINFER_CUDA_ARCH_LIST=12.0a + CUDA 12.8 → cutlass MoE",
                "v7 MoE CG only when moe_fused_backend=cutlass (moe_decode_cg_ok)",
                "A100/3090: Triton MoE → CG request does not capture MoE kernels",
            ],
        },
        "results": results,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2))
    md = OUT.with_suffix(".md")
    lines = [
        "# MoE e2e ports: vllm-v7 vs official",
        "",
        f"- Generated: `{payload['meta']['generated']}`",
        f"- Model: `{MOE}`",
        f"- Suite: `{args.suite}`",
        f"- Prompt tokens: `{len(ids)}`, max_new=`{args.max_tokens}`",
        f"- Warmup/repeats: {args.warmup}/{args.repeats}",
        f"- Modes: {modes}",
        "",
        "| Case | GPUs | Mode | v7 TTFT | off TTFT | v7/off TTFT | v7 decode | off decode | v7/off | v7 e2e tok/s | off e2e tok/s | v7 e2e s | off e2e s |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in results:
        v7, off = r["v7"], r["official"]
        mode = "eager" if v7["eager"] else "cg"
        lines.append(
            f"| {r['case']} | {v7['physical_gpus']} | {mode} | "
            f"{v7['ttft_ms']:.2f} | {off['ttft_ms']:.2f} | {r['ratio_ttft_v7_over_off']:.2f}x | "
            f"{v7['decode_tok_s']:.2f} | {off['decode_tok_s']:.2f} | {r['ratio_decode_v7_over_off']:.2f}x | "
            f"{v7['e2e_tok_s']:.2f} | {off['e2e_tok_s']:.2f} | "
            f"{v7['e2e_s']:.3f} | {off['e2e_s']:.3f} |"
        )
    lines += [
        "",
        "## Notes",
        "",
        "- PRO6000 cutlass+cg: both engines request CUDA Graph; v7 MoE CG requires cutlass.",
        "- Ampere Triton: v7 MoE stays eager even if `cg` mode is requested.",
        "- Ratios: `<1` on TTFT means v7 faster; `>1` on decode/e2e tok/s means v7 faster.",
        "",
    ]
    md.write_text("\n".join(lines))
    print(f"\nWrote {OUT}")
    print(f"Wrote {md}")
    print("\n======== SUMMARY ========")
    for r in results:
        print(
            f"{r['case']:<36} "
            f"ttft {r['v7']['ttft_ms']:.1f}/{r['official']['ttft_ms']:.1f} "
            f"({r['ratio_ttft_v7_over_off']:.2f}x)  "
            f"decode {r['v7']['decode_tok_s']:.1f}/{r['official']['decode_tok_s']:.1f} "
            f"({r['ratio_decode_v7_over_off']:.2f}x)  "
            f"e2e_tok/s {r['v7']['e2e_tok_s']:.1f}/{r['official']['e2e_tok_s']:.1f} "
            f"({r['ratio_e2e_tok_v7_over_off']:.2f}x)"
        )


if __name__ == "__main__":
    main()
