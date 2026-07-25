#!/usr/bin/env python3
"""Full vllm-v7 vs official vLLM matrix: text / VL / MoE × parallel × metrics.

Metrics per case:
  - TTFT (ms), decode tok/s, e2e tok/s, e2e latency (s)
  - optional batch: aggregate tok/s, per-req latency

Engines use separate interpreters (.venv vs .venv-vllm).
"""
from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
V7_DIR = ROOT / "vllm-v7"
V7_PY = ROOT / ".venv" / "bin" / "python"
OFF_PY = ROOT / ".venv-vllm" / "bin" / "python"
HF = Path("/mnt/nfs/zhangli/huggingface")
OUT = V7_DIR / "bench" / "results" / f"full_compare_{time.strftime('%Y%m%d_%H%M%S')}.json"

TEXT_MODEL = str(HF / "Qwen3-8B")
VL_MODEL = str(HF / "Qwen3-VL-8B-Instruct")
MOE_MODEL = str(
    next(
        (
            p
            for p in (
                HF / "Qwen1.5-MoE-A2.7B",
                HF / "Qwen1.5-MoE-A2.7B-Chat",
                HF / "Qwen3-30B-A3B",
            )
            if p.is_dir() and any(p.glob("*.safetensors"))
        ),
        HF / "Qwen1.5-MoE-A2.7B",
    )
)

# Prefer free homogeneous 3090s (GPU4 often occupied by other jobs).
GPU_3090 = [1, 5, 2, 4]
GPU_A100 = 3
GPU_BLACKWELL = 0
# PP may be hetero across stages; prefer free 3090 → A100 → Blackwell → tight 3090.
ALL_GPUS_PP = [1, 5, 2, 3, 0, 4]
MIN_FREE_MIB = 18000
MIN_FREE_TP4_MIB = 9000  # allow slightly occupied cards for TP4
MIN_FREE_PP_MIB = 6000  # PP shards weights → lower bar per card


def _nvidia_lib_path() -> str:
    site = subprocess.check_output(
        [str(OFF_PY), "-c", "import site; print(site.getsitepackages()[0])"],
        text=True,
    ).strip()
    libs: list[str] = []
    nvidia = Path(site) / "nvidia"
    if nvidia.is_dir():
        for p in nvidia.rglob("lib"):
            if p.is_dir():
                libs.append(str(p))
        # CUDA 13 wheels may use nvidia/cu13/lib
        cu13 = nvidia / "cu13" / "lib"
        if cu13.is_dir():
            libs.insert(0, str(cu13))
    return ":".join(dict.fromkeys(libs))


def _base_env() -> dict:
    env = os.environ.copy()
    env["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    env.setdefault("NCCL_IB_DISABLE", "1")
    env.setdefault("NCCL_NET", "Socket")
    return env


def _off_env(device_ids: list[int]) -> dict:
    """Official vLLM env: hide unused GPUs, modern nvcc, cu13 libs."""
    env = _base_env()
    env["CUDA_VISIBLE_DEVICES"] = ",".join(str(d) for d in device_ids)
    cuda_home = "/usr/local/cuda-12.8"
    if not Path(cuda_home).is_dir():
        cuda_home = "/usr/local/cuda"
    env["CUDA_HOME"] = cuda_home
    env["PATH"] = f"{cuda_home}/bin:" + env.get("PATH", "")
    libs = _nvidia_lib_path()
    extra = [libs]
    for p in (f"{cuda_home}/lib64",):
        if Path(p).is_dir():
            extra.append(p)
    if env.get("LD_LIBRARY_PATH"):
        extra.append(env["LD_LIBRARY_PATH"])
    env["LD_LIBRARY_PATH"] = ":".join(x for x in extra if x)
    # System /usr/bin/nvcc is CUDA 9.1; flashinfer JIT breaks on it.
    # Prefer torch sampler for stable benches (still comparable e2e).
    env["VLLM_USE_FLASHINFER_SAMPLER"] = "0"
    return env


def _v7_env(device_ids: list[int]) -> tuple[dict, list[int]]:
    """Hide Blackwell/other GPUs; remap device_ids to 0..n-1; modern nvcc for FI JIT."""
    env = _base_env()
    env["CUDA_VISIBLE_DEVICES"] = ",".join(str(d) for d in device_ids)
    env["PYTHONPATH"] = str(V7_DIR)
    cuda_home = "/usr/local/cuda-12.8"
    if not Path(cuda_home).is_dir():
        cuda_home = "/usr/local/cuda"
    env["CUDA_HOME"] = cuda_home
    # Drop ancient /usr/bin/nvcc (CUDA 9.x) — FlashInfer JIT needs modern nvcc.
    path_parts = [f"{cuda_home}/bin"]
    for p in env.get("PATH", "").split(":"):
        if p and p not in path_parts and p != "/usr/bin":
            path_parts.append(p)
    path_parts.append("/usr/bin")  # keep last for non-nvcc tools
    env["PATH"] = ":".join(path_parts)
    env["CUDA_NVCC_EXECUTABLE"] = f"{cuda_home}/bin/nvcc"
    # Pin FI arches to selected GPUs only (mixed hosts with SM12 + CUDA 12.8).
    env.setdefault("FLASHINFER_CUDA_ARCH_LIST", "8.0 8.6 9.0a")
    remapped = list(range(len(device_ids)))
    return env, remapped


def _gpu_free_mib() -> dict[int, int]:
    try:
        out = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=index,memory.free",
                "--format=csv,noheader,nounits",
            ],
            text=True,
        )
    except Exception:
        return {}
    free: dict[int, int] = {}
    for line in out.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) >= 2:
            free[int(parts[0])] = int(float(parts[1]))
    return free


def pick_gpus(
    n: int,
    candidates: list[int] | None = None,
    *,
    min_free: int | None = None,
    require: bool = False,
) -> list[int] | None:
    """Pick n GPUs with enough free memory (prefer candidate order, then freest).

    If ``require`` and fewer than ``n`` meet ``min_free``, return None.
    Otherwise fall back to the freest ``n`` candidates (may be tight).
    """
    cands = list(candidates or GPU_3090)
    floor = MIN_FREE_MIB if min_free is None else min_free
    free = _gpu_free_mib()
    chosen: list[int] = []
    for g in cands:
        if free.get(g, 0) >= floor:
            chosen.append(g)
        if len(chosen) >= n:
            return chosen
    if require:
        return None
    ranked = sorted(cands, key=lambda g: free.get(g, 0), reverse=True)
    out = ranked[:n]
    return out if len(out) == n else None


def _gpu_snapshot() -> str:
    free = _gpu_free_mib()
    return " ".join(f"{i}:{free.get(i, 0)}" for i in range(6))


def _kill_orphans() -> None:
    """Best-effort cleanup of leftover v7/vLLM bench children of this user.

    Only targets our temp bench scripts and vllm-v7 worker names — never
    blanket-kill every ``spawn_main`` on the machine (other jobs / prior
    runs share that cmdline pattern).
    """
    try:
        out = subprocess.check_output(
            ["ps", "-u", os.getlogin() if hasattr(os, "getlogin") else "zhangli",
             "-o", "pid=,cmd="],
            text=True,
        )
    except Exception:
        try:
            out = subprocess.check_output(
                ["ps", "-u", "zhangli", "-o", "pid=,cmd="], text=True
            )
        except Exception:
            return
    kill_pids: list[int] = []
    for line in out.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            pid_s, cmd = line.split(None, 1)
        except ValueError:
            continue
        low = cmd.lower()
        if "bench_vs_vllm_full.py" in cmd:
            continue
        if any(
            k in low
            for k in (
                "/tmp/tmp",  # our NamedTemporaryFile bench scripts
                "vllm-v7-",  # Process name prefix from transport.spawn
                "vllm.entrypoints",
                "vllm.worker",
                "enginecore",
            )
        ):
            kill_pids.append(int(pid_s))
    for pid in kill_pids:
        try:
            os.kill(pid, signal.SIGKILL)
        except OSError:
            pass
    if kill_pids:
        time.sleep(1.0)


def _run_script(py: Path, code: str, *, env: dict, cwd: Path, timeout: int = 1800) -> dict:
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False, dir="/tmp") as f:
        f.write(code)
        path = f.name
    proc = None
    try:
        proc = subprocess.Popen(
            [str(py), path],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env=env,
            cwd=str(cwd),
            start_new_session=True,  # own process group → clean kill
        )
        try:
            stdout, stderr = proc.communicate(timeout=timeout)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except OSError:
                proc.kill()
            _kill_orphans()
            raise RuntimeError(f"script timeout after {timeout}s")
        if proc.returncode != 0:
            _kill_orphans()
            raise RuntimeError(
                f"script failed rc={proc.returncode}:\n{stderr[-5000:]}\n{stdout[-2000:]}"
            )
        for line in stdout.splitlines():
            if line.startswith("JSON:"):
                return json.loads(line[5:])
        raise RuntimeError(f"no JSON line:\n{stdout[-2000:]}\n{stderr[-2000:]}")
    finally:
        if proc is not None and proc.poll() is None:
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except OSError:
                pass
        try:
            os.unlink(path)
        except OSError:
            pass
        _kill_orphans()
        time.sleep(0.5)


def run_v7_text(
    *,
    model: str,
    tp: int,
    pp: int,
    device_ids: list[int],
    prompt: str,
    max_tokens: int,
    batch: int,
    enforce_eager: bool,
    warmup: int,
    repeats: int,
    num_kv: int = 64,
    context_len: int = 2048,
    moe_cuda_graph: bool = True,
    moe_fused_backend: str = "auto",
) -> dict:
    # CRITICAL: wrap in ``if __name__ == "__main__"``. Spawn workers re-import
    # this file as ``__mp_main__``; without the guard they re-enter Engine()
    # and recursively spawn more workers → silent hang / zombie children.
    code = r"""
import gc, json, sys, time
sys.path.insert(0, {v7!r})

def free():
    import torch
    gc.collect()
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            with torch.cuda.device(i):
                torch.cuda.empty_cache(); torch.cuda.synchronize()

def main():
    import torch
    from src import Engine, EngineConfig, SamplingParams
    cfg = EngineConfig(
        model={model!r}, tp_size={tp}, pp_size={pp}, device_ids={ids!r},
        enforce_eager={eager}, max_num_seqs=max(8, {batch}),
        context_len={ctx}, num_kvcache_blocks={nkv},
        gpu_memory_utilization=0.85, nccl_port=0, rpc_backend="zmq",
        moe_cuda_graph={moe_cg},
        moe_fused_backend={moe_backend!r},
    )
    eng = Engine(cfg)
    prompt, n, batch = {prompt!r}, {n}, {batch}
    try:
        for _ in range({warmup}):
            if batch <= 1:
                seq = eng.add_request(prompt, SamplingParams(temperature=0.0, max_tokens=8, ignore_eos=True))
                while not seq.is_finished: eng.step()
            else:
                eng.generate_batch([prompt]*min(batch,2), SamplingParams(temperature=0.0, max_tokens=4, ignore_eos=True))
        times, ttfts = [], []
        if batch <= 1:
            for _ in range({repeats}):
                seq = eng.add_request(prompt, SamplingParams(temperature=0.0, max_tokens=n, ignore_eos=True))
                t0 = time.perf_counter(); first=None
                while not seq.is_finished:
                    eng.step()
                    if first is None and seq.num_completion_tokens >= 1:
                        first = time.perf_counter()
                t1 = time.perf_counter()
                times.append(t1-t0)
                ttfts.append((first-t0)*1000 if first else float("nan"))
            e2e = sum(times)/len(times); ttft=sum(ttfts)/len(ttfts)
            decode_s = max(e2e - ttft/1000.0, 1e-6)
            out = {{
                "engine":"vllm-v7","tp":{tp},"pp":{pp},"batch":1,"eager":{eager},
                "e2e_s": e2e, "ttft_ms": ttft,
                "e2e_tok_s": n/e2e, "decode_tok_s": (n-1)/decode_s,
                "prompt_tokens": len(eng.tokenizer.encode(prompt)),
                "max_tokens": n,
                "physical_gpus": {phys!r},
            }}
        else:
            prompts = [prompt]*batch
            sp = SamplingParams(temperature=0.0, max_tokens=n, ignore_eos=True)
            for _ in range({repeats}):
                t0 = time.perf_counter()
                eng.generate_batch(prompts, sp)
                times.append(time.perf_counter()-t0)
            e2e = sum(times)/len(times)
            out = {{
                "engine":"vllm-v7","tp":{tp},"pp":{pp},"batch":batch,"eager":{eager},
                "e2e_s": e2e, "ttft_ms": None,
                "e2e_tok_s": (n*batch)/e2e, "decode_tok_s": (n*batch)/e2e,
                "agg_tok_s": (n*batch)/e2e,
                "per_req_e2e_s": e2e,
                "prompt_tokens": len(eng.tokenizer.encode(prompt)),
                "max_tokens": n,
                "physical_gpus": {phys!r},
            }}
        print("JSON:"+json.dumps(out), flush=True)
    finally:
        eng.destroy(); free()

if __name__ == "__main__":
    main()
""".format(
        v7=str(V7_DIR),
        model=model,
        tp=tp,
        pp=pp,
        ids=list(range(len(device_ids))),
        phys=device_ids,
        eager=enforce_eager,
        prompt=prompt,
        n=max_tokens,
        batch=batch,
        warmup=warmup,
        repeats=repeats,
        nkv=num_kv,
        ctx=context_len,
        moe_cg=moe_cuda_graph,
        moe_backend=moe_fused_backend,
    )
    env, _ = _v7_env(device_ids)
    return _run_script(V7_PY, code, env=env, cwd=V7_DIR)


def run_off_text(
    *,
    model: str,
    tp: int,
    pp: int,
    device_ids: list[int],
    prompt: str,
    max_tokens: int,
    batch: int,
    enforce_eager: bool,
    warmup: int,
    repeats: int,
    enable_ep: bool = False,
    context_len: int = 2048,
    num_kv: int | None = None,  # accepted for API parity with run_v7_text; unused
    moe_cuda_graph: bool = True,  # v7-only; ignored by official
    moe_fused_backend: str = "auto",  # v7-only; ignored by official
) -> dict:
    del moe_cuda_graph, moe_fused_backend
    code = r"""
import json, time, os, gc
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
from vllm import LLM, SamplingParams
import vllm

def _ttft_from_metrics(outs):
    m = getattr(outs[0], "metrics", None)
    if m is None:
        return float("nan")
    try:
        arr = getattr(m, "arrival_time", None)
        first = getattr(m, "first_token_time", None)
        if arr is not None and first is not None and first == first and arr == arr:
            return (first - arr) * 1000.0
    except Exception:
        pass
    return float("nan")

def _shutdown(llm):
    try:
        llm.llm_engine.engine_core.shutdown()
    except Exception:
        pass
    del llm
    gc.collect()
    try:
        from vllm.distributed.parallel_state import (
            destroy_model_parallel, destroy_distributed_environment,
        )
        destroy_model_parallel()
        destroy_distributed_environment()
    except Exception:
        pass
    try:
        import torch.distributed as dist
        if dist.is_available() and dist.is_initialized():
            dist.destroy_process_group()
    except Exception:
        pass

kwargs = dict(
    model={model!r},
    tensor_parallel_size={tp},
    pipeline_parallel_size={pp},
    dtype="bfloat16",
    max_model_len={ctx},
    max_num_seqs=max(8, {batch}),
    gpu_memory_utilization=0.85,
    enforce_eager={eager},
    disable_log_stats=False,
)
if {ep}:
    kwargs["enable_expert_parallel"] = True
llm = LLM(**kwargs)
sp = SamplingParams(temperature=0.0, max_tokens={n}, ignore_eos=True)
sp1 = SamplingParams(temperature=0.0, max_tokens=1, ignore_eos=True)
prompt, batch = {prompt!r}, {batch}
for _ in range({warmup}):
    llm.generate([prompt]*min(batch,2), SamplingParams(temperature=0.0, max_tokens=4, ignore_eos=True))
times, ttfts = [], []
if batch <= 1:
    # Wall-clock TTFT = latency of max_tokens=1 (prefill + first token).
    # Metrics path is kept as a secondary source when available.
    for _ in range({repeats}):
        t0 = time.perf_counter()
        outs1 = llm.generate([prompt], sp1)
        wall_ttft = (time.perf_counter() - t0) * 1000.0
        m_ttft = _ttft_from_metrics(outs1)
        ttfts.append(m_ttft if m_ttft == m_ttft else wall_ttft)
        t0 = time.perf_counter()
        llm.generate([prompt], sp)
        times.append(time.perf_counter() - t0)
    e2e = sum(times)/len(times)
    ttft = sum(ttfts)/len(ttfts)
    decode_s = max(e2e - ttft/1000.0, 1e-6)
    out = {{
        "engine":"vllm-official","tp":{tp},"pp":{pp},"batch":1,"eager":{eager},
        "e2e_s": e2e, "ttft_ms": ttft,
        "e2e_tok_s": {n}/e2e,
        "decode_tok_s": ({n}-1)/decode_s,
        "max_tokens": {n},
        "vllm_version": getattr(vllm,"__version__","?"),
        "ttft_method": "wall_max_tokens_1",
    }}
else:
    prompts = [prompt]*batch
    for _ in range({repeats}):
        t0 = time.perf_counter()
        llm.generate(prompts, sp)
        times.append(time.perf_counter()-t0)
    e2e = sum(times)/len(times)
    out = {{
        "engine":"vllm-official","tp":{tp},"pp":{pp},"batch":batch,"eager":{eager},
        "e2e_s": e2e, "ttft_ms": None,
        "e2e_tok_s": ({n}*batch)/e2e, "decode_tok_s": ({n}*batch)/e2e,
        "agg_tok_s": ({n}*batch)/e2e,
        "max_tokens": {n},
        "vllm_version": getattr(vllm,"__version__","?"),
    }}
print("JSON:"+json.dumps(out))
_shutdown(llm)
""".format(
        model=model,
        tp=tp,
        pp=pp,
        eager=enforce_eager,
        prompt=prompt,
        n=max_tokens,
        batch=batch,
        warmup=warmup,
        repeats=repeats,
        ep=enable_ep,
        ctx=context_len,
    )
    return _run_script(OFF_PY, code, env=_off_env(device_ids), cwd=ROOT)


def run_v7_vl(
    *,
    model: str,
    tp: int,
    device_ids: list[int],
    n_images: int,
    max_tokens: int,
    enforce_eager: bool,
    warmup: int,
    repeats: int,
) -> dict:
    code = r"""
import gc, json, sys, time
sys.path.insert(0, {v7!r})

def main():
    from PIL import Image, ImageDraw
    import torch
    from src import Engine, EngineConfig

    def img(c, s=256):
        im = Image.new("RGB", (s,s), c)
        ImageDraw.Draw(im).rectangle([24,24,s-24,s-24], fill=(255,255,255))
        return im

    colors = [(30,80,220),(220,40,40),(40,180,60),(200,160,20)]
    images = [img(colors[i%len(colors)]) for i in range({n_img})]
    cfg = EngineConfig(
        model={model!r}, tp_size={tp}, pp_size=1, device_ids={ids!r},
        enforce_eager={eager}, max_num_seqs=4, context_len=4096,
        num_kvcache_blocks=128, gpu_memory_utilization=0.85,
        nccl_port=0, mm_encoder_tp_mode="data",
    )
    eng = Engine(cfg)
    prompt = "Describe briefly." if {n_img}==1 else "List main colors in order. Brief."
    n = {n}
    try:
        for _ in range({warmup}):
            list(eng.generate(prompt, max_new_tokens=4, temperature=0.0, images=images))
        times, ttfts = [], []
        for _ in range({repeats}):
            t0 = time.perf_counter(); first=None; toks=0
            for _tok in eng.generate(prompt, max_new_tokens=n, temperature=0.0, images=images):
                if first is None: first = time.perf_counter()
                toks += 1
            t1 = time.perf_counter()
            times.append(t1-t0)
            ttfts.append((first-t0)*1000 if first else float("nan"))
        t0 = time.perf_counter(); first=None
        for _tok in eng.generate("Yes or no: white square?", max_new_tokens=4, temperature=0.0, images=images[:1] if {n_img}>0 else None):
            if first is None: first = time.perf_counter()
        cache_ttft = (first-t0)*1000 if first else None
        e2e = sum(times)/len(times); ttft=sum(ttfts)/len(ttfts)
        decode_s = max(e2e - ttft/1000.0, 1e-6)
        print("JSON:"+json.dumps({{
            "engine":"vllm-v7","modality":"vl","tp":{tp},"n_images":{n_img},"eager":{eager},
            "e2e_s": e2e, "ttft_ms": ttft, "cache_reuse_ttft_ms": cache_ttft,
            "e2e_tok_s": n/e2e, "decode_tok_s": max(n-1,1)/decode_s,
            "max_tokens": n,
            "physical_gpus": {phys!r},
        }}), flush=True)
    finally:
        eng.destroy(); gc.collect(); torch.cuda.empty_cache()

if __name__ == "__main__":
    main()
""".format(
        v7=str(V7_DIR),
        model=model,
        tp=tp,
        ids=list(range(len(device_ids))),
        phys=device_ids,
        eager=enforce_eager,
        n_img=n_images,
        n=max_tokens,
        warmup=warmup,
        repeats=repeats,
    )
    env, _ = _v7_env(device_ids)
    return _run_script(V7_PY, code, env=env, cwd=V7_DIR, timeout=2400)


def run_off_vl(
    *,
    model: str,
    tp: int,
    device_ids: list[int],
    n_images: int,
    max_tokens: int,
    enforce_eager: bool,
    warmup: int,
    repeats: int,
) -> dict:
    code = r"""
import json, time, os, gc
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
from PIL import Image, ImageDraw
from vllm import LLM, SamplingParams
import vllm

def img(c, s=256):
    im = Image.new("RGB", (s,s), c)
    ImageDraw.Draw(im).rectangle([24,24,s-24,s-24], fill=(255,255,255))
    return im

def _shutdown(llm):
    try:
        llm.llm_engine.engine_core.shutdown()
    except Exception:
        pass
    del llm
    gc.collect()
    try:
        from vllm.distributed.parallel_state import (
            destroy_model_parallel, destroy_distributed_environment,
        )
        destroy_model_parallel()
        destroy_distributed_environment()
    except Exception:
        pass
    try:
        import torch.distributed as dist
        if dist.is_available() and dist.is_initialized():
            dist.destroy_process_group()
    except Exception:
        pass

colors = [(30,80,220),(220,40,40),(40,180,60),(200,160,20)]
images = [img(colors[i%len(colors)]) for i in range({n_img})]
llm = LLM(
    model={model!r}, tensor_parallel_size={tp}, dtype="bfloat16",
    max_model_len=4096, max_num_seqs=4, gpu_memory_utilization=0.85,
    enforce_eager={eager}, disable_log_stats=False,
    limit_mm_per_prompt={{"image": {n_img}}},
)
sp = SamplingParams(temperature=0.0, max_tokens={n}, ignore_eos=True)
sp1 = SamplingParams(temperature=0.0, max_tokens=1, ignore_eos=True)
prompt = "Describe briefly." if {n_img}==1 else "List main colors in order. Brief."
req = {{
    "prompt": prompt,
    "multi_modal_data": {{"image": images if {n_img}>1 else images[0]}},
}}
for _ in range({warmup}):
    llm.generate([req], SamplingParams(temperature=0.0, max_tokens=4, ignore_eos=True))
times, ttfts = [], []
for _ in range({repeats}):
    t0 = time.perf_counter()
    llm.generate([req], sp1)
    ttfts.append((time.perf_counter() - t0) * 1000.0)
    t0 = time.perf_counter()
    llm.generate([req], sp)
    times.append(time.perf_counter() - t0)
t0 = time.perf_counter()
llm.generate([req], SamplingParams(temperature=0.0, max_tokens=4, ignore_eos=True))
cache_ttft = (time.perf_counter() - t0) * 1000.0
e2e = sum(times)/len(times)
ttft = sum(ttfts)/len(ttfts)
decode_s = max(e2e - ttft/1000.0, 1e-6)
print("JSON:"+json.dumps({{
    "engine":"vllm-official","modality":"vl","tp":{tp},"n_images":{n_img},"eager":{eager},
    "e2e_s": e2e, "ttft_ms": ttft, "cache_reuse_ttft_ms": cache_ttft,
    "e2e_tok_s": {n}/e2e,
    "decode_tok_s": ({n}-1)/decode_s,
    "max_tokens": {n},
    "vllm_version": getattr(vllm,"__version__","?"),
    "ttft_method": "wall_max_tokens_1",
}}))
_shutdown(llm)
""".format(
        model=model,
        tp=tp,
        eager=enforce_eager,
        n_img=n_images,
        n=max_tokens,
        warmup=warmup,
        repeats=repeats,
    )
    return _run_script(OFF_PY, code, env=_off_env(device_ids), cwd=ROOT, timeout=2400)


def _pair(tag: str, v7: dict | None, off: dict | None, err7=None, erro=None) -> dict:
    row = {"case": tag, "v7": v7, "official": off, "error_v7": err7, "error_official": erro}
    if v7 and off and "e2e_tok_s" in v7 and "e2e_tok_s" in off:
        row["ratio_official_over_v7_e2e"] = off["e2e_tok_s"] / max(v7["e2e_tok_s"], 1e-9)
        if v7.get("decode_tok_s") and off.get("decode_tok_s"):
            row["ratio_official_over_v7_decode"] = off["decode_tok_s"] / max(
                v7["decode_tok_s"], 1e-9
            )
        if v7.get("ttft_ms") and off.get("ttft_ms") and off["ttft_ms"] == off["ttft_ms"]:
            row["ratio_v7_over_official_ttft"] = v7["ttft_ms"] / max(off["ttft_ms"], 1e-9)
    return row


def _write_report(path: Path, payload: dict) -> None:
    """Markdown performance report next to the JSON results."""
    lines = [
        "# vllm-v7 vs official vLLM — performance report",
        "",
        f"- Generated: `{payload.get('meta', {}).get('generated', '')}`",
        f"- Mode: `{payload.get('meta', {}).get('mode', '')}`",
        f"- Suites: `{', '.join(payload.get('meta', {}).get('suites', []))}`",
        f"- Warmup/repeats: "
        f"{payload.get('meta', {}).get('warmup')}/"
        f"{payload.get('meta', {}).get('repeats')}",
        f"- GPU free @ start: `{payload.get('meta', {}).get('gpu_free_start', '')}`",
        "",
        "| Case | GPUs | v7 decode | off decode | v7/off | v7 TTFT | off TTFT | Notes |",
        "|---|---|---:|---:|---:|---:|---:|---|",
    ]
    for r in payload.get("results", []):
        v7, off = r.get("v7") or {}, r.get("official") or {}
        vd = v7.get("decode_tok_s")
        od = off.get("decode_tok_s")
        ratio = ""
        if isinstance(vd, (int, float)) and isinstance(od, (int, float)) and vd > 0:
            ratio = f"{vd / od:.2f}x"
        notes = []
        if r.get("error_v7"):
            notes.append(f"v7:{str(r['error_v7'])[:60]}")
        if r.get("error_official"):
            notes.append(f"off:{str(r['error_official'])[:60]}")
        gpus = r.get("device_ids") or (v7.get("physical_gpus") if v7 else None) or ""
        lines.append(
            f"| {r.get('case', '?')} | {gpus} | "
            f"{vd if vd is not None else 'ERR'} | "
            f"{od if od is not None else 'ERR'} | {ratio or '-'} | "
            f"{v7.get('ttft_ms', 'ERR')} | {off.get('ttft_ms', 'ERR')} | "
            f"{'; '.join(notes) if notes else 'ok'} |"
        )
    lines += [
        "",
        "## How to read",
        "",
        "- **decode tok/s**: higher is better (generation throughput after first token).",
        "- **v7/off**: `v7_decode / official_decode` (>1 means v7 faster).",
        "- **TTFT**: time to first token (ms); lower is better. Prefill-heavy / VL vision dominate.",
        "- CG cases use `enforce_eager=False` on both engines.",
        "- PP>2 / PP6 may use hetero GPUs (3090+A100+Blackwell); official may reject hetero.",
        "",
    ]
    path.write_text("\n".join(lines))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--suite",
        default="text,vl,moe",
        help="comma: text,vl,moe",
    )
    ap.add_argument(
        "--mode",
        choices=("cg", "eager", "all"),
        default="cg",
        help="cg=CUDA Graph primary matrix (default); eager=eager only; all=both",
    )
    ap.add_argument("--warmup", type=int, default=1)
    ap.add_argument("--repeats", type=int, default=2)
    ap.add_argument("--quick", action="store_true", help="smaller matrix")
    args = ap.parse_args()
    suites = {s.strip() for s in args.suite.split(",") if s.strip()}
    want_cg = args.mode in ("cg", "all")
    want_eager = args.mode in ("eager", "all")

    results: list[dict] = []
    short = "The capital of France is"
    medium = ("In machine learning, transformers use attention. " * 40).strip()
    gpu_free_start = _gpu_snapshot()
    print(
        f"MODE={args.mode} suites={sorted(suites)} free_mib=[{gpu_free_start}]",
        flush=True,
    )

    def skip(tag: str, reason: str, gpus=None) -> None:
        print(f"SKIP {tag}: {reason} gpus={gpus}", flush=True)
        results.append(
            {
                "case": tag,
                "v7": None,
                "official": None,
                "error_v7": f"skipped: {reason}",
                "error_official": f"skipped: {reason}",
                "device_ids": gpus,
            }
        )
        OUT.parent.mkdir(parents=True, exist_ok=True)
        OUT.write_text(json.dumps({"partial": True, "results": results}, indent=2))

    def do_text_case(tag, **kw):
        ids = kw.get("device_ids")
        if not ids:
            skip(tag, "no device_ids", ids)
            return None
        print(f"\n=== TEXT {tag}: v7 gpus={ids} free=[{_gpu_snapshot()}] ===", flush=True)
        v7 = off = None
        e7 = eo = None
        try:
            kw_v7 = {k: v for k, v in kw.items() if k != "enable_ep"}
            v7 = run_v7_text(warmup=args.warmup, repeats=args.repeats, **kw_v7)
            print(
                f"  v7  ttft={v7.get('ttft_ms')} e2e_tok/s={v7['e2e_tok_s']:.2f} "
                f"decode={v7['decode_tok_s']:.2f}",
                flush=True,
            )
        except Exception as e:
            e7 = str(e)[:800]
            print(f"  v7 FAIL: {e7[:300]}", flush=True)
        print(f"=== TEXT {tag}: official ===", flush=True)
        try:
            off = run_off_text(warmup=args.warmup, repeats=args.repeats, **kw)
            print(
                f"  off ttft={off.get('ttft_ms')} e2e_tok/s={off['e2e_tok_s']:.2f} "
                f"decode={off['decode_tok_s']:.2f}",
                flush=True,
            )
        except Exception as e:
            eo = str(e)[:800]
            print(f"  off FAIL: {eo[:300]}", flush=True)
        row = _pair(tag, v7, off, e7, eo)
        row["device_ids"] = ids
        results.append(row)
        OUT.parent.mkdir(parents=True, exist_ok=True)
        OUT.write_text(json.dumps({"partial": True, "results": results}, indent=2))
        _kill_orphans()
        time.sleep(2.0)
        return row

    def do_vl(tag, **kw):
        ids = kw.get("device_ids")
        if not ids:
            skip(tag, "no device_ids", ids)
            return None
        print(f"\n=== VL {tag}: v7 gpus={ids} ===", flush=True)
        v7 = off = None
        e7 = eo = None
        try:
            v7 = run_v7_vl(warmup=args.warmup, repeats=args.repeats, **kw)
            print(
                f"  v7  ttft={v7.get('ttft_ms')} decode={v7['decode_tok_s']:.2f}",
                flush=True,
            )
        except Exception as e:
            e7 = str(e)[:800]
            print(f"  v7 FAIL: {e7[:300]}", flush=True)
        print(f"=== VL {tag}: official ===", flush=True)
        try:
            off = run_off_vl(warmup=args.warmup, repeats=args.repeats, **kw)
            print(
                f"  off ttft={off.get('ttft_ms')} decode={off['decode_tok_s']:.2f}",
                flush=True,
            )
        except Exception as e:
            eo = str(e)[:800]
            print(f"  off FAIL: {eo[:300]}", flush=True)
        row = _pair(tag, v7, off, e7, eo)
        row["device_ids"] = ids
        results.append(row)
        OUT.write_text(json.dumps({"partial": True, "results": results}, indent=2))
        _kill_orphans()
        time.sleep(2.0)
        return row

    # ------------------------------------------------------------------ text
    if "text" in suites:
        g1 = pick_gpus(1, require=True)
        g2 = pick_gpus(2, require=True)
        g4 = pick_gpus(4, min_free=MIN_FREE_TP4_MIB, require=True)
        g4_loose = pick_gpus(4, min_free=MIN_FREE_TP4_MIB, require=False)
        # PP: allow hetero (3090 + A100 + Blackwell)
        pp2 = pick_gpus(2, ALL_GPUS_PP, min_free=MIN_FREE_PP_MIB, require=True) or g2
        pp3 = pick_gpus(3, ALL_GPUS_PP, min_free=MIN_FREE_PP_MIB, require=True)
        pp4 = pick_gpus(4, ALL_GPUS_PP, min_free=MIN_FREE_PP_MIB, require=True)
        pp6 = pick_gpus(6, ALL_GPUS_PP, min_free=MIN_FREE_PP_MIB, require=True)
        # TP2×PP2 needs 4 homogeneous 3090s (TP within stage)
        tp2pp2 = pick_gpus(4, GPU_3090, min_free=MIN_FREE_TP4_MIB, require=True)
        print(
            f"GPU pick text: tp1={g1} tp2={g2} tp4={g4 or g4_loose} "
            f"pp2={pp2} pp3={pp3} pp4={pp4} pp6={pp6} tp2pp2={tp2pp2}",
            flush=True,
        )

        def text_short(tag, *, tp, pp, ids, eager, **extra):
            if not ids:
                skip(tag, "insufficient GPUs", ids)
                return
            do_text_case(
                tag,
                model=TEXT_MODEL,
                tp=tp,
                pp=pp,
                device_ids=ids,
                prompt=short,
                max_tokens=64,
                batch=1,
                enforce_eager=eager,
                **extra,
            )

        # --- CG primary ---
        if want_cg:
            text_short("text_tp1_cg_short", tp=1, pp=1, ids=g1, eager=False)
            text_short("text_tp2_cg_short", tp=2, pp=1, ids=g2, eager=False)
            text_short(
                "text_tp4_cg_short",
                tp=4,
                pp=1,
                ids=g4 or g4_loose,
                eager=False,
                num_kv=48,
                context_len=1024,
            )
            text_short("text_pp2_cg_short", tp=1, pp=2, ids=pp2, eager=False)
            if not args.quick:
                text_short("text_pp3_cg_short", tp=1, pp=3, ids=pp3, eager=False)
                text_short("text_pp4_cg_short", tp=1, pp=4, ids=pp4, eager=False)
                text_short(
                    "text_pp6_cg_short",
                    tp=1,
                    pp=6,
                    ids=pp6,
                    eager=False,
                    num_kv=32,
                    context_len=1024,
                )
                text_short(
                    "text_tp2pp2_cg_short",
                    tp=2,
                    pp=2,
                    ids=tp2pp2,
                    eager=False,
                    num_kv=48,
                    context_len=1024,
                )
                do_text_case(
                    "text_tp1_cg_batch4",
                    model=TEXT_MODEL,
                    tp=1,
                    pp=1,
                    device_ids=g1,
                    prompt=short,
                    max_tokens=64,
                    batch=4,
                    enforce_eager=False,
                )
                do_text_case(
                    "text_tp2_cg_batch4",
                    model=TEXT_MODEL,
                    tp=2,
                    pp=1,
                    device_ids=g2,
                    prompt=short,
                    max_tokens=64,
                    batch=4,
                    enforce_eager=False,
                )
                do_text_case(
                    "text_tp1_cg_medium",
                    model=TEXT_MODEL,
                    tp=1,
                    pp=1,
                    device_ids=pick_gpus(1, require=True),
                    prompt=medium,
                    max_tokens=64,
                    batch=1,
                    enforce_eager=False,
                    context_len=4096,
                    num_kv=128,
                )

        # --- eager baselines (mode=eager|all) ---
        if want_eager:
            text_short("text_tp1_eager_short", tp=1, pp=1, ids=g1, eager=True)
            text_short("text_tp2_eager_short", tp=2, pp=1, ids=g2, eager=True)
            text_short(
                "text_tp4_eager_short",
                tp=4,
                pp=1,
                ids=g4 or g4_loose,
                eager=True,
                num_kv=48,
                context_len=1024,
            )
            text_short("text_pp2_eager_short", tp=1, pp=2, ids=pp2, eager=True)
            if not args.quick:
                text_short("text_pp6_eager_short", tp=1, pp=6, ids=pp6, eager=True)
                text_short(
                    "text_tp2pp2_eager_short",
                    tp=2,
                    pp=2,
                    ids=tp2pp2,
                    eager=True,
                    num_kv=48,
                    context_len=1024,
                )
                do_text_case(
                    "text_tp1_eager_batch4",
                    model=TEXT_MODEL,
                    tp=1,
                    pp=1,
                    device_ids=g1,
                    prompt=short,
                    max_tokens=64,
                    batch=4,
                    enforce_eager=True,
                )

    # ------------------------------------------------------------------ VL
    if "vl" in suites and Path(VL_MODEL).is_dir():
        g1 = pick_gpus(1, require=True)
        g2 = pick_gpus(2, require=True)
        g4 = pick_gpus(4, min_free=MIN_FREE_TP4_MIB, require=True)
        if want_cg:
            do_vl(
                "vl_tp1_cg_1img",
                model=VL_MODEL,
                tp=1,
                device_ids=g1,
                n_images=1,
                max_tokens=32,
                enforce_eager=False,
            )
            do_vl(
                "vl_tp2_cg_1img",
                model=VL_MODEL,
                tp=2,
                device_ids=g2,
                n_images=1,
                max_tokens=32,
                enforce_eager=False,
            )
            if not args.quick:
                do_vl(
                    "vl_tp1_cg_3img",
                    model=VL_MODEL,
                    tp=1,
                    device_ids=pick_gpus(1, require=True),
                    n_images=3,
                    max_tokens=24,
                    enforce_eager=False,
                )
                do_vl(
                    "vl_tp4_cg_1img",
                    model=VL_MODEL,
                    tp=4,
                    device_ids=g4,
                    n_images=1,
                    max_tokens=24,
                    enforce_eager=False,
                )
        if want_eager:
            do_vl(
                "vl_tp1_eager_1img",
                model=VL_MODEL,
                tp=1,
                device_ids=pick_gpus(1, require=True),
                n_images=1,
                max_tokens=32,
                enforce_eager=True,
            )
            if not args.quick:
                do_vl(
                    "vl_tp2_eager_1img",
                    model=VL_MODEL,
                    tp=2,
                    device_ids=pick_gpus(2, require=True),
                    n_images=1,
                    max_tokens=32,
                    enforce_eager=True,
                )
    elif "vl" in suites:
        print(f"SKIP vl suite: model missing at {VL_MODEL}", flush=True)

    # ------------------------------------------------------------------ MoE (Triton fused; A100 / 3090 TP)
    # A2.7B ~28GB weights → TP1 needs A100/Blackwell; multi-GPU via TP2 on 3090s.
    if "moe" in suites and Path(MOE_MODEL).is_dir():
        a100_free = _gpu_free_mib().get(GPU_A100, 0)
        moe_kw = dict(
            model=MOE_MODEL,
            prompt=short,
            max_tokens=64,
            batch=1,
            enforce_eager=True,  # Triton path: no decode CG
            num_kv=32,
            context_len=1024,
            moe_cuda_graph=False,
            moe_fused_backend="triton",
        )
        print(
            f"MoE model={MOE_MODEL} backend=triton a100_free={a100_free}MiB",
            flush=True,
        )
        # Single-GPU: A100
        if a100_free >= 35000:
            do_text_case(
                "moe_tp1_a100_triton",
                tp=1,
                pp=1,
                device_ids=[GPU_A100],
                **moe_kw,
            )
        else:
            skip("moe_tp1_a100_triton", f"A100 free={a100_free}MiB < 35000", [GPU_A100])
        # Multi-GPU: TP2 on homogeneous 3090s
        tp2 = pick_gpus(2, candidates=GPU_3090)
        if tp2:
            do_text_case(
                "moe_tp2_3090_triton",
                tp=2,
                pp=1,
                device_ids=tp2,
                **moe_kw,
            )
        else:
            skip("moe_tp2_3090_triton", "no free 3090 pair", None)
        if want_cg:
            print(
                "NOTE: MoE Triton path is eager-only; "
                "skip cutlass/Blackwell CG cases in this suite.",
                flush=True,
            )
    elif "moe" in suites:
        print(f"SKIP moe suite: model missing at {MOE_MODEL}", flush=True)

    payload = {
        "meta": {
            "text_model": TEXT_MODEL,
            "vl_model": VL_MODEL,
            "moe_model": MOE_MODEL,
            "gpus_3090": GPU_3090,
            "gpu_a100": GPU_A100,
            "all_gpus_pp": ALL_GPUS_PP,
            "warmup": args.warmup,
            "repeats": args.repeats,
            "suites": sorted(suites),
            "mode": args.mode,
            "quick": args.quick,
            "generated": time.strftime("%Y-%m-%d %H:%M:%S"),
            "gpu_free_start": gpu_free_start,
            "gpu_free_end": _gpu_snapshot(),
        },
        "results": results,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2))
    md = OUT.with_suffix(".md")
    _write_report(md, payload)
    print(f"\nWrote {OUT}")
    print(f"Wrote {md}")

    print("\n======== SUMMARY ========")
    for r in results:
        v7, off = r.get("v7") or {}, r.get("official") or {}
        print(
            f"{r['case']:<32} "
            f"gpus={str(r.get('device_ids')):<16} "
            f"v7_decode={v7.get('decode_tok_s', 'ERR')!s:>8} "
            f"off_decode={off.get('decode_tok_s', 'ERR')!s:>8} "
            f"v7_ttft={v7.get('ttft_ms', 'ERR')!s:>8} "
            f"off_ttft={off.get('ttft_ms', 'ERR')!s:>8} "
            f"ratio_e2e={r.get('ratio_official_over_v7_e2e', float('nan'))!s}"
        )


if __name__ == "__main__":
    main()
