"""高并发连续批吞吐，尽量打满 4090 D。

同一 workload：prompt 128~256、out 96~128、T=0.6、ignore_eos，一次性入队后 drain。
扫 concurrency=32/64/128/256（v3 CUDA Graph 上限 256）。

v3: decode CG + compile + hash（与 nano / 官方默认一致）
nano: CUDA Graph + hash
官方: 0.26 默认 hash APC + 异步调度

用法（在 vllm-v3/ 下）:
    ../.venv2/bin/python bench/bench_concurrent.py --engine v3
    ../.venv2/bin/python bench/bench_concurrent.py --engine nano
    ../.venv-vllm/bin/python bench/bench_concurrent.py --engine vllm
"""
from __future__ import annotations

import argparse
import gc
import os
import random
import subprocess
import sys
import time
from pathlib import Path

import torch

BENCH = Path(__file__).resolve().parent
V3_ROOT = BENCH.parent
REPO = V3_ROOT.parent
NANO_ROOT = REPO / "nano-vllm"
OUT = BENCH / "out"
MODEL = str(Path.home() / "huggingface" / "Qwen3-0.6B")
TEMP = 0.6
CONCS = (32, 64, 128, 256)


def dummy_ids(n: int, salt: int = 0) -> list[int]:
    n = max(1, n)
    return [((i * 7919 + 10007 + salt * 97) % 150000) + 100 for i in range(n)]


def workload(n: int, *, shared_prefix: int = 0):
    rng = random.Random(0)
    prefix = dummy_ids(shared_prefix, salt=0) if shared_prefix else []
    suffix_lo, suffix_hi = (64, 128) if shared_prefix else (128, 256)
    prompts = [
        prefix + dummy_ids(rng.randint(suffix_lo, suffix_hi), salt=i + 1)
        for i in range(n)
    ]
    max_tokens = [rng.randint(96, 128) for _ in range(n)]
    return prompts, max_tokens


_HEADER = "engine | conc | wall (s) | out tokens | tok/s"


def _append_points(engine_key: str, points: list[dict]) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "bench_concurrent.txt"
    first = not path.exists() or path.stat().st_size == 0
    with path.open("a") as f:
        if first:
            f.write(_HEADER + "\n")
        for rec in points:
            f.write(
                f"{engine_key} | {rec['conc']} | {rec['wall']:.2f} | "
                f"{rec['n_out']}/{rec['want']} | {rec['tps']:.1f}\n"
            )
    print(f"→ {path}", flush=True)


def bench_v3(concs: list[int], device: str, *, prefix: str, shared_prefix: int) -> list[dict]:
    if str(BENCH) not in sys.path:
        sys.path.insert(0, str(BENCH))
    if str(V3_ROOT) not in sys.path:
        sys.path.insert(0, str(V3_ROOT))
    from common import destroy, drain, make_engine  # noqa: E402
    from src import SamplingParams  # noqa: E402

    torch.cuda.set_device(device)
    points = []
    for n in concs:
        prompts, max_tokens = workload(n, shared_prefix=shared_prefix)
        want = sum(max_tokens)
        print(f"  conc={n} init …", flush=True)
        engine = make_engine(
            cg=True,
            compile=True,
            prefix=prefix,
            max_num_seqs=n,
            max_batched=16384,
            gpu_mem=0.90,
        )
        sp_w = SamplingParams(temperature=TEMP, max_tokens=4, ignore_eos=True)
        for it in range(3):
            s = engine.add_request(dummy_ids(32, salt=it), sp_w)
            while not s.is_finished:
                engine.step()
        # Gumbel sampler 按 batch 编译：用目标 conc 短请求把 compile 踢出计时窗口。
        sp_b = SamplingParams(temperature=TEMP, max_tokens=2, ignore_eos=True)
        for i in range(n):
            engine.add_request(dummy_ids(16, salt=100 + i), sp_b)
        drain(engine)
        torch.cuda.synchronize()
        seqs = [
            engine.add_request(
                p, SamplingParams(temperature=TEMP, ignore_eos=True, max_tokens=mt)
            )
            for p, mt in zip(prompts, max_tokens)
        ]
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        drain(engine)
        torch.cuda.synchronize()
        wall = time.perf_counter() - t0
        n_out = sum(s.num_completion_tokens for s in seqs)
        tps = n_out / wall if wall > 0 else 0.0
        destroy(engine)
        rec = {"conc": n, "wall": wall, "n_out": n_out, "want": want, "tps": tps}
        points.append(rec)
        print(
            f"  conc={n}  {wall:.2f}s  {n_out}/{want}  {tps:.1f} tok/s",
            flush=True,
        )
    return points


def bench_nano(concs: list[int], *, shared_prefix: int = 0) -> list[dict]:
    if str(NANO_ROOT) not in sys.path:
        sys.path.insert(0, str(NANO_ROOT))
    from nanovllm import LLM, SamplingParams  # noqa: E402

    points = []
    for n in concs:
        prompts, max_tokens = workload(n, shared_prefix=shared_prefix)
        want = sum(max_tokens)
        print(f"  conc={n} init …", flush=True)
        llm = LLM(
            MODEL,
            enforce_eager=False,
            max_model_len=4096,
            max_num_seqs=n,
            max_num_batched_tokens=16384,
            gpu_memory_utilization=0.90,
        )
        llm.generate(
            [[7, 13, 29, 31]],
            SamplingParams(temperature=TEMP, max_tokens=4, ignore_eos=True),
            use_tqdm=False,
        )
        sps = [
            SamplingParams(temperature=TEMP, ignore_eos=True, max_tokens=mt)
            for mt in max_tokens
        ]
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        outs = llm.generate(prompts, sps, use_tqdm=False)
        torch.cuda.synchronize()
        wall = time.perf_counter() - t0
        n_out = sum(len(o["token_ids"]) for o in outs)
        tps = n_out / wall if wall > 0 else 0.0
        try:
            llm.exit()
        except Exception:
            pass
        del llm
        gc.collect()
        torch.cuda.empty_cache()
        rec = {"conc": n, "wall": wall, "n_out": n_out, "want": want, "tps": tps}
        points.append(rec)
        print(
            f"  conc={n}  {wall:.2f}s  {n_out}/{want}  {tps:.1f} tok/s",
            flush=True,
        )
    return points


def bench_vllm(concs: list[int], *, shared_prefix: int = 0) -> list[dict]:
    from vllm import LLM, SamplingParams  # noqa: E402

    try:
        from vllm import TokensPrompt  # noqa: E402

        pack = lambda ids: TokensPrompt(prompt_token_ids=ids)
    except Exception:
        pack = lambda ids: dict(prompt_token_ids=ids)

    points = []
    for n in concs:
        prompts, max_tokens = workload(n, shared_prefix=shared_prefix)
        want = sum(max_tokens)
        print(f"  conc={n} init …", flush=True)
        llm = LLM(
            model=MODEL,
            enforce_eager=False,
            max_model_len=4096,
            gpu_memory_utilization=0.90,
            max_num_seqs=n,
            max_num_batched_tokens=16384,
            enable_prefix_caching=True,
        )
        llm.generate(
            [pack([7, 13, 29, 31])],
            SamplingParams(temperature=TEMP, max_tokens=4, ignore_eos=True),
            use_tqdm=False,
        )
        sps = [
            SamplingParams(temperature=TEMP, ignore_eos=True, max_tokens=mt)
            for mt in max_tokens
        ]
        reqs = [pack(p) for p in prompts]
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        outs = llm.generate(reqs, sps, use_tqdm=False)
        torch.cuda.synchronize()
        wall = time.perf_counter() - t0
        n_out = sum(len(o.outputs[0].token_ids) for o in outs)
        tps = n_out / wall if wall > 0 else 0.0
        rec = {"conc": n, "wall": wall, "n_out": n_out, "want": want, "tps": tps}
        points.append(rec)
        print(
            f"  conc={n}  {wall:.2f}s  {n_out}/{want}  {tps:.1f} tok/s",
            flush=True,
        )
        del llm
        torch.cuda.empty_cache()
    return points


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--engine", choices=("v3", "nano", "vllm"), required=True)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--concs", default="32,64,128,256")
    p.add_argument("--prefix", choices=("hash", "radix"), default="hash")
    p.add_argument("--shared-prefix", type=int, default=0)
    args = p.parse_args()
    concs = [int(x) for x in args.concs.split(",") if x.strip()]
    print(
        f"# concurrent engine={args.engine} prefix={args.prefix} "
        f"shared={args.shared_prefix} concs={concs}\n",
        flush=True,
    )

    engine_key = args.engine
    if args.engine == "v3" and args.prefix == "radix":
        engine_key = "v3-radix"
    if args.shared_prefix:
        engine_key = f"{engine_key}-sp"

    child_extra = ["--prefix", args.prefix, "--shared-prefix", str(args.shared_prefix)]
    # nano CUDA Graph 退场后显存仍被占用，同进程重建会算出 0 个 KV block；逐 conc 起子进程。
    if args.engine in ("nano", "vllm") and len(concs) > 1 and os.environ.get("BENCH09_CHILD") != "1":
        for n in concs:
            env = {**os.environ, "BENCH09_CHILD": "1"}
            subprocess.check_call(
                [sys.executable, str(Path(__file__).resolve()),
                 "--engine", args.engine, "--device", args.device,
                 "--concs", str(n), *child_extra],
                env=env,
            )
        return

    if args.engine == "v3":
        points = bench_v3(
            concs, args.device, prefix=args.prefix, shared_prefix=args.shared_prefix
        )
    elif args.engine == "nano":
        points = bench_nano(concs, shared_prefix=args.shared_prefix)
    else:
        points = bench_vllm(concs, shared_prefix=args.shared_prefix)

    _append_points(engine_key, points)


if __name__ == "__main__":
    main()
