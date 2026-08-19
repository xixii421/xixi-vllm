"""TP=2 引擎级 A/B：argmax 快路径 vs 全量 logits 贪心，批量 decode 吞吐对比。

torchrun 每进程都执行本脚本，--old 时在所有进程里把 argmax 打回
"全量 logits + argmax"（旧行为），因此多卡 worker 同样生效。

用法（在 vllm-v4/ 下）:
    ../.venv2/bin/torchrun --nproc_per_node=2 bench/bench_ab_tp2.py --mode fast
    ../.venv2/bin/torchrun --nproc_per_node=2 bench/bench_ab_tp2.py --mode old
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

MODEL = str(Path.home() / "huggingface" / "Qwen3-0.6B")
PROMPT_LEN = 128
MAX_TOKENS = 48
N_REQS = 32


def dummy_ids(n: int, salt: int = 0) -> list[int]:
    n = max(1, n)
    return [((i * 7919 + 10007 + salt * 97) % 150000) + 100 for i in range(n)]


def _load_config(args) -> "EngineConfig":
    import yaml

    from src.config import EngineConfig  # noqa: E402

    cfg = {
        "dtype": "auto",
        "enforce_eager": True,
        "torch_compile": False,
        "context_len": 4096,
        "max_num_seqs": 32,
        "max_num_batched_tokens": 8192,
        "kvcache_block_size": 256,
        "prefix_backend": "none",
        "gpu_memory_utilization": 0.6,
        "num_kvcache_blocks": 0,
        "nccl_port": 29500,
        "master_addr": "127.0.0.1",
        "master_port": 29500,
        "nccl_ifname": "",
    }
    return EngineConfig(
        model=MODEL,
        tp_size=args.tp,
        pp_size=1,
        **cfg,
    )


def _worker_loop(worker) -> None:
    import torch.distributed as dist

    while True:
        sched_list = [None]
        dist.broadcast_object_list(sched_list, src=0)
        sched = sched_list[0]
        if sched is None:
            break
        result = worker.execute_model(sched)
        sample = result if result is not None else ([], [])
        dist.gather_object(sample, None, dst=0)


def _run_driver(config, worker, args) -> None:
    from src import Engine, SamplingParams  # noqa: E402
    from src.compute.executor import TorchrunExecutor  # noqa: E402

    executor = TorchrunExecutor(config, worker.device, worker.dtype, worker)
    engine = Engine(config, executor=executor)
    try:
        # 预热（compile + CG 录图不计时）
        sp_w = SamplingParams(temperature=0.0, max_tokens=4, ignore_eos=True)
        for it in range(2):
            for i in range(N_REQS):
                engine.add_request(dummy_ids(PROMPT_LEN, salt=it * 13 + i), sp_w)
            while not engine.scheduler.is_finished():
                engine.step()

        sp = SamplingParams(temperature=0.0, ignore_eos=True, max_tokens=MAX_TOKENS)
        seqs = [
            engine.add_request(dummy_ids(PROMPT_LEN, salt=100 + i), sp)
            for i in range(N_REQS)
        ]
        import torch

        torch.cuda.synchronize()
        t0 = time.perf_counter()
        while not engine.scheduler.is_finished():
            engine.step()
        torch.cuda.synchronize()
        e2e = time.perf_counter() - t0

        n = sum(len(s.token_ids) - s.num_prompt_tokens for s in seqs)
        tag = "old(全量 logits)" if args.mode == "old" else "fast(argmax 快路径)"
        print(f"[{tag}] tp={args.tp} {N_REQS} reqs x {MAX_TOKENS} tok: "
              f"{n} tokens in {e2e:.3f}s ({n / e2e:.1f} tok/s)", flush=True)
    finally:
        engine.destroy()
        executor.shutdown()


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=("fast", "old"), default="fast")
    p.add_argument("--tp", type=int, default=2)
    args = p.parse_args()

    if args.mode == "old":
        from src.compute.models.qwen3 import Qwen3ForCausalLM  # noqa: E402

        Qwen3ForCausalLM.argmax = lambda self, h: self.compute_logits(h).argmax(-1)

    from src.compute.worker import Worker  # noqa: E402
    from src.distributed import set_nccl_env  # noqa: E402

    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    config = _load_config(args)
    set_nccl_env(config.nccl_ifname)

    worker = Worker(config, rank, rank % args.tp, rank // args.tp, local_rank=local_rank)
    worker.init_environment(init_method="env://")
    worker.initialize()

    try:
        if rank == 0:
            _run_driver(config, worker, args)
        else:
            _worker_loop(worker)
    finally:
        worker.destroy_environment()


if __name__ == "__main__":
    main()
