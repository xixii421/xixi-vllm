"""PP 中间张量点对点通信的 CPU 正确性验证（gloo，tp=2 pp=2，world=4）。

用法：python test/test_pp_comm.py

验证生产路径 send_tensors / recv_tensors：相邻 PP stage 按约定顺序
收发 hidden_states 与 residual，接收端写入预分配 buffer。
"""

from __future__ import annotations

import os
import sys
import traceback
from queue import Empty

import torch
import torch.multiprocessing as mp

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.distributed import (  # noqa: E402
    destroy_distributed_environment,
    destroy_model_parallel,
    get_pp_group,
    get_tp_group,
    init_distributed_environment,
    initialize_model_parallel,
    recv_tensors,
    send_tensors,
    tensor_model_parallel_all_reduce,
)

TP, PP = 2, 2
WORLD = TP * PP
INIT_METHOD = "tcp://127.0.0.1:29511"


def _worker(rank: int, q: mp.Queue) -> None:
    torch.set_default_dtype(torch.float64)
    torch.set_grad_enabled(False)
    init_distributed_environment(
        world_size=WORLD, rank=rank, backend="gloo", init_method=INIT_METHOD,
        timeout=30,
    )
    initialize_model_parallel(TP, PP, tp_rank=rank % TP, pp_rank=rank // TP)
    try:
        tp_sum = tensor_model_parallel_all_reduce(torch.tensor(float(rank + 1)))
        assert tp_sum.item() == sum(r + 1 for r in get_tp_group().ranks)
        pp = get_pp_group()
        g = torch.Generator().manual_seed(1234)
        h = torch.randn(2, 4, dtype=torch.float64, generator=g)
        r = torch.randn(2, 4, dtype=torch.float64, generator=g)

        if pp.is_first_rank:
            send_tensors((h, r), (pp.group_rank + 1) % pp.size, pp)
            q.put(True)
        elif pp.is_last_rank:
            rh, rr = torch.empty_like(h), torch.empty_like(r)
            recv_tensors((rh, rr), (pp.group_rank - 1) % pp.size, pp)
            ok = torch.allclose(rh, h, atol=1e-10)
            ok &= torch.allclose(rr, r, atol=1e-10)
            print(f"  [rank {rank} recv] {'PASS' if ok else 'FAIL'}")
            q.put(bool(ok))
        else:
            q.put(True)
    except Exception:  # noqa: BLE001
        traceback.print_exc()
        q.put(False)
    finally:
        destroy_model_parallel()
        destroy_distributed_environment()


def main() -> int:
    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    procs = [ctx.Process(target=_worker, args=(r, q)) for r in range(WORLD)]
    try:
        for p in procs:
            p.start()
        results = [q.get(timeout=60) for _ in range(WORLD)]
        for p in procs:
            p.join(timeout=10)
        ok = all(results) and all(p.exitcode == 0 for p in procs)
        print("\n" + ("PP send/recv ALL PASS" if ok else "PP send/recv FAIL"))
        return 0 if ok else 1
    except Empty:
        print("PP send/recv FAIL: worker timeout", flush=True)
        return 1
    finally:
        for p in procs:
            if p.is_alive():
                p.terminate()
        for p in procs:
            if p.pid is not None:
                p.join(timeout=5)
        q.close()


if __name__ == "__main__":
    sys.exit(main())
