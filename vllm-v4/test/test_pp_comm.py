"""PP 中间张量点对点通信的 CPU 正确性验证（gloo，tp=2 pp=2，world=4）。

用法：python test/test_pp_comm.py

验证 send_tensor_dict / recv_tensor_dict 的 send-allgather 优化：
first PP stage 的每个 tp rank 只发自己切片，last stage 经 tp all-gather 还原完整张量。
"""

from __future__ import annotations

import os
import sys
import traceback

import torch
import torch.multiprocessing as mp

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.distributed import (  # noqa: E402
    IntermediateTensors,
    destroy_distributed_environment,
    destroy_model_parallel,
    get_pp_group,
    get_tp_group,
    init_distributed_environment,
    initialize_model_parallel,
    recv_tensor_dict,
    send_tensor_dict,
)

TP, PP = 2, 2
WORLD = TP * PP
INIT_METHOD = "tcp://127.0.0.1:29511"


def _worker(rank: int, q: mp.Queue) -> None:
    torch.set_default_dtype(torch.float64)
    torch.set_grad_enabled(False)
    init_distributed_environment(
        world_size=WORLD, rank=rank, backend="gloo", init_method=INIT_METHOD
    )
    initialize_model_parallel(TP, PP, tp_rank=rank % TP, pp_rank=rank // TP)
    try:
        pp = get_pp_group()
        # 所有 rank 用同一种子生成相同张量，last stage 据此校验接收结果。
        g = torch.Generator().manual_seed(1234)
        h = torch.randn(2, 4, dtype=torch.float64, generator=g)
        r = torch.randn(2, 4, dtype=torch.float64, generator=g)

        if pp.is_first_rank:
            send_tensor_dict(
                IntermediateTensors(hidden_states=h, residual=r),
                group=pp,
                all_gather_group=get_tp_group(),
            )
            q.put(True)
        elif pp.is_last_rank:
            recv = recv_tensor_dict(group=pp, all_gather_group=get_tp_group())
            assert recv is not None
            ok = torch.allclose(recv["hidden_states"], h, atol=1e-10)
            ok &= torch.allclose(recv["residual"], r, atol=1e-10)
            print(f"  [rank {rank} recv] {'PASS' if ok else 'FAIL'}")
            q.put(bool(ok))
        else:  # pp 中段（pp>2 时）透传；此处 pp=2 不触发
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
    for p in procs:
        p.start()
    results = [q.get() for _ in range(WORLD)]
    for p in procs:
        p.join()
    ok = all(results)
    print("\n" + ("PP send/recv ALL PASS" if ok else "PP send/recv FAIL"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
