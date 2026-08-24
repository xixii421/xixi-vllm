"""进程组与 rank 布局（仅 TP / PP；DP 不进入 NCCL world）。

并行决策（与 ``control/client.py`` / ``control/executor.py`` 对齐）::

    dp_size > 1  → MPClient 起多个独立 Engine 副本（各副本内 dp=1，各自一套 NCCL）
    tp/pp > 1    → 同一 Engine 内 MultiprocExecutor，WORLD = tp_size * pp_size

本模块只服务「单 Engine 内的 TP×PP」：

    global_rank = pp_rank * tp_size + tp_rank

DP 是请求级副本（ZMQ 路由），不是 Megatron 式 NCCL DP 组，因此这里没有 ``DP`` /
``get_dp_group``。
"""

from __future__ import annotations

import datetime
import logging
import os
from contextlib import contextmanager

import torch
import torch.distributed as dist
from torch.distributed import ProcessGroup

logger = logging.getLogger(__name__)


class Group:
    """逻辑并行组：NCCL（GPU 张量）+ Gloo（CPU / 元数据）各一份 ProcessGroup。"""

    def __init__(self, ranks: list[int], rank: int):
        self.ranks = ranks
        self.rank = rank
        self.size = len(ranks)
        self.group_rank = ranks.index(rank)
        self.device_group: ProcessGroup = dist.new_group(
            ranks=ranks, use_local_synchronization=True
        )
        self.cpu_group: ProcessGroup = dist.new_group(
            ranks=ranks, backend="gloo", use_local_synchronization=True
        )

    @property
    def is_first_rank(self) -> bool:
        return self.group_rank == 0

    @property
    def is_last_rank(self) -> bool:
        return self.group_rank == self.size - 1

    @property
    def prev_group_rank(self) -> int:
        """组内 rank，供 NCCL ``group_peer=``。线性 PP，首 stage 无前驱。"""
        if self.group_rank <= 0:
            raise RuntimeError(
                "first PP stage has no previous peer (linear pipeline, not a ring)"
            )
        return self.group_rank - 1

    @property
    def next_group_rank(self) -> int:
        """组内 rank，供 NCCL ``group_peer=``。线性 PP，末 stage 无后继。"""
        if self.group_rank >= self.size - 1:
            raise RuntimeError(
                "last PP stage has no next peer (linear pipeline, not a ring)"
            )
        return self.group_rank + 1

    @property
    def prev_rank(self) -> int:
        """全局 rank。控制面 Gloo send/recv 用；NCCL ``group_peer`` 不要用。"""
        return self.ranks[self.prev_group_rank]

    @property
    def next_rank(self) -> int:
        """全局 rank。控制面 Gloo send/recv 用；NCCL ``group_peer`` 不要用。"""
        return self.ranks[self.next_group_rank]

    def barrier(self) -> None:
        if self.size > 1:
            dist.barrier(self.cpu_group)

    def destroy(self) -> None:
        dist.destroy_process_group(self.device_group)
        dist.destroy_process_group(self.cpu_group)
        del self.device_group
        del self.cpu_group


WORLD: Group | None = None
TP: Group | None = None
PP: Group | None = None


def get_world_group() -> Group:
    assert WORLD is not None, "Distributed process group is not initialized."
    return WORLD


def get_tp_group() -> Group:
    assert TP is not None, "Tensor parallel process group is not initialized."
    return TP


def get_pp_group() -> Group:
    assert PP is not None, "Pipeline parallel process group is not initialized."
    return PP


def initialize_model_parallel(
    tp_size: int,
    pp_size: int,
    tp_rank: int,
    pp_rank: int,
) -> None:
    """划分 TP / PP 组（单 Engine 内）。

    rank 布局::

        global_rank = pp_rank * tp_size + tp_rank

    - TP 组：同一 ``pp_rank`` 下连续的 ``tp_size`` 个 global rank（机内 NVLink）。
    - PP 组：同一 ``tp_rank`` 下步进 ``tp_size`` 的 ``pp_size`` 个 global rank。

    DP 不在此建组：``dp_size>1`` 由 ``MPClient`` 起独立 Engine，每副本各自调用本函数且
    其 ``world_size = tp_size * pp_size``。
    """
    global TP, PP
    if not (0 <= tp_rank < tp_size and 0 <= pp_rank < pp_size):
        raise ValueError(
            f"invalid ranks: tp_rank={tp_rank}/{tp_size}, pp_rank={pp_rank}/{pp_size}"
        )
    my_rank = pp_rank * tp_size + tp_rank
    tp_ranks = [pp_rank * tp_size + k for k in range(tp_size)]
    pp_ranks = [k * tp_size + tp_rank for k in range(pp_size)]
    TP = Group(ranks=tp_ranks, rank=my_rank)
    PP = Group(ranks=pp_ranks, rank=my_rank)


def destroy_model_parallel() -> None:
    global TP, PP
    if TP is not None:
        TP.destroy()
    if PP is not None:
        PP.destroy()
    TP = None
    PP = None


def init_distributed_environment(
    world_size: int = -1,
    rank: int = -1,
    local_rank: int = -1,
    backend: str = "nccl",
    init_method: str = "env://",
    timeout: int = 600,
) -> None:
    global WORLD
    if dist.is_initialized():
        logger.info("Distributed process group is already initialized.")
        return
    device_id = local_rank if local_rank >= 0 else rank
    if backend == "nccl":
        torch.cuda.set_device(device_id)
    dist.init_process_group(
        backend=backend,
        init_method=init_method,
        world_size=world_size,
        rank=rank,
        timeout=datetime.timedelta(seconds=timeout),
        device_id=torch.device("cuda", device_id) if backend == "nccl" else None,
    )
    WORLD = Group(ranks=list(range(world_size)), rank=rank)


def destroy_distributed_environment() -> None:
    global WORLD
    if WORLD is not None:
        WORLD.destroy()
    WORLD = None
    if dist.is_initialized():
        dist.destroy_process_group()


def is_initialized() -> bool:
    return dist.is_initialized()


@contextmanager
def graph_capture(device: torch.device):
    """CUDA Graph 捕获窗口：进出前后对 TP/PP 做 barrier，避免图内集合通信死锁。

    PP 的 send/recv 在 piecewise CG 下位于图外，此处 barrier 主要对齐图内 TP 通信。
    """
    torch.cuda.synchronize(device)
    get_tp_group().barrier()
    get_pp_group().barrier()
    try:
        yield
    finally:
        torch.cuda.synchronize(device)
        get_tp_group().barrier()
        get_pp_group().barrier()


def set_nccl_env(ifname: str = "", *, disable_ib: bool = False) -> None:
    def _set(name: str, value: str) -> None:
        os.environ.setdefault(name, value)

    if ifname:
        _set("NCCL_SOCKET_IFNAME", ifname)
    if disable_ib:
        _set("NCCL_IB_DISABLE", "1")
    else:
        _set("NCCL_IB_GID_INDEX", "3")
        _set("NCCL_NET_GDR_LEVEL", "5")
    _set("NCCL_P2P_LEVEL", "PXB")
    _set("NCCL_DEBUG", "WARN")
