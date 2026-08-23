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

    @classmethod
    def new(cls, ranks: list[int], rank: int) -> Group | None:
        """全体 rank 必须以相同 ``ranks`` 序列调用；非成员返回 None。"""
        device_group = dist.new_group(
            ranks=ranks, use_local_synchronization=True
        )
        cpu_group = dist.new_group(
            ranks=ranks, backend="gloo", use_local_synchronization=True
        )
        if rank not in ranks:
            return None
        g = cls.__new__(cls)
        g.ranks = ranks
        g.rank = rank
        g.size = len(ranks)
        g.group_rank = ranks.index(rank)
        g.device_group = device_group
        g.cpu_group = cpu_group
        return g

    @property
    def is_first_rank(self) -> bool:
        return self.group_rank == 0

    @property
    def is_last_rank(self) -> bool:
        return self.group_rank == self.size - 1

    @property
    def prev_group_rank(self) -> int:
        """组内 rank，供 ``group_peer=`` 使用。线性邻居，首 stage 越界即报错。"""
        if self.group_rank <= 0:
            raise RuntimeError(
                "first PP stage has no previous peer (linear pipeline, not a ring)"
            )
        return self.group_rank - 1

    @property
    def next_group_rank(self) -> int:
        """组内 rank，供 ``group_peer=`` 使用。线性邻居，末 stage 越界即报错。"""
        if self.group_rank >= self.size - 1:
            raise RuntimeError(
                "last PP stage has no next peer (linear pipeline, not a ring)"
            )
        return self.group_rank + 1

    @property
    def prev_rank(self) -> int:
        """全局 rank。控制面 Gloo send/recv 用这个；NCCL ``group_peer`` 不要用。"""
        return self.ranks[self.prev_group_rank]

    @property
    def next_rank(self) -> int:
        """全局 rank。控制面 Gloo send/recv 用这个；NCCL ``group_peer`` 不要用。"""
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
DP: Group | None = None  # 数据并行副本组；吞吐走 DP 复制，而不是把同步 PP 做成 1F1B。


def get_world_group() -> Group:
    assert WORLD is not None, "Distributed process group is not initialized."
    return WORLD


def get_tp_group() -> Group:
    assert TP is not None, "Tensor parallel process group is not initialized."
    return TP


def get_pp_group() -> Group:
    assert PP is not None, "Pipeline parallel process group is not initialized."
    return PP


def get_dp_group() -> Group:
    assert DP is not None, "Data parallel process group is not initialized."
    return DP


def initialize_model_parallel(
    tp_size: int,
    pp_size: int,
    tp_rank: int,
    pp_rank: int,
    dp_size: int = 1,
    dp_rank: int = 0,
) -> None:
    """划分 TP / PP / DP 组。

    rank 布局（与 Megatron 一致，DP 为最外层）::

        global = dp_rank * (pp_size * tp_size) + pp_rank * tp_size + tp_rank

    ``dp_size=1`` 时退化为原来的 ``global = pp_rank * tp_size + tp_rank``。
    DP 组只做副本对齐（同一 tp/pp 坐标跨 replica）；本引擎不实现 1F1B PP。
    要吞吐请起 ``dp_size`` 个独立 Engine 副本，而不是加深同步流水线。
    """
    global TP, PP, DP
    replica = pp_size * tp_size
    tp_global_ranks = [dp_rank * replica + pp_rank * tp_size + k for k in range(tp_size)]
    pp_global_ranks = [dp_rank * replica + k * tp_size + tp_rank for k in range(pp_size)]
    TP = Group(ranks=tp_global_ranks, rank=tp_global_ranks[tp_rank])
    PP = Group(ranks=pp_global_ranks, rank=pp_global_ranks[pp_rank])
    # new_group 必须全体 rank 用同一组划分调用。dp=1 时不再建多余组。
    DP = None
    if dp_size > 1:
        my_rank = dp_rank * replica + pp_rank * tp_size + tp_rank
        for p in range(pp_size):
            for t in range(tp_size):
                ranks = [k * replica + p * tp_size + t for k in range(dp_size)]
                group = Group.new(ranks, my_rank)
                if group is not None:
                    DP = group


def destroy_model_parallel() -> None:
    global TP, PP, DP
    if TP is not None:
        TP.destroy()
    if PP is not None:
        PP.destroy()
    if DP is not None:
        DP.destroy()
    TP = None
    PP = None
    DP = None


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
