"""进程组与模型并行 rank 布局。

每个逻辑组（WORLD/TP/PP）对应两个物理 ProcessGroup：
    - device_group：NCCL，传输 GPU 张量
    - cpu_group：Gloo，传输元数据与 CPU 张量

rank 约定：global rank = pp_rank * tp_size + tp_rank。
TP 在组内连续（适配机内 NVLink），PP 跨组步进（适配跨机）。
"""

from __future__ import annotations

import datetime
import logging

import torch
import torch.distributed as dist
from torch.distributed import ProcessGroup

logger = logging.getLogger(__name__)


class Group:
    def __init__(self, ranks: list[int], rank: int):
        self.ranks = ranks
        self.rank = rank
        self.size = len(ranks)
        # 所有 rank 都必须调用 new_group，即使不在组内。
        # use_local_synchronization=True 允许仅组内 rank 参与 new_group 的同步。
        self.device_group: ProcessGroup = dist.new_group(
            ranks=ranks, use_local_synchronization=True
        )
        self.cpu_group: ProcessGroup = dist.new_group(
            ranks=ranks, backend="gloo", use_local_synchronization=True
        )

    @property
    def group_rank(self) -> int:
        return self.ranks.index(self.rank)

    @property
    def first_rank(self) -> int:
        return self.ranks[0]

    @property
    def last_rank(self) -> int:
        return self.ranks[-1]

    @property
    def is_first_rank(self) -> bool:
        return self.rank == self.first_rank

    @property
    def is_last_rank(self) -> bool:
        return self.rank == self.last_rank

    @property
    def prev_rank(self) -> int:
        return self.ranks[(self.group_rank - 1) % self.size]

    @property
    def next_rank(self) -> int:
        return self.ranks[(self.group_rank + 1) % self.size]

    def barrier(self) -> None:
        if self.size <= 1:
            return
        dist.barrier(self.cpu_group)

    def destroy(self) -> None:
        dist.destroy_process_group(self.device_group)
        del self.device_group
        dist.destroy_process_group(self.cpu_group)
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
    tp_size: int, pp_size: int, tp_rank: int, pp_rank: int
) -> None:
    global TP, PP

    tp_global_ranks = [pp_rank * tp_size + k for k in range(tp_size)]
    pp_global_ranks = [tp_rank + k * tp_size for k in range(pp_size)]

    TP = Group(ranks=tp_global_ranks, rank=tp_global_ranks[tp_rank])
    PP = Group(ranks=pp_global_ranks, rank=pp_global_ranks[pp_rank])


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

    # rank 是全局 rank（NCCL rendezvous 用）；local_rank 是机内 GPU 下标（设备用）。
    # 单机时两者相同；多机时 local_rank = rank % nproc_per_node。
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


def set_nccl_env(ifname: str = "", *, disable_ib: bool = False) -> None:
    """多机 NCCL 网络配置：仅在变量未设置时写入默认，避免覆盖用户显式配置。

    - ``NCCL_SOCKET_IFNAME``：指定机间通信网卡（TP 锁机内不依赖它，PP 跨机才需要）。
    - ``NCCL_IB_*``：无 InfiniBand 时禁用，走 TCP；有 IB/RDMA 时保持默认启用。
    - ``NCCL_P2P_LEVEL``：机内 TP 用 NVLink/NVLink 之外 P2P，设为 PXB 避免跨 PCIe 交换机。
    """
    os = __import__("os")

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
