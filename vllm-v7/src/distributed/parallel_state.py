from __future__ import annotations

import datetime
import logging

import torch
import torch.distributed as dist
from torch.distributed import ProcessGroup

logger = logging.getLogger(__name__)


class RankGroup:
    def __init__(self, ranks: list[int], rank: int) -> None:
        if not ranks:
            raise ValueError("ranks must be non-empty")
        if rank not in ranks:
            raise ValueError(f"rank {rank} not in ranks {ranks}")
        self.ranks = list(ranks)
        self.rank = int(rank)
        self.size = len(self.ranks)

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


class Communicator(RankGroup):
    """Topology plus owned/borrowed process groups for collectives."""

    def __init__(
        self,
        ranks: list[int],
        rank: int,
        *,
        device_group: ProcessGroup | None,
        cpu_group: ProcessGroup | None = None,
        owns_device_group: bool = False,
        owns_cpu_group: bool = False,
    ) -> None:
        super().__init__(ranks, rank)
        self.device_group = device_group
        self.cpu_group = cpu_group
        self._owns_device_group = owns_device_group
        self._owns_cpu_group = owns_cpu_group

    def barrier(self) -> None:
        if self.size <= 1:
            return
        if self.cpu_group is None:
            raise RuntimeError("barrier requires a cpu_group")
        dist.barrier(self.cpu_group)

    def destroy(self) -> None:
        if self._owns_device_group and self.device_group is not None:
            dist.destroy_process_group(self.device_group)
            self.device_group = None
            self._owns_device_group = False
        if self._owns_cpu_group and self.cpu_group is not None:
            dist.destroy_process_group(self.cpu_group)
            self.cpu_group = None
            self._owns_cpu_group = False


WORLD: Communicator | None = None
TP: Communicator | None = None
PP: RankGroup | None = None
_OWNS_DEFAULT_PG: bool = False


def get_world_group() -> Communicator:
    if WORLD is None:
        raise RuntimeError("Distributed process group is not initialized.")
    return WORLD


def get_tp_group() -> Communicator:
    if TP is None:
        raise RuntimeError("Tensor parallel process group is not initialized.")
    return TP


def get_pp_group() -> RankGroup:
    if PP is None:
        raise RuntimeError("Pipeline parallel topology is not initialized.")
    return PP


def initialize_model_parallel(
    tp_size: int, pp_size: int, tp_rank: int, pp_rank: int
) -> None:
    global TP, PP
    if WORLD is None:
        raise RuntimeError("Call init_distributed_environment before initialize_model_parallel")
    if tp_size < 1 or pp_size < 1:
        raise ValueError(f"tp_size and pp_size must be >= 1, got {tp_size}, {pp_size}")
    if WORLD.size != tp_size * pp_size:
        raise ValueError(
            f"world_size ({WORLD.size}) != tp_size * pp_size ({tp_size * pp_size})"
        )
    if not (0 <= tp_rank < tp_size and 0 <= pp_rank < pp_size):
        raise ValueError(
            f"invalid tp_rank/pp_rank: {tp_rank}/{pp_rank} for sizes {tp_size}/{pp_size}"
        )
    if TP is not None or PP is not None:
        raise RuntimeError("Model parallel groups are already initialized")

    tp_global_ranks = [pp_rank * tp_size + k for k in range(tp_size)]
    pp_global_ranks = [tp_rank + k * tp_size for k in range(pp_size)]
    my_rank = WORLD.rank

    if tp_global_ranks == WORLD.ranks:
        # TP spans the whole world: reuse WORLD's NCCL group, no extra Gloo.
        TP = Communicator(
            ranks=tp_global_ranks,
            rank=my_rank,
            device_group=WORLD.device_group,
            cpu_group=None,
            owns_device_group=False,
            owns_cpu_group=False,
        )
    elif tp_size == 1:
        TP = Communicator(
            ranks=tp_global_ranks,
            rank=tp_global_ranks[0],
            device_group=None,
            cpu_group=None,
            owns_device_group=False,
            owns_cpu_group=False,
        )
    else:
        device_group = dist.new_group(
            ranks=tp_global_ranks, use_local_synchronization=True
        )
        TP = Communicator(
            ranks=tp_global_ranks,
            rank=tp_global_ranks[tp_rank],
            device_group=device_group,
            cpu_group=None,
            owns_device_group=True,
            owns_cpu_group=False,
        )

    # PP is topology only — tensor P2P uses WORLD.
    PP = RankGroup(ranks=pp_global_ranks, rank=pp_global_ranks[pp_rank])


def destroy_model_parallel() -> None:
    global TP, PP
    if TP is not None:
        TP.destroy()
    TP = None
    PP = None


def init_distributed_environment(
    world_size: int = -1,
    rank: int = -1,
    backend: str = "nccl",
    init_method: str = "env://",
    local_device: int | None = None,
) -> None:
    global WORLD, _OWNS_DEFAULT_PG
    if WORLD is not None:
        logger.info("Distributed environment already initialized.")
        return

    if world_size < 1:
        raise ValueError(f"world_size must be >= 1, got {world_size}")
    if not (0 <= rank < world_size):
        raise ValueError(f"rank must be in [0, {world_size}), got {rank}")

    if backend == "nccl":
        if local_device is None:
            raise ValueError("local_device is required when backend='nccl'")
        torch.cuda.set_device(int(local_device))

    if dist.is_initialized():
        _OWNS_DEFAULT_PG = False
        logger.info("Using externally initialized torch.distributed process group.")
    else:
        dist.init_process_group(
            backend=backend,
            init_method=init_method,
            world_size=world_size,
            rank=rank,
            timeout=datetime.timedelta(seconds=60),
            device_id=(
                torch.device("cuda", int(local_device)) if backend == "nccl" else None
            ),
        )
        _OWNS_DEFAULT_PG = True

    ranks = list(range(world_size))
    if world_size == 1:
        # Singleton: no extra Gloo communicator.
        WORLD = Communicator(
            ranks=ranks,
            rank=rank,
            device_group=dist.group.WORLD,
            cpu_group=None,
            owns_device_group=False,
            owns_cpu_group=False,
        )
        return

    cpu_group = dist.new_group(
        ranks=ranks, backend="gloo", use_local_synchronization=True
    )
    WORLD = Communicator(
        ranks=ranks,
        rank=rank,
        device_group=dist.group.WORLD,
        cpu_group=cpu_group,
        owns_device_group=False,
        owns_cpu_group=True,
    )


def destroy_distributed_environment() -> None:
    global WORLD, _OWNS_DEFAULT_PG
    destroy_model_parallel()
    if WORLD is not None:
        WORLD.destroy()
    WORLD = None
    if _OWNS_DEFAULT_PG and dist.is_initialized():
        dist.destroy_process_group()
    _OWNS_DEFAULT_PG = False


def is_initialized() -> bool:
    return WORLD is not None
