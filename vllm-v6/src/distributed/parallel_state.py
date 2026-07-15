"""Megatron-style WORLD / TP / PP process groups."""

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
        # NOTE: all ranks must execute new_group when creating a new process group,
        # even if they are not members of the group.
        self.device_group: ProcessGroup = dist.new_group(
            ranks=ranks, use_local_synchronization=True
        )  # type: ignore
        self.cpu_group: ProcessGroup = dist.new_group(
            ranks=ranks, backend="gloo", use_local_synchronization=True
        )  # type: ignore

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
        group_rank = self.group_rank
        return self.ranks[(group_rank - 1) % self.size]

    @property
    def next_rank(self) -> int:
        group_rank = self.group_rank
        return self.ranks[(group_rank + 1) % self.size]

    def barrier(self):
        if self.size <= 1:
            return
        dist.barrier(self.cpu_group)

    def destroy(self):
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
    """Get the tensor parallel process group."""
    assert TP is not None, "Tensor parallel process group is not initialized."
    return TP


def get_pp_group() -> Group:
    """Get the pipeline parallel process group."""
    assert PP is not None, "Pipeline parallel process group is not initialized."
    return PP


def initialize_model_parallel(tp_size: int, pp_size: int, tp_rank: int, pp_rank: int):
    global TP, PP

    tp_global_ranks = [pp_rank * tp_size + k for k in range(tp_size)]
    pp_global_ranks = [tp_rank + k * tp_size for k in range(pp_size)]

    TP = Group(ranks=tp_global_ranks, rank=tp_global_ranks[tp_rank])
    PP = Group(ranks=pp_global_ranks, rank=pp_global_ranks[pp_rank])


def destroy_model_parallel():
    global TP, PP
    if TP is not None:
        TP.destroy()
    if PP is not None:
        PP.destroy()
    TP = None
    PP = None


def init_distributed_environment(
    word_size: int = -1,
    rank: int = -1,
    backend: str = "nccl",
    init_method: str = "env://",
    local_device: int | None = None,
):
    """Initialize the process group.

    ``local_device`` is the physical CUDA device index for this rank
    (e.g. ``device_ids[rank]``). When ``None``, defaults to ``rank`` (ullm behaviour).
    """
    global WORLD
    if dist.is_initialized():
        logger.info("Distributed process group is already initialized.")
        return

    device = rank if local_device is None else int(local_device)

    if backend == "nccl":
        torch.cuda.set_device(device)

    dist.init_process_group(
        backend=backend,
        init_method=init_method,
        world_size=word_size,
        rank=rank,
        timeout=datetime.timedelta(seconds=60),
        device_id=torch.device("cuda", device) if backend == "nccl" else None,
    )
    WORLD = Group(ranks=list(range(word_size)), rank=rank)


def destroy_distributed_environment():
    global WORLD
    if WORLD is not None:
        WORLD.destroy()
    WORLD = None

    if dist.is_initialized():
        dist.destroy_process_group()


def is_initialized() -> bool:
    return dist.is_initialized()
