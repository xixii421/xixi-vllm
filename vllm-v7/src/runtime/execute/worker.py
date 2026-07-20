from __future__ import annotations

import torch

from ...config import EngineConfig
from ...distributed import (
    destroy_distributed_environment,
    destroy_model_parallel,
    get_pp_group,
    get_tp_group,
    init_distributed_environment,
    initialize_model_parallel,
    recv_intermediate_tensors,
    send_intermediate_tensors,
)
from ...distributed.pp import IntermediateTensors
from ...scheduling import Sequence
from ..runner import ModelRunner


class Worker:
    def __init__(
        self,
        config: EngineConfig,
        rank: int,
        tp_rank: int,
        pp_rank: int,
        local_device: int,
    ):
        self.config = config
        self.rank = rank
        self.tp_rank = tp_rank
        self.pp_rank = pp_rank
        self.local_device = local_device
        self.tp_size = config.tp_size
        self.pp_size = config.pp_size
        self.world_size = config.world_size
        self.nccl_port = config.nccl_port
        self.model_runner: ModelRunner | None = None

    def init_environment(self) -> None:
        init_method = self.config.nccl_init_method
        init_distributed_environment(
            world_size=self.world_size,
            rank=self.rank,
            backend="nccl",
            init_method=init_method,
            local_device=self.local_device,
        )
        initialize_model_parallel(self.tp_size, self.pp_size, self.tp_rank, self.pp_rank)
        self.device = torch.device(f"cuda:{self.local_device}")
        self.model_runner = ModelRunner(
            self.config,
            self.device,
            rank=self.rank,
            tp_rank=self.tp_rank,
            pp_rank=self.pp_rank,
        )
        print(
            f"[Worker {self.rank}] ready device={self.device} "
            f"TP={self.tp_rank}/{self.tp_size} PP={self.pp_rank}/{self.pp_size} "
            f"rpc={self.config.rpc_backend} nccl={init_method}"
        )

    def destroy_environment(self) -> None:
        torch.cuda.synchronize()
        if self.model_runner is not None:
            self.model_runner.destroy()
            self.model_runner = None
        from ...distributed import clear_pp_buffer_cache

        clear_pp_buffer_cache()
        destroy_model_parallel()
        destroy_distributed_environment()

    def get_num_kvcache_blocks(self) -> int:
        assert self.model_runner is not None
        return self.model_runner.num_blocks

    def register_mm_pixels(self, pixels: dict[str, torch.Tensor]) -> None:
        """Store image pixels by uid for the next encode (prefill wire has no tensors)."""
        if not pixels or self.model_runner is None:
            return
        model = self.model_runner.model
        enc = getattr(model, "mm_encoder", None)
        if enc is not None:
            enc.pixel_store.put_many(pixels)

    def execute_model(
        self, seqs: list[Sequence], is_prefill: bool
    ) -> tuple[list[int] | None, list | None]:
        assert self.model_runner is not None
        from ...utils.profiler import Timer

        intermediate_tensors = None
        ag = get_tp_group() if get_tp_group().size > 1 else None
        pp = get_pp_group()
        if not pp.is_first_rank:
            with Timer("worker.pp_recv"):
                intermediate_tensors = recv_intermediate_tensors(
                    src=pp.prev_rank, all_gather_group=ag
                )
        with Timer("worker.forward"):
            output, mrope_updates = self.model_runner.run(
                seqs, is_prefill, intermediate_tensors=intermediate_tensors
            )
        if not pp.is_last_rank:
            assert isinstance(output, IntermediateTensors)
            with Timer("worker.pp_send"):
                send_intermediate_tensors(
                    output, dst=pp.next_rank, all_gather_group=ag
                )
            return None, mrope_updates
        assert isinstance(output, list)
        return output, mrope_updates

    def reset_sys_profile(self) -> None:
        from ...utils.profiler import reset

        reset()

    def get_sys_profile(self) -> dict:
        from ...utils.profiler import summary

        return summary()
