"""Per-rank worker: NCCL init, model load, PP recv/send, execute."""

from __future__ import annotations

import torch

from ..config import EngineConfig
from ..distributed import (
    destroy_distributed_environment,
    destroy_model_parallel,
    get_pp_group,
    get_tp_group,
    init_distributed_environment,
    initialize_model_parallel,
    recv_tensor_dict,
    send_tensor_dict,
)
from ..layers.utils_parallel import IntermediateTensors
from ..memory import Sequence
from .model_runner import ModelRunner


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
        init_distributed_environment(
            word_size=self.world_size,
            rank=self.rank,
            backend="nccl",
            init_method=f"tcp://127.0.0.1:{self.nccl_port}",
            local_device=self.local_device,
        )
        initialize_model_parallel(
            self.tp_size,
            self.pp_size,
            self.tp_rank,
            self.pp_rank,
        )
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
            f"TP={self.tp_rank}/{self.tp_size} PP={self.pp_rank}/{self.pp_size}"
        )

    def destroy_environment(self) -> None:
        torch.cuda.synchronize()
        if self.model_runner is not None:
            self.model_runner.destroy()
            self.model_runner = None
        destroy_model_parallel()
        destroy_distributed_environment()

    def get_num_kvcache_blocks(self) -> int:
        assert self.model_runner is not None
        return self.model_runner.num_blocks

    def execute_model(
        self, seqs: list[Sequence], is_prefill: bool
    ) -> list[int] | None:
        assert self.model_runner is not None
        intermediate_tensors = None
        # send-allgather optimization only helps when TP>1; skip for TP=1 to avoid
        # reshape round-trips that can amplify fp16 drift across PP boundaries.
        ag = get_tp_group() if get_tp_group().size > 1 else None
        if not get_pp_group().is_first_rank:
            recv = recv_tensor_dict(group=get_pp_group(), all_gather_group=ag)
            assert recv is not None
            intermediate_tensors = IntermediateTensors(recv)

        output = self.model_runner.run(
            seqs, is_prefill, intermediate_tensors=intermediate_tensors
        )

        if not get_pp_group().is_last_rank:
            assert isinstance(output, IntermediateTensors)
            send_tensor_dict(output, group=get_pp_group(), all_gather_group=ag)
            return None

        assert isinstance(output, list)
        return output
