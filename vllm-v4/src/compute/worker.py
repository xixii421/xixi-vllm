"""单个 rank 的 worker：负责进程组初始化、模型/KV 装配与一步前向。

每个 worker 进程对应一张卡，rank = pp_rank * tp_size + tp_rank。
PP 的中间张量（hidden_states + residual）通过 NCCL 点对点在相邻 stage 间传递。
"""

from __future__ import annotations

import torch

from ..config import EngineConfig
from ..data import KVManager, SchedulerOutput
from ..distributed import (
    destroy_distributed_environment,
    destroy_model_parallel,
    get_pp_group,
    init_distributed_environment,
    initialize_model_parallel,
    set_pp_layer_ranges,
)
from .model_runner import ModelRunner
from .models.loader import default_dtype_context, load_model, skip_param_init
from .models.qwen3 import Qwen3ForCausalLM
from .sampler import Sampler


class Worker:
    def __init__(
        self,
        config: EngineConfig,
        rank: int,
        tp_rank: int,
        pp_rank: int,
        local_rank: int | None = None,
    ):
        self.config = config
        self.rank = rank
        self.tp_rank = tp_rank
        self.pp_rank = pp_rank
        self.tp_size = config.tp_size
        self.pp_size = config.pp_size
        self.world_size = self.tp_size * self.pp_size
        # local_rank 是机内 GPU 下标：单机 spawn 时等于 rank；多机 torchrun 时为 LOCAL_RANK。
        self.local_rank = rank if local_rank is None else local_rank
        self.model_runner: ModelRunner | None = None

    def init_environment(self, init_method: str | None = None) -> None:
        init_method = init_method or (
            f"tcp://{self.config.master_addr}:{self.config.nccl_port}"
        )
        device_id = self._device_id()
        init_distributed_environment(
            world_size=self.world_size,
            rank=self.rank,
            local_rank=device_id,
            backend="nccl",
            init_method=init_method,
        )
        initialize_model_parallel(self.tp_size, self.pp_size, self.tp_rank, self.pp_rank)
        # 不均分 PP：按 config.pp_layer_counts 覆盖等分层区间（如 12/28 适配 12G+24G 异卡）。
        if self.config.pp_layer_counts is not None:
            counts = self.config.pp_layer_counts
            assert len(counts) == self.pp_size, "pp_layer_counts 长度须等于 pp_size"
            assert sum(counts) == self.config.hf_config.num_hidden_layers, (
                "pp_layer_counts 之和须等于模型层数"
            )
            set_pp_layer_ranges(counts)
        self.device = torch.device(f"cuda:{device_id}")
        self.dtype = self._resolve_dtype()

    def _device_id(self) -> int:
        # device_ids 显式映射（如单卡调试 [0,0]）优先，否则用机内 local_rank。
        if self.config.device_ids is not None:
            return self.config.device_ids[self.rank]
        return self.local_rank

    def initialize(self) -> None:
        model = self._load_model()
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        self.sampler = Sampler()
        self.kv = self._make_kv()
        self.model_runner = ModelRunner(
            model, self.kv, self.sampler, config=self.config, device=self.device
        )

    def _resolve_dtype(self) -> torch.dtype:
        if self.config.dtype != "auto":
            return getattr(torch, self.config.dtype)
        dt = getattr(self.config.hf_config, "torch_dtype", None)
        return dt if isinstance(dt, torch.dtype) else torch.float16

    def _load_model(self) -> torch.nn.Module:
        with default_dtype_context(self.dtype):
            with skip_param_init():
                model = Qwen3ForCausalLM(self.config.hf_config)
        load_model(model, self.config.model)
        return model.to(dtype=self.dtype, device=self.device).eval()

    def _make_kv(self) -> KVManager:
        if self.config.num_kvcache_blocks > 0:
            num_blocks = self.config.num_kvcache_blocks
        else:
            num_blocks = KVManager.auto_num_blocks(
                self.config.hf_config,
                block_size=self.config.block_size,
                dtype=self.dtype,
                device=self.device,
                gpu_memory_utilization=self.config.gpu_memory_utilization,
            )
            self.config.num_kvcache_blocks = num_blocks
        return KVManager(
            self.config.hf_config,
            num_blocks=num_blocks,
            block_size=self.config.block_size,
            device=self.device,
            dtype=self.dtype,
            prefix_backend=self.config.prefix_backend,
        )

    def get_kv_cache_size(self) -> int:
        assert self.kv is not None, "initialize() must be called first"
        return self.kv.num_blocks_total

    def execute_model(self, sched: SchedulerOutput):
        """执行一步。入参为最小增量 SchedulerOutput；非 last PP 返回 None，last PP 返回采样 id。"""
        assert self.model_runner is not None
        result = self.model_runner.run_pipeline(sched)
        if not get_pp_group().is_last_rank:
            return None
        n_p, n_d = result
        return self.model_runner.read_samples(n_p, n_d)

    def destroy_environment(self) -> None:
        if torch.cuda.is_available():
            try:
                torch.cuda.synchronize()
            except RuntimeError:
                pass
        if self.model_runner is not None:
            self.model_runner.destroy()
        self.model_runner = None
        self.kv = None
        set_pp_layer_ranges(None)
        destroy_model_parallel()
        destroy_distributed_environment()
