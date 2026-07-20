from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Literal

from transformers import AutoConfig, PretrainedConfig


def _is_vl_config(cfg: PretrainedConfig) -> bool:
    arch = getattr(cfg, "architectures", None) or []
    name = type(cfg).__name__
    return (
        any(("VL" in a or "Vision" in a for a in arch))
        or "VL" in name
        or hasattr(cfg, "vision_config")
    )


def _is_moe_config(cfg: PretrainedConfig) -> bool:
    arch = getattr(cfg, "architectures", None) or []
    name = type(cfg).__name__
    if any("Moe" in a or "MoE" in a for a in arch) or "Moe" in name or "MoE" in name:
        return True
    return int(getattr(cfg, "num_experts", 0) or 0) > 0


def _free_tcp_port() -> int:
    import socket

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


@dataclass
class EngineConfig:
    model: str
    context_len: int = 4096
    kvcache_block_size: int = 256
    max_num_seqs: int = 256
    max_num_batched_tokens: int = 8192
    gpu_memory_utilization: float = 0.9
    enforce_eager: bool = False
    dtype: str = "auto"
    num_kvcache_blocks: int = 0
    enable_vit_cuda_graph: bool = False
    moe_cuda_graph: bool = True
    # auto → cutlass if可用 else triton
    moe_fused_backend: Literal["auto", "cutlass", "triton"] = "auto"
    vit_cg_min_budget: int = 64
    vit_cg_max_budget: int = 256
    tp_size: int = 1
    pp_size: int = 1
    nccl_port: int = 0
    device_ids: list[int] | None = None
    mm_encoder_tp_mode: str = "data"
    rpc_backend: str = "zmq"
    zmq_ipc_prefix: str = "/tmp/vllm_v7"
    nccl_host: str = "127.0.0.1"
    nnodes: int = 1
    node_rank: int = 0
    enable_radix: bool = True
    allow_hetero_tp: bool = False
    hf_config: PretrainedConfig = field(init=False)
    is_multimodal: bool = field(init=False, default=False)
    is_moe: bool = field(init=False, default=False)

    def __post_init__(self) -> None:
        self.model = os.path.expanduser(self.model)
        if not os.path.isdir(self.model):
            raise FileNotFoundError(
                f"Model path not found: {self.model}. Use a local directory "
                f"(e.g. /data/huggingface-local/Qwen3-VL-8B-Instruct)."
            )
        if self.tp_size < 1 or self.pp_size < 1:
            raise ValueError("tp_size and pp_size must be >= 1")
        if self.rpc_backend not in ("zmq", "pipe"):
            raise ValueError(f"rpc_backend must be 'zmq' or 'pipe', got {self.rpc_backend!r}")
        if self.moe_fused_backend not in ("auto", "cutlass", "triton"):
            raise ValueError(
                f"moe_fused_backend must be auto|cutlass|triton, got {self.moe_fused_backend!r}"
            )
        if self.device_ids is not None and len(self.device_ids) != self.world_size:
            raise ValueError(
                f"device_ids length ({len(self.device_ids)}) must equal "
                f"tp_size * pp_size ({self.world_size})"
            )
        self._validate_tp_homogeneous_devices()
        if self.kvcache_block_size % 256 != 0:
            raise ValueError("kvcache_block_size must be a multiple of 256 (FlashInfer)")
        # LocalExecutor (tp=pp=1) still opens a TCPStore; never hardcode 29500
        # or concurrent benches collide with EADDRINUSE.
        if self.nccl_port <= 0:
            self.nccl_port = _free_tcp_port()
        self.hf_config = AutoConfig.from_pretrained(self.model, trust_remote_code=True)
        self.is_multimodal = _is_vl_config(self.hf_config)
        self.is_moe = _is_moe_config(self.hf_config)
        if self.is_multimodal and self.is_moe:
            raise ValueError("Qwen3 MoE + VL is not supported in this teaching build")
        text_cfg = getattr(self.hf_config, "text_config", self.hf_config)
        n_heads = int(text_cfg.num_attention_heads)
        n_kv = int(text_cfg.num_key_value_heads)
        inter = int(text_cfg.intermediate_size)
        if n_heads % self.tp_size != 0:
            raise ValueError(
                f"num_attention_heads ({n_heads}) must be divisible by tp_size ({self.tp_size})"
            )
        if n_kv >= self.tp_size:
            if n_kv % self.tp_size != 0:
                raise ValueError(
                    f"num_key_value_heads ({n_kv}) must be divisible by tp_size ({self.tp_size})"
                )
        elif self.tp_size % n_kv != 0:
            raise ValueError(
                f"tp_size ({self.tp_size}) must be divisible by num_key_value_heads ({n_kv})"
            )
        if inter % self.tp_size != 0:
            raise ValueError(
                f"intermediate_size ({inter}) must be divisible by tp_size ({self.tp_size})"
            )
        moe_inter = getattr(text_cfg, "moe_intermediate_size", None)
        if self.is_moe:
            if moe_inter is None:
                raise ValueError("MoE config missing moe_intermediate_size")
            moe_inter = int(moe_inter)
            n_experts = int(getattr(text_cfg, "num_experts", 0) or 0)
            if n_experts <= 0:
                raise ValueError("MoE config has num_experts <= 0")
            if moe_inter % self.tp_size != 0:
                raise ValueError(
                    f"moe_intermediate_size ({moe_inter}) must be divisible by "
                    f"tp_size ({self.tp_size})"
                )
            if self.tp_size > n_experts:
                raise ValueError(
                    f"tp_size ({self.tp_size}) > num_experts ({n_experts})"
                )
            shared_inter = int(getattr(text_cfg, "shared_expert_intermediate_size", 0) or 0)
            if shared_inter and shared_inter % self.tp_size != 0:
                raise ValueError(
                    f"shared_expert_intermediate_size ({shared_inter}) must be divisible by "
                    f"tp_size ({self.tp_size})"
                )
        if self.pp_size > int(text_cfg.num_hidden_layers):
            raise ValueError(
                f"pp_size ({self.pp_size}) > num_hidden_layers ({text_cfg.num_hidden_layers})"
            )
        if self.mm_encoder_tp_mode not in ("data", "weights"):
            raise ValueError(
                f"mm_encoder_tp_mode must be 'data' or 'weights', got {self.mm_encoder_tp_mode!r}"
            )
        # 'weights' = true ViT Weight-TP (Column/Row/QKV parallel); 'data' = DP images.
        if self.is_multimodal and self.mm_encoder_tp_mode == "weights" and self.tp_size > 1:
            vc = getattr(self.hf_config, "vision_config", None)
            if vc is not None:
                v_heads = int(getattr(vc, "num_heads", 1))
                v_inter = int(getattr(vc, "intermediate_size", 1))
                merge = int(getattr(vc, "spatial_merge_size", 2))
                merge_h = int(getattr(vc, "hidden_size", 1)) * merge * merge
                if v_heads % self.tp_size != 0:
                    raise ValueError(
                        f"vision num_heads ({v_heads}) must be divisible by tp_size "
                        f"({self.tp_size}) for mm_encoder_tp_mode='weights'"
                    )
                if v_inter % self.tp_size != 0:
                    raise ValueError(
                        f"vision intermediate_size ({v_inter}) must be divisible by tp_size "
                        f"({self.tp_size}) for mm_encoder_tp_mode='weights'"
                    )
                if merge_h % self.tp_size != 0:
                    raise ValueError(
                        f"vision merger hidden ({merge_h}) must be divisible by tp_size "
                        f"({self.tp_size}) for mm_encoder_tp_mode='weights'"
                    )
        # DeepStack × PP is validated in Engine (needs PP split math without
        # pulling distributed into this config layer).
        max_pos = getattr(text_cfg, "max_position_embeddings", self.context_len)
        self.context_len = min(self.context_len, max_pos)

    @property
    def world_size(self) -> int:
        return self.tp_size * self.pp_size

    @property
    def block_size(self) -> int:
        return self.kvcache_block_size

    @property
    def text_config(self) -> PretrainedConfig:
        return getattr(self.hf_config, "text_config", self.hf_config)

    @property
    def nccl_init_method(self) -> str:
        return f"tcp://{self.nccl_host}:{self.nccl_port}"

    def resolved_device_ids(self) -> list[int]:
        if self.device_ids is not None:
            return list(self.device_ids)
        return list(range(self.world_size))

    def _validate_tp_homogeneous_devices(self) -> None:
        if self.tp_size <= 1 or self.allow_hetero_tp:
            return
        import torch

        if not torch.cuda.is_available():
            return
        ids = self.resolved_device_ids()
        for pp_rank in range(self.pp_size):
            stage = ids[pp_rank * self.tp_size : (pp_rank + 1) * self.tp_size]
            props = []
            for d in stage:
                if d < 0 or d >= torch.cuda.device_count():
                    raise ValueError(
                        f"device_ids contains invalid GPU index {d} "
                        f"(visible count={torch.cuda.device_count()})"
                    )
                p = torch.cuda.get_device_properties(d)
                props.append((p.name, int(p.total_memory)))
            names = {n for n, _ in props}
            mems = [m for _, m in props]
            mem_ratio = (max(mems) / min(mems)) if min(mems) > 0 else 1.0
            if len(names) > 1 or mem_ratio > 1.15:
                raise ValueError(
                    f"Heterogeneous TP within PP stage {pp_rank}: devices={stage} "
                    f"props={props}. Use same-SKU GPUs per TP group, map different "
                    f"SKUs across PP stages, or set allow_hetero_tp=True."
                )
