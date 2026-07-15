from __future__ import annotations

import os
from dataclasses import dataclass, field
from transformers import AutoConfig, PretrainedConfig

from .pd import PDConfig, PDRole


def _is_vl_config(cfg: PretrainedConfig) -> bool:
    arch = getattr(cfg, "architectures", None) or []
    name = type(cfg).__name__
    return (
        any(("VL" in a or "Vision" in a for a in arch))
        or "VL" in name
        or hasattr(cfg, "vision_config")
    )


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
    vit_cg_min_budget: int = 64
    vit_cg_max_budget: int = 256
    tp_size: int = 1
    pp_size: int = 1
    nccl_port: int = 0
    device_ids: list[int] | None = None
    mm_encoder_tp_mode: str = "data"
    # v7: ZMQ control-plane RPC (ullm-style). "pipe" keeps v6 multiprocessing.Pipe.
    rpc_backend: str = "zmq"
    zmq_ipc_prefix: str = "/tmp/vllm_v7"
    # Multi-node NCCL: tcp://MASTER_ADDR:MASTER_PORT (not 127.0.0.1).
    nccl_host: str = "127.0.0.1"
    nnodes: int = 1
    node_rank: int = 0
    # Prefix cache: hash table (v6) or Radix tree (ullm-inspired).
    enable_radix: bool = True
    # Async scheduling: placeholder-token overlap (ullm); disabled when pp_size>1.
    enable_async_scheduling: bool = False
    # PD disaggregation role (stub: only COLLOCATED runs today).
    pd_mode: str = "collocated"
    pd_kv_backend: str = "none"
    pd_remote_host: str | None = None
    pd_remote_port: int = 29600
    hf_config: PretrainedConfig = field(init=False)
    is_multimodal: bool = field(init=False, default=False)
    pd: PDConfig = field(init=False)

    def __post_init__(self) -> None:
        self.model = os.path.expanduser(self.model)
        if not os.path.isdir(self.model):
            raise FileNotFoundError(
                f"Model path not found: {self.model}. Use a local directory "
                f"(e.g. ~/huggingface/Qwen3-VL-2B-Instruct)."
            )
        if self.tp_size < 1 or self.pp_size < 1:
            raise ValueError("tp_size and pp_size must be >= 1")
        if self.rpc_backend not in ("zmq", "pipe"):
            raise ValueError(f"rpc_backend must be 'zmq' or 'pipe', got {self.rpc_backend!r}")
        if self.device_ids is not None and len(self.device_ids) != self.world_size:
            raise ValueError(
                f"device_ids length ({len(self.device_ids)}) must equal "
                f"tp_size * pp_size ({self.world_size})"
            )
        if self.kvcache_block_size % 256 != 0:
            raise ValueError("kvcache_block_size must be a multiple of 256 (FlashInfer)")
        if self.nccl_port <= 0 and self.world_size > 1:
            self.nccl_port = _free_tcp_port()
        elif self.nccl_port <= 0:
            self.nccl_port = 29500
        if self.enable_async_scheduling and self.pp_size > 1:
            raise ValueError("enable_async_scheduling is incompatible with pp_size>1")
        role_map = {
            "collocated": PDRole.COLLOCATED,
            "prefill": PDRole.PREFILL,
            "decode": PDRole.DECODE,
        }
        if self.pd_mode not in role_map:
            raise ValueError(f"pd_mode must be one of {list(role_map)}, got {self.pd_mode!r}")
        self.pd = PDConfig(
            role=role_map[self.pd_mode],
            kv_transfer_backend=self.pd_kv_backend,
            remote_host=self.pd_remote_host,
            remote_port=self.pd_remote_port,
        )
        if self.pd.role != PDRole.COLLOCATED and self.pd.kv_transfer_backend == "none":
            raise ValueError(
                "pd_mode=prefill|decode requires a KV transfer backend "
                "(set pd_kv_backend); see src/pd/transfer.py"
            )
        self.hf_config = AutoConfig.from_pretrained(self.model, trust_remote_code=True)
        self.is_multimodal = _is_vl_config(self.hf_config)
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
        if self.pp_size > int(text_cfg.num_hidden_layers):
            raise ValueError(
                f"pp_size ({self.pp_size}) > num_hidden_layers ({text_cfg.num_hidden_layers})"
            )
        if self.mm_encoder_tp_mode not in ("data", "weights"):
            raise ValueError(
                f"mm_encoder_tp_mode must be 'data' or 'weights', got {self.mm_encoder_tp_mode!r}"
            )
        if self.is_multimodal and self.mm_encoder_tp_mode == "weights" and self.tp_size > 1:
            vc = getattr(self.hf_config, "vision_config", None)
            if vc is not None:
                v_heads = int(getattr(vc, "num_heads", 1))
                v_inter = int(getattr(vc, "intermediate_size", 1))
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
        if self.is_multimodal and self.pp_size > 1:
            from .distributed.pp import get_pp_indices

            vc = getattr(self.hf_config, "vision_config", None)
            n_ds = len(getattr(vc, "deepstack_visual_indexes", None) or []) if vc else 0
            if n_ds > 0:
                _s, end0 = get_pp_indices(
                    int(text_cfg.num_hidden_layers), pp_rank=0, pp_size=self.pp_size
                )
                if end0 < n_ds:
                    raise ValueError(
                        f"pp_size={self.pp_size} gives first-stage layers [0,{end0}), "
                        f"but DeepStack needs layers 0..{n_ds - 1}; reduce pp_size"
                    )
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
