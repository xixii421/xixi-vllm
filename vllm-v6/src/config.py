import os
from dataclasses import dataclass, field

from transformers import AutoConfig, PretrainedConfig


def _is_vl_config(cfg: PretrainedConfig) -> bool:
    arch = getattr(cfg, "architectures", None) or []
    name = type(cfg).__name__
    return (
        any("VL" in a or "Vision" in a for a in arch)
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
    # Multimodal: capture budget-based ViT CUDA Graphs (orthogonal to decode CG).
    enable_vit_cuda_graph: bool = False
    vit_cg_min_budget: int = 64
    vit_cg_max_budget: int = 256
    # Distributed (Megatron-style TP + PP)
    tp_size: int = 1
    pp_size: int = 1
    # 0 = auto-pick a free TCP port (avoids EADDRINUSE after prior runs).
    nccl_port: int = 0
    device_ids: list[int] | None = None

    hf_config: PretrainedConfig = field(init=False)
    is_multimodal: bool = field(init=False, default=False)

    def __post_init__(self) -> None:
        self.model = os.path.expanduser(self.model)
        if not os.path.isdir(self.model):
            raise FileNotFoundError(
                f"Model path not found: {self.model}. "
                "Use a local directory (e.g. ~/huggingface/Qwen3-VL-2B-Instruct)."
            )
        if self.tp_size < 1 or self.pp_size < 1:
            raise ValueError("tp_size and pp_size must be >= 1")
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

        self.hf_config = AutoConfig.from_pretrained(self.model, trust_remote_code=True)
        self.is_multimodal = _is_vl_config(self.hf_config)

        if self.world_size > 1 and self.is_multimodal:
            raise ValueError(
                "Multimodal (Qwen3-VL) is only supported for world_size==1 "
                f"(got tp_size={self.tp_size}, pp_size={self.pp_size})."
            )

        # PP mid-stage send/recv is incompatible with decode CUDA Graph.
        if self.pp_size > 1:
            self.enforce_eager = True

        # For VL, lengths / heads live on text_config.
        text_cfg = getattr(self.hf_config, "text_config", self.hf_config)
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

    def resolved_device_ids(self) -> list[int]:
        """CUDA device index for each global rank."""
        if self.device_ids is not None:
            return list(self.device_ids)
        return list(range(self.world_size))
