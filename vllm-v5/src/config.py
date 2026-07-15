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

    hf_config: PretrainedConfig = field(init=False)
    is_multimodal: bool = field(init=False, default=False)

    def __post_init__(self) -> None:
        self.model = os.path.expanduser(self.model)
        if not os.path.isdir(self.model):
            raise FileNotFoundError(
                f"Model path not found: {self.model}. "
                "Use a local directory (e.g. ~/huggingface/Qwen3-VL-2B-Instruct)."
            )
        if self.kvcache_block_size % 256 != 0:
            raise ValueError("kvcache_block_size must be a multiple of 256 (FlashInfer)")
        self.hf_config = AutoConfig.from_pretrained(self.model, trust_remote_code=True)
        self.is_multimodal = _is_vl_config(self.hf_config)

        # For VL, lengths / heads live on text_config.
        text_cfg = getattr(self.hf_config, "text_config", self.hf_config)
        max_pos = getattr(text_cfg, "max_position_embeddings", self.context_len)
        self.context_len = min(self.context_len, max_pos)

    @property
    def block_size(self) -> int:
        return self.kvcache_block_size

    @property
    def text_config(self) -> PretrainedConfig:
        return getattr(self.hf_config, "text_config", self.hf_config)
