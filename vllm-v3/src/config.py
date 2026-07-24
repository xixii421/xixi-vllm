import os
from dataclasses import dataclass, field

from transformers import AutoConfig, PretrainedConfig


@dataclass
class EngineConfig:
    model: str
    context_len: int = 4096
    kvcache_block_size: int = 256
    max_num_seqs: int = 256
    max_num_batched_tokens: int = 8192
    gpu_memory_utilization: float = 0.9
    enforce_eager: bool = True  # False → Decode CUDA Graph (--compile / bench --v3-compile)
    dtype: str = "auto"
    # Prefix cache backend (v2-compatible). radix is completed beyond v2 (evict/invalidate/dedup).
    prefix_backend: str = "hash"
    # Decode-first + one prefill chunk per step (v2 fairness path). False = prefill-first.
    mix_prefill_decode: bool = True
    # 0 = auto-size from VRAM; >0 = fixed block count (bench override).
    num_kvcache_blocks: int = 0
    # FlashInfer decode: Tensor Cores (aligns with v7 / upstream vLLM when True).
    flashinfer_use_tensor_cores: bool = True
    # FlashInfer plan + CG indices H2D: async when True (hides copy behind compute).
    flashinfer_non_blocking: bool = True

    hf_config: PretrainedConfig = field(init=False)

    def __post_init__(self) -> None:
        self.model = os.path.expanduser(self.model)
        if not os.path.isdir(self.model):
            raise FileNotFoundError(
                f"Model path not found: {self.model}. "
                "Use a local directory (e.g. ~/huggingface/Qwen3-0.6B)."
            )
        if self.kvcache_block_size % 256 != 0:
            raise ValueError("kvcache_block_size must be a multiple of 256 (FlashInfer)")
        if self.prefix_backend not in ("none", "hash", "radix"):
            raise ValueError(
                f"prefix_backend must be none|hash|radix, got {self.prefix_backend!r}"
            )
        self.hf_config = AutoConfig.from_pretrained(self.model, trust_remote_code=True)
        self.context_len = min(
            self.context_len,
            getattr(self.hf_config, "max_position_embeddings", self.context_len),
        )

    @property
    def block_size(self) -> int:
        return self.kvcache_block_size
