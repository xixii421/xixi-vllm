from __future__ import annotations

import os
from dataclasses import dataclass, field

from transformers import AutoConfig, PretrainedConfig


@dataclass
class EngineConfig:
    """引擎配置：调度器 / 内存 / 算子 / 采样参数。"""

    #  模型 
    model: str  # 本地模型路径

    #  调度器 
    context_len: int = 4096
    max_num_seqs: int = 256
    max_num_batched_tokens: int = 8192
    mix_prefill_decode: bool = True  # True=decode优先混合调度，False=全prefill优先

    #  内存 
    kvcache_block_size: int = 256  # 必须是256的倍数（FlashInfer约束）
    gpu_memory_utilization: float = 0.9
    num_kvcache_blocks: int = 0  # 0=自动按显存计算，>0=固定块数(基准测试用)
    prefix_backend: str = "hash"  # 前缀缓存后端: none / hash / radix

    #  算子 
    enforce_eager: bool = True  # False=启用Decode CUDA Graph
    dtype: str = "auto"
    flashinfer_use_tensor_cores: bool = True  # FlashInfer启用Tensor Core
    flashinfer_non_blocking: bool = True  # FlashInfer plan异步H2D

    #  运行时 
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
