"""引擎配置：模型路径、上下文、精度、KV / 前缀缓存。"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Literal

from transformers import AutoConfig, PretrainedConfig


@dataclass
class EngineConfig:
    """从本地模型目录构造推理所需配置。"""

    model: str
    context_len: int = 2048
    enforce_eager: bool = True
    dtype: str = "auto"
    # KV / prefix (v1 新增相对 v0)
    prefix_backend: Literal["none", "hash", "radix"] = "hash"
    block_size: int = 16
    num_kv_blocks: int = 0  # 0 → 按 context_len 推导
    hf_config: PretrainedConfig = field(init=False)

    def __post_init__(self) -> None:
        self.model = os.path.expanduser(self.model)
        self.hf_config = AutoConfig.from_pretrained(self.model, trust_remote_code=True)
        if self.prefix_backend not in ("none", "hash", "radix"):
            raise ValueError(
                f"prefix_backend must be none|hash|radix, got {self.prefix_backend!r}"
            )
        if self.block_size <= 0:
            raise ValueError("block_size must be > 0")
        if self.num_kv_blocks <= 0:
            # 双倍于单序列满窗，留出跨请求缓存空间
            need = (self.context_len + self.block_size - 1) // self.block_size
            self.num_kv_blocks = max(need * 2, need + 8)
