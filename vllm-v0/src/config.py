from __future__ import annotations
import os
from dataclasses import dataclass, field
from transformers import AutoConfig, PretrainedConfig


@dataclass
class EngineConfig:
    model: str
    context_len: int = 2048
    enforce_eager: bool = True
    dtype: str = "auto"
    hf_config: PretrainedConfig = field(init=False)

    def __post_init__(self) -> None:
        self.model = os.path.expanduser(self.model)
        # 启动时拉取 HF 结构超参；真正的权重在 Engine 里再 load
        self.hf_config = AutoConfig.from_pretrained(self.model, trust_remote_code=True)
