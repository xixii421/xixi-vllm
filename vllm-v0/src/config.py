"""引擎配置：模型路径、上下文长度、精度与 eager / compile。"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

from transformers import AutoConfig, PretrainedConfig


@dataclass
class EngineConfig:
    """从本地模型目录构造推理所需配置。

    Attributes:
        model: 含 config.json 与 *.safetensors 的目录。
        context_len: 生成时允许的最大序列长度（含 prompt）。
        enforce_eager: True 时不用 torch.compile，便于调试。
        dtype: "auto" 跟 checkpoint；否则如 "float16" / "bfloat16"。
    """

    model: str
    context_len: int = 2048
    enforce_eager: bool = True
    dtype: str = "auto"
    hf_config: PretrainedConfig = field(init=False)

    def __post_init__(self) -> None:
        self.model = os.path.expanduser(self.model)
        # 启动时拉取 HF 结构超参；真正的权重在 Engine 里再 load
        self.hf_config = AutoConfig.from_pretrained(self.model, trust_remote_code=True)
