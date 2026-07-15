import os
from dataclasses import dataclass, field
from transformers import AutoConfig, PretrainedConfig


@dataclass
class EngineConfig:
    model: str
    context_len: int = 2048
    gpu_memory_utilization: float = 0.8
    enforce_eager: bool = False
    dtype: str = "auto"

    hf_config: PretrainedConfig = field(init=False)

    def __post_init__(self):
        self.model = os.path.expanduser(self.model)
        self.hf_config = AutoConfig.from_pretrained(
            self.model, trust_remote_code=True
        )