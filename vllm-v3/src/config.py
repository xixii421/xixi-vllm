import math
import os
from dataclasses import dataclass, field

from transformers import AutoConfig, PretrainedConfig


@dataclass
class EngineConfig:
    model: str
    context_len: int = 2048
    block_size: int = 16
    max_num_seqs: int = 8
    max_num_batched_tokens: int = 4096
    gpu_memory_utilization: float = 0.8
    enforce_eager: bool = False
    dtype: str = "auto"
    num_gpu_blocks: int | None = None

    hf_config: PretrainedConfig = field(init=False)

    def __post_init__(self):
        self.model = os.path.expanduser(self.model)
        self.hf_config = AutoConfig.from_pretrained(
            self.model, trust_remote_code=True
        )
        if self.block_size <= 0:
            raise ValueError("block_size must be positive")
        if self.max_num_seqs <= 0:
            raise ValueError("max_num_seqs must be positive")
        if self.max_num_batched_tokens <= 0:
            raise ValueError("max_num_batched_tokens must be positive")
        if self.num_gpu_blocks is None:
            pages_per_seq = math.ceil(self.context_len / self.block_size)
            self.num_gpu_blocks = pages_per_seq * self.max_num_seqs
        if self.num_gpu_blocks <= 0:
            raise ValueError("num_gpu_blocks must be positive")
