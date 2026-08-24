from __future__ import annotations

import os
from dataclasses import dataclass, field

from transformers import AutoConfig, PretrainedConfig


@dataclass
class EngineConfig:
    """引擎配置：调度 / 内存 / 算子 / 并行。

    并行语义（三者正交，分叉点不同）::

        dp_size  → 独立 Engine 副本数（``MPClient`` / ZMQ）；不进入 NCCL world
        tp_size  → 同 Engine 内张量并行度（NCCL all-reduce / gather）
        pp_size  → 同 Engine 内流水线 stage 数（NCCL P2P hidden+residual）

    单 Engine 的 NCCL ``world_size = tp_size * pp_size``。总 GPU 数约为
    ``dp_size * tp_size * pp_size``（由 ``MPClient`` 给每副本切 ``device_ids``）。
    """

    #  模型
    model: str  # 本地模型路径

    #  调度器
    context_len: int = 4096
    max_num_seqs: int = 256
    max_num_batched_tokens: int = 8192
    mix_prefill_decode: bool = True  # True=decode 优先混合；False=prefill 优先

    #  内存
    kvcache_block_size: int = 256  # 必须是 256 的倍数（FA2 paged 约束）
    gpu_memory_utilization: float = 0.9
    num_kvcache_blocks: int = 0  # 0=自动按显存；>0=固定块数（基准测试）
    prefix_backend: str = "hash"  # none / hash / radix

    #  算子
    enforce_eager: bool = True  # False=启用 decode piecewise CUDA Graph
    torch_compile: bool = False  # True=prefill 首 chunk 走 torch.compile
    compile_mode: str = "reduce-overhead"
    compile_dynamic: bool = True
    dtype: str = "auto"

    #  并行
    tp_size: int = 1  # 张量并行：同 Engine、同 schedule、NCCL 切权重
    pp_size: int = 1  # 流水线并行：同 Engine、sync lockstep 切层
    dp_size: int = 1  # 数据并行：独立 Engine 副本数（ZMQ）；每副本内等价 dp=1
    nccl_port: int = 29500
    master_addr: str = "127.0.0.1"
    master_port: int = 29500
    device_ids: list[int] | None = None  # 长度应为 tp*pp（单副本）或 dp*tp*pp（门面）
    nccl_ifname: str = ""
    pp_layer_counts: list[int] | None = None  # 不均分 PP；None=等分

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
            raise ValueError("kvcache_block_size must be a multiple of 256 (FA2 paged)")
        if self.prefix_backend not in ("none", "hash", "radix"):
            raise ValueError(
                f"prefix_backend must be none|hash|radix, got {self.prefix_backend!r}"
            )
        if self.tp_size < 1 or self.pp_size < 1 or self.dp_size < 1:
            raise ValueError("tp_size / pp_size / dp_size must be >= 1")
        self.hf_config = AutoConfig.from_pretrained(self.model, trust_remote_code=True)
        self.context_len = min(
            self.context_len,
            getattr(self.hf_config, "max_position_embeddings", self.context_len),
        )

    @property
    def world_size(self) -> int:
        """单 Engine 的 NCCL world（不含 DP 副本）。"""
        return self.tp_size * self.pp_size

    @property
    def num_gpus(self) -> int:
        """门面视角下的总 GPU 数：``dp_size * tp_size * pp_size``。"""
        return self.dp_size * self.tp_size * self.pp_size

    @property
    def block_size(self) -> int:
        return self.kvcache_block_size
