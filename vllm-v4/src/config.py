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
    kvcache_block_size: int = 256  # 必须是256的倍数（FA2 paged 约束）
    gpu_memory_utilization: float = 0.9
    num_kvcache_blocks: int = 0  # 0=自动按显存计算，>0=固定块数(基准测试用)
    prefix_backend: str = "hash"  # 前缀缓存后端: none / hash / radix

    #  算子 
    enforce_eager: bool = True  # False=启用 Decode CUDA Graph
    torch_compile: bool = False  # True=prefill 首 chunk 走 torch.compile（全图编译，无 piecewise/静态 buffer copy）
    compile_mode: str = "reduce-overhead"  # torch.compile mode: default / reduce-overhead / max-autotune
    compile_dynamic: bool = True  # True=符号形状（一次编译覆盖任意长度，避免按长度重编译）；False=固定形状重放（仅离线固定长度，新长度会重编译 ~2-12s）
    dtype: str = "auto"

    #  并行 
    tp_size: int = 1  # 张量并行度（机内 NVLink）
    pp_size: int = 1  # 流水线并行度（跨机低带宽）
    nccl_port: int = 29500  # NCCL 单机 rendezvous 端口
    master_addr: str = "127.0.0.1"  # 多机 NCCL rendezvous 主节点地址
    master_port: int = 29500  # 多机 NCCL rendezvous 端口
    device_ids: list[int] | None = None  # 每进程可见 GPU 列表（与 tp_size*pp_size 对齐）
    nccl_ifname: str = ""  # 多机网卡名（如 ib0/eth0）；空=交由 NCCL 自动探测
    pp_layer_counts: list[int] | None = None  # 不均分 PP 每 stage 层数（如 [12,28]）；None=等分

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
        self.hf_config = AutoConfig.from_pretrained(self.model, trust_remote_code=True)
        self.context_len = min(
            self.context_len,
            getattr(self.hf_config, "max_position_embeddings", self.context_len),
        )

    @property
    def block_size(self) -> int:
        return self.kvcache_block_size
