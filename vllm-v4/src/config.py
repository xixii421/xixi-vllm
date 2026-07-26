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
    # None → text defaults to eager; VL defaults to Decode CUDA Graph on.
    enforce_eager: bool | None = None
    # Prefill torch.compile (inductor). None → on when Decode CG is on
    # (``enforce_eager=False``). Decode still uses CUDA Graph; compile only
    # accelerates the eager prefill forward (TTFT).
    compile_prefill: bool | None = None
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
    # Vision encoder CUDA Graph (budget replay). Default on for VL when not eager.
    enable_vit_cuda_graph: bool | None = None
    vit_cg_token_budgets: tuple[int, ...] = (64, 128, 256, 512, 1024)
    vit_cg_max_batch_size: int = 1
    # Max merged vision tokens to encode in one step (0 = same as max_num_batched_tokens).
    encoder_compute_budget: int = 0

    hf_config: PretrainedConfig = field(init=False)
    is_multimodal: bool = field(init=False, default=False)

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
        self.is_multimodal = _is_vl_config(self.hf_config)
        if self.enforce_eager is None:
            # VL: decode CG on by default; text stays eager unless --compile.
            self.enforce_eager = not self.is_multimodal
        if self.compile_prefill is None:
            self.compile_prefill = not bool(self.enforce_eager)
        if self.enable_vit_cuda_graph is None:
            self.enable_vit_cuda_graph = self.is_multimodal and not self.enforce_eager
        if self.encoder_compute_budget <= 0:
            self.encoder_compute_budget = self.max_num_batched_tokens
        # For VL, lengths / heads live on text_config.
        max_pos = getattr(self.text_config, "max_position_embeddings", self.context_len)
        self.context_len = min(self.context_len, max_pos)

    @property
    def block_size(self) -> int:
        return self.kvcache_block_size

    @property
    def text_config(self) -> PretrainedConfig:
        return getattr(self.hf_config, "text_config", self.hf_config)
