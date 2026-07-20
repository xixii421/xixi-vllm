"""Model construction, weight load, and paged-KV pool allocation."""

from __future__ import annotations

import torch
from torch import nn

from ...config import EngineConfig
from ...distributed import (
    get_pp_group,
    get_pp_indices,
    get_tp_group,
    init_distributed_environment,
    initialize_model_parallel,
    is_initialized,
)
from ...models.loader import load_model
from ...models.qwen3 import Qwen3ForCausalLM


def ensure_single_process_parallel(config: EngineConfig, local_device: int) -> None:
    if is_initialized():
        return
    init_distributed_environment(
        world_size=1,
        rank=0,
        backend="nccl",
        init_method=f"tcp://127.0.0.1:{config.nccl_port}",
        local_device=local_device,
    )
    initialize_model_parallel(1, 1, 0, 0)


def resolve_dtype(config: EngineConfig) -> torch.dtype:
    if config.dtype != "auto":
        return getattr(torch, config.dtype)
    cfg = config.text_config
    dt = (
        getattr(cfg, "dtype", None)
        or getattr(cfg, "torch_dtype", None)
        or getattr(config.hf_config, "dtype", None)
        or getattr(config.hf_config, "torch_dtype", None)
    )
    return dt if isinstance(dt, torch.dtype) else torch.float16


def build_and_load_model(
    config: EngineConfig,
    *,
    device: torch.device,
    dtype: torch.dtype,
    rank: int,
    tp_rank: int,
    tp_size: int,
    pp_size: int,
) -> nn.Module:
    """Construct the CausalLM (dense / MoE / VL) and load weights."""
    from ...models.weight_utils import default_dtype_context, move_module

    with default_dtype_context(dtype):
        if config.is_multimodal:
            from ...models.qwen3_vl import Qwen3VLForCausalLM

            model = Qwen3VLForCausalLM(
                config.hf_config, mm_encoder_tp_mode=config.mm_encoder_tp_mode
            )
            model.load_weights(
                config.model,
                device,
                dtype,
                rank=rank,
                tp_size=tp_size,
                pp_size=pp_size,
            )
            model = model.eval()
            if get_pp_group().is_first_rank and model.vision_tower is not None:
                from ..vision import VisionEncoderRunner

                model.mm_encoder = VisionEncoderRunner(
                    model.vision_tower,
                    mm_encoder_tp_mode=config.mm_encoder_tp_mode,
                    num_deepstack=model._ndps,
                )
            return model

        if config.is_moe:
            from ...config.moe_backend import require_moe_backend
            from ...layers.moe import MoEParallelConfig, set_moe_parallel_config
            from ...models.qwen3_moe import Qwen3MoeForCausalLM

            backend = require_moe_backend(
                config.moe_fused_backend,
                device=device.index if device.index is not None else 0,
            )
            set_moe_parallel_config(
                MoEParallelConfig.make(
                    tp_size=tp_size,
                    tp_rank=tp_rank,
                    fused_backend=backend,
                )
            )
            model = Qwen3MoeForCausalLM(config.hf_config)
        else:
            model = Qwen3ForCausalLM(config.hf_config)

        load_model(
            model,
            config.model,
            rank=rank,
            tp_size=tp_size,
            pp_size=pp_size,
            prefer_sharded=True,
            device=device,
            dtype=dtype,
        )
        return move_module(model, device=device, dtype=dtype)


def layer_shard_meta(config: EngineConfig) -> tuple[int, int, int, int, int, int]:
    """Return (start, end, num_layers, num_qo_heads, num_kv_heads, head_dim)."""
    cfg = config.text_config
    tp_size = get_tp_group().size
    start, end = get_pp_indices(
        cfg.num_hidden_layers, get_pp_group().group_rank, get_pp_group().size
    )
    num_layers = end - start
    num_qo_heads = cfg.num_attention_heads // tp_size
    num_kv_heads = max(1, cfg.num_key_value_heads // tp_size)
    head_dim = getattr(cfg, "head_dim", cfg.hidden_size // cfg.num_attention_heads)
    return start, end, num_layers, num_qo_heads, num_kv_heads, head_dim


def alloc_kv_cache(
    *,
    config: EngineConfig,
    device: torch.device,
    dtype: torch.dtype,
    num_layers: int,
    num_kv_heads: int,
    head_dim: int,
    start_layer: int,
    end_layer: int,
    rank: int,
    decode_cg_planned: bool,
) -> torch.Tensor:
    torch.cuda.empty_cache()
    torch.cuda.synchronize(device)
    block_size = config.kvcache_block_size
    block_bytes = 2 * num_layers * block_size * num_kv_heads * head_dim * dtype.itemsize
    if config.num_kvcache_blocks > 0:
        num_blocks = config.num_kvcache_blocks
    else:
        free, total = torch.cuda.mem_get_info(device)
        available = int(total * config.gpu_memory_utilization) - (total - free)
        if decode_cg_planned:
            available -= int(2.0 * 1024**3)
        num_blocks = max(4, available // block_bytes)
        if rank == 0 or config.world_size == 1:
            config.num_kvcache_blocks = num_blocks
    if decode_cg_planned and num_blocks < 2:
        raise RuntimeError(
            f"CUDA Graph requires >= 2 KV blocks after allocation, got {num_blocks}"
        )
    kv = torch.empty(
        2,
        num_layers,
        num_blocks,
        block_size,
        num_kv_heads,
        head_dim,
        dtype=dtype,
        device=device,
    )
    if rank == 0 or config.world_size == 1:
        print(
            f"KV pool: {num_blocks} blocks × {block_size} tok "
            f"layers=[{start_layer}:{end_layer}] kv_heads={num_kv_heads} "
            f"({num_blocks * block_bytes / 1024**3:.2f} GiB)"
        )
    return kv
