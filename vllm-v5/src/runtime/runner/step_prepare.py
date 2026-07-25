"""Prepare tensors and sampling flags for one ModelRunner step."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from ...config import EngineConfig
from ...distributed import get_pp_group
from ...layers.attention_backend import FlashInferBackend, compute_paged_kv_metadata
from ...scheduling import Sequence
from .model_inputs import ModelInputs, build_model_inputs
from .mrope_sync import MRopeStateUpdate, snapshot_mrope_updates


@dataclass
class PreparedStep:
    inputs: ModelInputs
    positions: torch.Tensor
    inputs_embeds: torch.Tensor | None
    deepstack: object | None
    mrope_updates: list[MRopeStateUpdate] | None
    temperatures: torch.Tensor
    top_ps: torch.Tensor
    top_ks: torch.Tensor
    min_ps: torch.Tensor
    all_greedy: bool
    any_greedy: bool
    top_k_enabled: bool
    top_p_enabled: bool
    min_p_enabled: bool
    max_top_k: int
    attn_meta: object | None  # filled for eager path; None when CG will run


def _seq_lens(inputs: ModelInputs, is_prefill: bool, seqs: list[Sequence]) -> list[int]:
    if is_prefill:
        return [s.num_cached_tokens + s.num_scheduled_tokens for s in seqs]
    return [len(s) for s in seqs]


def _cu_q(seqs: list[Sequence]) -> list[int]:
    cu = [0]
    for s in seqs:
        cu.append(cu[-1] + s.num_scheduled_tokens)
    return cu


def build_attn_meta(
    *,
    prepared: PreparedStep,
    seqs: list[Sequence],
    is_prefill: bool,
    block_size: int,
    device: torch.device,
    backend: FlashInferBackend,
    kv_cache: torch.Tensor,
) -> object:
    """Build FlashInfer metadata for an already-prepared step (CG miss path)."""
    from ...utils.profiler import Timer

    with Timer("runner.attn_meta"):
        seq_lens = _seq_lens(prepared.inputs, is_prefill, seqs)
        bt = (
            prepared.inputs.block_tables
            if prepared.inputs.block_tables is not None
            else torch.zeros(len(seqs), 1, dtype=torch.int32, device=device)
        )
        indptr_cpu, idx_gpu, last_cpu = compute_paged_kv_metadata(
            seq_lens, bt, block_size, device
        )
        cu_cpu = (
            torch.tensor(_cu_q(seqs), dtype=torch.int32, device="cpu")
            if is_prefill
            else None
        )
        return backend.build_metadata(
            indptr_cpu,
            idx_gpu,
            last_cpu,
            prepared.inputs.slot_mapping,
            is_prefill,
            kv_cache,
            cu_cpu,
        )


def prepare_step(
    *,
    model,
    config: EngineConfig,
    seqs: list[Sequence],
    is_prefill: bool,
    block_size: int,
    device: torch.device,
    backend: FlashInferBackend,
    kv_cache: torch.Tensor,
    use_cuda_graph: bool,
) -> PreparedStep:
    from ...utils.profiler import Timer

    with Timer("runner.build_inputs"):
        inputs = build_model_inputs(seqs, is_prefill, block_size, device)

    inputs_embeds = None
    deepstack = None
    positions = inputs.positions
    mrope_updates: list[MRopeStateUpdate] | None = None
    mm_prefill = (
        is_prefill
        and config.is_multimodal
        and any(s.has_multimodal for s in seqs)
    )
    if mm_prefill:
        with Timer("runner.mm_encode"):
            if get_pp_group().is_first_rank:
                inputs_embeds, deepstack, positions = model.encode_and_merge(
                    inputs.input_ids, seqs, device
                )
                inputs.inputs_embeds = inputs_embeds
                inputs.deepstack_input_embeds = deepstack
                inputs.positions = positions
            else:
                from ..vision import mrope_positions

                vc = getattr(config.hf_config, "vision_config", None)
                merge_size = int(getattr(vc, "spatial_merge_size", 2)) if vc else 2
                positions = mrope_positions(
                    inputs.input_ids, seqs, merge_size, device
                )
                inputs.positions = positions
        mrope_updates = snapshot_mrope_updates(seqs)

    temperature_values = [float(s.temperature) for s in seqs]
    top_p_values = [1.0 if s.top_p is None else float(s.top_p) for s in seqs]
    top_k_values = [0 if s.top_k is None else int(s.top_k) for s in seqs]
    min_p_values = [0.0 if s.min_p is None else float(s.min_p) for s in seqs]
    all_greedy = all(t <= 1e-5 for t in temperature_values)
    any_greedy = any(t <= 1e-5 for t in temperature_values)
    top_k_enabled = any(k > 0 for k in top_k_values)
    top_p_enabled = any(p < 1.0 - 1e-12 for p in top_p_values)
    min_p_enabled = any(p > 0.0 for p in min_p_values)
    max_top_k = max((k for k in top_k_values if k > 0), default=1)
    temperatures = torch.tensor(
        temperature_values, dtype=torch.float32, pin_memory=True
    ).to(device, non_blocking=True)
    top_ps = torch.tensor(top_p_values, dtype=torch.float32, pin_memory=True).to(
        device, non_blocking=True
    )
    top_ks = torch.tensor(top_k_values, dtype=torch.int32, pin_memory=True).to(
        device, non_blocking=True
    )
    min_ps = torch.tensor(min_p_values, dtype=torch.float32, pin_memory=True).to(
        device, non_blocking=True
    )

    prepared = PreparedStep(
        inputs=inputs,
        positions=positions,
        inputs_embeds=inputs_embeds,
        deepstack=deepstack,
        mrope_updates=mrope_updates,
        temperatures=temperatures,
        top_ps=top_ps,
        top_ks=top_ks,
        min_ps=min_ps,
        all_greedy=all_greedy,
        any_greedy=any_greedy,
        top_k_enabled=top_k_enabled,
        top_p_enabled=top_p_enabled,
        min_p_enabled=min_p_enabled,
        max_top_k=max_top_k,
        attn_meta=None,
    )
    if not use_cuda_graph:
        prepared.attn_meta = build_attn_meta(
            prepared=prepared,
            seqs=seqs,
            is_prefill=is_prefill,
            block_size=block_size,
            device=device,
            backend=backend,
            kv_cache=kv_cache,
        )
    return prepared
