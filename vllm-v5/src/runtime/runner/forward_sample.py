"""Eager / CUDA-Graph forward + sampling for one prepared step."""

from __future__ import annotations

import torch

from ...distributed import get_pp_group
from ...distributed.pp import IntermediateTensors
from ...layers.attention import attention_context
from ..context import forward_context
from .model_inputs import ModelInputs
from .sampler import Sampler
from .step_prepare import PreparedStep


def hidden_for_sample(hidden: torch.Tensor, inputs: ModelInputs) -> torch.Tensor:
    if inputs.is_prefill and inputs.cu_seqlens_q is not None:
        last_idx = inputs.cu_seqlens_q[1:] - 1
        return hidden[last_idx]
    return hidden


def sample_from_hidden(
    *,
    model,
    sampler: Sampler,
    prepared: PreparedStep,
    hidden: torch.Tensor,
    sample_host: torch.Tensor,
    sample_done: torch.cuda.Event,
    device: torch.device,
) -> list[int]:
    from ...utils.profiler import Timer

    def _host_tokens(tokens: torch.Tensor) -> list[int]:
        host = sample_host[: tokens.numel()]
        host.copy_(tokens, non_blocking=True)
        sample_done.record(torch.cuda.current_stream(device))
        sample_done.synchronize()
        return host.tolist()

    def _sample(logits: torch.Tensor) -> list[int]:
        with Timer("runner.sample"):
            tokens = sampler(
                logits,
                prepared.temperatures,
                top_ps=prepared.top_ps,
                top_ks=prepared.top_ks,
                min_ps=prepared.min_ps,
                all_greedy=prepared.all_greedy,
                any_greedy=prepared.any_greedy,
                top_k_enabled=prepared.top_k_enabled,
                max_top_k=prepared.max_top_k,
                top_p_enabled=prepared.top_p_enabled,
                min_p_enabled=prepared.min_p_enabled,
            )
            return _host_tokens(tokens)

    hidden = hidden_for_sample(hidden, prepared.inputs)
    lm_head = getattr(model, "lm_head", None)
    if (
        prepared.all_greedy
        and lm_head is not None
        and hasattr(lm_head, "greedy_tokens")
    ):
        with Timer("runner.logits"):
            tokens = lm_head.greedy_tokens(hidden)
        with Timer("runner.sample"):
            return _host_tokens(tokens)
    with Timer("runner.logits"):
        logits = model.compute_logits(hidden)
    return _sample(logits)


def forward_eager(
    *,
    model,
    prepared: PreparedStep,
    seqs,
    is_prefill: bool,
    intermediate_tensors: IntermediateTensors | None,
) -> torch.Tensor | IntermediateTensors:
    from ...utils.profiler import Timer

    inputs = prepared.inputs
    with Timer("runner.forward"):
        with forward_context(
            is_prefill=is_prefill,
            seqs=seqs,
            slot_mapping=inputs.slot_mapping,
            cu_seqlens_q=inputs.cu_seqlens_q,
            cu_seqlens_k=inputs.cu_seqlens_k,
            max_seqlen_q=inputs.max_seqlen_q,
            max_seqlen_k=inputs.max_seqlen_k,
            context_lens=inputs.context_lens,
            block_tables=inputs.block_tables,
        ):
            with attention_context(prepared.attn_meta):
                h = model(
                    input_ids=(
                        None
                        if prepared.inputs_embeds is not None
                        or not get_pp_group().is_first_rank
                        else inputs.input_ids
                    ),
                    positions=prepared.positions,
                    inputs_embeds=(
                        prepared.inputs_embeds
                        if get_pp_group().is_first_rank
                        else None
                    ),
                    deepstack_input_embeds=(
                        prepared.deepstack if get_pp_group().is_first_rank else None
                    ),
                    intermediate_tensors=intermediate_tensors,
                )
        from ...utils.profiler import enabled as _prof_on

        if _prof_on():
            torch.cuda.synchronize()
    return h


def forward_cuda_graph(
    *,
    cuda_graph_runner,
    prepared: PreparedStep,
    seqs,
    intermediate_tensors: IntermediateTensors | None,
) -> torch.Tensor | IntermediateTensors:
    from ...utils.profiler import Timer

    with Timer("runner.cg_replay"):
        h = cuda_graph_runner.replay(
            prepared.inputs.input_ids if get_pp_group().is_first_rank else None,
            prepared.positions,
            prepared.inputs.slot_mapping,
            seq_lens=[len(s) for s in seqs],
            block_tables=[list(s.block_table) for s in seqs],
            intermediate_tensors=intermediate_tensors,
        )
        from ...utils.profiler import enabled as _prof_on

        if _prof_on():
            torch.cuda.synchronize()
    return h
