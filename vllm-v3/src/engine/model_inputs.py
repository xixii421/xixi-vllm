from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Optional

import torch

if TYPE_CHECKING:
    from ..memory import Sequence


@dataclass
class ModelInputs:
    input_ids: Optional[torch.Tensor] = None
    positions: Optional[torch.Tensor] = None
    is_prefill: bool = False
    cu_seqlens_q: Optional[torch.Tensor] = None
    cu_seqlens_k: Optional[torch.Tensor] = None
    max_seqlen_q: int = 0
    max_seqlen_k: int = 0
    context_lens: Optional[torch.Tensor] = None
    slot_mapping: Optional[torch.Tensor] = None
    block_tables: Optional[torch.Tensor] = None
    seqs: list["Sequence"] = field(default_factory=list)


def build_model_inputs(
    seqs: list["Sequence"],
    is_prefill: bool,
    block_size: int,
    device: torch.device,
) -> ModelInputs:
    from .input_builder import prepare_decode, prepare_prefill

    if is_prefill:
        input_ids, positions, ctx = prepare_prefill(seqs, block_size, device)
    else:
        input_ids, positions, ctx = prepare_decode(seqs, block_size, device)

    return ModelInputs(
        input_ids=input_ids,
        positions=positions,
        is_prefill=is_prefill,
        seqs=seqs,
        **ctx,
    )
