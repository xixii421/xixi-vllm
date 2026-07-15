from __future__ import annotations
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Optional
import torch
_CTX = None

@dataclass
class ForwardContext:
    is_prefill: bool = False
    cu_seqlens_q: Optional[torch.Tensor] = None
    cu_seqlens_k: Optional[torch.Tensor] = None
    max_seqlen_q: int = 0
    max_seqlen_k: int = 0
    context_lens: Optional[torch.Tensor] = None
    slot_mapping: Optional[torch.Tensor] = None
    block_tables: Optional[torch.Tensor] = None
    seqs: list = field(default_factory=list)

def get_context() -> ForwardContext:
    global _CTX
    if _CTX is None:
        return ForwardContext()
    return _CTX

@contextmanager
def forward_context(is_prefill: bool, seqs: list, *, slot_mapping: Optional[torch.Tensor]=None, cu_seqlens_q: Optional[torch.Tensor]=None, cu_seqlens_k: Optional[torch.Tensor]=None, max_seqlen_q: int=0, max_seqlen_k: int=0, context_lens: Optional[torch.Tensor]=None, block_tables: Optional[torch.Tensor]=None):
    global _CTX
    old = _CTX
    _CTX = ForwardContext(is_prefill=is_prefill, cu_seqlens_q=cu_seqlens_q, cu_seqlens_k=cu_seqlens_k, max_seqlen_q=max_seqlen_q, max_seqlen_k=max_seqlen_k, slot_mapping=slot_mapping, context_lens=context_lens, block_tables=block_tables, seqs=seqs)
    try:
        yield
    finally:
        _CTX = old

def reset_context() -> None:
    global _CTX
    _CTX = None
