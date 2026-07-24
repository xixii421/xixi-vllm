from __future__ import annotations

import torch
from torch import nn

from ..kv import ForwardContext, Sequence, set_forward_context
from ..pagedattention import slots_tensor
from ..sampler import Sampler
from ..kv.manager import KVManager


class ModelRunner:
    def __init__(
        self,
        model: nn.Module,
        kv: KVManager,
        sampler: Sampler,
        *,
        device: torch.device,
    ) -> None:
        self.model = model
        self.kv = kv
        self.sampler = sampler
        self.block_size = kv.block_size
        self.device = device

    def _tensor(self, data, dtype=torch.long) -> torch.Tensor:
        return torch.tensor(data, device=self.device, dtype=dtype)

    def _sample_one(self, logits: torch.Tensor, seq: Sequence) -> int:
        if seq.temperature is not None and seq.temperature <= 0:
            return int(torch.argmax(logits, dim=-1).item())

        def opt(v, dtype):
            return None if v is None else torch.tensor([v], device=self.device, dtype=dtype)

        token = self.sampler(
            logits,
            opt(seq.temperature, torch.float) if seq.temperature and seq.temperature > 0 else None,
            opt(seq.min_p, torch.float),
            opt(seq.top_p, torch.float),
            opt(seq.top_k, torch.int),
        )
        return int(token.item())

    @torch.inference_mode()
    def run_prefill_one(self, seq: Sequence) -> int:
        start = seq.num_cached_tokens
        end = start + seq.num_scheduled_tokens
        tokens = seq.token_ids[start:end]
        slots = slots_tensor(
            seq.block_table, start, end, self.block_size, device=self.device
        )
        set_forward_context(
            ForwardContext(
                is_prefill=True,
                store=self.kv.store,
                block_tables=[seq.block_table],
                slot_mapping=slots,
                query_lens=[end - start],
                context_lens=[end],
                cached_lens=[start],
            )
        )
        try:
            ids = self._tensor([tokens])
            positions = self._tensor([list(range(start, end))])
            h = self.model(ids, positions)
            logits = self.model.compute_logits(h[:, -1:]).squeeze(1)
            return self._sample_one(logits, seq)
        finally:
            set_forward_context(None)

    @torch.inference_mode()
    def run_decode_batch(self, seqs: list[Sequence]) -> list[int]:
        input_ids = [seq.last_token for seq in seqs]
        positions = [len(seq) - 1 for seq in seqs]
        slots = [
            seq.block_table[-1] * self.block_size + ((len(seq) - 1) % self.block_size)
            for seq in seqs
        ]
        set_forward_context(
            ForwardContext(
                is_prefill=False,
                store=self.kv.store,
                block_tables=[list(s.block_table) for s in seqs],
                slot_mapping=self._tensor(slots),
                query_lens=[1] * len(seqs),
                context_lens=[len(s) for s in seqs],
            )
        )
        try:
            ids = self._tensor(input_ids).unsqueeze(-1)
            pos = self._tensor(positions).unsqueeze(-1)
            h = self.model(ids, pos)
            logits = self.model.compute_logits(h[:, -1:]).squeeze(1)
            return [self._sample_one(logits[i : i + 1], seq) for i, seq in enumerate(seqs)]
        finally:
            set_forward_context(None)

    def run(self, prefills: list[Sequence], decodes: list[Sequence]) -> tuple[list[int], list[int]]:
        d_ids = self.run_decode_batch(decodes) if decodes else []
        p_ids = [self.run_prefill_one(s) for s in prefills] if prefills else []
        return p_ids, d_ids
