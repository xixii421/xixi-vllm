from __future__ import annotations

import torch
from torch import nn

from ..memory import PagedKVCache, Sequence, slots_for_range
from ..utils import ForwardContext, set_forward_context
from .sampler import Sampler


class ModelRunner:
    def __init__(
        self,
        model: nn.Module,
        kv_cache: PagedKVCache,
        sampler: Sampler,
        *,
        block_size: int,
        device: torch.device,
    ) -> None:
        self.model = model
        self.kv_cache = kv_cache
        self.sampler = sampler
        self.block_size = block_size
        self.device = device

    def _tensor(self, data, dtype=torch.long) -> torch.Tensor:
        return torch.tensor(data, device=self.device, dtype=dtype)

    def _sample_one(self, logits: torch.Tensor, seq: Sequence) -> int:
        """logits: [1, vocab]."""

        def opt(v, dtype):
            if v is None:
                return None
            return torch.tensor([v], device=self.device, dtype=dtype)

        temps = None
        if seq.temperature is not None and seq.temperature > 0:
            temps = opt(seq.temperature, torch.float)
        elif seq.temperature == 0:
            temps = None  # greedy path when all filters None
        # Prefer explicit greedy when temperature <= 0
        if seq.temperature is not None and seq.temperature <= 0:
            return int(torch.argmax(logits, dim=-1).item())

        token = self.sampler(
            logits,
            temps,
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
        slots = slots_for_range(seq.block_table, start, end, self.block_size)

        set_forward_context(
            ForwardContext(
                is_prefill=True,
                kv_cache=self.kv_cache,
                block_tables=[seq.block_table],
                slot_mapping=self._tensor(slots),
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
        slots = []
        for seq in seqs:
            pos = len(seq) - 1
            slots.append(
                seq.block_table[-1] * self.block_size + (pos % self.block_size)
            )
        context_lens = [len(seq) for seq in seqs]

        set_forward_context(
            ForwardContext(
                is_prefill=False,
                kv_cache=self.kv_cache,
                block_tables=[list(s.block_table) for s in seqs],
                slot_mapping=self._tensor(slots),
                query_lens=[1] * len(seqs),
                context_lens=context_lens,
            )
        )
        try:
            ids = self._tensor(input_ids).unsqueeze(-1)  # [B, 1]
            pos = self._tensor(positions).unsqueeze(-1)
            h = self.model(ids, pos)
            logits = self.model.compute_logits(h[:, -1:])  # [B, 1, V]
            logits = logits.squeeze(1)
            return [self._sample_one(logits[i : i + 1], seq) for i, seq in enumerate(seqs)]
        finally:
            set_forward_context(None)

    def run(self, seqs: list[Sequence], is_prefill: bool) -> list[int]:
        if not seqs:
            return []
        if is_prefill:
            # Prefill: run each scheduled sequence sequentially (teaching path).
            # Scheduling itself is continuous; Decode batch is where GPU concurrency wins.
            return [self.run_prefill_one(seq) for seq in seqs]
        return self.run_decode_batch(seqs)
