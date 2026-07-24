from __future__ import annotations

from collections import deque

from ..kv import KVManager, Sequence, SequenceStatus


class Scheduler:
    def __init__(
        self,
        kv: KVManager,
        *,
        max_num_seqs: int,
        max_num_batched_tokens: int,
        eos_token_id: int,
        mix_prefill_decode: bool = True,
    ) -> None:
        self.kv = kv
        self.max_num_seqs = max_num_seqs
        self.max_num_batched_tokens = max_num_batched_tokens
        self.eos = eos_token_id
        self.mix_prefill_decode = mix_prefill_decode
        self.block_size = kv.block_size
        self.waiting: deque[Sequence] = deque()
        self.running: deque[Sequence] = deque()

    def is_finished(self) -> bool:
        return not self.waiting and not self.running

    def add(self, seq: Sequence) -> None:
        seq.status = SequenceStatus.WAITING
        self.waiting.append(seq)

    def schedule(self) -> tuple[list[Sequence], list[Sequence]]:
        if self.mix_prefill_decode:
            return self._schedule_mixed()
        return self._schedule_prefill_first()

    def _room(self, prefills: list[Sequence], decodes: list[Sequence]) -> bool:
        return len(prefills) + len(decodes) < self.max_num_seqs

    def _try_alloc(self, seq: Sequence) -> bool:
        if seq.block_table:
            return True
        if not self.kv.can_allocate(seq):
            return False
        self.kv.allocate(seq)
        return True

    def _schedule_decode(
        self,
        prefills: list[Sequence],
        decodes: list[Sequence],
        tokens_used: int,
    ) -> int:
        while self.running and self._room(prefills, decodes):
            if tokens_used + 1 > self.max_num_batched_tokens:
                break
            seq = self.running.popleft()
            while not self.kv.can_append(seq):
                if self.running:
                    self.preempt(self.running.pop())
                else:
                    self.preempt(seq)
                    break
            else:
                seq.num_scheduled_tokens = 1
                seq.is_prefill = False
                self.kv.may_append(seq)
                decodes.append(seq)
                tokens_used += 1
        if decodes:
            self.running.extendleft(reversed(decodes))
        return tokens_used

    def _pull_prefill_chunk(
        self,
        prefills: list[Sequence],
        decodes: list[Sequence],
        tokens_used: int,
    ) -> int:
        if not self.waiting or not self._room(prefills, decodes):
            return tokens_used
        remaining = self.max_num_batched_tokens - tokens_used
        if remaining <= 0:
            return tokens_used

        seq = self.waiting[0]
        if not self._try_alloc(seq):
            return tokens_used

        need = seq.num_tokens - seq.num_cached_tokens
        if need <= 0:
            return tokens_used

        chunk = min(need, remaining)
        seq.num_scheduled_tokens = chunk
        seq.is_prefill = True
        prefills.append(seq)
        if seq.num_cached_tokens + chunk == seq.num_tokens:
            seq.status = SequenceStatus.RUNNING
            self.waiting.popleft()
            self.running.append(seq)
        return tokens_used + chunk

    def _schedule_prefill_first(self) -> tuple[list[Sequence], list[Sequence]]:
        prefills: list[Sequence] = []
        decodes: list[Sequence] = []
        tokens_used = 0

        while self.waiting and self._room(prefills, decodes):
            remaining = self.max_num_batched_tokens - tokens_used
            if remaining <= 0:
                break
            seq = self.waiting[0]
            if not self._try_alloc(seq):
                break
            need = seq.num_tokens - seq.num_cached_tokens
            if need <= 0:
                break
            if remaining < need and prefills:
                break

            chunk = min(need, remaining)
            seq.num_scheduled_tokens = chunk
            seq.is_prefill = True
            tokens_used += chunk
            prefills.append(seq)
            if seq.num_cached_tokens + chunk == seq.num_tokens:
                seq.status = SequenceStatus.RUNNING
                self.waiting.popleft()
                self.running.append(seq)
            else:
                break  # one partial chunk

        if prefills:
            return prefills, []
        self._schedule_decode(prefills, decodes, 0)
        return [], decodes

    def _schedule_mixed(self) -> tuple[list[Sequence], list[Sequence]]:
        prefills: list[Sequence] = []
        decodes: list[Sequence] = []
        used = self._schedule_decode(prefills, decodes, 0)
        if decodes:
            self._pull_prefill_chunk(prefills, decodes, used)
            return prefills, decodes
        return self._schedule_prefill_first()

    def preempt(self, seq: Sequence) -> None:
        if seq.status == SequenceStatus.FINISHED:
            return
        seq.status = SequenceStatus.WAITING
        seq.is_prefill = True
        self.kv.deallocate(seq, publish=False)
        self.waiting.appendleft(seq)

    def postprocess(
        self,
        seqs: list[Sequence],
        token_ids: list[int],
        *,
        is_prefill: bool,
    ) -> list[Sequence]:
        finished: list[Sequence] = []
        for seq, token_id in zip(seqs, token_ids):
            if seq.num_scheduled_tokens > 0:
                seq.num_cached_tokens += seq.num_scheduled_tokens
                self.kv.sync_prefix(seq)
            seq.num_scheduled_tokens = 0

            if is_prefill and seq.num_cached_tokens < seq.num_prompt_tokens:
                continue

            seq.append_token(token_id)
            hit_eos = (not seq.ignore_eos) and token_id == self.eos
            hit_len = seq.num_completion_tokens >= seq.max_tokens
            if hit_eos or hit_len:
                seq.status = SequenceStatus.FINISHED
                self.kv.deallocate(seq, publish=True)
                if seq in self.running:
                    self.running.remove(seq)
                finished.append(seq)
        return finished
