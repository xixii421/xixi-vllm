from __future__ import annotations

from collections import deque

from ..memory import BlockManager, Sequence, SequenceStatus


class Scheduler:

    def __init__(
        self,
        *,
        max_num_seqs: int,
        max_num_batched_tokens: int,
        num_blocks: int,
        block_size: int,
        eos_token_id: int,
    ) -> None:
        self.max_num_seqs = max_num_seqs
        self.max_num_batched_tokens = max_num_batched_tokens
        self.eos = eos_token_id
        self.block_size = block_size
        Sequence.block_size = block_size

        self.block_manager = BlockManager(num_blocks, block_size)
        self.waiting: deque[Sequence] = deque()
        self.running: deque[Sequence] = deque()

    def is_finished(self) -> bool:
        return not self.waiting and not self.running

    def add(self, seq: Sequence) -> None:
        seq.status = SequenceStatus.WAITING
        self.waiting.append(seq)

    def schedule(self) -> tuple[list[Sequence], bool]:
        scheduled: list[Sequence] = []
        num_batched_tokens = 0

        # ---- Prefill ----
        while self.waiting and len(scheduled) < self.max_num_seqs:
            seq = self.waiting[0]
            remaining = self.max_num_batched_tokens - num_batched_tokens
            if remaining <= 0:
                break

            if not seq.block_table:
                num_cached_blocks = self.block_manager.can_allocate(seq)
                if num_cached_blocks == -1:
                    break
                num_tokens = seq.num_tokens - num_cached_blocks * self.block_size
            else:
                # Chunked resume (v3 first cut usually finishes in one go).
                num_tokens = seq.num_tokens - seq.num_cached_tokens
                num_cached_blocks = 0

            if num_tokens <= 0:
                # Full prefix hit (or empty remainder): promote without a compute step.
                if not seq.block_table:
                    self.block_manager.allocate(seq, num_cached_blocks)
                seq.num_scheduled_tokens = 0
                seq.is_prefill = True
                seq.status = SequenceStatus.RUNNING
                self.waiting.popleft()
                self.running.append(seq)
                continue

            # Only the first seq in the batch may be partially scheduled.
            if remaining < num_tokens and scheduled:
                break

            if not seq.block_table:
                self.block_manager.allocate(seq, num_cached_blocks)

            seq.num_scheduled_tokens = min(num_tokens, remaining)
            seq.is_prefill = True
            num_batched_tokens += seq.num_scheduled_tokens

            if seq.num_cached_tokens + seq.num_scheduled_tokens == seq.num_tokens:
                seq.status = SequenceStatus.RUNNING
                self.waiting.popleft()
                self.running.append(seq)
            scheduled.append(seq)

        if scheduled:
            return scheduled, True

        # ---- Decode ----
        while self.running and len(scheduled) < self.max_num_seqs:
            seq = self.running.popleft()
            while not self.block_manager.can_append(seq):
                if self.running:
                    self.preempt(self.running.pop())
                else:
                    self.preempt(seq)
                    break
            else:
                seq.num_scheduled_tokens = 1
                seq.is_prefill = False
                self.block_manager.may_append(seq)
                scheduled.append(seq)

        if not scheduled:
            return [], False

        self.running.extendleft(reversed(scheduled))
        return scheduled, False

    def preempt(self, seq: Sequence) -> None:
        """Recompute preemption: free pages, requeue as WAITING for full re-prefill."""
        if seq.status == SequenceStatus.FINISHED:
            return
        seq.status = SequenceStatus.WAITING
        seq.is_prefill = True
        self.block_manager.deallocate(seq)
        self.waiting.appendleft(seq)

    def postprocess(
        self,
        seqs: list[Sequence],
        token_ids: list[int],
        is_prefill: bool,
    ) -> list[Sequence]:
        """Advance caches, append sampled tokens, finish / free. Returns newly finished."""
        finished: list[Sequence] = []
        for seq, token_id in zip(seqs, token_ids):
            if seq.num_scheduled_tokens > 0:
                self.block_manager.hash_blocks(seq)
                seq.num_cached_tokens += seq.num_scheduled_tokens
            seq.num_scheduled_tokens = 0

            if is_prefill and seq.num_cached_tokens < seq.num_tokens:
                # Chunked / re-prefill incomplete (align nano: compare full seq length).
                continue

            # max_tokens=0: finish without emitting a completion token.
            if seq.max_tokens <= 0:
                seq.status = SequenceStatus.FINISHED
                self.block_manager.deallocate(seq)
                if seq in self.running:
                    self.running.remove(seq)
                if seq in self.waiting:
                    self.waiting.remove(seq)
                finished.append(seq)
                continue

            seq.append_token(token_id)
            hit_eos = (not seq.ignore_eos) and token_id == self.eos
            hit_len = seq.num_completion_tokens >= seq.max_tokens
            if hit_eos or hit_len:
                seq.status = SequenceStatus.FINISHED
                self.block_manager.deallocate(seq)
                if seq in self.running:
                    self.running.remove(seq)
                if seq in self.waiting:
                    self.waiting.remove(seq)
                finished.append(seq)
        return finished
