"""Waiting / running continuous batcher (prefill-first)."""

from __future__ import annotations

from collections import deque

from ..utils.profiler import Timer
from .block_manager import BlockManager
from .output import SchedulerOutput
from .sequence import Sequence, SequenceStatus


class Scheduler:
    def __init__(
        self,
        *,
        max_num_seqs: int,
        max_num_batched_tokens: int,
        num_blocks: int,
        block_size: int,
        eos_token_id: int,
        enable_radix: bool = True,
    ) -> None:
        self.max_num_seqs = max_num_seqs
        self.max_num_batched_tokens = max_num_batched_tokens
        self.eos = eos_token_id
        self.block_size = block_size
        self.block_manager = BlockManager(
            num_blocks,
            block_size,
            enable_radix=enable_radix,
        )
        self.waiting: deque[Sequence] = deque()
        self.running: deque[Sequence] = deque()

    def is_finished(self) -> bool:
        return not self.waiting and not self.running

    def add(self, seq: Sequence) -> None:
        seq.status = SequenceStatus.WAITING
        self.waiting.append(seq)

    def schedule(self) -> SchedulerOutput:
        with Timer("sched.schedule"):
            return self._schedule_body()

    def _schedule_body(self) -> SchedulerOutput:
        scheduled: list[Sequence] = []
        num_batched_tokens = 0
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
                num_tokens = seq.num_tokens - seq.num_cached_tokens
                num_cached_blocks = 0
            if num_tokens <= 0:
                if not seq.block_table:
                    self.block_manager.allocate(seq, num_cached_blocks)
                seq.num_scheduled_tokens = 0
                seq.is_prefill = True
                seq.status = SequenceStatus.RUNNING
                self.waiting.popleft()
                self.running.append(seq)
                continue
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
            return SchedulerOutput(seqs=tuple(scheduled), is_prefill=True)

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
            return SchedulerOutput.empty()
        self.running.extendleft(reversed(scheduled))
        return SchedulerOutput(seqs=tuple(scheduled), is_prefill=False)

    def preempt(self, seq: Sequence) -> None:
        if seq.status == SequenceStatus.FINISHED:
            return
        seq.status = SequenceStatus.WAITING
        seq.is_prefill = True
        self.block_manager.deallocate(seq)
        seq.reset_mrope_state()
        self.waiting.appendleft(seq)

    def abort(self, seq: Sequence) -> None:
        if seq.status == SequenceStatus.FINISHED:
            return
        seq.status = SequenceStatus.FINISHED
        self.block_manager.deallocate(seq)
        try:
            self.running.remove(seq)
        except ValueError:
            pass
        try:
            self.waiting.remove(seq)
        except ValueError:
            pass

    def postprocess(
        self, seqs: list[Sequence] | tuple[Sequence, ...], token_ids: list[int], is_prefill: bool
    ) -> list[Sequence]:
        with Timer("sched.postprocess"):
            return self._postprocess_body(list(seqs), token_ids, is_prefill)

    def _finish(self, seq: Sequence, finished: list[Sequence]) -> None:
        seq.status = SequenceStatus.FINISHED
        self.block_manager.deallocate(seq)
        try:
            self.running.remove(seq)
        except ValueError:
            pass
        try:
            self.waiting.remove(seq)
        except ValueError:
            pass
        finished.append(seq)

    def _postprocess_body(
        self, seqs: list[Sequence], token_ids: list[int], is_prefill: bool
    ) -> list[Sequence]:
        finished: list[Sequence] = []
        for seq, token_id in zip(seqs, token_ids):
            if seq.num_scheduled_tokens > 0:
                self.block_manager.hash_blocks(seq)
                seq.num_cached_tokens += seq.num_scheduled_tokens
            seq.num_scheduled_tokens = 0
            if is_prefill and seq.num_cached_tokens < seq.num_tokens:
                continue
            if seq.max_tokens <= 0:
                self._finish(seq, finished)
                continue
            seq.append_token(token_id)
            hit_eos = not seq.ignore_eos and token_id == self.eos
            hit_len = seq.num_completion_tokens >= seq.max_tokens
            if hit_eos or hit_len:
                self._finish(seq, finished)
        return finished
