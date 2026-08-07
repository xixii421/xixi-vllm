from __future__ import annotations

from collections import deque

from ..memory import KVManager, Sequence, SequenceStatus


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

    def _promote_full_hit(self, seq: Sequence) -> None:
        if not seq.block_table:
            if not self._try_alloc(seq):
                return
        seq.num_scheduled_tokens = 0
        seq.is_prefill = True
        seq.status = SequenceStatus.RUNNING
        self.waiting.popleft()
        self.running.append(seq)

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
        """一次尽量消化多个 waiting 请求（v4 迁移）。

        原来只拉 waiting 队首一个，prefill 批恒为 bs=1，bs>1 的 prefill CG
        piece 全废。循环版让同一步里并发多个 prefill，既提升吞吐，又让
        prefill CG 能按 (bs>1, L) piece 整批重放。
        约束：本步最多留一个未完成 chunk（partial），保证 chunked prefill
        正确性且不让单个长请求饿死后续。
        """
        if not self.waiting or not self._room(prefills, decodes):
            return tokens_used
        n_partial = 0
        for seq in list(self.waiting):
            if not self._room(prefills, decodes):
                break
            remaining = self.max_num_batched_tokens - tokens_used
            if remaining <= 0:
                break
            if not self._try_alloc(seq):
                break
            need = seq.num_tokens - seq.num_cached_tokens
            if need <= 0:
                self._promote_full_hit(seq)
                continue
            chunk = min(need, remaining)
            seq.num_scheduled_tokens = chunk
            seq.is_prefill = True
            prefills.append(seq)
            tokens_used += chunk
            if seq.num_cached_tokens + chunk == seq.num_tokens:
                seq.status = SequenceStatus.RUNNING
                self.waiting.remove(seq)
                self.running.append(seq)
            else:
                n_partial += 1
                if n_partial >= 1:
                    break  # 本步最多留一个未完成 chunk
        return tokens_used

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
                self._promote_full_hit(seq)
                continue
            if remaining < need and prefills:
                break

            chunk = min(need, remaining)
            if remaining < chunk and prefills:
                break
            seq.num_scheduled_tokens = chunk
            seq.is_prefill = True
            tokens_used += chunk
            prefills.append(seq)
            if seq.num_cached_tokens + chunk == seq.num_tokens:
                seq.status = SequenceStatus.RUNNING
                self.waiting.popleft()
                self.running.append(seq)
            else:
                break

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
                # v2 contract: advance first, then publish full blocks.
                seq.num_cached_tokens += seq.num_scheduled_tokens
                self.kv.sync_prefix(seq)
            seq.num_scheduled_tokens = 0

            if is_prefill and seq.num_cached_tokens < seq.num_prompt_tokens:
                continue

            if seq.max_tokens <= 0:
                seq.status = SequenceStatus.FINISHED
                self.kv.deallocate(seq, publish=True)
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
                self.kv.deallocate(seq, publish=True)
                if seq in self.running:
                    self.running.remove(seq)
                if seq in self.waiting:
                    self.waiting.remove(seq)
                finished.append(seq)
        return finished
