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
        encoder_compute_budget: int = 0,
    ) -> None:
        self.kv = kv
        self.max_num_seqs = max_num_seqs
        self.max_num_batched_tokens = max_num_batched_tokens
        self.eos = eos_token_id
        self.mix_prefill_decode = mix_prefill_decode
        self.encoder_compute_budget = (
            encoder_compute_budget if encoder_compute_budget > 0 else max_num_batched_tokens
        )
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
            self._promote_full_hit(seq)
            return tokens_used

        chunk = min(need, remaining)
        chunk = self._clamp_chunk_for_encoder_budget(seq, chunk)
        if chunk <= 0:
            return tokens_used
        seq.num_scheduled_tokens = chunk
        seq.is_prefill = True
        prefills.append(seq)
        if seq.num_cached_tokens + chunk == seq.num_tokens:
            seq.status = SequenceStatus.RUNNING
            self.waiting.popleft()
            self.running.append(seq)
        return tokens_used + chunk

    def _clamp_chunk_for_encoder_budget(self, seq: Sequence, chunk: int) -> int:
        """Shorten chunk so overlapping vision tokens fit the encoder budget.

        If the next image alone exceeds the budget but the cursor is already at
        that image, still schedule the image (official all-or-nothing exception)
        so the request cannot stall forever.
        """
        if not seq.has_multimodal or chunk <= 0:
            return chunk
        budget = self.encoder_compute_budget
        start = seq.num_cached_tokens
        end = start + chunk
        used = 0
        for mf in seq.mm_features:
            img_start = mf.start
            img_end = mf.start + mf.num_encoder_tokens
            if img_end <= start:
                continue
            if img_start >= end:
                break
            if used + mf.num_encoder_tokens > budget:
                # Truncate before the image when text prefix remains.
                if img_start > start:
                    return img_start - start
                # Already at the oversized image → schedule it alone.
                return min(chunk, img_end - start)
            used += mf.num_encoder_tokens
        return chunk

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
            chunk = self._clamp_chunk_for_encoder_budget(seq, chunk)
            if chunk <= 0:
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
        # KV was dropped → the sequence re-prefills from scratch, so its MRoPE
        # cursor must reset too (otherwise chunked position math is stale).
        if seq.has_multimodal:
            seq.reset_mrope_state()
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
