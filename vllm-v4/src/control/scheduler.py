from __future__ import annotations
from collections import deque
from ..data import KVManager, ScheduledBatch, Sequence, SequenceStatus


class Scheduler:
    """LLM 推理请求调度器。

    负责管理等待队列与运行队列，并根据最大序列数、最大 token 预算
    以及 KV cache 可用块数，输出每一步可执行的 prefill 与 decode 序列。
    """

    def __init__(
        self,
        kv: KVManager,
        *,
        max_num_seqs: int,
        max_num_batched_tokens: int,
        eos_token_id: int,
        mix_prefill_decode: bool = True,
    ) -> None:
        # KV cache 管理器，负责块分配、释放、前缀匹配与回收。
        self.kv = kv

        # 单步最多调度的序列数。
        self.max_num_seqs = max_num_seqs

        # 单步最多处理的 token 总数。
        self.max_num_batched_tokens = max_num_batched_tokens

        # EOS token id，用于判断生成是否结束。
        self.eos = eos_token_id

        # 是否允许 prefill 与 decode 在同一步混合调度。
        self.mix_prefill_decode = mix_prefill_decode

        # 等待队列：新请求或被抢占后需要重新分配 KV 的序列。
        self.waiting: deque[Sequence] = deque()

        # 运行队列：已开始生成或已完成 prefill、可参与 decode 的序列。
        self.running: deque[Sequence] = deque()

    def is_finished(self) -> bool:
        """判断是否所有请求都已执行完成。"""
        return not self.waiting and not self.running

    def add(self, seq: Sequence) -> None:
        """将新序列加入等待队列。"""
        seq.status = SequenceStatus.WAITING
        self.waiting.append(seq)

    def schedule(self) -> ScheduledBatch:
        """调度一步可执行的序列。"""
        if self.mix_prefill_decode:
            p, d = self._schedule_mixed()
        else:
            p, d = self._schedule_prefill_first()
        return ScheduledBatch(prefills=p, decodes=d)

    def _room(self, prefills: list[Sequence], decodes: list[Sequence]) -> bool:
        """判断当前批次是否还能容纳新的序列。"""
        return len(prefills) + len(decodes) < self.max_num_seqs

    def _try_alloc(self, seq: Sequence) -> bool:
        """尝试为序列分配 KV 块。

        如果序列已经拥有 block table，则视为已分配并直接返回成功；
        否则先检查 KV cache 是否足够，再进行分配，分配过程包含前缀匹配。
        """
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
        """从运行队列头部拉取 decode 序列。

        如果某个序列无法继续追加 token，则优先从运行队列尾部抢占其他序列，
        释放其 KV 块腾出空间；如果抢占后仍无法满足该序列，
        则抢占该序列自身，并结束本轮 decode 调度。
        """
        while self.running and self._room(prefills, decodes):
            if tokens_used + 1 > self.max_num_batched_tokens:
                break

            seq = self.running.popleft()

            # KV 不足时，从队尾开始抢占其他运行序列，为当前序列腾出空间。
            while not self.kv.can_append(seq) and self.running:
                self.preempt(self.running.pop())

            # 抢占其他序列后仍无法满足当前序列，则只能抢占当前序列并放弃本轮。
            if not self.kv.can_append(seq):
                self.preempt(seq)
                break

            seq.num_scheduled_tokens = 1
            self.kv.may_append(seq)
            decodes.append(seq)
            tokens_used += 1

        # 本步参与 decode 的序列仍处于运行状态，放回运行队列头部。
        if decodes:
            self.running.extendleft(reversed(decodes))

        return tokens_used

    def _pull_prefill_chunk(
        self,
        prefills: list[Sequence],
        decodes: list[Sequence],
        tokens_used: int,
        *,
        allow_trailing_partial: bool = True,
    ) -> int:
        """从等待队列拉取 prefill chunk，直到序列预算或 token 预算耗尽。

        Args:
            prefills: 输出参数，收集本步可调度的 prefill 序列。
            decodes: 当前批次中已有的 decode 序列，用于判断序列预算。
            tokens_used: 当前批次已使用的 token 数。
            allow_trailing_partial: 是否允许在已有 prefill 之后追加部分 chunk。

        Returns:
            int: 拉取 prefill chunk 后本步累计使用的 token 数。
        """
        if not self.waiting or not self._room(prefills, decodes):
            return tokens_used

        # 记录已产生的部分 prefill chunk 数量。
        # 每轮最多只允许一个序列处于未完成 prefill 状态，避免过多半截 prefill。
        n_partial = 0

        for seq in list(self.waiting):
            if not self._room(prefills, decodes):
                break

            remaining = self.max_num_batched_tokens - tokens_used
            if remaining <= 0:
                break

            if not self._try_alloc(seq):
                break

            # 该序列本次还需要计算的 token 数。
            need = seq.num_tokens - seq.num_cached_tokens

            if need <= 0:
                # 前缀完全命中：无需执行 prefill，直接进入运行队列作为 decode 候选。
                seq.status = SequenceStatus.RUNNING
                self.waiting.remove(seq)
                self.running.append(seq)
                continue

            # 如果不允许在已有 prefill 后追加部分 chunk，且当前预算不足，则停止拉取。
            if remaining < need and prefills and not allow_trailing_partial:
                break

            # 计算本步为该序列调度的 chunk 大小。
            chunk = min(need, remaining)
            seq.num_scheduled_tokens = chunk
            prefills.append(seq)
            tokens_used += chunk

            # 本步完成后 prompt 全部进入缓存：转入运行队列，本步需采样。
            if seq.num_cached_tokens + chunk == seq.num_tokens:
                seq.need_sample = True
                seq.status = SequenceStatus.RUNNING
                self.waiting.remove(seq)
                self.running.append(seq)
            else:
                # 该序列本步只完成部分 prefill，仍留在等待队列中。
                seq.need_sample = False
                n_partial += 1
                if n_partial >= 1:
                    break

        return tokens_used

    def _schedule_prefill_first(self) -> tuple[list[Sequence], list[Sequence]]:
        """prefill 优先调度策略。

        优先尝试调度 prefill；如果没有可执行的 prefill，再调度 decode。
        """
        prefills: list[Sequence] = []
        decodes: list[Sequence] = []

        # prefill 优先模式下，不允许在已有 prefill 后追加部分 chunk。
        self._pull_prefill_chunk(prefills, decodes, 0, allow_trailing_partial=False)

        if prefills:
            return prefills, []

        self._schedule_decode(prefills, decodes, 0)
        return [], decodes

    def _schedule_mixed(self) -> tuple[list[Sequence], list[Sequence]]:
        """prefill 与 decode 混合调度策略。

        先尽量调度已运行序列的 decode，再用剩余预算拉取新的 prefill chunk；
        如果没有任何 decode，可退化为 prefill 优先策略。
        """
        prefills: list[Sequence] = []
        decodes: list[Sequence] = []

        used = self._schedule_decode(prefills, decodes, 0)

        if decodes:
            self._pull_prefill_chunk(prefills, decodes, used)
            return prefills, decodes

        return self._schedule_prefill_first()

    def preempt(self, seq: Sequence) -> None:
        """抢占一个序列。

        释放其占用的 KV 块，并将其放回等待队列头部。
        此处不发布前缀缓存，保留该序列后续重新计算的机会。
        """
        if seq.status == SequenceStatus.FINISHED:
            return

        seq.status = SequenceStatus.WAITING
        self.kv.deallocate(seq, publish=False)
        self.waiting.appendleft(seq)

    def postprocess(
        self,
        seqs: list[Sequence],
        token_ids: list[int],
        *,
        is_prefill: bool,
    ) -> list[Sequence]:
        """前向结束后更新序列状态。

        主要工作包括：
            1. 推进每个序列的已缓存 token 数。
            2. 同步前缀缓存。
            3. 对完成 prompt 的序列追加采样 token。
            4. 判断是否命中 EOS 或达到最大生成长度。
            5. 回收已完成序列的 KV cache。

        Args:
            seqs: 本步参与前向计算的序列。
            token_ids: 与 seqs 一一对应的采样 token id。
            is_prefill: 本批是否为 prefill。

        Returns:
            list[Sequence]: 本步完成的序列列表。
        """
        finished: list[Sequence] = []

        for seq, token_id in zip(seqs, token_ids):
            # 被调度的序列本步一定有 num_scheduled_tokens 大于 0。
            seq.num_cached_tokens += seq.num_scheduled_tokens

            # 将本步计算结果同步到前缀缓存。
            self.kv.sync_prefix(seq)
            seq.num_scheduled_tokens = 0

            # 未完成 prompt 的 prefill chunk 本步不采样，也不判断是否完成。
            if is_prefill and seq.num_cached_tokens < seq.num_prompt_tokens:
                continue

            # max_tokens 小于等于 0 表示只需要处理 prompt，不生成新 token。
            if seq.max_tokens <= 0:
                seq.status = SequenceStatus.FINISHED
                self.kv.deallocate(seq, publish=True)
                finished.append(seq)
                continue

            # 追加本步采样得到的 token，并判断是否完成。
            seq.append_token(token_id)
            hit_eos = (not seq.ignore_eos) and token_id == self.eos
            hit_len = seq.num_completion_tokens >= seq.max_tokens

            if hit_eos or hit_len:
                seq.status = SequenceStatus.FINISHED

                # 完成序列释放 KV cache，并发布前缀供后续请求复用。
                self.kv.deallocate(seq, publish=True)
                finished.append(seq)

        # 已完成序列需要从运行队列中移除。
        # 未完成的 partial prefill 已在上面跳过，不会出现在 finished 中。
        if finished:
            done = {s.seq_id for s in finished}
            self.running = deque(s for s in self.running if s.seq_id not in done)

        return finished