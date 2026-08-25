"""Batch：常驻 GPU buffer（计算面）。

prefill / decode / CUDA Graph 共用同一套静态 GPU 张量，每步只 ``copy_`` 实际数据进
buffer；CPU 侧配套预分配的 pinned staging 中转区（``_Staging``），热路径不再动态
分配锁页内存。padding 区域保持捕获时的安全值（slot=-1、seqlen=0、block=-1），
CUDA Graph 重放无需额外清理。
"""

from __future__ import annotations

import torch

from ..data.batch import SeqSchedule
from ..data.sequence import slots_for_range


class _Staging:
    """与 GPU 张量同形状的 pinned CPU 中转区，热路径复用。

    decode / prefill 各持一份：``mix_prefill_decode`` 开启时同一步内先
    build_decode 再 build_prefill，前者发起的异步 H2D 拷贝尚未执行完，
    共用一份 staging 会被后者覆盖而读到脏数据。block_tables 预填 -1，
    每步只写有效块、逐行清理 padding。
    """

    def __init__(self, *, max_bs: int, max_tokens: int, max_blocks: int) -> None:
        self.input_ids = self._alloc(max_tokens, torch.int64)
        self.positions = self._alloc(max_tokens, torch.int64)
        self.slot_mapping = self._alloc(max_tokens, torch.int32, fill=-1)
        self.cache_seqlens = self._alloc(max_bs, torch.int32)
        self.cu_seqlens_q = self._alloc(max_bs + 1, torch.int32)
        self.cu_seqlens_k = self._alloc(max_bs + 1, torch.int32)
        self.block_tables = self._alloc((max_bs, max_blocks), torch.int32, fill=-1)
        self.temperatures = self._alloc(max_bs, torch.float32)
        self.top_ps = self._alloc(max_bs, torch.float32)
        self.top_ks = self._alloc(max_bs, torch.int32)
        self.min_ps = self._alloc(max_bs, torch.float32)

    @staticmethod
    def _alloc(
        shape: int | tuple[int, ...],
        dtype: torch.dtype,
        fill: int | float | None = None,
    ) -> torch.Tensor:
        if isinstance(shape, int):
            shape = (shape,)
        if fill is None:
            return torch.empty(shape, dtype=dtype, pin_memory=True)
        return torch.full(shape, fill, dtype=dtype, pin_memory=True)


class Batch:
    def __init__(
        self,
        *,
        max_bs: int,
        max_tokens: int,
        max_blocks: int,
        vocab_size: int,
        device: torch.device,
        block_size: int,
    ) -> None:
        self.max_bs = max_bs
        self.max_tokens = max_tokens
        self.max_blocks = max_blocks
        self.vocab_size = vocab_size
        self.device = device
        self.block_size = block_size

        self.input_ids = torch.zeros(max_tokens, dtype=torch.int64, device=device)
        self.positions = torch.zeros(max_tokens, dtype=torch.int64, device=device)
        self.slot_mapping = torch.full(
            (max_tokens,), -1, dtype=torch.int32, device=device
        )
        self.cache_seqlens = torch.zeros(max_bs, dtype=torch.int32, device=device)
        self.cu_seqlens_q = torch.zeros(max_bs + 1, dtype=torch.int32, device=device)
        self.cu_seqlens_k = torch.zeros(max_bs + 1, dtype=torch.int32, device=device)
        self.block_tables = torch.zeros(
            max_bs, max_blocks, dtype=torch.int32, device=device
        )

        self.temperatures = torch.zeros(max_bs, dtype=torch.float32, device=device)
        self.top_ps = torch.zeros(max_bs, dtype=torch.float32, device=device)
        self.top_ks = torch.zeros(max_bs, dtype=torch.int32, device=device)
        self.min_ps = torch.zeros(max_bs, dtype=torch.float32, device=device)

        self.all_greedy = True
        self.any_greedy = False
        self.use_top_k = False
        self.use_top_p = False
        self.use_min_p = False
        self.max_top_k = 0

        self._decode_staging = _Staging(
            max_bs=max_bs, max_tokens=max_tokens, max_blocks=max_blocks
        )
        self._prefill_staging = _Staging(
            max_bs=max_bs, max_tokens=max_tokens, max_blocks=max_blocks
        )

    @staticmethod
    def _put(gpu: torch.Tensor, staging: torch.Tensor, values: list) -> None:
        """list → pinned staging → GPU：复用缓冲，单次异步 H2D，无新增分配。"""
        staging.numpy()[:] = values
        gpu.copy_(staging, non_blocking=True)

    def build_decode(self, seqs: list[SeqSchedule]) -> None:
        bs = len(seqs)
        st = self._decode_staging
        # decode 时 num_cached_tokens 已指向本步 token 的位置：position = num_cached_tokens，
        # cache_seqlen = num_cached_tokens + 1（含本步写入的 KV）。
        self._put(
            self.input_ids[:bs],
            st.input_ids[:bs],
            [s.token_ids[-1] for s in seqs],
        )
        self._put(
            self.positions[:bs],
            st.positions[:bs],
            [s.num_cached_tokens for s in seqs],
        )
        self._put(
            self.slot_mapping[:bs],
            st.slot_mapping[:bs],
            [self._decode_slot(s) for s in seqs],
        )
        self._put(
            self.cache_seqlens[:bs],
            st.cache_seqlens[:bs],
            [s.num_cached_tokens + 1 for s in seqs],
        )
        self._fill_block_tables(st, seqs)
        self._fill_sampling(st, seqs)

    def build_prefill(self, seqs: list[SeqSchedule]) -> tuple[int, int, int, bool]:
        st = self._prefill_staging
        ids: list[int] = []
        positions: list[int] = []
        slots: list[int] = []
        cu_q = [0]
        cu_k = [0]
        max_q = 0
        max_k = 0
        for s in seqs:
            start = s.num_cached_tokens
            n = len(s.token_ids)
            end = start + n
            ids.extend(s.token_ids)
            positions.extend(range(start, end))
            slots.extend(slots_for_range(s.block_table, start, end, self.block_size))
            cu_q.append(cu_q[-1] + n)
            cu_k.append(cu_k[-1] + end)
            max_q = max(max_q, n)
            max_k = max(max_k, end)

        n = len(ids)
        self._put(self.input_ids[:n], st.input_ids[:n], ids)
        self._put(self.positions[:n], st.positions[:n], positions)
        self._put(self.slot_mapping[:n], st.slot_mapping[:n], slots)
        self._put(
            self.cu_seqlens_q[: len(cu_q)],
            st.cu_seqlens_q[: len(cu_q)],
            cu_q,
        )
        self._put(
            self.cu_seqlens_k[: len(cu_k)],
            st.cu_seqlens_k[: len(cu_k)],
            cu_k,
        )

        has_cache = cu_k[-1] > cu_q[-1]
        if has_cache:
            self._fill_block_tables(st, seqs)
        self._fill_sampling(st, seqs)
        return n, max_q, max_k, has_cache

    def _fill_block_tables(self, st: _Staging, seqs: list[SeqSchedule]) -> None:
        # 每行只写有效块，剩余清 -1（行复用可能残留旧 seq 的块 id，须保证
        # padding 恒为安全值）；整行一次全宽异步拷贝到 GPU。
        view = st.block_tables.numpy()
        for i, s in enumerate(seqs):
            row = view[i]
            k = len(s.block_table)
            row[:k] = s.block_table
            row[k:] = -1
        self.block_tables[: len(seqs)].copy_(
            st.block_tables[: len(seqs)], non_blocking=True
        )

    def _fill_sampling(self, st: _Staging, seqs: list[SeqSchedule]) -> None:
        bs = len(seqs)
        vocab = self.vocab_size
        temps = [s.temperature for s in seqs]
        top_ps = [1.0 if s.top_p is None else s.top_p for s in seqs]
        top_ks = [
            vocab if (s.top_k is None or s.top_k <= 0) else min(s.top_k, vocab)
            for s in seqs
        ]
        min_ps = [0.0 if s.min_p is None else s.min_p for s in seqs]

        self._put(self.temperatures[:bs], st.temperatures[:bs], temps)
        self._put(self.top_ps[:bs], st.top_ps[:bs], top_ps)
        self._put(self.top_ks[:bs], st.top_ks[:bs], top_ks)
        self._put(self.min_ps[:bs], st.min_ps[:bs], min_ps)

        self.all_greedy = all(t <= 1e-5 for t in temps)
        self.any_greedy = any(t <= 1e-5 for t in temps)
        self.use_top_k = any(k < vocab for k in top_ks)
        self.use_top_p = any(p < 1.0 for p in top_ps)
        self.use_min_p = any(m > 0.0 for m in min_ps)
        self.max_top_k = max(top_ks) if self.use_top_k else 0

    def _decode_slot(self, seq: SeqSchedule) -> int:
        n = seq.num_cached_tokens + 1
        return (
            seq.block_table[(n - 1) // self.block_size] * self.block_size
            + (n - 1) % self.block_size
        )
