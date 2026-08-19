from __future__ import annotations

from dataclasses import dataclass, field

import msgspec
import torch

from .sequence import Sequence, slots_for_range


@dataclass
class ScheduledBatch:
    """调度器产出的本步批次。compute 层只消费这个显式接口，不触碰 Sequence 内部。"""

    prefills: list[Sequence] = field(default_factory=list)
    decodes: list[Sequence] = field(default_factory=list)

    def is_empty(self) -> bool:
        return not self.prefills and not self.decodes

    def to_wire(self) -> "SchedulerOutput":
        """转成最小增量 wire 结构：只携带本步真正需要的数据，丢弃 Sequence 全历史。"""
        return SchedulerOutput(
            prefills=[_prefill_schedule(s) for s in self.prefills],
            decodes=[_decode_schedule(s) for s in self.decodes],
        )


def _sampling_fields(s: Sequence) -> tuple[float, float | None, int | None, float | None]:
    return s.temperature, s.top_p, s.top_k, s.min_p


def _decode_schedule(s: Sequence) -> "SeqSchedule":
    """decode 每序列只发 [last_token]（1 个 id）+ 已缓存 token 数 + block_table。"""
    assert len(s.token_ids) == s.num_cached_tokens + 1, (
        f"decode invariant: len(token_ids)={len(s.token_ids)} vs "
        f"num_cached_tokens+1={s.num_cached_tokens + 1}"
    )
    temperature, top_p, top_k, min_p = _sampling_fields(s)
    return SeqSchedule(
        seq_id=s.seq_id,
        block_table=list(s.block_table),
        num_cached_tokens=s.num_cached_tokens,
        token_ids=[s.last_token],
        need_sample=False,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        min_p=min_p,
    )


def _prefill_schedule(s: Sequence) -> "SeqSchedule":
    """prefill 每序列只发本步 scheduled 的 token 切片，不含已缓存前缀与后续未调度部分。"""
    start = s.num_cached_tokens
    end = start + s.num_scheduled_tokens
    temperature, top_p, top_k, min_p = _sampling_fields(s)
    return SeqSchedule(
        seq_id=s.seq_id,
        block_table=list(s.block_table),
        num_cached_tokens=s.num_cached_tokens,
        token_ids=s.token_ids[start:end],
        need_sample=s.need_sample,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        min_p=min_p,
    )


class SeqSchedule(msgspec.Struct):
    """单条序列的调度增量（wire 格式）。

    相对完整 Sequence，只保留 build_decode / build_prefill 真正读取的字段：
    - token_ids：decode 为 [last_token]，prefill 为本步 scheduled 切片；
    - num_cached_tokens：本步前已缓存的 token 数（同时用于推导 position / cache_seqlen）；
    - block_table：完整块表（block 数远小于 token 数，不可再压缩）。
    """

    seq_id: int
    block_table: list[int]
    num_cached_tokens: int
    token_ids: list[int]
    need_sample: bool = False
    temperature: float = 0.0
    top_p: float | None = None
    top_k: int | None = None
    min_p: float | None = None


class SchedulerOutput(msgspec.Struct):
    """控制面 → 计算面的最小调度结果（wire 格式）。"""

    prefills: list[SeqSchedule] = field(default_factory=list)
    decodes: list[SeqSchedule] = field(default_factory=list)


class Batch:
    """常驻 GPU buffer：prefill / decode / CUDA Graph 共用同一套静态张量。

    每步只 copy_ 实际数据进 buffer，避免反复分配 GPU 张量；padding 区域保持
    捕获时的安全值（slot=-1、seqlen=0、block=0），使 CUDA Graph 重放无需额外清理。
    """

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

    @staticmethod
    def _cp(dst: torch.Tensor, data: list, dtype: torch.dtype) -> None:
        cpu = torch.tensor(data, dtype=dtype, pin_memory=True)
        dst.copy_(cpu, non_blocking=True)

    def build_decode(self, seqs: list[SeqSchedule]) -> None:
        bs = len(seqs)
        # decode 时 num_cached_tokens 已指向本步 token 的位置：position = num_cached_tokens，
        # cache_seqlen = num_cached_tokens + 1（含本步写入的 KV）。
        self._cp(self.input_ids[:bs], [s.token_ids[-1] for s in seqs], torch.int64)
        self._cp(
            self.positions[:bs], [s.num_cached_tokens for s in seqs], torch.int64
        )
        self._cp(
            self.slot_mapping[:bs],
            [self._decode_slot(s) for s in seqs],
            torch.int32,
        )
        self._cp(
            self.cache_seqlens[:bs],
            [s.num_cached_tokens + 1 for s in seqs],
            torch.int32,
        )
        self._fill_block_tables(seqs)
        self._fill_sampling(seqs)

    def build_prefill(self, seqs: list[SeqSchedule]) -> tuple[int, int, int, bool]:
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
        self._cp(self.input_ids[:n], ids, torch.int64)
        self._cp(self.positions[:n], positions, torch.int64)
        self._cp(self.slot_mapping[:n], slots, torch.int32)
        self._cp(self.cu_seqlens_q[: len(cu_q)], cu_q, torch.int32)
        self._cp(self.cu_seqlens_k[: len(cu_k)], cu_k, torch.int32)

        has_cache = cu_k[-1] > cu_q[-1]
        if has_cache:
            self._fill_block_tables(seqs)
        self._fill_sampling(seqs)
        return n, max_q, max_k, has_cache

    def _fill_block_tables(self, seqs: list[SeqSchedule]) -> None:
        width = self.max_blocks
        flat: list[int] = []
        for s in seqs:
            flat.extend(s.block_table)
            flat.extend([-1] * (width - len(s.block_table)))
        self._cp(self.block_tables[: len(seqs)].reshape(-1), flat, torch.int32)

    def _fill_sampling(self, seqs: list[SeqSchedule]) -> None:
        bs = len(seqs)
        vocab = self.vocab_size
        temps = [s.temperature for s in seqs]
        top_ps = [1.0 if s.top_p is None else s.top_p for s in seqs]
        top_ks = [
            vocab if (s.top_k is None or s.top_k <= 0) else min(s.top_k, vocab)
            for s in seqs
        ]
        min_ps = [0.0 if s.min_p is None else s.min_p for s in seqs]

        self._cp(self.temperatures[:bs], temps, torch.float32)
        self._cp(self.top_ps[:bs], top_ps, torch.float32)
        self._cp(self.top_ks[:bs], top_ks, torch.int32)
        self._cp(self.min_ps[:bs], min_ps, torch.float32)

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
