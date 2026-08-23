"""Data plane wire structures：调度产物与引擎输出（纯 CPU，无 torch 依赖）。

``Batch``（GPU 缓冲）已迁到 ``compute/batch.py``；本文件只保留控制面 → 计算面
的最小增量 wire 格式，以及 EngineCore → 门面的输出格式。
"""

from __future__ import annotations

from dataclasses import dataclass, field

import msgspec

from .sequence import Sequence


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

    prefills: list[SeqSchedule] = msgspec.field(default_factory=list)
    decodes: list[SeqSchedule] = msgspec.field(default_factory=list)


class EngineCoreOutput(msgspec.Struct):
    """EngineCore → 客户端（Engine）的单序列增量输出。

    每步对每条调度到的序列最多产出一个新 token（decode 或 prefill 采样），
    ``new_token_ids`` 为空表示本步未采样（partial prefill 或 max_tokens<=0）。
    """

    seq_id: int
    new_token_ids: list[int] = msgspec.field(default_factory=list)
    is_finished: bool = False
