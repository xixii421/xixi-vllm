from __future__ import annotations

from dataclasses import dataclass, field

import torch

from .sequence import Sequence, slots_for_range


@dataclass
class ScheduledBatch:
    """调度器产出的本步批次。compute 层只消费这个显式接口，不触碰 Sequence 内部。"""

    prefills: list[Sequence] = field(default_factory=list)
    decodes: list[Sequence] = field(default_factory=list)

    def is_empty(self) -> bool:
        return not self.prefills and not self.decodes


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
    ) -> None:
        self.max_bs = max_bs
        self.max_tokens = max_tokens
        self.max_blocks = max_blocks
        self.vocab_size = vocab_size
        self.device = device

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
        """ CPU侧数据拷贝进GPU """
        cpu = torch.tensor(data, dtype=dtype, pin_memory=True)
        dst.copy_(cpu, non_blocking=True)

    def build_decode(self, seqs: list[Sequence]) -> None:
        """ 把一批要做 decode 的请求，填进已经提前分配好的静态 GPU 缓冲区 """
        bs = len(seqs)
        block_size = seqs[0].block_size
        self._cp(self.input_ids[:bs], [s.last_token for s in seqs], torch.int64) # 输入token id
        self._cp(self.positions[:bs], [len(s) - 1 for s in seqs], torch.int64) # 输入token pos
        self._cp(
            self.slot_mapping[:bs],
            [self._decode_slot(s, block_size) for s in seqs],
            torch.int32,
        ) # 输入token对应的物理地址
        self._cp(self.cache_seqlens[:bs], [len(s) for s in seqs], torch.int32) # 已缓存token数
        self._fill_block_tables(seqs) # 填充KVCache映射表
        self._fill_sampling(seqs)

    def build_prefill(self, seqs: list[Sequence]) -> tuple[int, int, int, bool]:
        """ 组装 Prefill ，把 CPU上的 Sequence 列表数据填入静态 GPU Buffer """
        block_size = seqs[0].block_size
        ids: list[int] = []
        positions: list[int] = []
        slots: list[int] = []
        cu_q = [0]
        cu_k = [0]
        max_q = 0 # 最大Q处理数量，用来分配Q侧需要的共享内存
        max_k = 0
        for s in seqs:
            start = s.num_cached_tokens
            n = s.num_scheduled_tokens
            end = start + n
            ids.extend(s.token_ids[start:end]) # 收集本轮全部输入token id
            positions.extend(range(start, end))
            slots.extend(slots_for_range(s.block_table, start, end, block_size))
            cu_q.append(cu_q[-1] + n) # 本轮新产生的Q累计数量
            cu_k.append(cu_k[-1] + end) # 本轮历史K累计数量
            max_q = max(max_q, n)
            max_k = max(max_k, end)

        n = len(ids)
        self._cp(self.input_ids[:n], ids, torch.int64)
        self._cp(self.positions[:n], positions, torch.int64)
        self._cp(self.slot_mapping[:n], slots, torch.int32)
        self._cp(self.cu_seqlens_q[: len(cu_q)], cu_q, torch.int32)
        self._cp(self.cu_seqlens_k[: len(cu_k)], cu_k, torch.int32)

        has_cache = cu_k[-1] > cu_q[-1]
        if has_cache: # 存在历史KVCache，则需要更新 GPU 缓存
            self._fill_block_tables(seqs)
        self._fill_sampling(seqs)
        return n, max_q, max_k, has_cache

    def _fill_block_tables(self, seqs: list[Sequence]) -> None:
        """ 将 CPU 侧 KVCache 表填充进 GPU """
        width = self.max_blocks
        flat: list[int] = []
        for s in seqs:
            flat.extend(s.block_table)
            flat.extend([-1] * (width - len(s.block_table))) # 补到最大宽度
        self._cp(self.block_tables[: len(seqs)].reshape(-1), flat, torch.int32)

    def _fill_sampling(self, seqs: list[Sequence]) -> None:
        """ 将每条请求的采样参数拷贝进 GPU """
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

        self.all_greedy = all(t <= 1e-5 for t in temps) # 贪心快速路径，跳过概率重分布
        self.any_greedy = any(t <= 1e-5 for t in temps)
        self.use_top_k = any(k < vocab for k in top_ks)
        self.use_top_p = any(p < 1.0 for p in top_ps)
        self.use_min_p = any(m > 0.0 for m in min_ps)
        self.max_top_k = max(top_ks) if self.use_top_k else 0

    @staticmethod
    def _decode_slot(seq: Sequence, block_size: int) -> int:
        """ 逻辑地址转物理地址 """
        n = len(seq)
        return seq.block_table[(n - 1) // block_size] * block_size + (n - 1) % block_size
