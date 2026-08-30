from __future__ import annotations

import torch

from .layers.attention import AttnInputs, AttnMetadata, attention_context


class CUDAGraphRunner:
    """decode CUDA Graph：静态 buffer 来自 Batch，捕获后仅 memcpy 重放。"""

    def __init__(
        self,
        model,
        batch,
        meta: AttnMetadata,
        *,
        max_batch_size: int,
        hidden_size: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> None:
        self.model = model
        self.batch = batch
        self._meta = meta
        self.device = device
        self.max_bs = max_batch_size
        self.bs_list = self._make_bs_list(max_batch_size)
        self.graphs: dict[int, torch.cuda.CUDAGraph] = {} # 不同档位对应的CG对象
        self._graph_pool = None # CG专用显存池，多张graph共用一个pool
        self.outputs = torch.zeros(
            max_batch_size, hidden_size, dtype=dtype, device=device
        )

    def capture(self) -> None:
        torch.cuda.synchronize(self.device) # 等待GPU设备同步
        torch.cuda.empty_cache() # 释放空闲显存块
        self.batch.slot_mapping.fill_(-1)
        for bs in sorted(self.bs_list, reverse=True): # 从大batch往小batch遍历
            attn = AttnInputs(
                slot_mapping=self.batch.slot_mapping[:bs],
                is_prefill=False,
                cache_seqlens=self.batch.cache_seqlens[:bs],
                block_tables=self.batch.block_tables[:bs],
            )
            with attention_context(self._meta):
                self.outputs[:bs] = self.model(
                    self.batch.input_ids[:bs], self.batch.positions[:bs], attn
                ) # 热身，触发算子编译，初始化，缓存， 初始化kvcache状态

            graph = torch.cuda.CUDAGraph() # 开始捕获
            with attention_context(self._meta):
                with torch.cuda.graph(graph, pool=self._graph_pool):
                    self.outputs[:bs] = self.model(
                        self.batch.input_ids[:bs], self.batch.positions[:bs], attn
                    )
            if self._graph_pool is None:
                self._graph_pool = graph.pool()
            self.graphs[bs] = graph
        self.batch.slot_mapping.fill_(-1) # 清除录制脏值
        torch.cuda.synchronize(self.device)

    def can_use(self, bs: int) -> bool:
        return bs <= self.max_bs

    def replay(self, bs: int) -> torch.Tensor:
        """ 重放 """
        pbs = self._pad_bs(bs)
        self.graphs[pbs].replay()
        return self.outputs[:bs]

    def _pad_bs(self, bs: int) -> int:
        """ 给定真实运行的 batch 大小，向上取整 """
        for b in self.bs_list:
            if b >= bs:
                return b
        return bs

    @staticmethod
    def _make_bs_list(max_bs: int) -> list[int]:
        """ 生成录制档位，稀疏采样，减少显存开销 """
        bs = [1, 2, 4, 8] + list(range(16, max_bs + 1, 16))
        bs = [b for b in bs if b <= max_bs]
        if max_bs not in bs:
            bs.append(max_bs)
        return bs

    def destroy(self) -> None:
        self.graphs.clear()
        self._graph_pool = None
        del self.outputs
