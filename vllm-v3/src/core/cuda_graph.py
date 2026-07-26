from __future__ import annotations

import gc

import torch
from flashinfer import BatchDecodeWithPagedKVCacheWrapper

from ..layers.attention import AttentionMetadata, attention_context
from ..utils.context import forward_context


class CUDAGraphRunner:
    def __init__(
        self,
        model,
        max_batch_size: int,
        max_model_len: int,
        block_size: int,
        hidden_size: int,
        num_kv_heads: int,
        head_dim: int,
        device: torch.device,
        dtype: torch.dtype,
        *,
        backend: object,
        kv_cache_pool: torch.Tensor,
        dummy_page_id: int,
        workspace_bytes: int = 256 * 1024 * 1024,
    ) -> None:
        self.model = model
        self.device = device
        self.max_bs = max_batch_size
        self.block_size = block_size
        self.dtype = dtype
        self.backend = backend
        self.kv_cache_pool = kv_cache_pool
        self.dummy_page_id = int(dummy_page_id)
        self.bs_list = self._make_bs_list(max_batch_size)

        self._max_blocks = (max_model_len + block_size - 1) // block_size

        # 1. 预分配模型 I/O 静态 Buffer（CUDA Graph 强制要求内存地址与形状绝对固定）
        self.workspace_buffer = torch.empty(
            workspace_bytes, dtype=torch.uint8, device=device
        )
        self.input_ids = torch.zeros(max_batch_size, dtype=torch.int64, device=device)
        self.positions = torch.zeros(max_batch_size, dtype=torch.int64, device=device)
        self.slot_mapping = torch.full(
            (max_batch_size,), -1, dtype=torch.int32, device=device
        )
        self.outputs = torch.zeros(max_batch_size, hidden_size, dtype=dtype, device=device)

        # 2. 预分配 FlashInfer 所需的 Paged KV Cache 索引 Buffer（双缓冲机制）
        # Device 端 Buffer: Graph 内部直接读取，地址在生命周期内保持固定
        self._cg_bufs: dict[int, dict[str, torch.Tensor]] = {
            bs: {
                "indptr": torch.zeros(bs + 1, dtype=torch.int32, device=device),
                "indices": torch.zeros(bs * self._max_blocks, dtype=torch.int32, device=device),
                "last_page": torch.zeros(bs, dtype=torch.int32, device=device),
            }
            for bs in self.bs_list
        }
        # Host 端 Pinned Memory Buffer: 供 CPU 极速重组动态索引后，通过 DMA 异步零拷贝至 Device
        self._host_bufs: dict[int, dict[str, torch.Tensor]] = {
            bs: {
                "indptr": torch.zeros(bs + 1, dtype=torch.int32, pin_memory=True),
                "last_page": torch.ones(bs, dtype=torch.int32, pin_memory=True),
                "indices": torch.zeros(bs * self._max_blocks, dtype=torch.int32, pin_memory=True),
            }
            for bs in self.bs_list
        }

        self.cg_wrappers: dict[int, BatchDecodeWithPagedKVCacheWrapper] = {}
        self.graphs: dict[int, torch.cuda.CUDAGraph] = {}
        self._metas: dict[int, AttentionMetadata] = {}
        self._captured = False

        self._sm_scale = head_dim**-0.5
        self._use_tensor_cores = bool(getattr(backend.cfg, "use_tensor_cores", True))
        self._non_blocking = bool(getattr(backend.cfg, "non_blocking", True))

    def capture(self) -> None:
        """捕获所有预定义 batch size 的 CUDA Graph"""
        if self._captured:
            return

        # 环境净化：确保录制起点显存状态纯净，无历史残留干扰
        torch.cuda.synchronize(self.device)
        torch.cuda.empty_cache()
        self.slot_mapping.fill_(-1)

        graph_pool = None
        # 降序捕获策略：优先为最大 batch size 分配显存池，后续较小 graph 强制复用该池，避免显存碎片化
        for bs in sorted(self.bs_list, reverse=True):
            cg_wrapper = self._make_cg_wrapper(bs)
            self.cg_wrappers[bs] = cg_wrapper

            meta = AttentionMetadata(
                wrapper=cg_wrapper,
                kv_cache_pool=self.kv_cache_pool,
                slot_mapping=self.slot_mapping[:bs],
            )
            self._metas[bs] = meta
            graph = torch.cuda.CUDAGraph()

            # 阶段 A: Warmup Run（热身运行）。强制底层算子（如 FlashInfer）完成动态 Workspace 分配，使其进入静态就绪状态
            with forward_context(is_prefill=False, seqs=[], slot_mapping=self.slot_mapping[:bs]):
                with attention_context(meta):
                    self.outputs[:bs] = self.model(
                        input_ids=self.input_ids[:bs],
                        positions=self.positions[:bs],
                    )

            # 阶段 B: Capture Run（正式录制）。仅记录 Kernel 调用序列与固定内存地址的依赖关系，不实际执行计算
            with forward_context(is_prefill=False, seqs=[], slot_mapping=self.slot_mapping[:bs]):
                with attention_context(meta):
                    with torch.cuda.graph(graph, pool=graph_pool):
                        self.outputs[:bs] = self.model(
                            input_ids=self.input_ids[:bs],
                            positions=self.positions[:bs],
                        )

            if graph_pool is None:
                graph_pool = graph.pool()
            self.graphs[bs] = graph

        # 状态固化与收尾：清零 KV Cache 防止脏数据污染，标记捕获完成
        self.kv_cache_pool.zero_()
        torch.cuda.synchronize(self.device)
        self._captured = True

    def _make_cg_wrapper(self, bs: int) -> BatchDecodeWithPagedKVCacheWrapper:
        """为指定 batch size 创建并初始化 FlashInfer Wrapper"""
        c = self.backend.cfg
        bufs = self._cg_bufs[bs]
        host = self._host_bufs[bs]

        wrapper = BatchDecodeWithPagedKVCacheWrapper(
            self.workspace_buffer,
            "NHD",
            use_tensor_cores=self._use_tensor_cores,
            use_cuda_graph=True,  # 关键：告知 FlashInfer 禁用内部动态内存分配，严格使用预分配 Buffer
            paged_kv_indptr_buffer=bufs["indptr"],
            paged_kv_indices_buffer=bufs["indices"],
            paged_kv_last_page_len_buffer=bufs["last_page"],
        )

        # 注入 Dummy 数据并执行 plan，确保 capture 时计算图的数据流路径有效且稳定
        host["indptr"].copy_(torch.arange(0, bs + 1, dtype=torch.int32))
        host["last_page"].fill_(1)
        host["indices"][:bs].fill_(self.dummy_page_id)
        bufs["indices"][:bs].fill_(self.dummy_page_id)

        wrapper.plan(
            indptr=host["indptr"],
            indices=bufs["indices"][:bs],
            last_page_len=host["last_page"],
            num_qo_heads=c.num_qo_heads,
            num_kv_heads=c.num_kv_heads,
            head_dim=c.head_dim,
            page_size=c.block_size,
            sm_scale=self._sm_scale,
            q_data_type=c.dtype,
            kv_data_type=c.dtype,
            non_blocking=self._non_blocking,
        )
        return wrapper

    def can_use(self, bs: int) -> bool:
        """检查当前 batch size 是否满足已捕获 Graph 的使用条件"""
        return self._captured and bs <= self.max_bs

    def _pad_bs(self, bs: int) -> int:
        """将实际 batch size 向上对齐至预捕获的离散 bs_list，以复用已有计算图"""
        for b in self.bs_list:
            if b >= bs:
                return b
        return bs

    def replay(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        slot_mapping: torch.Tensor,
        *,
        seq_lens: list[int],
        block_tables: list[list[int]],
    ) -> torch.Tensor:
        """执行实际的推理重放（核心热路径）"""
        bs = input_ids.shape[0]
        assert bs == len(seq_lens) == len(block_tables), "Batch size mismatch"
        pbs = self._pad_bs(bs)

        # 步骤 1: 原地更新输入数据至静态 Buffer（保持内存地址不变）
        self.input_ids[:bs] = input_ids
        self.positions[:bs] = positions

        # 仅重置未使用的 padding 部分，避免全量 fill_ 造成不必要的显存带宽浪费
        if bs < self.max_bs:
            self.slot_mapping[bs:] = -1
        self.slot_mapping[:bs] = slot_mapping

        # 步骤 2: 根据真实的 block_tables 极速重组 FlashInfer 索引并异步注入
        self._replan_cg(pbs, bs, seq_lens, block_tables)

        # 步骤 3: 触发 CUDA Graph 重放，GPU 以零调度开销执行固化后的 Kernel 序列
        meta = self._metas[pbs]
        with forward_context(is_prefill=False, seqs=[], slot_mapping=self.slot_mapping[:pbs]):
            with attention_context(meta):
                self.graphs[pbs].replay()

        # 步骤 4: 截取并返回有效 batch size 的输出结果
        return self.outputs[:bs]

    def _replan_cg(
        self,
        pbs: int,
        real_bs: int,
        seq_lens: list[int],
        block_tables: list[list[int]],
    ) -> None:
        """动态索引极速重组与算子元数据重绑定"""
        c = self.backend.cfg
        block = self.block_size
        bufs = self._cg_bufs[pbs]
        host = self._host_bufs[pbs]
        dummy = self.dummy_page_id

        # 获取 Pinned Memory 的 NumPy 视图，利用连续内存布局在 CPU 端极速拼接 CSR 格式索引
        indptr_np = host["indptr"].numpy()
        last_np = host["last_page"].numpy()
        idx_np = host["indices"].numpy()

        acc = 0
        indptr_np[0] = 0

        # 填充真实序列的动态索引
        for i in range(real_bs):
            s = seq_lens[i]
            nb = (s + block - 1) // block
            table = block_tables[i]

            if nb > len(table):
                raise RuntimeError(
                    f"block_table shorter than needed pages: need {nb}, have {len(table)}"
                )

            idx_np[acc : acc + nb] = table[:nb]
            acc += nb
            indptr_np[i + 1] = acc
            last_np[i] = (s - 1) % block + 1

        # 填充 Padding 序列的 Dummy 索引，保持图结构完整性
        for i in range(real_bs, pbs):
            idx_np[acc] = dummy
            acc += 1
            indptr_np[i + 1] = acc
            last_np[i] = 1

        # 异步非阻塞拷贝：DMA 绕过 CPU 直接将重组后的索引搬运至 GPU 静态 Buffer
        bufs["indices"][:acc].copy_(host["indices"][:acc], non_blocking=self._non_blocking)

        # 算子元数据重绑定：更新 FlashInfer 内部指针，使其指向固定地址中刚刚覆盖的最新数据
        self.cg_wrappers[pbs].plan(
            indptr=host["indptr"],
            indices=bufs["indices"][:acc],
            last_page_len=host["last_page"],
            num_qo_heads=c.num_qo_heads,
            num_kv_heads=c.num_kv_heads,
            head_dim=c.head_dim,
            page_size=block,
            sm_scale=self._sm_scale,
            q_data_type=c.dtype,
            kv_data_type=c.dtype,
            non_blocking=self._non_blocking,
        )

    def destroy(self) -> None:
        """显式释放所有资源，防止显存泄漏"""
        self.graphs.clear()
        self.cg_wrappers.clear()
        self._metas.clear()
        self._captured = False

        # 显式置 None 协助 Python GC 与 PyTorch C++ 后端立即回收显存
        self.workspace_buffer = None
        self.input_ids = None
        self.positions = None
        self.slot_mapping = None
        self.outputs = None
        self._cg_bufs.clear()
        self._host_bufs.clear()

        gc.collect()
        torch.cuda.empty_cache()

    @staticmethod
    def _make_bs_list(max_bs: int) -> list[int]:
        """生成离散的 batch size 列表，在图捕获时间开销与运行时覆盖率之间取得平衡"""
        bs = [1, 2, 4, 8] + list(range(16, max_bs + 1, 16))
        bs = [b for b in bs if b <= max_bs]
        if max_bs not in bs:
            bs.append(max_bs)
        return sorted(set(bs))