from __future__ import annotations

import gc

import torch
from flashinfer import BatchDecodeWithPagedKVCacheWrapper

from ..layers.attention import AttentionMetadata, AttnInputs, attention_context
from ..layers.attention_backend import FlashInferBackend


class CUDAGraphRunner:
    def __init__(
        self,
        model,
        *,
        max_batch_size: int,
        max_model_len: int,
        hidden_size: int,
        backend: FlashInferBackend,
        kv_cache_pool: torch.Tensor,
        dummy_page_id: int,
        device: torch.device,
        dtype: torch.dtype,
        workspace_bytes: int | None = None,
    ) -> None:
        self.model = model
        self.device = device
        self.dtype = dtype
        self.kv_cache_pool = kv_cache_pool
        self._non_blocking = backend.cfg.non_blocking

        # 已捕获的 CUDA Graph 按 padded batch size 存储。
        # _metas 保存各 batch size 对应的 attention 元数据，用于 capture/replay 上下文。
        # _graph_pool 由首个捕获的图创建，后续图复用同一 pool，以稳定地址并共享显存。
        self.graphs: dict[int, torch.cuda.CUDAGraph] = {}
        self._metas: dict[int, AttentionMetadata] = {}
        self._captured = False
        self._graph_pool = None

        self.backend = backend
        self.dummy_page_id = int(dummy_page_id)
        self.block_size = backend.cfg.block_size
        self.max_bs = max_batch_size
        self.bs_list = self._make_bs_list(max_batch_size)
        self._max_blocks = (max_model_len + self.block_size - 1) // self.block_size

        # 模型输入/输出静态 buffer：CUDA Graph 捕获后地址固定，运行时原地更新。
        workspace = workspace_bytes or backend.cfg.workspace_bytes
        self.workspace_buffer = torch.empty(workspace, dtype=torch.uint8, device=device)
        self.input_ids = torch.zeros(max_batch_size, dtype=torch.int64, device=device)
        self.positions = torch.zeros(max_batch_size, dtype=torch.int64, device=device)
        self.slot_mapping = torch.full(
            (max_batch_size,), -1, dtype=torch.int32, device=device
        )
        self.outputs = torch.zeros(max_batch_size, hidden_size, dtype=dtype, device=device)
        self._dummy_cu_seqlens = torch.zeros(2, dtype=torch.int32, device=device)

        # FlashInfer paged KV 索引双缓冲：
        # device buffer 供图内 kernel 读取；pinned host buffer 供 CPU 重组后异步拷贝到 device。
        # 每个 batch size 档位单独维护一组 buffer。
        self._cg_bufs: dict[int, dict[str, torch.Tensor]] = {
            bs: {
                "indptr": torch.zeros(bs + 1, dtype=torch.int32, device=device),
                "indices": torch.zeros(bs * self._max_blocks, dtype=torch.int32, device=device),
                "last_page": torch.zeros(bs, dtype=torch.int32, device=device),
            }
            for bs in self.bs_list
        }
        self._host_bufs: dict[int, dict[str, torch.Tensor]] = {
            bs: {
                "indptr": torch.zeros(bs + 1, dtype=torch.int32, pin_memory=True),
                "last_page": torch.ones(bs, dtype=torch.int32, pin_memory=True),
                "indices": torch.zeros(bs * self._max_blocks, dtype=torch.int32, pin_memory=True),
            }
            for bs in self.bs_list
        }

        self.cg_wrappers: dict[int, BatchDecodeWithPagedKVCacheWrapper] = {}
        self._sm_scale = backend.cfg.head_dim**-0.5
        self._use_tensor_cores = backend.cfg.use_tensor_cores

    # ------------------------------------------------------------------ #
    # 捕获
    # ------------------------------------------------------------------ #

    def capture(self) -> None:
        """捕获所有预定义 batch size 的 CUDA Graph。"""
        if self._captured:
            return
        torch.cuda.synchronize(self.device)
        torch.cuda.empty_cache()
        self.slot_mapping.fill_(-1)
        # 按 batch size 从大到小捕获：最大图先建立共享 pool，小图复用显存，降低碎片。
        for bs in sorted(self.bs_list, reverse=True):
            self._capture_bs(bs)
        self.kv_cache_pool.zero_()
        torch.cuda.synchronize(self.device)
        self._captured = True

    def _capture_one(self, key: int, run_fn) -> torch.cuda.CUDAGraph:
        """将 run_fn 固化为 CUDA Graph。

        首个图创建共享 pool，后续图复用该 pool，保证地址稳定并减少显存碎片。
        """
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, pool=self._graph_pool):
            run_fn()
        if self._graph_pool is None:
            self._graph_pool = graph.pool()
        self.graphs[key] = graph
        return graph

    def _capture_bs(self, bs: int) -> None:
        """为单个 batch size 创建 FlashInfer wrapper、执行 warmup，并捕获 CUDA Graph。"""
        wrapper = self._make_cg_wrapper(bs)
        self.cg_wrappers[bs] = wrapper
        self._metas[bs] = AttentionMetadata(
            wrapper=wrapper,
            kv_cache_pool=self.kv_cache_pool,
        )

        def _run_capture() -> None:
            self.outputs[:bs] = self.model(
                input_ids=self.input_ids[:bs],
                positions=self.positions[:bs],
                attn=AttnInputs(
                    slot_mapping=self.slot_mapping[:bs],
                    cu_seqlens=self._dummy_cu_seqlens,
                    max_seqlen=0,
                    use_fa2=False,
                ),
            )

        # 先 warmup，使 FlashInfer 在图外完成动态 workspace 分配；图内不允许分配。
        with attention_context(self._metas[bs]):
            _run_capture()
        # 正式捕获固化 kernel 序列。
        with attention_context(self._metas[bs]):
            self._capture_one(bs, _run_capture)

    def _make_cg_wrapper(self, bs: int) -> BatchDecodeWithPagedKVCacheWrapper:
        """创建指定 batch size 的 FlashInfer decode wrapper，并用 dummy 索引完成 plan。"""
        c = self.backend.cfg
        bufs = self._cg_bufs[bs]
        host = self._host_bufs[bs]

        wrapper = BatchDecodeWithPagedKVCacheWrapper(
            self.workspace_buffer,
            "NHD",
            use_tensor_cores=self._use_tensor_cores,
            # 启用 CUDA Graph 模式并禁用 FlashInfer 内部动态分配，仅使用预分配 buffer。
            use_cuda_graph=True,
            paged_kv_indptr_buffer=bufs["indptr"],
            paged_kv_indices_buffer=bufs["indices"],
            paged_kv_last_page_len_buffer=bufs["last_page"],
        )

        # 填充 dummy 索引，使捕获期间 FlashInfer 的读取路径稳定有效。
        host["indptr"].copy_(torch.arange(0, bs + 1, dtype=torch.int32))
        host["last_page"].fill_(1)
        host["indices"][:bs].fill_(self.dummy_page_id)
        bufs["indices"][:bs].fill_(self.dummy_page_id)
        self._plan_cg(wrapper, host["indptr"], bufs["indices"][:bs], host["last_page"])
        return wrapper

    def _plan_cg(
        self,
        wrapper: BatchDecodeWithPagedKVCacheWrapper,
        indptr: torch.Tensor,
        indices: torch.Tensor,
        last_page_len: torch.Tensor,
    ) -> None:
        """将 wrapper 绑定到给定的 paged KV CSR 索引 buffer。"""
        c = self.backend.cfg
        wrapper.plan(
            indptr=indptr,
            indices=indices,
            last_page_len=last_page_len,
            num_qo_heads=c.num_qo_heads,
            num_kv_heads=c.num_kv_heads,
            head_dim=c.head_dim,
            page_size=c.block_size,
            sm_scale=self._sm_scale,
            q_data_type=c.dtype,
            kv_data_type=c.dtype,
            non_blocking=self._non_blocking,
        )

    # ------------------------------------------------------------------ #
    # 重放（热路径）
    # ------------------------------------------------------------------ #

    def can_use(self, bs: int) -> bool:
        """判断当前 batch size 是否可用已捕获图。"""
        return self._captured and bs <= self.max_bs

    def _pad_bs(self, bs: int) -> int:
        """将实际 batch size 向上对齐到最近的已捕获 batch size 档位。"""
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
        """执行一次 CUDA Graph 重放，返回实际有效 batch 的输出。"""
        bs = input_ids.shape[0]
        assert bs == len(seq_lens) == len(block_tables), "Batch size mismatch"
        pbs = self._pad_bs(bs)

        # 原地更新静态输入 buffer，不改变图捕获时的地址。
        self.input_ids[:bs] = input_ids
        self.positions[:bs] = positions
        self.slot_mapping[:bs] = slot_mapping
        # 仅重置 padding 区间的 slot_mapping，避免全量 fill。
        if pbs > bs:
            self.slot_mapping[bs:pbs] = -1

        # 根据真实 KV 页表重建 FlashInfer 索引并重新 plan。
        self._replan_cg(seq_lens, block_tables)

        # 重放已捕获图，执行固化后的 kernel 序列。
        with attention_context(self._metas[pbs]):
            self.graphs[pbs].replay()

        return self.outputs[:bs]

    def _replan_cg(
        self,
        seq_lens: list[int],
        block_tables: list[list[int]],
    ) -> None:
        """根据真实 seq_lens 和 block_tables 重建 FlashInfer CSR 索引。"""
        real_bs = len(seq_lens)
        pbs = self._pad_bs(real_bs)
        c = self.backend.cfg
        block = self.block_size
        bufs = self._cg_bufs[pbs]
        host = self._host_bufs[pbs]
        dummy = self.dummy_page_id

        indptr_np = host["indptr"].numpy()
        last_np = host["last_page"].numpy()
        idx_np = host["indices"].numpy()

        # 真实序列：将每个序列的 block_table 按页写入 indices，并用 indptr 记录 CSR 页边界。
        acc = 0
        indptr_np[0] = 0
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

        # padding 序列：每个 padding 序列使用单页 dummy page，保持图结构完整。
        for i in range(real_bs, pbs):
            idx_np[acc] = dummy
            acc += 1
            indptr_np[i + 1] = acc
            last_np[i] = 1

        # 将 host 索引异步拷贝到 device 静态 buffer，然后重新绑定 wrapper。
        bufs["indices"][:acc].copy_(host["indices"][:acc], non_blocking=self._non_blocking)
        self._plan_cg(
            self.cg_wrappers[pbs],
            host["indptr"],
            bufs["indices"][:acc],
            host["last_page"],
        )

    # ------------------------------------------------------------------ #
    # 清理
    # ------------------------------------------------------------------ #

    def _release(self) -> None:
        """释放大型 buffer 引用，帮助回收 GPU/Python 资源。"""
        self.cg_wrappers.clear()
        self.workspace_buffer = None
        self.input_ids = None
        self.positions = None
        self.slot_mapping = None
        self.outputs = None
        self._cg_bufs.clear()
        self._host_bufs.clear()

    def destroy(self) -> None:
        """释放所有 CUDA Graph、wrapper 和 buffer，并重置状态。"""
        self.graphs.clear()
        self._metas.clear()
        self._graph_pool = None
        self._captured = False
        self._release()
        gc.collect()
        torch.cuda.empty_cache()

    @staticmethod
    def _make_bs_list(max_bs: int) -> list[int]:
        """生成离散 batch size 网格：小 batch 更密集，大 batch 更稀疏。"""
        bs = [1, 2, 4, 8] + list(range(16, max_bs + 1, 16))
        bs = [b for b in bs if b <= max_bs]
        if max_bs not in bs:
            bs.append(max_bs)
        return bs