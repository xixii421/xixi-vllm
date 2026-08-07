""" Piecewise Prefill CUDA Graph Runner

prefill 的 token 总数与每请求 seq 长度都在变，无法像 decode 那样按 batch
捕获一张静态图。piecewise 思路：按 (bs, L) 捕获一组「固定形状」的 CUDA Graph，
重放时把真实 batch 的每个 seq 统一 pad 到长度 L（真实 token 放前、padding 放后），
保持 qo_indptr/kv_indptr/last_page_len 与捕获完全一致（FlashInfer 的 kernel
launch grid 依赖这三个分布，改分布会重放出错误结果），只更新页号与数据。

  ┌── 捕获 (bs=2, L=256) ─────────────────────────────┐
  │ qo_indptr   [0,256,512]   kv_indptr [0,P,2P]     │
  │ last_page   [16,16]  (每 seq 全 256 token)        │
  └───────────────────────────────────────────────────┘
  ┌── 重放 真实 [150, 200] → pad 成 [256, 256] ────────┐
  │ 行 0..149   = seq0 真实 token（slot→真实页）       │
  │ 行 150..255 = seq0 padding（slot=-1, 不写 KV）     │
  │ 行 256..455 = seq1 真实 token                     │
  │ 行 456..511 = seq1 padding                       │
  │ 输出取 行 0..149 与 256..455                      │
  └───────────────────────────────────────────────────┘

限制：仅支持无前缀缓存的首 chunk（每 seq 的 KV 恰好等于本次 q，分布均匀）。
带缓存前缀的续 chunk（num_cached_tokens > 0）KV 分布不均，无法命中捕获图，
由上层回退到 eager/compile。

收益：prefill 每步省掉 ~N 次 Python→kernel launch（28 层 × 若干 kernel），
CPU 开销从每步 ~ms 级降到 ~µs 级。
"""

import gc
import torch
from typing import Any

from flashinfer import BatchPrefillWithPagedKVCacheWrapper

from ..layers.attention import AttentionMetadata, attention_context
from ..utils.context import forward_context


class PrefillCUDAGraphRunner:
    """prefill piecewise CUDA Graph：按 (bs, L) 捕获固定形状的图并重放。"""

    def __init__(
        self,
        model: torch.nn.Module,
        *,
        max_batch_size: int,
        max_model_len: int,
        max_batched_tokens: int,
        block_size: int,
        hidden_size: int,
        num_qo_heads: int,
        num_kv_heads: int,
        head_dim: int,
        device: torch.device,
        dtype: torch.dtype,
        backend: Any,
        kv_cache_pool: torch.Tensor,
        dummy_page_id: int,
        workspace_bytes: int = 0,
    ) -> None:
        self.model = model
        self.device = device
        self.block_size = block_size
        self.dtype = dtype
        self.backend = backend
        self.kv_cache_pool = kv_cache_pool
        self.dummy_page_id = int(dummy_page_id)
        self.hidden_size = hidden_size
        self._ws_override = workspace_bytes

        self.pieces = self._make_pieces(
            max_batch_size, max_model_len, max_batched_tokens
        )

        self._buffers: dict[tuple, dict[str, torch.Tensor]] = {}
        self._metas: dict[tuple, AttentionMetadata] = {}
        self.graphs: dict[tuple, torch.cuda.CUDAGraph] = {}
        self._captured = False

        self._sm_scale = head_dim ** -0.5
        self._non_blocking = bool(getattr(backend.cfg, "non_blocking", True))
        self._heads = (num_qo_heads, num_kv_heads, head_dim)

    def _make_pieces(self, max_bs: int, max_len: int, max_tokens: int) -> list[tuple]:
        """生成 (bs, L) piece 集合：bs×L ≤ max_batched_tokens。"""
        bs_list = [b for b in (1, 2, 4, 8, 16) if b <= max_bs]
        L_list = [L for L in (128, 256, 512, 1024, 2048, 4096, 8192) if L <= max_len]
        pieces = [(bs, L) for bs in bs_list for L in L_list if bs * L <= max_tokens]
        # 从大到小捕获：最大 piece 先占显存池，后续复用避免碎片
        return sorted(pieces, key=lambda p: (p[0] * p[1], p[0]), reverse=True)

    def _measure_workspace(self, piece: tuple, bs_pages: int) -> int:
        """估算 FlashInfer plan 需要的 workspace 字节数。

        plan 在 workspace 内按对齐偏移一次性分配多个临时 buffer
        （batch_prefill_tmp_s / tmp_v 等），总量约 M×16KB + 28MB 固定开销。
        用保守公式（×1.5 余量）直接计算，避免逐 piece 探测的解析脆弱性。
        """
        if self._ws_override > 0:
            return self._ws_override
        M = piece[0] * piece[1]
        return max(64 * 1024 * 1024, M * 24 * 1024)

    def capture(self) -> None:
        if self._captured:
            return
        torch.cuda.synchronize(self.device)
        torch.cuda.empty_cache()

        graph_pool = None
        for piece in self.pieces:
            self._make_piece_buffers(piece)
            self._run_forward(piece)              # warmup（编译 Triton kernel）
            torch.cuda.synchronize(self.device)

            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, pool=graph_pool):
                self._run_forward(piece)
            if graph_pool is None:
                graph_pool = graph.pool()
            self.graphs[piece] = graph
            print(f"  prefill CG (bs={piece[0]}, L={piece[1]}) captured")

        self.kv_cache_pool.zero_()
        torch.cuda.synchronize(self.device)
        self._captured = True

    def _run_forward(self, piece: tuple) -> None:
        bs, L = piece
        bufs = self._buffers[piece]
        meta = self._metas[piece]
        with forward_context(
            is_prefill=True,
            seqs=[],
            slot_mapping=bufs["slot_mapping"],
            cu_seqlens_q=bufs["qo_indptr"],
            cu_seqlens_k=bufs["kv_indptr"],
            max_seqlen_q=L,
            max_seqlen_k=L,
            block_tables=None,
        ):
            with attention_context(meta):
                bufs["outputs"][:] = self.model(
                    input_ids=bufs["input_ids"][: bs * L],
                    positions=bufs["positions"][: bs * L],
                )

    def _make_piece_buffers(self, piece: tuple) -> None:
        bs, L = piece
        M = bs * L
        pages = (L + self.block_size - 1) // self.block_size

        bufs = {
            "input_ids": torch.zeros(M, dtype=torch.int64, device=self.device),
            "positions": torch.zeros(M, dtype=torch.int64, device=self.device),
            "slot_mapping": torch.full((M,), -1, dtype=torch.int32, device=self.device),
            "qo_indptr": torch.zeros(bs + 1, dtype=torch.int32, device=self.device),
            "kv_indptr": torch.zeros(bs + 1, dtype=torch.int32, device=self.device),
            "kv_indices": torch.full((bs * pages,), self.dummy_page_id, dtype=torch.int32, device=self.device),
            "kv_last_page": torch.zeros(bs, dtype=torch.int32, device=self.device),
            "outputs": torch.zeros(M, self.hidden_size, dtype=self.dtype, device=self.device),
            "pages": pages,
        }
        # 捕获固定分布：每 seq 恰好 L 个 token
        bufs["qo_indptr"].copy_(torch.arange(0, (bs + 1) * L, L, dtype=torch.int32))
        bufs["kv_indptr"].copy_(torch.arange(0, (bs + 1) * pages, pages, dtype=torch.int32))
        bufs["kv_last_page"].fill_((L - 1) % self.block_size + 1)

        # FlashInfer plan 需要的工作空间随模型 heads 与 M 放大（4B/8B 远超
        # 0.6B）。用「溢出则加倍重试」自适应，避免硬编码公式失效。
        ws = self._measure_workspace(piece, pages)
        while True:
            try:
                bufs["workspace"] = torch.empty(
                    ws, dtype=torch.uint8, device=self.device
                )
                wrapper = self._build_wrapper(piece, bufs)
                break
            except RuntimeError as e:
                if "Buffer overflow" not in str(e):
                    raise
                ws *= 2
                torch.cuda.empty_cache()
                continue

        self._buffers[piece] = bufs
        self._metas[piece] = AttentionMetadata(
            wrapper=wrapper,
            kv_cache_pool=self.kv_cache_pool,
            slot_mapping=bufs["slot_mapping"],
        )

    def _build_wrapper(self, piece: tuple, bufs: dict[str, torch.Tensor]):
        """按 piece 创建并 plan FlashInfer prefill wrapper（供捕获/重放复用）。"""
        bs, L = piece
        n_qh, n_kvh, hd = self._heads
        wrapper = BatchPrefillWithPagedKVCacheWrapper(
            bufs["workspace"],
            "NHD",
            use_cuda_graph=True,
            qo_indptr_buf=bufs["qo_indptr"],
            paged_kv_indptr_buf=bufs["kv_indptr"],
            paged_kv_indices_buf=bufs["kv_indices"],
            paged_kv_last_page_len_buf=bufs["kv_last_page"],
        )
        wrapper.plan(
            qo_indptr=bufs["qo_indptr"],
            paged_kv_indptr=bufs["kv_indptr"],
            paged_kv_indices=bufs["kv_indices"],
            paged_kv_last_page_len=bufs["kv_last_page"],
            num_qo_heads=n_qh,
            num_kv_heads=n_kvh,
            head_dim_qk=hd,
            head_dim_vo=hd,
            page_size=self.block_size,
            causal=True,
            sm_scale=self._sm_scale,
            q_data_type=self.dtype,
            kv_data_type=self.dtype,
            non_blocking=self._non_blocking,
        )
        return wrapper

    def can_use(self, bs: int, seq_lens: list[int]) -> tuple | None:
        """找可用 piece 中 bs×L 最小者：bs_piece ≥ bs 且 L_piece ≥ max(seq_lens)。"""
        if not self._captured:
            return None
        need_len = max(seq_lens) if seq_lens else 0
        best = None
        best_cost = None
        for piece in self.pieces:
            pbs, L = piece
            if pbs >= bs and L >= need_len:
                cost = pbs * L
                if best_cost is None or cost < best_cost:
                    best, best_cost = piece, cost
        return best

    def replay(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        slot_mapping: torch.Tensor,
        seq_lens: list[int],
        block_tables: list[list[int]],
    ) -> torch.Tensor:
        """重放 piecewise prefill graph，返回真实 token 对应的 hidden 行。"""
        bs = len(seq_lens)
        piece = self.can_use(bs, seq_lens)
        assert piece is not None, "no piece covers this prefill batch"
        pbs, L = piece
        bufs = self._buffers[piece]
        pages = bufs["pages"]

        # 1) 输入数据：seq i 真实 token 放行 [i*L, i*L+len_i)，padding 行保持 0/-1
        in_ids, pos, slot = bufs["input_ids"], bufs["positions"], bufs["slot_mapping"]
        in_ids.zero_()
        pos.zero_()
        slot.fill_(-1)
        off = 0
        for i in range(bs):
            n = seq_lens[i]
            lo = i * L
            in_ids[lo : lo + n].copy_(input_ids[off : off + n])
            pos[lo : lo + n].copy_(positions[off : off + n])
            slot[lo : lo + n].copy_(slot_mapping[off : off + n])
            off += n
        assert off == input_ids.numel(), "input_ids size mismatch"

        # 2) 页号：seq i 真实页放入 [i*pages, i*pages+real_pages)，余下保持 dummy
        idx_buf = bufs["kv_indices"]
        idx_buf.fill_(self.dummy_page_id)
        for i in range(bs):
            base = i * pages
            real_pages = (seq_lens[i] + self.block_size - 1) // self.block_size
            if real_pages > 0:
                tab = torch.tensor(
                    block_tables[i][:real_pages], dtype=torch.int32, device=self.device
                )
                idx_buf[base : base + real_pages].copy_(tab)

        # 3) 重放
        meta = self._metas[piece]
        with forward_context(
            is_prefill=True,
            seqs=[],
            slot_mapping=slot,
            cu_seqlens_q=bufs["qo_indptr"],
            cu_seqlens_k=bufs["kv_indptr"],
            max_seqlen_q=L,
            max_seqlen_k=L,
            block_tables=None,
        ):
            with attention_context(meta):
                self.graphs[piece].replay()

        # 4) 取回真实 token 行（跳过每 seq 的 padding 区）
        out = bufs["outputs"]
        rows = [out[i * L : i * L + seq_lens[i]] for i in range(bs)]
        return torch.cat(rows, dim=0)

    def destroy(self) -> None:
        self.graphs.clear()
        self._metas.clear()
        self._buffers.clear()
        self._captured = False
        gc.collect()
        torch.cuda.empty_cache()
