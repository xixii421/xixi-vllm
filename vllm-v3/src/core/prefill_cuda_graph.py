import gc
import torch
from typing import Any

from flashinfer import BatchPrefillWithPagedKVCacheWrapper

from ..layers.attention import AttentionMetadata, attention_context
from ..utils.context import forward_context


class PrefillCUDAGraphRunner:

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
        cg_max_tokens: int = 1536,
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
            max_batch_size, max_model_len, max_batched_tokens, cg_max_tokens
        )
        self._ws_bytes = self._workspace_bytes()
        self._ws: torch.Tensor | None = None

        self._buffers: dict[tuple, dict[str, torch.Tensor]] = {}
        self._metas: dict[tuple, AttentionMetadata] = {}
        self.graphs: dict[tuple, torch.cuda.CUDAGraph] = {}
        self._captured = False

        self._sm_scale = head_dim ** -0.5
        self._non_blocking = bool(getattr(backend.cfg, "non_blocking", True))
        self._heads = (num_qo_heads, num_kv_heads, head_dim)

    def _make_pieces(
        self, max_bs: int, max_len: int, max_tokens: int, cg_max_tokens: int
    ) -> list[tuple]:
        """生成 (bs, L) piece 集合：bs×L ≤ min(max_batched_tokens, cg_max_tokens)。

        cg_max_tokens 上限很关键：它同时约束 ① 捕获显存（每个 piece 一个 CUDA graph
        私有 pool，4B/8B 上 8192-token 网格直接 OOM）和 ② 收益区间（>1536 token 的
        长 prefill 已 GPU-bound，padded CG 反而不如 eager，无需捕获）。

        bs=1 用中点网格（128/192/256/384/512/576/…），把「pad 到下一个 2 的幂」的
        最坏浪费从 ~1.86x 压到 ~1.25x（短文本 TTFT 最多省 ~40%）。
        bs≥2 按 bs 定制网格，在盈利上界内加密的同时控制 piece 数量（显存/捕获时间）。
        """
        cap = min(max_tokens, cg_max_tokens)
        bs_list = [b for b in (1, 2, 4, 8, 16) if b <= max_bs]
        fine_L = (128, 192, 256, 384, 512, 576, 640, 704, 768, 896, 1024, 1152, 1280, 1536)
        # bs≥2 按 bs 定制：bs*L ≤ 1536 内加密，修复如 (2,512) 差 1 token 用不上、
        # bs=8 只有 L=128 的覆盖缺口，同时控制 piece 数量。
        per_bs_L = {
            2: (128, 192, 256, 384, 512, 640, 768),   # max 1536
            4: (128, 192, 256, 320, 384),             # max 1536
            8: (128, 160, 192),                       # max 1536
            16: (),
        }
        pieces: list[tuple] = []
        for bs in bs_list:
            L_list = fine_L if bs == 1 else per_bs_L.get(bs, (128, 256, 512, 1024))
            for L in L_list:
                if L <= max_len and bs * L <= cap:
                    pieces.append((bs, L))
        # 从大到小捕获：最大 piece 先占显存池，后续复用避免碎片
        return sorted(pieces, key=lambda p: (p[0] * p[1], p[0]), reverse=True)

    def _workspace_bytes(self) -> int:
        """FlashInfer plan 需要的 workspace 字节数（全 runner 共享一份）。

        取所有 piece 中最大的 M=bs*L 估算（plan 在 workspace 内按对齐偏移
        分配临时 buffer，总量约 M×24KB，下限 64MB）。所有 piece 从不同时
        运行（"或"关系），因此共享一块即可，避免 29×64MB 的重复占用。
        """
        if self._ws_override > 0:
            return self._ws_override
        M = max(p[0] * p[1] for p in self.pieces)
        return max(64 * 1024 * 1024, M * 24 * 1024)

    def _calibrate(self) -> None:
        """估计 eager 的「CPU 发射地板」S。

        直接 forward 的 wall 在 CPU-bound 区间 = 360 次 kernel 发射的串行化时间
        （GPU 在等 CPU），该值近似等于引擎里 eager prefill 的单步成本中 CG 能
        消除的部分。CG 用一次 replay 发射替换全部 kernel 发射，因此：
            CG 有益  ⇔  piece_replay_gpu < S

        用 n=512 的 forward wall 作为 S：0.6B/1.7B 上它落在 CPU-bound 区间
        （GPU 时间 ~7ms 但 wall ~22ms），4B 上它是 GPU-bound（~32ms），两种
        语义下都恰好构成「CG 打不赢的门槛」。warmup 一次再计时，排除编译。
        """
        import time

        piece = (1, 512)
        self._prepare_piece(piece)  # warmup：编译/autotune 不计时
        torch.cuda.synchronize(self.device)
        start, end = torch.cuda.Event(True), torch.cuda.Event(True)
        t0 = time.perf_counter()
        start.record()
        self._run_forward(piece)
        end.record()
        torch.cuda.synchronize(self.device)
        self._eager_floor_ms = (time.perf_counter() - t0) * 1000 - 2.0
        print(f"[prefill CG] calibrate: eager_floor(n=512)={self._eager_floor_ms:.1f}ms")

    def _prepare_piece(self, piece: tuple) -> None:
        """准备 piece 资源并 warmup forward；共享 workspace 不够则自动加倍重建。

        plan 在 _make_piece_buffers 内就可能溢出（大模型需要更大的 FlashInfer
        tmp buffer），因此 try 覆盖整个资源准备；重建时 _ws=None 释放旧 buffer，
        让 _shared_ws 按新 _ws_bytes 重新分配。
        """
        try:
            self._make_piece_buffers(piece)
            self._run_forward(piece)
        except RuntimeError as e:
            if "Buffer overflow" not in str(e):
                raise
            self._ws_bytes *= 2
            self._ws = None
            torch.cuda.empty_cache()
            self._make_piece_buffers(piece)
            self._run_forward(piece)

    def _measure_piece_replay(self, piece: tuple) -> None:
        """捕获后用 replay 实测 piece 的真实 GPU 时间（直接 forward 含发射串行化，
        虚高 3x，不能用）。"""
        graph = self.graphs[piece]
        start, end = torch.cuda.Event(True), torch.cuda.Event(True)
        start.record()
        graph.replay()
        end.record()
        torch.cuda.synchronize(self.device)
        self._piece_gpu_ms[piece] = start.elapsed_time(end)

    def capture(self) -> None:
        if self._captured:
            return
        torch.cuda.synchronize(self.device)
        torch.cuda.empty_cache()

        # inference_mode：warmup/校准/捕获的 forward 都不能构建 autograd 图，
        # 否则激活内存与耗时放大数倍（引擎真实路径也全程 inference_mode）。
        with torch.inference_mode():
            self._calibrate()
            self._piece_gpu_ms: dict[tuple, float] = {}

            graph_pool = None
            for piece in self.pieces:
                self._prepare_piece(piece)
                torch.cuda.synchronize(self.device)

                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph, pool=graph_pool):
                    self._run_forward(piece)
                if graph_pool is None:
                    graph_pool = graph.pool()
                self.graphs[piece] = graph
                self._measure_piece_replay(piece)
                print(
                    f"  prefill CG (bs={piece[0]}, L={piece[1]}) "
                    f"gpu={self._piece_gpu_ms[piece]:.1f}ms captured"
                )

            self.kv_cache_pool.zero_()
            torch.cuda.synchronize(self.device)
            self._captured = True

    def _run_forward(self, piece: tuple) -> None:
        bs, L = piece
        bufs = self._buffers[piece]
        meta = self._metas[piece]
        # 首 chunk prefill 走 FA2 varlen：K/V 就是本 chunk 的 q/k/v（无缓存前缀），
        # 因此 cu_seqlens_k 必须与 cu_seqlens_q 同为 token 粒度（kv_indptr 是页粒度，
        # 只喂给 FlashInfer paged wrapper，不能作为 FA2 的 cu_seqlens_k，否则每条
        # K 序列被当成 1 个 token）。
        with forward_context(
            is_prefill=True,
            seqs=[],
            slot_mapping=bufs["slot_mapping"],
            cu_seqlens_q=bufs["qo_indptr"],
            cu_seqlens_k=bufs["qo_indptr"],
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
        """为一个 (bs, L) piece 准备全部静态资源：buffers → 分布 → wrapper → meta。

        这些资源在捕获时固定形状，重放时只改数据不改形状。拆分小方法让
        每个职责独立可读（见 _alloc_buffers / _init_capture_layout / _shared_ws）。
        """
        bs, L = piece
        M = bs * L
        bufs = self._alloc_buffers(M, bs, L)

        # 捕获固定分布：每 seq 恰好 L 个 token（重放时真实长度 ≤ L，余下留 padding）
        self._init_capture_layout(bufs, bs, L)

        bufs["workspace"] = self._shared_ws()
        wrapper = self._build_wrapper(piece, bufs)

        self._buffers[piece] = bufs
        self._metas[piece] = self._build_metadata(bufs, wrapper)

    def _alloc_buffers(self, M: int, bs: int, L: int) -> dict[str, Any]:
        """分配静态 tensor 与派生量（pages 非 tensor，仅供 replay 使用）。"""
        pages = (L + self.block_size - 1) // self.block_size
        return {
            "input_ids": torch.zeros(M, dtype=torch.int64, device=self.device),
            "positions": torch.zeros(M, dtype=torch.int64, device=self.device),
            "slot_mapping": torch.full((M,), -1, dtype=torch.int32, device=self.device),
            "qo_indptr": torch.zeros(bs + 1, dtype=torch.int32, device=self.device),
            "kv_indptr": torch.zeros(bs + 1, dtype=torch.int32, device=self.device),
            "kv_indices": torch.full(
                (bs * pages,), self.dummy_page_id, dtype=torch.int32, device=self.device
            ),
            "kv_last_page": torch.zeros(bs, dtype=torch.int32, device=self.device),
            "outputs": torch.zeros(
                M, self.hidden_size, dtype=self.dtype, device=self.device
            ),
            "pages": pages,
        }

    def _init_capture_layout(self, bufs: dict[str, Any], bs: int, L: int) -> None:
        """把捕获时的固定分布写进 buffer：每 seq L 个 token / pages 个 KV 页。"""
        pages = bufs["pages"]
        bufs["qo_indptr"].copy_(
            torch.arange(0, (bs + 1) * L, L, dtype=torch.int32)
        )
        bufs["kv_indptr"].copy_(
            torch.arange(0, (bs + 1) * pages, pages, dtype=torch.int32)
        )
        bufs["kv_last_page"].fill_((L - 1) % self.block_size + 1)

    def _shared_ws(self) -> torch.Tensor:
        """FlashInfer workspace：全 runner 共享一块（pieces 从不同时运行，安全）。

        plan 溢出则整块加倍重建（capture 外层重试）。
        """
        if self._ws is None:
            self._ws = torch.empty(
                self._ws_bytes, dtype=torch.uint8, device=self.device
            )
        return self._ws

    def _build_metadata(
        self, bufs: dict[str, Any], wrapper: Any
    ) -> AttentionMetadata:
        return AttentionMetadata(
            wrapper=wrapper,
            kv_cache_pool=self.kv_cache_pool,
            slot_mapping=bufs["slot_mapping"],
            # CG piece 内用 FA2（连续 KV 直算，比 FlashInfer paged prefill 快）。
            # FA2 对整块 L 做 causal 注意力，padding 行由 position 0 的输入产生，
            # 与 FlashInfer 把 padding 纳入 kv_indptr 的语义一致。
            use_fa2=True,
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
        """找覆盖批次的 piece，且其 replay GPU 时间低于 eager 地板时才用 CG。

        eager 单步成本 ≈ 发射串行化地板 S（_eager_floor_ms）；CG 成本 ≈ piece
        replay GPU 时间。因此「piece_gpu < S」才是 CG 真正有益的区间。这一判断
        用捕获时实测的 replay 时间，天然吸收了不同模型规模/形状的 GEMM 效率差异
        （0.6B 上 ≤~1500 token 有效，4B 上只有 ≤~300 token 有效）。
        """
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
        if best is None:
            return None
        if self._piece_gpu_ms.get(best, float("inf")) < self._eager_floor_ms:
            return best
        return None

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
        # FA2 varlen 需要 token 粒度 cu_seqlens_k（== qo_indptr，首 chunk K=本 chunk）
        with forward_context(
            is_prefill=True,
            seqs=[],
            slot_mapping=slot,
            cu_seqlens_q=bufs["qo_indptr"],
            cu_seqlens_k=bufs["qo_indptr"],
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
        self._ws = None
        self._captured = False
        gc.collect()
        torch.cuda.empty_cache()
