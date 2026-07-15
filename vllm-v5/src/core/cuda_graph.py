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
        self.bs_list = _make_bs_list(max_batch_size)

        max_blocks = (max_model_len + block_size - 1) // block_size
        self._max_blocks = max_blocks

        self.workspace_buffer = torch.empty(
            workspace_bytes, dtype=torch.uint8, device=device
        )

        self.input_ids = torch.zeros(max_batch_size, dtype=torch.int64, device=device)
        self.positions = torch.zeros(max_batch_size, dtype=torch.int64, device=device)
        self.slot_mapping = torch.full(
            (max_batch_size,), -1, dtype=torch.int32, device=device
        )
        self.outputs = torch.zeros(max_batch_size, hidden_size, dtype=dtype, device=device)

        self._cg_bufs: dict[int, dict[str, torch.Tensor]] = {
            bs: {
                "indptr": torch.zeros(bs + 1, dtype=torch.int32, device=device),
                "indices": torch.zeros(bs * max_blocks, dtype=torch.int32, device=device),
                "last_page": torch.zeros(bs, dtype=torch.int32, device=device),
            }
            for bs in self.bs_list
        }
        self._host_bufs: dict[int, dict[str, torch.Tensor]] = {
            bs: {
                "indptr": torch.zeros(bs + 1, dtype=torch.int32, pin_memory=True),
                "last_page": torch.ones(bs, dtype=torch.int32, pin_memory=True),
                "indices": torch.zeros(bs * max_blocks, dtype=torch.int32, pin_memory=True),
            }
            for bs in self.bs_list
        }
        self.cg_wrappers: dict[int, BatchDecodeWithPagedKVCacheWrapper] = {}
        self.graphs: dict[int, torch.cuda.CUDAGraph] = {}
        self._metas: dict[int, AttentionMetadata] = {}
        self._captured = False
        self._sm_scale = head_dim**-0.5

    def capture(self) -> None:
        if self._captured:
            return
        torch.cuda.synchronize(self.device)
        torch.cuda.empty_cache()

        self.slot_mapping.fill_(-1)

        graph_pool = None
        for bs in sorted(self.bs_list, reverse=True):
            cg = self._make_cg_wrapper(bs)
            self.cg_wrappers[bs] = cg
            meta = AttentionMetadata(
                wrapper=cg,
                kv_cache_pool=self.kv_cache_pool,
                slot_mapping=self.slot_mapping[:bs],
            )
            self._metas[bs] = meta
            graph = torch.cuda.CUDAGraph()
            with forward_context(
                is_prefill=False,
                seqs=[],
                slot_mapping=self.slot_mapping[:bs],
            ):
                with attention_context(meta):
                    self.outputs[:bs] = self.model(
                        input_ids=self.input_ids[:bs],
                        positions=self.positions[:bs],
                    )
                    with torch.cuda.graph(graph, pool=graph_pool):
                        self.outputs[:bs] = self.model(
                            input_ids=self.input_ids[:bs],
                            positions=self.positions[:bs],
                        )
            if graph_pool is None:
                graph_pool = graph.pool()
            self.graphs[bs] = graph

        self.kv_cache_pool.zero_()
        torch.cuda.synchronize(self.device)
        self._captured = True

    def _make_cg_wrapper(self, bs: int) -> BatchDecodeWithPagedKVCacheWrapper:
        c = self.backend.cfg
        bufs = self._cg_bufs[bs]
        host = self._host_bufs[bs]
        w = BatchDecodeWithPagedKVCacheWrapper(
            self.workspace_buffer,
            "NHD",
            use_tensor_cores=False,
            use_cuda_graph=True,
            paged_kv_indptr_buffer=bufs["indptr"],
            paged_kv_indices_buffer=bufs["indices"],
            paged_kv_last_page_len_buffer=bufs["last_page"],
        )

        host["indptr"].copy_(torch.arange(0, bs + 1, dtype=torch.int32))
        host["last_page"].fill_(1)
        host["indices"][:bs].fill_(self.dummy_page_id)
        bufs["indices"][:bs].fill_(self.dummy_page_id)
        w.plan(
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
            non_blocking=False,
        )
        return w

    def can_use(self, bs: int) -> bool:
        return self._captured and bs <= self.max_bs

    def _pad_bs(self, bs: int) -> int:
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
        bs = input_ids.shape[0]
        assert bs == len(seq_lens) == len(block_tables)
        pbs = self._pad_bs(bs)

        self.input_ids[:bs] = input_ids
        self.positions[:bs] = positions
        self.slot_mapping.fill_(-1)
        self.slot_mapping[:bs] = slot_mapping

        self._replan_cg(pbs, bs, seq_lens, block_tables)

        meta = self._metas[pbs]
        with forward_context(
            is_prefill=False,
            seqs=[],
            slot_mapping=self.slot_mapping[:pbs],
        ):
            with attention_context(meta):
                self.graphs[pbs].replay()

        return self.outputs[:bs]

    def _replan_cg(
        self,
        pbs: int,
        real_bs: int,
        seq_lens: list[int],
        block_tables: list[list[int]],
    ) -> None:
        c = self.backend.cfg
        block = self.block_size
        bufs = self._cg_bufs[pbs]
        host = self._host_bufs[pbs]
        dummy = self.dummy_page_id

        nbs = [(s + block - 1) // block for s in seq_lens]

        indptr_h = host["indptr"]
        last_h = host["last_page"]
        idx_h = host["indices"]
        indptr_np = indptr_h.numpy()
        last_np = last_h.numpy()
        idx_np = idx_h.numpy()

        indptr_np[:] = 0
        last_np[:] = 1

        acc = 0
        for i, (s, nb, table) in enumerate(zip(seq_lens, nbs, block_tables)):
            if nb > len(table):
                raise RuntimeError(
                    f"block_table shorter than needed pages: need {nb}, have {len(table)}"
                )
            idx_np[acc : acc + nb] = table[:nb]
            acc += nb
            indptr_np[i + 1] = acc
            last_np[i] = (s - 1) % block + 1

        for i in range(real_bs, pbs):
            idx_np[acc] = dummy
            acc += 1
            indptr_np[i + 1] = acc
            last_np[i] = 1

        bufs["indices"][:acc].copy_(idx_h[:acc], non_blocking=False)

        self.cg_wrappers[pbs].plan(
            indptr=indptr_h,
            indices=bufs["indices"][:acc],
            last_page_len=last_h,
            num_qo_heads=c.num_qo_heads,
            num_kv_heads=c.num_kv_heads,
            head_dim=c.head_dim,
            page_size=block,
            sm_scale=self._sm_scale,
            q_data_type=c.dtype,
            kv_data_type=c.dtype,
            non_blocking=False,
        )

    def destroy(self) -> None:
        del self.graphs
        self.graphs = {}
        self.cg_wrappers.clear()
        self._metas.clear()
        self._captured = False
        gc.collect()
        torch.cuda.empty_cache()


def _make_bs_list(max_bs: int) -> list[int]:
    bs = [1, 2, 4, 8] + list(range(16, max_bs + 1, 16))
    bs = [b for b in bs if b <= max_bs]
    if max_bs not in bs:
        bs.append(max_bs)
    return sorted(set(bs))
