from __future__ import annotations

import torch

from ..config import EngineConfig
from ..core.cuda_graph import CUDAGraphRunner
from ..core.prefill_cuda_graph import PrefillCUDAGraphRunner
from ..layers.attention import attention_context
from ..layers.attention_backend import (
    BackendConfig,
    FlashInferBackend,
    compute_paged_kv_metadata,
)
from ..memory import KVManager, Sequence
from ..utils.context import forward_context
from .model_inputs import ModelInputs, build_model_inputs
from .sampler import Sampler


class ModelRunner:
    def __init__(
        self,
        model: torch.nn.Module,
        kv: KVManager,
        sampler: Sampler,
        *,
        config: EngineConfig,
        device: torch.device,
    ) -> None:
        self.model = model
        self.kv = kv
        self.sampler = sampler.to(device=device)
        self.config = config
        self.device = device
        self.block_size = kv.block_size
        self.kv_cache = kv.kv_cache
        self.dtype = kv.dtype
        self.backend = self._make_backend()

        self._warmup()
        self.cuda_graph_runner: CUDAGraphRunner | None = None
        if not config.enforce_eager:
            self.cuda_graph_runner = self._capture_cg()
        self.prefill_cg_runner: PrefillCUDAGraphRunner | None = None
        if config.prefill_cuda_graph:
            self.prefill_cg_runner = self._capture_prefill_cg()

    def _make_backend(self) -> FlashInferBackend:
        cfg = self.config.hf_config
        hd = getattr(cfg, "head_dim", cfg.hidden_size // cfg.num_attention_heads)
        return FlashInferBackend(
            BackendConfig(
                num_qo_heads=cfg.num_attention_heads,
                num_kv_heads=cfg.num_key_value_heads,
                head_dim=hd,
                block_size=self.block_size,
                dtype=self.dtype,
                device=self.device,
                use_tensor_cores=self.config.flashinfer_use_tensor_cores,
                non_blocking=self.config.flashinfer_non_blocking,
            )
        )

    def _warmup(self) -> None:
        torch.cuda.empty_cache()
        n = min(128, self.config.context_len)
        bs = 2
        total = bs * n
        n_blocks = (n + self.block_size - 1) // self.block_size
        need = bs * n_blocks
        usable = self.kv.num_blocks_schedulable
        if need > usable:
            n_blocks = max(1, usable // bs)
            n = min(n, n_blocks * self.block_size)
            total = bs * n

        ids = torch.arange(1, total + 1, dtype=torch.int64, device=self.device)
        pos = torch.arange(total, dtype=torch.int64, device=self.device)
        cu = torch.tensor([0, n, total], dtype=torch.int32, device=self.device)
        cu_cpu = torch.tensor([0, n, total], dtype=torch.int32, device="cpu")
        sm = torch.arange(total, dtype=torch.int32, device=self.device)
        bt = torch.full((bs, n_blocks), -1, dtype=torch.int32, device=self.device)
        bt[0, :n_blocks] = torch.arange(n_blocks, device=self.device)
        bt[1, :n_blocks] = torch.arange(n_blocks, 2 * n_blocks, device=self.device)
        assert int(bt.max().item()) < usable

        indptr_cpu, idx_gpu, last_cpu = compute_paged_kv_metadata(
            [n, n], bt, self.block_size, self.device
        )
        meta = self.backend.build_metadata(
            bs, indptr_cpu, idx_gpu, last_cpu, sm, True, self.kv_cache, cu_cpu
        )
        with forward_context(
            is_prefill=True,
            seqs=[],
            cu_seqlens_q=cu,
            cu_seqlens_k=cu,
            max_seqlen_q=n,
            max_seqlen_k=n,
            block_tables=bt,
            slot_mapping=sm,
        ):
            with attention_context(meta):
                self.model(input_ids=ids, positions=pos)

        self.kv_cache.zero_()
        torch.cuda.empty_cache()

    def _capture_cg(self) -> CUDAGraphRunner:
        cfg = self.config.hf_config
        hd = getattr(cfg, "head_dim", cfg.hidden_size // cfg.num_attention_heads)
        max_bs = min(self.config.max_num_seqs, 256)
        dummy_page_id = self.kv.dummy_page_id
        if dummy_page_id < 0:
            raise RuntimeError("CUDA Graph capture requires a reserved dummy page")
        runner = CUDAGraphRunner(
            model=self.model,
            max_batch_size=max_bs,
            max_model_len=self.config.context_len,
            block_size=self.block_size,
            hidden_size=cfg.hidden_size,
            num_kv_heads=cfg.num_key_value_heads,
            head_dim=hd,
            device=self.device,
            dtype=self.dtype,
            backend=self.backend,
            kv_cache_pool=self.kv_cache,
            dummy_page_id=dummy_page_id,
        )
        runner.capture()
        print(
            f"CUDA Graph captured for decode "
            f"(max_bs={max_bs}, dummy_page={dummy_page_id})"
        )
        return runner

    def _capture_prefill_cg(self) -> PrefillCUDAGraphRunner:
        cfg = self.config.hf_config
        hd = getattr(cfg, "head_dim", cfg.hidden_size // cfg.num_attention_heads)
        max_bs = min(self.config.max_num_seqs, 256)
        dummy_page_id = self.kv.dummy_page_id
        if dummy_page_id < 0:
            raise RuntimeError("CUDA Graph capture requires a reserved dummy page")
        runner = PrefillCUDAGraphRunner(
            model=self.model,
            max_batch_size=max_bs,
            max_model_len=self.config.context_len,
            max_batched_tokens=self.config.max_num_batched_tokens,
            block_size=self.block_size,
            hidden_size=cfg.hidden_size,
            num_qo_heads=cfg.num_attention_heads,
            num_kv_heads=cfg.num_key_value_heads,
            head_dim=hd,
            device=self.device,
            dtype=self.dtype,
            backend=self.backend,
            kv_cache_pool=self.kv_cache,
            dummy_page_id=dummy_page_id,
            cg_max_tokens=self.config.prefill_cg_max_tokens,
        )
        runner.capture()
        print(
            f"CUDA Graph captured for prefill piecewise "
            f"({len(runner.pieces)} pieces, dummy_page={dummy_page_id})"
        )
        return runner

    @torch.inference_mode()
    def run(
        self, prefills: list[Sequence], decodes: list[Sequence]
    ) -> tuple[list[int], list[int]]:
        """v2 contract: decode batch first, then prefill batch (two forwards)."""
        d_ids = self._run_batch(decodes, is_prefill=False) if decodes else []
        p_ids = self._run_batch(prefills, is_prefill=True) if prefills else []
        return p_ids, d_ids

    def _run_batch(self, seqs: list[Sequence], *, is_prefill: bool) -> list[int]:
        """执行一批序列，返回采样的 token id。

        流程：构造输入 → 选最快可用的执行路径 → 采样。
        采样参数统一打包，避免每条路径重复构造。
        """
        if not seqs:
            return []
        inputs = build_model_inputs(seqs, is_prefill, self.block_size, self.device)
        sampling = self._build_sampling_inputs(seqs)
        hidden = self._forward(seqs, inputs, is_prefill)
        if is_prefill and not self._need_sample(seqs):
            return [0] * len(seqs)
        logits = self._logits_for_sample(hidden, inputs)
        return self.sampler(
            logits,
            sampling["temperatures"],
            top_ps=sampling["top_ps"],
            top_ks=sampling["top_ks"],
            min_ps=sampling["min_ps"],
        ).tolist()

    @staticmethod
    def _need_sample(seqs: list[Sequence]) -> bool:
        """是否有序列在本步完成 prefill（否则跳过 LM-head/采样）。"""
        return any(
            s.num_cached_tokens + s.num_scheduled_tokens >= s.num_prompt_tokens
            for s in seqs
        )

    def _build_sampling_inputs(
        self, seqs: list[Sequence],
    ) -> dict[str, torch.Tensor]:
        """把每条 seq 的采样参数整批打包成 GPU tensor。

        一次构造整 batch 的参数 + 一次 non_blocking H2D，避免逐条临时构造
        小张量、逐条同步（v2 的逐条 `.item()` 取回是并发瓶颈）。
        """
        return {
            "temperatures": torch.tensor(
                [s.temperature for s in seqs], dtype=torch.float32, pin_memory=True
            ).to(self.device, non_blocking=True),
            "top_ps": torch.tensor(
                [1.0 if s.top_p is None else float(s.top_p) for s in seqs],
                dtype=torch.float32, pin_memory=True,
            ).to(self.device, non_blocking=True),
            "top_ks": torch.tensor(
                [0 if s.top_k is None else int(s.top_k) for s in seqs],
                dtype=torch.int32, pin_memory=True,
            ).to(self.device, non_blocking=True),
            "min_ps": torch.tensor(
                [0.0 if s.min_p is None else float(s.min_p) for s in seqs],
                dtype=torch.float32, pin_memory=True,
            ).to(self.device, non_blocking=True),
        }

    def _forward(
        self,
        seqs: list[Sequence],
        inputs: ModelInputs,
        is_prefill: bool,
    ) -> torch.Tensor:
        """按优先级选执行路径，返回整批 hidden states。

        优先级设计（谁最快用谁，不可用则回退下一档）：
          decode        : decode CG → eager
          prefill 首 chunk : piece CG → eager
          prefill 续 chunk : eager（前缀命中跳过 KV 重算，剩余走 eager）
        """
        if not is_prefill:
            hidden = self._decode_cg(seqs, inputs)
            if hidden is not None:
                return hidden
            return self._eager_forward(seqs, inputs, is_prefill)
        if all(s.num_cached_tokens == 0 for s in seqs):
            hidden = self._piece_cg(seqs, inputs)
            if hidden is not None:
                return hidden
        return self._eager_forward(seqs, inputs, is_prefill)

    def _decode_cg(self, seqs: list[Sequence], inputs: ModelInputs) -> torch.Tensor | None:
        """decode CUDA Graph：固定 bs 的图重放（bs 不在网格则回退 eager）。"""
        runner = self.cuda_graph_runner
        if runner is None or not runner.can_use(len(seqs)):
            return None
        return runner.replay(
            inputs.input_ids,
            inputs.positions,
            inputs.slot_mapping,
            seq_lens=[len(s) for s in seqs],
            block_tables=[list(s.block_table) for s in seqs],
        )

    def _piece_cg(self, seqs: list[Sequence], inputs: ModelInputs) -> torch.Tensor | None:
        """piecewise prefill CG：仅首 chunk（无缓存前缀，KV 分布均匀）。

        形状不在 (bs, L) 网格，或该 piece 的 GPU 时间不优于 eager 地板时回退。
        """
        runner = self.prefill_cg_runner
        if runner is None:
            return None
        seq_lens = [s.num_scheduled_tokens for s in seqs]
        if not runner.can_use(len(seqs), seq_lens):
            return None
        return runner.replay(
            inputs.input_ids,
            inputs.positions,
            inputs.slot_mapping,
            seq_lens=seq_lens,
            block_tables=[list(s.block_table) for s in seqs],
        )

    def _eager_forward(
        self, seqs: list[Sequence], inputs: ModelInputs, is_prefill: bool
    ) -> torch.Tensor:
        """回退路径：eager 前向。"""
        seq_lens = (
            [s.num_cached_tokens + s.num_scheduled_tokens for s in seqs]
            if is_prefill
            else [len(s) for s in seqs]
        )
        indptr_cpu, idx_gpu, last_cpu = compute_paged_kv_metadata(
            seq_lens,
            inputs.block_tables
            if inputs.block_tables is not None
            else torch.zeros(len(seqs), 1, dtype=torch.int32, device=self.device),
            self.block_size,
            self.device,
        )
        cu_cpu = (
            torch.tensor(_cu_q(seqs), dtype=torch.int32, device="cpu")
            if is_prefill
            else None
        )
        meta = self.backend.build_metadata(
            len(seqs),
            indptr_cpu,
            idx_gpu,
            last_cpu,
            inputs.slot_mapping,
            is_prefill,
            self.kv_cache,
            cu_cpu,
            use_fa2=is_prefill and all(s.num_cached_tokens == 0 for s in seqs),
        )
        with forward_context(
            is_prefill=is_prefill,
            seqs=seqs,
            slot_mapping=inputs.slot_mapping,
            cu_seqlens_q=inputs.cu_seqlens_q,
            cu_seqlens_k=inputs.cu_seqlens_k,
            max_seqlen_q=inputs.max_seqlen_q,
            max_seqlen_k=inputs.max_seqlen_k,
            context_lens=inputs.context_lens,
            block_tables=inputs.block_tables,
        ):
            with attention_context(meta):
                return self.model(
                    input_ids=inputs.input_ids, positions=inputs.positions
                )

    def _logits_for_sample(self, hidden: torch.Tensor, inputs: ModelInputs) -> torch.Tensor:
        if inputs.is_prefill and inputs.cu_seqlens_q is not None:
            last_idx = inputs.cu_seqlens_q[1:] - 1
            hidden = hidden[last_idx]
        return self.model.compute_logits(hidden)

    def destroy(self) -> None:
        if self.cuda_graph_runner is not None:
            self.cuda_graph_runner.destroy()
        if self.prefill_cg_runner is not None:
            self.prefill_cg_runner.destroy()


def _cu_q(seqs: list[Sequence]) -> list[int]:
    cu = [0]
    for s in seqs:
        cu.append(cu[-1] + s.num_scheduled_tokens)
    return cu
