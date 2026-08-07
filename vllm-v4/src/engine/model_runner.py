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
import torch._dynamo as dynamo



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
        self.is_multimodal = config.is_multimodal
        self.backend = self._make_backend()
        # Eager module kept for Decode CUDA Graph capture/replay.
        # Prefill may use a torch.compile wrapper sharing the same weights.
        self._prefill_model: torch.nn.Module | None = None

        self._warmup()
        self.cuda_graph_runner: CUDAGraphRunner | None = None
        if not config.enforce_eager:
            self.cuda_graph_runner = self._capture_cg()
        self.prefill_cg_runner: PrefillCUDAGraphRunner | None = None
        if config.prefill_cuda_graph:
            self.prefill_cg_runner = self._capture_prefill_cg()
        if config.compile_prefill:
            self._prefill_model = self._compile_prefill_model()

    def _make_backend(self) -> FlashInferBackend:
        cfg = self.config.text_config
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
        cfg = self.config.text_config
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
        cfg = self.config.text_config
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
        )
        runner.capture()
        print(
            f"CUDA Graph captured for prefill piecewise "
            f"({len(runner.pieces)} pieces, dummy_page={dummy_page_id})"
        )
        return runner

    def _compile_prefill_model(self) -> torch.nn.Module:
        # 关键：prefill 的 seq_len 每次请求都变（513、1022…），必须让 inductor
        # 生成「动态 shape」的单份代码，而不是每个 shape 重新编译 / 重录 CUDA Graph。
        #   1) dynamic=True        → 符号化 seq/batch 维，避免每个长度重编译
        #   2) cudagraph_skip_dynamic_graphs=True
        #                          → 动态 shape 时跳过 CUDA Graph 重录（重录比编译还慢，
        #                            这是 reduce-overhead 在动态 prefill 上的致命坑）
        # 收益：任意 prompt 长度首次 prefill 即命中编译好的图，TTFT 稳定。
        torch._inductor.config.triton.cudagraph_skip_dynamic_graphs = True

        dynamo.config.cache_size_limit = max(int(dynamo.config.cache_size_limit), 64)
        compiled = torch.compile(
            self.model,
            dynamic=True,
            fullgraph=False,
            mode="reduce-overhead",
        )
        # Trigger on a few prefill sizes (text prompt bounds).
        warm_lens = sorted({128, 256, 512, 1024})
        warm_lens = [n for n in warm_lens if 0 < n <= self.config.context_len]
        usable = self.kv.num_blocks_schedulable
        with torch.inference_mode():
            for n in warm_lens:
                n_blocks = (n + self.block_size - 1) // self.block_size
                if n_blocks > usable:
                    continue
                ids = torch.arange(1, n + 1, dtype=torch.int64, device=self.device)
                pos = torch.arange(n, dtype=torch.int64, device=self.device)
                cu = torch.tensor([0, n], dtype=torch.int32, device=self.device)
                cu_cpu = torch.tensor([0, n], dtype=torch.int32, device="cpu")
                sm = torch.arange(n, dtype=torch.int32, device=self.device)
                bt = torch.full((1, n_blocks), -1, dtype=torch.int32, device=self.device)
                bt[0, :n_blocks] = torch.arange(n_blocks, device=self.device)
                indptr_cpu, idx_gpu, last_cpu = compute_paged_kv_metadata(
                    [n], bt, self.block_size, self.device
                )
                meta = self.backend.build_metadata(
                    1, indptr_cpu, idx_gpu, last_cpu, sm, True, self.kv_cache, cu_cpu
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
                        compiled(input_ids=ids, positions=pos)
                if self.is_multimodal:
                    self._warm_compiled_mm(compiled, n, bt, sm, cu, cu_cpu, indptr_cpu, idx_gpu, last_cpu)
        self.kv_cache.zero_()
        torch.cuda.empty_cache()
        print(f"Prefill torch.compile ready (warm_lens={warm_lens}, mm={'yes' if self.is_multimodal else 'no'})")
        return compiled

    def _warm_compiled_mm(
        self, compiled, n, bt, sm, cu, cu_cpu, indptr_cpu, idx_gpu, last_cpu
    ) -> None:
        """Warm the compiled model on an MM prefill shape (inputs_embeds + 3D MRoPE)."""
        hidden = self.config.text_config.hidden_size
        emb = torch.randn(n, hidden, device=self.device, dtype=self.dtype)
        pos3d = torch.arange(n, dtype=torch.int64, device=self.device).repeat(3, 1)
        deepstack = None
        n_ds = int(getattr(self.model, "_ndps", 0))
        if n_ds:
            deepstack = [
                torch.zeros(n, hidden, device=self.device, dtype=self.dtype)
                for _ in range(n_ds)
            ]
        rope = getattr(self.model, "_mrope_module", None)
        if callable(rope):
            rope().ensure_cache_size(n + 1)
        meta = self.backend.build_metadata(
            1, indptr_cpu, idx_gpu, last_cpu, sm, True, self.kv_cache, cu_cpu
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
                compiled(
                    input_ids=None,
                    positions=pos3d,
                    inputs_embeds=emb,
                    deepstack_input_embeds=deepstack,
                )

    @torch.inference_mode()
    def run(
        self, prefills: list[Sequence], decodes: list[Sequence]
    ) -> tuple[list[int], list[int]]:
        """v2 contract: decode batch first, then prefill batch (two forwards)."""
        d_ids = self._run_batch(decodes, is_prefill=False) if decodes else []
        p_ids = self._run_batch(prefills, is_prefill=True) if prefills else []
        return p_ids, d_ids

    def _run_batch(self, seqs: list[Sequence], *, is_prefill: bool) -> list[int]:
        if not seqs:
            return []

        inputs = build_model_inputs(seqs, is_prefill, self.block_size, self.device)
        temperatures = torch.tensor(
            [s.temperature for s in seqs],
            dtype=torch.float32,
            pin_memory=True,
        ).to(self.device, non_blocking=True)
        top_ps = torch.tensor(
            [1.0 if s.top_p is None else float(s.top_p) for s in seqs],
            dtype=torch.float32,
            pin_memory=True,
        ).to(self.device, non_blocking=True)
        top_ks = torch.tensor(
            [0 if s.top_k is None else int(s.top_k) for s in seqs],
            dtype=torch.int32,
            pin_memory=True,
        ).to(self.device, non_blocking=True)
        min_ps = torch.tensor(
            [0.0 if s.min_p is None else float(s.min_p) for s in seqs],
            dtype=torch.float32,
            pin_memory=True,
        ).to(self.device, non_blocking=True)

        def _sample(logits: torch.Tensor) -> list[int]:
            return self.sampler(
                logits, temperatures, top_ps=top_ps, top_ks=top_ks, min_ps=min_ps
            ).tolist()

        need_sample = True
        if is_prefill:
            need_sample = any(
                s.num_cached_tokens + s.num_scheduled_tokens >= s.num_prompt_tokens
                for s in seqs
            )

        # Multimodal prefill: ViT (+EncoderCache), scatter embeds, 3-axis MRoPE.
        inputs_embeds = None
        deepstack = None
        if is_prefill and self.is_multimodal and any(s.has_multimodal for s in seqs):
            inputs_embeds, deepstack, positions3d = self.model.encode_and_merge(
                inputs.input_ids, seqs, self.device
            )
            inputs.inputs_embeds = inputs_embeds
            inputs.deepstack_input_embeds = deepstack
            inputs.positions = positions3d
        elif self.is_multimodal:
            rope_fn = getattr(self.model, "_mrope_module", None)
            if callable(rope_fn):
                host_max = 0
                for s in seqs:
                    if is_prefill:
                        host_max = max(
                            host_max, s.num_cached_tokens + s.num_scheduled_tokens
                        )
                    else:
                        host_max = max(host_max, len(s) + int(s.rope_delta))
                rope_fn().ensure_cache_size(host_max + 1)

        if (
            not is_prefill
            and self.cuda_graph_runner is not None
            and self.cuda_graph_runner.can_use(len(seqs))
        ):
            h = self.cuda_graph_runner.replay(
                inputs.input_ids,
                inputs.positions,
                inputs.slot_mapping,
                seq_lens=[len(s) for s in seqs],
                block_tables=[list(s.block_table) for s in seqs],
            )
            return _sample(self._logits_for_sample(h, inputs))

        # Piecewise prefill CUDA Graph：仅首 chunk（无缓存前缀，KV 分布均匀）。
        # 续 chunk / 前缀缓存命中 / 多模态走 eager/compile。
        if (
            is_prefill
            and self.prefill_cg_runner is not None
            and inputs_embeds is None
            and deepstack is None
            and all(s.num_cached_tokens == 0 for s in seqs)
        ):
            seq_lens = [s.num_scheduled_tokens for s in seqs]
            if self.prefill_cg_runner.can_use(len(seqs), seq_lens):
                h = self.prefill_cg_runner.replay(
                    inputs.input_ids,
                    inputs.positions,
                    inputs.slot_mapping,
                    seq_lens=seq_lens,
                    block_tables=[list(s.block_table) for s in seqs],
                )
                if not need_sample:
                    return [0] * len(seqs)
                return _sample(self._logits_for_sample(h, inputs))

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
        )

        use_compiled = is_prefill and self._prefill_model is not None
        model = self._prefill_model if use_compiled else self.model

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
                h = model(
                    input_ids=None if inputs_embeds is not None else inputs.input_ids,
                    positions=inputs.positions,
                    inputs_embeds=inputs_embeds,
                    deepstack_input_embeds=deepstack,
                )

        if not need_sample:
            return [0] * len(seqs)
        return _sample(self._logits_for_sample(h, inputs))

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
