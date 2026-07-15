from __future__ import annotations

import torch
from torch import nn

from ..config import EngineConfig
from ..core.cuda_graph import CUDAGraphRunner
from ..layers.attention import attention_context
from ..layers.attention_backend import BackendConfig, FlashInferBackend, compute_paged_kv_metadata
from ..memory import Sequence
from ..models.loader import load_model
from ..models.qwen3 import Qwen3ForCausalLM
from ..utils.context import forward_context
from .model_inputs import ModelInputs, build_model_inputs
from .sampler import Sampler


def _text_cfg(config: EngineConfig):
    return config.text_config


class ModelRunner:
    def __init__(self, config: EngineConfig, device: torch.device) -> None:
        self.config = config
        self.device = device
        self.block_size = config.kvcache_block_size
        self.dtype = self._resolve_dtype(config)
        self.is_multimodal = config.is_multimodal

        if self.is_multimodal:
            from ..models.qwen3_vl import Qwen3VLForCausalLM

            self.model = Qwen3VLForCausalLM(config.hf_config)  # type: ignore[arg-type]
            self.model.load_weights(config.model, device, self.dtype)
            self.model = self.model.eval()
        else:
            self.model = Qwen3ForCausalLM(config.hf_config)
            load_model(self.model, config.model)
            self.model = self.model.to(dtype=self.dtype, device=self.device).eval()

        self.kv_cache = self._alloc_kv()
        self.sampler = Sampler().to(device=self.device)
        self.backend = self._make_backend()

        self._warmup()
        self.cuda_graph_runner: CUDAGraphRunner | None = None
        self.vit_cg_runner = None
        if not config.enforce_eager:
            self.cuda_graph_runner = self._capture_cg()
        if self.is_multimodal and config.enable_vit_cuda_graph:
            self.vit_cg_runner = self._capture_vit_cg()

    def _resolve_dtype(self, config: EngineConfig) -> torch.dtype:
        if config.dtype != "auto":
            return getattr(torch, config.dtype)

        cfg = _text_cfg(config)
        dt = (
            getattr(cfg, "dtype", None)
            or getattr(cfg, "torch_dtype", None)
            or getattr(config.hf_config, "dtype", None)
            or getattr(config.hf_config, "torch_dtype", None)
        )
        return dt if isinstance(dt, torch.dtype) else torch.float16

    def _alloc_kv(self) -> torch.Tensor:
        cfg = _text_cfg(self.config)
        torch.cuda.empty_cache()
        torch.cuda.synchronize(self.device)

        nkv = cfg.num_key_value_heads
        hd = getattr(cfg, "head_dim", cfg.hidden_size // cfg.num_attention_heads)
        layers = cfg.num_hidden_layers
        block_bytes = 2 * layers * self.block_size * nkv * hd * self.dtype.itemsize

        if self.config.num_kvcache_blocks > 0:
            num_blocks = self.config.num_kvcache_blocks
        else:
            free, total = torch.cuda.mem_get_info(self.device)
            available = int(total * self.config.gpu_memory_utilization) - (total - free)
            if not self.config.enforce_eager:
                available -= int(2.0 * 1024**3)
            num_blocks = max(4, available // block_bytes)
            self.config.num_kvcache_blocks = num_blocks

        if not self.config.enforce_eager and num_blocks < 2:
            raise RuntimeError(
                f"CUDA Graph requires >= 2 KV blocks after allocation, got {num_blocks}"
            )

        kv = torch.empty(
            2,
            layers,
            num_blocks,
            self.block_size,
            nkv,
            hd,
            dtype=self.dtype,
            device=self.device,
        )
        print(
            f"KV pool: {num_blocks} blocks × {self.block_size} tok "
            f"({num_blocks * block_bytes / (1024**3):.2f} GiB)"
        )
        return kv

    def _make_backend(self) -> FlashInferBackend:
        cfg = _text_cfg(self.config)
        hd = getattr(cfg, "head_dim", cfg.hidden_size // cfg.num_attention_heads)
        return FlashInferBackend(
            BackendConfig(
                num_qo_heads=cfg.num_attention_heads,
                num_kv_heads=cfg.num_key_value_heads,
                head_dim=hd,
                block_size=self.block_size,
                dtype=self.dtype,
                device=self.device,
            )
        )

    def _warmup(self) -> None:
        torch.cuda.empty_cache()
        n = min(128, self.config.context_len)
        bs = 2
        total = bs * n
        n_blocks = (n + self.block_size - 1) // self.block_size
        need = bs * n_blocks
        # Keep last page reserved for CG dummy when present.
        usable = max(1, self.kv_cache.shape[2] - (0 if self.config.enforce_eager else 1))
        if need > usable:
            # Shrink warmup so we never OOB the KV pool.
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
        cfg = _text_cfg(self.config)
        hd = getattr(cfg, "head_dim", cfg.hidden_size // cfg.num_attention_heads)
        max_bs = min(self.config.max_num_seqs, 256)
        dummy_page_id = self.kv_cache.shape[2] - 1
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

    def _capture_vit_cg(self):
        from ..core.vit_cuda_graph import ViTCUDAGraphRunner

        assert self.is_multimodal
        cfg = self.model.get_vit_cg_config()
        cfg.min_budget = self.config.vit_cg_min_budget
        cfg.max_budget = self.config.vit_cg_max_budget
        runner = ViTCUDAGraphRunner(
            vit_forward=self.model.vit_forward_cg,
            metadata_fn=self.model.vision_tower.prepare_metadata,
            config=cfg,
            device=self.device,
            dtype=self.dtype,
        )
        runner.capture()
        self.model._vit_cg = runner
        return runner

    @torch.inference_mode()
    def run(self, seqs: list[Sequence], is_prefill: bool) -> list[int]:
        if not seqs:
            return []

        inputs = build_model_inputs(seqs, is_prefill, self.block_size, self.device)
        inputs_embeds = None
        deepstack = None
        positions = inputs.positions
        if (
            is_prefill
            and self.is_multimodal
            and any(s.has_multimodal for s in seqs)
        ):
            inputs_embeds, deepstack, positions = self.model.encode_and_merge(
                inputs.input_ids, seqs, self.device
            )
            inputs.inputs_embeds = inputs_embeds
            inputs.deepstack_input_embeds = deepstack
            inputs.positions = positions

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

        if (
            not is_prefill
            and self.cuda_graph_runner is not None
            and self.cuda_graph_runner.can_use(len(seqs))
            and positions.ndim == 1
        ):
            h = self.cuda_graph_runner.replay(
                inputs.input_ids,
                positions,
                inputs.slot_mapping,
                seq_lens=[len(s) for s in seqs],
                block_tables=[list(s.block_table) for s in seqs],
            )
            logits = self._logits_for_sample(h, inputs)
            return _sample(logits)

        seq_lens = _seq_lens(inputs, is_prefill, seqs)
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
                h = self.model(
                    input_ids=None if inputs_embeds is not None else inputs.input_ids,
                    positions=positions,
                    inputs_embeds=inputs_embeds,
                    deepstack_input_embeds=deepstack,
                )

        logits = self._logits_for_sample(h, inputs)
        return _sample(logits)

    def _logits_for_sample(self, hidden: torch.Tensor, inputs: ModelInputs) -> torch.Tensor:
        if inputs.is_prefill and inputs.cu_seqlens_q is not None:
            last_idx = inputs.cu_seqlens_q[1:] - 1
            hidden = hidden[last_idx]
        return self.model.compute_logits(hidden)

    @property
    def bytes_per_token(self) -> int:
        cfg = _text_cfg(self.config)
        nkv = cfg.num_key_value_heads
        hd = getattr(cfg, "head_dim", cfg.hidden_size // cfg.num_attention_heads)
        return 2 * cfg.num_hidden_layers * nkv * hd * self.dtype.itemsize

    @property
    def pool_nbytes(self) -> int:
        return self.kv_cache.nbytes

    @property
    def num_blocks(self) -> int:
        return self.kv_cache.shape[2]

    def destroy(self) -> None:
        if self.cuda_graph_runner is not None:
            self.cuda_graph_runner.destroy()
        if self.vit_cg_runner is not None:
            self.vit_cg_runner.destroy()
            if getattr(self.model, "_vit_cg", None) is not None:
                self.model._vit_cg = None


def _seq_lens(inputs: ModelInputs, is_prefill: bool, seqs: list[Sequence]) -> list[int]:
    if is_prefill:
        return [s.num_cached_tokens + s.num_scheduled_tokens for s in seqs]
    return [len(s) for s in seqs]


def _cu_q(seqs: list[Sequence]) -> list[int]:
    cu = [0]
    for s in seqs:
        cu.append(cu[-1] + s.num_scheduled_tokens)
    return cu
