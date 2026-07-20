"""GPU model step orchestrator (load → prepare → forward/sample)."""

from __future__ import annotations

import torch

from ...config import EngineConfig
from ...cuda_graph.decode import CUDAGraphRunner
from ...distributed import (
    destroy_distributed_environment,
    destroy_model_parallel,
    get_pp_group,
    get_tp_group,
    is_initialized,
)
from ...distributed.pp import IntermediateTensors
from ...layers.attention import attention_context
from ...layers.attention_backend import (
    BackendConfig,
    FlashInferBackend,
    compute_paged_kv_metadata,
)
from ...scheduling import Sequence
from ..context import forward_context
from .forward_sample import (
    forward_cuda_graph,
    forward_eager,
    sample_from_hidden,
)
from .load_kv import (
    alloc_kv_cache,
    build_and_load_model,
    ensure_single_process_parallel,
    layer_shard_meta,
    resolve_dtype,
)
from .sampler import Sampler
from .step_prepare import build_attn_meta, prepare_step

__all__ = ["ModelRunner", "ensure_single_process_parallel"]


class ModelRunner:
    def __init__(
        self,
        config: EngineConfig,
        device: torch.device,
        rank: int = 0,
        tp_rank: int = 0,
        pp_rank: int = 0,
    ) -> None:
        self.config = config
        self.device = device
        self.rank = rank
        self.tp_rank = tp_rank
        self.pp_rank = pp_rank
        self.block_size = config.kvcache_block_size
        self.dtype = resolve_dtype(config)
        self.is_multimodal = config.is_multimodal
        if config.world_size == 1 and (not is_initialized()):
            ensure_single_process_parallel(config, device.index or 0)

        (
            self.start_layer,
            self.end_layer,
            self.num_layers,
            self.num_qo_heads,
            self.num_kv_heads,
            self.head_dim,
        ) = layer_shard_meta(config)
        self.tp_size = get_tp_group().size
        self.pp_size = get_pp_group().size

        self.model = build_and_load_model(
            config,
            device=device,
            dtype=self.dtype,
            rank=rank,
            tp_rank=tp_rank,
            tp_size=self.tp_size,
            pp_size=self.pp_size,
        )
        self.kv_cache = alloc_kv_cache(
            config=config,
            device=device,
            dtype=self.dtype,
            num_layers=self.num_layers,
            num_kv_heads=self.num_kv_heads,
            head_dim=self.head_dim,
            start_layer=self.start_layer,
            end_layer=self.end_layer,
            rank=rank,
            decode_cg_planned=self._decode_cg_planned(),
        )
        self.sampler = Sampler().to(device=self.device)
        self._sample_host = torch.empty(
            config.max_num_seqs, dtype=torch.int64, pin_memory=True
        )
        self._sample_done = torch.cuda.Event(blocking=False)
        self.backend = FlashInferBackend(
            BackendConfig(
                num_heads=self.num_qo_heads,
                num_kv_heads=self.num_kv_heads,
                head_dim=self.head_dim,
                block_size=self.block_size,
                dtype=self.dtype,
                device=self.device,
            )
        )
        self._warmup()
        self.cuda_graph_runner: CUDAGraphRunner | None = None
        self.vit_cg_runner = None
        if self._decode_cg_planned():
            self.cuda_graph_runner = self._capture_cg()
        if self.is_multimodal and config.enable_vit_cuda_graph:
            self.vit_cg_runner = self._capture_vit_cg()

    def _decode_cg_planned(self) -> bool:
        if self.config.enforce_eager:
            return False
        if not self.config.is_moe:
            return True
        from ...config.moe_backend import moe_decode_cg_ok
        from ...layers.moe import get_moe_parallel_config

        return bool(self.config.moe_cuda_graph) and moe_decode_cg_ok(
            get_moe_parallel_config().fused_backend
        )

    def _warmup(self) -> None:
        torch.cuda.empty_cache()
        # MoE cutlass: keep warmup small; dense keeps a slightly larger touch.
        if self.config.is_moe:
            n, bs = min(16, self.config.context_len), 1
        else:
            n, bs = min(128, self.config.context_len), 2
        total = bs * n
        n_blocks = (n + self.block_size - 1) // self.block_size
        need = bs * n_blocks
        usable = max(
            1, self.kv_cache.shape[2] - (0 if self.config.enforce_eager else 1)
        )
        if need > usable:
            n_blocks = max(1, usable // bs)
            n = min(n, n_blocks * self.block_size)
            total = bs * n
        ids = torch.arange(1, total + 1, dtype=torch.int64, device=self.device)
        pos = torch.arange(total, dtype=torch.int64, device=self.device)
        cu_lens = [0]
        for i in range(bs):
            cu_lens.append(cu_lens[-1] + n)
        cu = torch.tensor(cu_lens, dtype=torch.int32, device=self.device)
        cu_cpu = torch.tensor(cu_lens, dtype=torch.int32, device="cpu")
        sm = torch.arange(total, dtype=torch.int32, device=self.device)
        bt = torch.full((bs, n_blocks), -1, dtype=torch.int32, device=self.device)
        for i in range(bs):
            bt[i, :n_blocks] = torch.arange(
                i * n_blocks, (i + 1) * n_blocks, device=self.device
            )
        assert int(bt.max().item()) < usable
        indptr_cpu, idx_gpu, last_cpu = compute_paged_kv_metadata(
            [n] * bs, bt, self.block_size, self.device
        )
        meta = self.backend.build_metadata(
            indptr_cpu, idx_gpu, last_cpu, sm, True, self.kv_cache, cu_cpu
        )
        print(
            f"[warmup] rank={self.rank} pp={self.pp_rank} moe={self.config.is_moe} "
            f"bs={bs} n={n} device={self.device}",
            flush=True,
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
                inter = None
                if not get_pp_group().is_first_rank:
                    h = self.config.text_config.hidden_size
                    inter = IntermediateTensors(
                        hidden_states=torch.zeros(
                            total, h, dtype=self.dtype, device=self.device
                        ),
                        residual=torch.zeros(
                            total, h, dtype=self.dtype, device=self.device
                        ),
                    )
                self.model(
                    input_ids=ids if get_pp_group().is_first_rank else None,
                    positions=pos,
                    intermediate_tensors=inter,
                )
        print(f"[warmup] rank={self.rank} done", flush=True)
        self.kv_cache.zero_()
        torch.cuda.empty_cache()

    def _capture_cg(self) -> CUDAGraphRunner:
        max_bs = min(self.config.max_num_seqs, 256)
        dummy_page_id = self.kv_cache.shape[2] - 1
        first = get_pp_group().is_first_rank
        last = get_pp_group().is_last_rank
        if first and last:
            pp_role = "full"
        elif first:
            pp_role = "first"
        elif last:
            pp_role = "last"
        else:
            pp_role = "middle"
        runner = CUDAGraphRunner(
            model=self.model,
            max_batch_size=max_bs,
            max_model_len=self.config.context_len,
            block_size=self.block_size,
            hidden_size=self.config.text_config.hidden_size,
            head_dim=self.head_dim,
            device=self.device,
            dtype=self.dtype,
            backend=self.backend,
            kv_cache_pool=self.kv_cache,
            dummy_page_id=dummy_page_id,
            pp_role=pp_role,
        )
        runner.capture()
        return runner

    def _capture_vit_cg(self):
        from ...cuda_graph.vit import ViTCUDAGraphRunner

        assert self.is_multimodal
        if (
            not get_pp_group().is_first_rank
            or getattr(self.model, "vision_tower", None) is None
        ):
            return None
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
        if getattr(self.model, "mm_encoder", None) is not None:
            self.model.mm_encoder.vit_cg = runner
        return runner

    @torch.inference_mode()
    def run(
        self,
        seqs: list[Sequence],
        is_prefill: bool,
        intermediate_tensors: IntermediateTensors | None = None,
    ) -> tuple[list[int] | IntermediateTensors, list | None]:
        if not seqs:
            if get_pp_group().is_last_rank:
                return ([], None)
            h = self.config.text_config.hidden_size
            empty = IntermediateTensors(
                hidden_states=torch.empty(
                    0, h, dtype=self.dtype, device=self.device
                ),
                residual=torch.empty(0, h, dtype=self.dtype, device=self.device),
            )
            return (empty, None)

        # Always prepare tensors + sampling flags; skip attn_meta when CG likely.
        cg_likely = (
            not is_prefill
            and self.cuda_graph_runner is not None
            and self.cuda_graph_runner.can_use(len(seqs))
        )
        prepared = prepare_step(
            model=self.model,
            config=self.config,
            seqs=seqs,
            is_prefill=is_prefill,
            block_size=self.block_size,
            device=self.device,
            backend=self.backend,
            kv_cache=self.kv_cache,
            use_cuda_graph=cg_likely,
        )
        use_cg = (
            cg_likely
            and prepared.positions.ndim == 1
            and self.cuda_graph_runner is not None
            and self.cuda_graph_runner.can_use(len(seqs))
        )
        if use_cg:
            h = forward_cuda_graph(
                cuda_graph_runner=self.cuda_graph_runner,
                prepared=prepared,
                seqs=seqs,
                intermediate_tensors=intermediate_tensors,
            )
        else:
            # CG miss: reuse inputs/sampling/MM; only build FlashInfer metadata.
            if prepared.attn_meta is None:
                prepared.attn_meta = build_attn_meta(
                    prepared=prepared,
                    seqs=seqs,
                    is_prefill=is_prefill,
                    block_size=self.block_size,
                    device=self.device,
                    backend=self.backend,
                    kv_cache=self.kv_cache,
                )
            h = forward_eager(
                model=self.model,
                prepared=prepared,
                seqs=seqs,
                is_prefill=is_prefill,
                intermediate_tensors=intermediate_tensors,
            )

        if isinstance(h, IntermediateTensors):
            return (h, prepared.mrope_updates)
        tokens = sample_from_hidden(
            model=self.model,
            sampler=self.sampler,
            prepared=prepared,
            hidden=h,
            sample_host=self._sample_host,
            sample_done=self._sample_done,
            device=self.device,
        )
        return (tokens, prepared.mrope_updates)

    @property
    def bytes_per_token(self) -> int:
        return (
            2
            * self.num_layers
            * self.num_kv_heads
            * self.head_dim
            * self.dtype.itemsize
        )

    @property
    def pool_nbytes(self) -> int:
        return self.kv_cache.nbytes

    @property
    def num_blocks(self) -> int:
        return self.kv_cache.shape[2]

    def destroy(self) -> None:
        if self.cuda_graph_runner is not None:
            self.cuda_graph_runner.destroy()
            self.cuda_graph_runner = None
        if self.vit_cg_runner is not None:
            self.vit_cg_runner.destroy()
            enc = getattr(self.model, "mm_encoder", None)
            if enc is not None:
                enc.vit_cg = None
            self.vit_cg_runner = None
        if self.config.is_moe:
            from ...layers.moe import clear_fused_state, set_moe_parallel_config

            clear_fused_state()
            set_moe_parallel_config(None)
        if self.config.world_size == 1 and is_initialized():
            torch.cuda.synchronize()
            destroy_model_parallel()
            destroy_distributed_environment()
