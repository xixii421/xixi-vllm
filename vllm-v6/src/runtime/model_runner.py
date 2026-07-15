from __future__ import annotations
import torch
from torch import nn
from ..config import EngineConfig
from ..cuda_graph.decode import CUDAGraphRunner
from ..distributed import destroy_distributed_environment, destroy_model_parallel, get_pp_group, get_pp_indices, get_tp_group, init_distributed_environment, initialize_model_parallel, is_initialized
from ..layers.attention import attention_context
from ..layers.attention_backend import BackendConfig, FlashInferBackend, compute_paged_kv_metadata
from ..distributed.pp import IntermediateTensors
from ..scheduling import Sequence
from ..models.loader import load_model
from ..models.qwen3 import Qwen3ForCausalLM
from ..context import forward_context
from .model_inputs import ModelInputs, build_model_inputs
from .sampler import Sampler

def _text_cfg(config: EngineConfig):
    return config.text_config

def ensure_single_process_parallel(config: EngineConfig, local_device: int) -> None:
    if is_initialized():
        return
    init_distributed_environment(word_size=1, rank=0, backend='nccl', init_method=f'tcp://127.0.0.1:{config.nccl_port}', local_device=local_device)
    initialize_model_parallel(1, 1, 0, 0)

class ModelRunner:

    def __init__(self, config: EngineConfig, device: torch.device, rank: int=0, tp_rank: int=0, pp_rank: int=0) -> None:
        self.config = config
        self.device = device
        self.rank = rank
        self.tp_rank = tp_rank
        self.pp_rank = pp_rank
        self.block_size = config.kvcache_block_size
        self.dtype = self._resolve_dtype(config)
        self.is_multimodal = config.is_multimodal
        if config.world_size == 1 and (not is_initialized()):
            ensure_single_process_parallel(config, device.index or 0)
        cfg = _text_cfg(config)
        self.tp_size = get_tp_group().size
        self.pp_size = get_pp_group().size
        self.start_layer, self.end_layer = get_pp_indices(cfg.num_hidden_layers, get_pp_group().group_rank, get_pp_group().size)
        self.num_layers = self.end_layer - self.start_layer
        self.num_qo_heads = cfg.num_attention_heads // self.tp_size
        self.num_kv_heads = max(1, cfg.num_key_value_heads // self.tp_size)
        self.head_dim = getattr(cfg, 'head_dim', cfg.hidden_size // cfg.num_attention_heads)
        if self.is_multimodal:
            from ..models.qwen3_vl import Qwen3VLForCausalLM
            self.model = Qwen3VLForCausalLM(config.hf_config, mm_encoder_tp_mode=config.mm_encoder_tp_mode)
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
        if config.dtype != 'auto':
            return getattr(torch, config.dtype)
        cfg = _text_cfg(config)
        dt = getattr(cfg, 'dtype', None) or getattr(cfg, 'torch_dtype', None) or getattr(config.hf_config, 'dtype', None) or getattr(config.hf_config, 'torch_dtype', None)
        return dt if isinstance(dt, torch.dtype) else torch.float16

    def _alloc_kv(self) -> torch.Tensor:
        torch.cuda.empty_cache()
        torch.cuda.synchronize(self.device)
        nkv = self.num_kv_heads
        hd = self.head_dim
        layers = self.num_layers
        block_bytes = 2 * layers * self.block_size * nkv * hd * self.dtype.itemsize
        if self.config.num_kvcache_blocks > 0:
            num_blocks = self.config.num_kvcache_blocks
        else:
            free, total = torch.cuda.mem_get_info(self.device)
            available = int(total * self.config.gpu_memory_utilization) - (total - free)
            if not self.config.enforce_eager:
                available -= int(2.0 * 1024 ** 3)
            num_blocks = max(4, available // block_bytes)
            if self.rank == 0 or self.config.world_size == 1:
                self.config.num_kvcache_blocks = num_blocks
        if not self.config.enforce_eager and num_blocks < 2:
            raise RuntimeError(f'CUDA Graph requires >= 2 KV blocks after allocation, got {num_blocks}')
        kv = torch.empty(2, layers, num_blocks, self.block_size, nkv, hd, dtype=self.dtype, device=self.device)
        if self.rank == 0 or self.config.world_size == 1:
            print(f'KV pool: {num_blocks} blocks × {self.block_size} tok layers=[{self.start_layer}:{self.end_layer}] kv_heads={nkv} ({num_blocks * block_bytes / 1024 ** 3:.2f} GiB)')
        return kv

    def _make_backend(self) -> FlashInferBackend:
        return FlashInferBackend(BackendConfig(num_qo_heads=self.num_qo_heads, num_kv_heads=self.num_kv_heads, head_dim=self.head_dim, block_size=self.block_size, dtype=self.dtype, device=self.device))

    def _warmup(self) -> None:
        torch.cuda.empty_cache()
        n = min(128, self.config.context_len)
        bs = 2
        total = bs * n
        n_blocks = (n + self.block_size - 1) // self.block_size
        need = bs * n_blocks
        usable = max(1, self.kv_cache.shape[2] - (0 if self.config.enforce_eager else 1))
        if need > usable:
            n_blocks = max(1, usable // bs)
            n = min(n, n_blocks * self.block_size)
            total = bs * n
        ids = torch.arange(1, total + 1, dtype=torch.int64, device=self.device)
        pos = torch.arange(total, dtype=torch.int64, device=self.device)
        cu = torch.tensor([0, n, total], dtype=torch.int32, device=self.device)
        cu_cpu = torch.tensor([0, n, total], dtype=torch.int32, device='cpu')
        sm = torch.arange(total, dtype=torch.int32, device=self.device)
        bt = torch.full((bs, n_blocks), -1, dtype=torch.int32, device=self.device)
        bt[0, :n_blocks] = torch.arange(n_blocks, device=self.device)
        bt[1, :n_blocks] = torch.arange(n_blocks, 2 * n_blocks, device=self.device)
        assert int(bt.max().item()) < usable
        indptr_cpu, idx_gpu, last_cpu = compute_paged_kv_metadata([n, n], bt, self.block_size, self.device)
        meta = self.backend.build_metadata(bs, indptr_cpu, idx_gpu, last_cpu, sm, True, self.kv_cache, cu_cpu)
        with forward_context(is_prefill=True, seqs=[], cu_seqlens_q=cu, cu_seqlens_k=cu, max_seqlen_q=n, max_seqlen_k=n, block_tables=bt, slot_mapping=sm):
            with attention_context(meta):
                inter = None
                if not get_pp_group().is_first_rank:
                    inter = IntermediateTensors(hidden_states=torch.zeros(total, _text_cfg(self.config).hidden_size, dtype=self.dtype, device=self.device), residual=torch.zeros(total, _text_cfg(self.config).hidden_size, dtype=self.dtype, device=self.device))
                self.model(input_ids=ids if get_pp_group().is_first_rank else None, positions=pos, intermediate_tensors=inter)
        self.kv_cache.zero_()
        torch.cuda.empty_cache()

    def _capture_cg(self) -> CUDAGraphRunner:
        max_bs = min(self.config.max_num_seqs, 256)
        dummy_page_id = self.kv_cache.shape[2] - 1
        first = get_pp_group().is_first_rank
        last = get_pp_group().is_last_rank
        if first and last:
            pp_role = 'full'
        elif first:
            pp_role = 'first'
        elif last:
            pp_role = 'last'
        else:
            pp_role = 'middle'
        runner = CUDAGraphRunner(model=self.model, max_batch_size=max_bs, max_model_len=self.config.context_len, block_size=self.block_size, hidden_size=_text_cfg(self.config).hidden_size, num_kv_heads=self.num_kv_heads, head_dim=self.head_dim, device=self.device, dtype=self.dtype, backend=self.backend, kv_cache_pool=self.kv_cache, dummy_page_id=dummy_page_id, pp_role=pp_role)
        runner.capture()
        print(f'CUDA Graph captured for decode (max_bs={max_bs}, dummy_page={dummy_page_id}, pp_role={pp_role}, tp={self.tp_size})')
        return runner

    def _capture_vit_cg(self):
        from ..cuda_graph.vit import ViTCUDAGraphRunner
        assert self.is_multimodal
        if not get_pp_group().is_first_rank or getattr(self.model, 'vision_tower', None) is None:
            return None
        cfg = self.model.get_vit_cg_config()
        cfg.min_budget = self.config.vit_cg_min_budget
        cfg.max_budget = self.config.vit_cg_max_budget
        runner = ViTCUDAGraphRunner(vit_forward=self.model.vit_forward_cg, metadata_fn=self.model.vision_tower.prepare_metadata, config=cfg, device=self.device, dtype=self.dtype)
        runner.capture()
        self.model._vit_cg = runner
        return runner

    @torch.inference_mode()
    def run(self, seqs: list[Sequence], is_prefill: bool, intermediate_tensors: IntermediateTensors | None=None) -> tuple[list[int] | IntermediateTensors, list | None]:
        from .mrope_sync import MRopeStateUpdate, snapshot_mrope_updates
        mrope_updates: list[MRopeStateUpdate] | None = None
        if not seqs:
            empty: list[int] | IntermediateTensors = [] if get_pp_group().is_last_rank else IntermediateTensors()
            return (empty, None)
        inputs = build_model_inputs(seqs, is_prefill, self.block_size, self.device)
        inputs_embeds = None
        deepstack = None
        positions = inputs.positions
        mm_prefill = is_prefill and self.is_multimodal and any((s.has_multimodal for s in seqs))
        if mm_prefill:
            if get_pp_group().is_first_rank:
                inputs_embeds, deepstack, positions = self.model.encode_and_merge(inputs.input_ids, seqs, self.device)
                inputs.inputs_embeds = inputs_embeds
                inputs.deepstack_input_embeds = deepstack
                inputs.positions = positions
            else:
                from ..multimodal import mrope_positions
                vc = getattr(self.config.hf_config, 'vision_config', None)
                merge_size = int(getattr(vc, 'spatial_merge_size', 2)) if vc else 2
                positions = mrope_positions(inputs.input_ids, seqs, merge_size, self.device)
                inputs.positions = positions
            mrope_updates = snapshot_mrope_updates(seqs)
        temperatures = torch.tensor([s.temperature for s in seqs], dtype=torch.float32, pin_memory=True).to(self.device, non_blocking=True)
        top_ps = torch.tensor([1.0 if s.top_p is None else float(s.top_p) for s in seqs], dtype=torch.float32, pin_memory=True).to(self.device, non_blocking=True)
        top_ks = torch.tensor([0 if s.top_k is None else int(s.top_k) for s in seqs], dtype=torch.int32, pin_memory=True).to(self.device, non_blocking=True)
        min_ps = torch.tensor([0.0 if s.min_p is None else float(s.min_p) for s in seqs], dtype=torch.float32, pin_memory=True).to(self.device, non_blocking=True)

        def _sample(logits: torch.Tensor) -> list[int]:
            return self.sampler(logits, temperatures, top_ps=top_ps, top_ks=top_ks, min_ps=min_ps).tolist()
        if not is_prefill and self.cuda_graph_runner is not None and self.cuda_graph_runner.can_use(len(seqs)) and (positions.ndim == 1):
            h = self.cuda_graph_runner.replay(inputs.input_ids if get_pp_group().is_first_rank else None, positions, inputs.slot_mapping, seq_lens=[len(s) for s in seqs], block_tables=[list(s.block_table) for s in seqs], intermediate_tensors=intermediate_tensors)
            if isinstance(h, IntermediateTensors):
                return (h, mrope_updates)
            logits = self._logits_for_sample(h, inputs)
            return (_sample(logits), mrope_updates)
        seq_lens = _seq_lens(inputs, is_prefill, seqs)
        indptr_cpu, idx_gpu, last_cpu = compute_paged_kv_metadata(seq_lens, inputs.block_tables if inputs.block_tables is not None else torch.zeros(len(seqs), 1, dtype=torch.int32, device=self.device), self.block_size, self.device)
        cu_cpu = torch.tensor(_cu_q(seqs), dtype=torch.int32, device='cpu') if is_prefill else None
        meta = self.backend.build_metadata(len(seqs), indptr_cpu, idx_gpu, last_cpu, inputs.slot_mapping, is_prefill, self.kv_cache, cu_cpu)
        with forward_context(is_prefill=is_prefill, seqs=seqs, slot_mapping=inputs.slot_mapping, cu_seqlens_q=inputs.cu_seqlens_q, cu_seqlens_k=inputs.cu_seqlens_k, max_seqlen_q=inputs.max_seqlen_q, max_seqlen_k=inputs.max_seqlen_k, context_lens=inputs.context_lens, block_tables=inputs.block_tables):
            with attention_context(meta):
                h = self.model(input_ids=None if inputs_embeds is not None or not get_pp_group().is_first_rank else inputs.input_ids, positions=positions, inputs_embeds=inputs_embeds if get_pp_group().is_first_rank else None, deepstack_input_embeds=deepstack if get_pp_group().is_first_rank else None, intermediate_tensors=intermediate_tensors)
        if isinstance(h, IntermediateTensors):
            return (h, mrope_updates)
        logits = self._logits_for_sample(h, inputs)
        return (_sample(logits), mrope_updates)

    def _logits_for_sample(self, hidden: torch.Tensor, inputs: ModelInputs) -> torch.Tensor:
        if inputs.is_prefill and inputs.cu_seqlens_q is not None:
            last_idx = inputs.cu_seqlens_q[1:] - 1
            hidden = hidden[last_idx]
        return self.model.compute_logits(hidden)

    @property
    def bytes_per_token(self) -> int:
        return 2 * self.num_layers * self.num_kv_heads * self.head_dim * self.dtype.itemsize

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
            if getattr(self.model, '_vit_cg', None) is not None:
                self.model._vit_cg = None
            self.vit_cg_runner = None
        if self.config.world_size == 1 and is_initialized():
            try:
                import torch
                torch.cuda.synchronize()
            except Exception:
                pass
            destroy_model_parallel()
            destroy_distributed_environment()

def _seq_lens(inputs: ModelInputs, is_prefill: bool, seqs: list[Sequence]) -> list[int]:
    if is_prefill:
        return [s.num_cached_tokens + s.num_scheduled_tokens for s in seqs]
    return [len(s) for s in seqs]

def _cu_q(seqs: list[Sequence]) -> list[int]:
    cu = [0]
    for s in seqs:
        cu.append(cu[-1] + s.num_scheduled_tokens)
    return cu
