from __future__ import annotations
import time
from typing import NamedTuple
import torch
from ..config import EngineConfig
from ..core.cuda_graph import CUDAGraphRunner
from ..layers.attention import AttentionMetadata, AttnInputs, attention_context
from ..layers.attention_backend import (
    BackendConfig,
    FlashInferBackend,
    compute_paged_kv_metadata,
)
from ..memory import KVManager, Sequence
from ..utils import h2d
from .input_builder import ModelInputs, build_model_inputs, prepare_unified
from .sampler import Sampler


class SamplingInputs(NamedTuple):
    """一次采样所需的批量化参数。

    所有参数均打包为 GPU 张量；字段为 None 表示整批未启用对应采样策略。
    """

    temperatures: torch.Tensor      # [batch_size] 采样温度。
    top_ps: torch.Tensor | None     # [batch_size] nucleus 采样阈值；None 表示未启用。
    top_ks: torch.Tensor | None     # [batch_size] top-k 采样阈值；None 表示未启用。
    min_ps: torch.Tensor | None     # [batch_size] min-p 采样阈值；None 表示未启用。


class ModelRunner:
    """模型执行器。

    负责将调度器给出的 prefill / decode 序列转换为模型输入，
    并选择合适的执行路径完成前向计算与采样。
    """

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

        # 首 chunk FA2 前向使用的 attention 元数据；wrapper 为 None 时走 flash-attn varlen。
        self._fa2_meta = AttentionMetadata(kv_cache_pool=self.kv_cache, wrapper=None)

        self._warmup()
        self.cuda_graph_runner: CUDAGraphRunner | None = None
        if not config.enforce_eager:
            self.cuda_graph_runner = self._capture_cg()

        self.prefill_fn = None
        if config.torch_compile:
            self.prefill_fn = self._build_compiled_prefill()

    def _head_dim(self) -> int:
        """返回 attention head 维度。"""
        cfg = self.config.hf_config
        return getattr(cfg, "head_dim", cfg.hidden_size // cfg.num_attention_heads)

    def _dummy_page_id(self) -> int:
        """获取 CUDA Graph 捕获所需的 dummy page id。"""
        dummy = self.kv.dummy_page_id
        if dummy < 0:
            raise RuntimeError("CUDA Graph capture requires a reserved dummy page")
        return dummy

    def _make_backend(self) -> FlashInferBackend:
        """根据模型配置创建 FlashInfer attention backend。"""
        cfg = self.config.hf_config
        return FlashInferBackend(
            BackendConfig(
                num_qo_heads=cfg.num_attention_heads,
                num_kv_heads=cfg.num_key_value_heads,
                head_dim=self._head_dim(),
                block_size=self.block_size,
                dtype=self.dtype,
                device=self.device,
                use_tensor_cores=self.config.flashinfer_use_tensor_cores,
            )
        )

    def _warmup(self) -> None:
        """执行一次小规模前向预热，初始化 kernel 和显存分配路径。"""
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
            bs, indptr_cpu, idx_gpu, last_cpu, True, self.kv_cache, cu_cpu
        )
        attn = AttnInputs(slot_mapping=sm, cu_seqlens=cu, max_seqlen=n, use_fa2=False)
        with attention_context(meta):
            self.model(input_ids=ids, positions=pos, attn=attn)

        self.kv_cache.zero_()
        torch.cuda.empty_cache()

    def _capture_cg(self) -> CUDAGraphRunner:
        """捕获 decode CUDA Graph，并返回对应的 graph runner。"""
        cfg = self.config.hf_config
        max_bs = min(self.config.max_num_seqs, 256)
        dummy = self._dummy_page_id()
        runner = CUDAGraphRunner(
            model=self.model,
            max_batch_size=max_bs,
            max_model_len=self.config.context_len,
            hidden_size=cfg.hidden_size,
            device=self.device,
            dtype=self.dtype,
            backend=self.backend,
            kv_cache_pool=self.kv_cache,
            dummy_page_id=dummy,
        )
        runner.capture()
        print(f"CUDA Graph captured for decode (max_bs={max_bs}, dummy_page={dummy})")
        return runner

    def _prefill_forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        slot_mapping: torch.Tensor,
        cu_seqlens: torch.Tensor,
        max_seqlen: int,
    ) -> torch.Tensor:
        """首 chunk prefill 前向。

        该函数作为 torch.compile 的编译单元，只接受纯 tensor / int 输入，
        使模型前向能够编译成单一计算图。
        """
        attn = AttnInputs(slot_mapping, cu_seqlens, max_seqlen, True)
        return self.model(input_ids, positions, attn)

    def _build_compiled_prefill(self):
        """编译 prefill 前向，并在启动阶段执行一次预热。"""
        t0 = time.perf_counter()
        fn = torch.compile(
            self._prefill_forward,
            mode=self.config.compile_mode,
            dynamic=self.config.compile_dynamic,
        )
        n = min(256, self.config.context_len)
        ids = torch.arange(1, n + 1, dtype=torch.int64, device=self.device)
        pos = torch.arange(n, dtype=torch.int64, device=self.device)
        cu = torch.tensor([0, n], dtype=torch.int32, device=self.device)
        sm = torch.arange(n, dtype=torch.int32, device=self.device)
        with torch.inference_mode():
            with attention_context(self._fa2_meta):
                fn(ids, pos, sm, cu, n)
        torch.cuda.synchronize()
        self.kv_cache.zero_()
        print(
            f"torch.compile prefill ready in {time.perf_counter() - t0:.2f}s "
            f"(mode={self.config.compile_mode})"
        )
        return fn

    @torch.inference_mode()
    def run(
        self, prefills: list[Sequence], decodes: list[Sequence]
    ) -> tuple[list[int], list[int]]:
        """执行一步前向与采样。

        Args:
            prefills: 本步需要执行 prefill 的序列。
            decodes: 本步需要执行 decode 的序列。

        Returns:
            tuple:
                - prefill 序列采样得到的 token ids。
                - decode 序列采样得到的 token ids。
        """
        # eager 模式下，混合 prefill / decode 批次合并为一次统一前向。
        if prefills and decodes and self.cuda_graph_runner is None:
            return self._run_unified(prefills, decodes)

        # CUDA Graph 模式下，prefill 与 decode 分别执行。
        d_ids = self._run_batch(decodes, is_prefill=False) if decodes else []
        p_ids = self._run_batch(prefills, is_prefill=True) if prefills else []
        return p_ids, d_ids

    def _run_unified(
        self, prefills: list[Sequence], decodes: list[Sequence]
    ) -> tuple[list[int], list[int]]:
        """将 decode 与 prefill 展平为一次统一前向。"""
        inputs, kv_lens = prepare_unified(
            decodes, prefills, self.block_size, self.device
        )
        inputs.is_prefill = True
        seqs = inputs.seqs
        sampling = self._build_sampling_inputs(seqs)
        hidden = self._paged_forward(seqs, inputs, kv_lens, is_prefill=True, use_fa2=False)
        tokens = self._sample(sampling, hidden, inputs)
        n_dec = len(decodes)
        return tokens[n_dec:], tokens[:n_dec]

    def _run_batch(self, seqs: list[Sequence], *, is_prefill: bool) -> list[int]:
        """执行一批同阶段序列，并返回采样 token ids。"""
        if not seqs:
            return []
        inputs = build_model_inputs(seqs, is_prefill, self.block_size, self.device)
        sampling = self._build_sampling_inputs(seqs)
        hidden = self._forward(seqs, inputs, is_prefill)

        # 若本步没有序列完成 prompt，则无需执行 LM head 和采样。
        if is_prefill and not self._need_sample(seqs):
            return [0] * len(seqs)
        return self._sample(sampling, hidden, inputs)

    def _sample(
        self, sampling: SamplingInputs, hidden: torch.Tensor, inputs: ModelInputs
    ) -> list[int]:
        """从 hidden states 提取待采样 logits，并执行整批采样。"""
        logits = self._logits_for_sample(hidden, inputs)
        return self.sampler(
            logits,
            sampling.temperatures,
            top_ps=sampling.top_ps,
            top_ks=sampling.top_ks,
            min_ps=sampling.min_ps,
        ).tolist()

    @staticmethod
    def _need_sample(seqs: list[Sequence]) -> bool:
        """判断是否有序列在本步完成 prompt，从而需要采样。"""
        return any(
            s.num_cached_tokens + s.num_scheduled_tokens >= s.num_prompt_tokens
            for s in seqs
        )

    @staticmethod
    def _is_first_chunk(seqs: list[Sequence]) -> bool:
        """判断是否为首 chunk prefill，即没有 prefix cache 命中。"""
        return all(s.num_cached_tokens == 0 for s in seqs)

    def _build_sampling_inputs(self, seqs: list[Sequence]) -> SamplingInputs:
        """将每条序列的采样参数打包为整批 GPU 张量。"""
        temps = h2d([s.temperature for s in seqs], torch.float32, self.device)

        def pack(attr: str, default, dtype: torch.dtype) -> torch.Tensor | None:
            """打包单个采样参数；若整批均未启用则返回 None。"""
            vals = [getattr(s, attr) for s in seqs]
            if all(v is None for v in vals):
                return None
            arr = [default if v is None else v for v in vals]
            return h2d(arr, dtype, self.device)

        return SamplingInputs(
            temperatures=temps,
            top_ps=pack("top_p", 1.0, torch.float32),
            top_ks=pack("top_k", 0, torch.int32),
            min_ps=pack("min_p", 0.0, torch.float32),
        )

    def _forward(
        self,
        seqs: list[Sequence],
        inputs: ModelInputs,
        is_prefill: bool,
    ) -> torch.Tensor:
        """选择前向执行路径并返回整批 hidden states。"""
        if not is_prefill:
            hidden = self._decode_cg(seqs, inputs)
            return hidden if hidden is not None else self._eager_forward(seqs, inputs, False)

        # 首 chunk prefill 优先使用 torch.compile 路径。
        if self.prefill_fn is not None and self._is_first_chunk(seqs):
            return self._compiled_prefill(inputs)
        return self._eager_forward(seqs, inputs, True)

    def _compiled_prefill(self, inputs: ModelInputs) -> torch.Tensor:
        """执行 torch.compile 编译后的首 chunk prefill 前向。"""
        with attention_context(self._fa2_meta):
            return self.prefill_fn(
                inputs.input_ids,
                inputs.positions,
                inputs.slot_mapping,
                inputs.cu_seqlens_q,
                inputs.max_seqlen_q,
            )

    def _decode_cg(self, seqs: list[Sequence], inputs: ModelInputs) -> torch.Tensor | None:
        """尝试使用 CUDA Graph 重放 decode 前向。"""
        runner = self.cuda_graph_runner
        if runner is None or not runner.can_use(len(seqs)):
            return None
        return runner.replay(
            inputs.input_ids,
            inputs.positions,
            inputs.slot_mapping,
            seq_lens=[len(s) for s in seqs],
            block_tables=[s.block_table for s in seqs],
        )

    def _eager_forward(
        self, seqs: list[Sequence], inputs: ModelInputs, is_prefill: bool
    ) -> torch.Tensor:
        """执行 eager 模式前向。"""
        if is_prefill:
            seq_lens = [s.num_cached_tokens + s.num_scheduled_tokens for s in seqs]
            use_fa2 = self._is_first_chunk(seqs)
        else:
            seq_lens = [len(s) for s in seqs]
            use_fa2 = False
        return self._paged_forward(seqs, inputs, seq_lens, is_prefill, use_fa2)

    def _build_paged_meta(
        self,
        seqs: list[Sequence],
        inputs: ModelInputs,
        seq_lens: list[int],
        is_prefill: bool,
    ) -> AttentionMetadata:
        """构造 FlashInfer paged KV attention 元数据。"""
        indptr_cpu, idx_gpu, last_cpu = compute_paged_kv_metadata(
            seq_lens, inputs.block_tables, self.block_size, self.device
        )
        cu_cpu = None
        if is_prefill:
            cu = [0]
            for s in seqs:
                cu.append(cu[-1] + s.num_scheduled_tokens)
            cu_cpu = torch.tensor(cu, dtype=torch.int32, device="cpu")
        return self.backend.build_metadata(
            len(seqs), indptr_cpu, idx_gpu, last_cpu, is_prefill, self.kv_cache, cu_cpu
        )

    def _paged_forward(
        self,
        seqs: list[Sequence],
        inputs: ModelInputs,
        seq_lens: list[int],
        is_prefill: bool,
        use_fa2: bool,
    ) -> torch.Tensor:
        """构造 attention 元数据并执行模型前向。"""
        meta = (
            self._fa2_meta
            if use_fa2
            else self._build_paged_meta(seqs, inputs, seq_lens, is_prefill)
        )
        attn = AttnInputs(
            slot_mapping=inputs.slot_mapping,
            cu_seqlens=inputs.cu_seqlens_q,
            max_seqlen=inputs.max_seqlen_q,
            use_fa2=use_fa2,
        )
        with attention_context(meta):
            return self.model(input_ids=inputs.input_ids, positions=inputs.positions, attn=attn)

    def _logits_for_sample(self, hidden: torch.Tensor, inputs: ModelInputs) -> torch.Tensor:
        """提取待采样位置的 hidden states，并计算 logits。"""
        # prefill 时只取每个序列的最后一个 token；decode 时 hidden 已是每序列一行。
        if inputs.is_prefill and inputs.cu_seqlens_q is not None:
            last_idx = inputs.cu_seqlens_q[1:] - 1
            hidden = hidden[last_idx]
        return self.model.compute_logits(hidden)

    def destroy(self) -> None:
        """释放 CUDA Graph runner 持有的资源。"""
        if self.cuda_graph_runner is not None:
            self.cuda_graph_runner.destroy()