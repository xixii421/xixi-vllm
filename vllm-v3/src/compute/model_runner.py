from __future__ import annotations

import torch

from ..config import EngineConfig
from ..data import Batch, KVManager, ScheduledBatch
from ..data.sequence import Sequence
from .cuda_graph import CUDAGraphRunner
from .layers.attention import AttnInputs, AttnMetadata, attention_context
from .sampler import Sampler


class ModelRunner:
    def __init__(
        self,
        model,
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
        self._meta = AttnMetadata(kv_cache=self.kv_cache)
        # attention 上下文元数据，持有 kv_cache 引用，给 attention 算子用

        max_bs = min(config.max_num_seqs, 256)
        max_blocks = (config.context_len + self.block_size - 1) // self.block_size
        self.batch = Batch(
            max_bs=max_bs,
            max_tokens=max(config.max_num_batched_tokens, max_bs),
            max_blocks=max_blocks,
            vocab_size=config.hf_config.vocab_size,
            device=device,
        )

        # 采样结果异步 D2H：独立 copy stream + pinned buffer，采样 kernel 与拷贝解耦。
        self._copy_stream = torch.cuda.Stream(device=device)
        # cuda.Stream GPU上的任务队列 _copy_stream 是一条专用流，做 GPU CPU 拷贝
        self._copy_event = torch.cuda.Event()
        # cuda.Event GPU上的标记点，用来做跨流同步
        self._sample_cpu = torch.empty(2 * max_bs, dtype=torch.int64, pin_memory=True)
        # CPU侧的接收缓冲区
        # pin_memory 锁页内存，GPU DMA 可用直接访问 D2H 拷贝速度明显更快，且支持异步拷贝

        self.cuda_graph_runner: CUDAGraphRunner | None = None
        if not config.enforce_eager:
            self.cuda_graph_runner = CUDAGraphRunner(
                model=self.model,
                batch=self.batch,
                meta=self._meta,
                max_batch_size=max_bs,
                hidden_size=config.hf_config.hidden_size,
                dtype=self.dtype,
                device=device,
            )
            self.cuda_graph_runner.capture() # 实例时直接捕获

        self.prefill_fn = None # compile 函数
        if config.torch_compile:
            self.prefill_fn = self._build_compiled_prefill()
            # 构建 compile 包装函数
            # 这里构建的是无前缀缓存的场景

    @torch.inference_mode() # 推理模式
    def run(self, sched: ScheduledBatch) -> tuple[int, int]:
        d = self._run_decode(sched.decodes) if sched.decodes else None
        p = self._run_prefill(sched.prefills) if sched.prefills else None
        return self._stage_samples(p, d)

    def _stage_samples(
        self, p: torch.Tensor | None, d: torch.Tensor | None
    ) -> tuple[int, int]:
        """ 把采样结果异步拷入 pinned buffer，返回 (prefill 数, decode 数) """
        n_p = 0 if p is None else p.numel()
        n_d = 0 if d is None else d.numel()
        cur = torch.cuda.current_stream() # 当前 GPU 默认流
        self._copy_stream.wait_stream(cur) # 拷贝任务需要等待主任务流完成才能继续
        with torch.cuda.stream(self._copy_stream):
            if d is not None:
                self._sample_cpu[:n_d].copy_(d, non_blocking=True)
            if p is not None:
                self._sample_cpu[n_d : n_d + n_p].copy_(p, non_blocking=True)
        self._copy_event.record(self._copy_stream) # 拷贝任务完成标记，用于同步
        return n_p, n_d

    def read_samples(self, n_p: int, n_d: int) -> tuple[list[int], list[int]]:
        """ 同步读回采样结果给 CPU """
        self._copy_event.synchronize()
        p_ids = self._sample_cpu[n_d : n_d + n_p].tolist() if n_p else []
        d_ids = self._sample_cpu[:n_d].tolist() if n_d else []
        return p_ids, d_ids

    def _run_decode(self, seqs: list[Sequence]) -> torch.Tensor:
        self.batch.build_decode(seqs)
        bs = len(seqs)
        runner = self.cuda_graph_runner
        if runner is not None and runner.can_use(bs):
            # CUDA Graph 路径
            hidden = runner.replay(bs)
        else:
            # 正常 model 前向路径
            attn = AttnInputs(
                slot_mapping=self.batch.slot_mapping[:bs],
                is_prefill=False,
                cache_seqlens=self.batch.cache_seqlens[:bs],
                block_tables=self.batch.block_tables[:bs],
            )
            with attention_context(self._meta):
                hidden = self.model(
                    self.batch.input_ids[:bs], self.batch.positions[:bs], attn
                )
        logits = self.model.compute_logits(hidden)
        return self._sample(bs, logits)

    def _run_prefill(self, seqs: list[Sequence]) -> torch.Tensor:
        n_tokens, max_q, max_k, has_cache = self.batch.build_prefill(seqs)
        bs = len(seqs)
        if self.prefill_fn is not None and all(
            s.num_cached_tokens == 0 for s in seqs
        ):  # 存在 compile 包装函数且所有请求都没有前缀缓存
            with attention_context(self._meta):
                hidden = self.prefill_fn(
                    self.batch.input_ids[:n_tokens],
                    self.batch.positions[:n_tokens],
                    self.batch.slot_mapping[:n_tokens],
                    self.batch.cu_seqlens_q[: bs + 1],
                    max_q,
                )
        else:
            attn = AttnInputs(
                slot_mapping=self.batch.slot_mapping[:n_tokens],
                is_prefill=True,
                cu_seqlens_q=self.batch.cu_seqlens_q[: bs + 1],
                cu_seqlens_k=self.batch.cu_seqlens_k[: bs + 1],
                max_seqlen_q=max_q,
                max_seqlen_k=max_k,
                block_tables=self.batch.block_tables[:bs] if has_cache else None,
            )
            with attention_context(self._meta):
                hidden = self.model(
                    self.batch.input_ids[:n_tokens],
                    self.batch.positions[:n_tokens],
                    attn,
                )
        if not any(s.need_sample for s in seqs):
            # 不生成 token
            return torch.zeros(bs, dtype=torch.int64, device=self.device)
        last = self.batch.cu_seqlens_q[1 : bs + 1] - 1
        logits = self.model.compute_logits(hidden[last])
        return self._sample(bs, logits)

    def _sample(self, bs: int, logits: torch.Tensor) -> torch.Tensor:
        return self.sampler(
            logits,
            self.batch.temperatures[:bs],
            top_ps=self.batch.top_ps[:bs],
            top_ks=self.batch.top_ks[:bs],
            min_ps=self.batch.min_ps[:bs],
            all_greedy=self.batch.all_greedy,
            any_greedy=self.batch.any_greedy,
            use_top_k=self.batch.use_top_k,
            use_top_p=self.batch.use_top_p,
            use_min_p=self.batch.use_min_p,
            max_top_k=self.batch.max_top_k,
        )
        # 不采用 GPU CPU 同步，不在 forward 做分支判断
        # CPU 传布尔值，GPU 在 batch 负责逻辑判断

    def _build_compiled_prefill(self):
        """ 构建 prefill 阶段 compile 函数 """
        fn = torch.compile(
            self._prefill_forward,
            mode=self.config.compile_mode,
            dynamic=self.config.compile_dynamic,
        )
        # 假输入
        n = min(256, self.config.context_len)
        ids = torch.arange(1, n + 1, dtype=torch.int64, device=self.device)
        pos = torch.arange(n, dtype=torch.int64, device=self.device)
        cu = torch.tensor([0, n], dtype=torch.int32, device=self.device)
        sm = torch.arange(n, dtype=torch.int32, device=self.device)
        with attention_context(self._meta), torch.inference_mode():
            fn(ids, pos, sm, cu, n) # 热身
        torch.cuda.synchronize()
        self.kv_cache.zero_()
        return fn

    def _prefill_forward(
        self, input_ids, positions, slot_mapping, cu_seqlens, max_seqlen
    ):
        """ 一次 prefill 逻辑，供 compile 包装"""
        attn = AttnInputs(
            slot_mapping=slot_mapping,
            is_prefill=True,
            cu_seqlens_q=cu_seqlens, # 累计序列长度
            cu_seqlens_k=cu_seqlens,
            max_seqlen_q=max_seqlen, # 单条 prompt 最大长度
            max_seqlen_k=max_seqlen,
        )
        return self.model(input_ids, positions, attn)

    def destroy(self) -> None:
        if self.cuda_graph_runner is not None:
            self.cuda_graph_runner.destroy()
            self.cuda_graph_runner = None
        self.prefill_fn = None
