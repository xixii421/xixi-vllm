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
        self._copy_event = torch.cuda.Event()
        self._sample_cpu = torch.empty(2 * max_bs, dtype=torch.int64, pin_memory=True)

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
            self.cuda_graph_runner.capture()

        self.prefill_fn = None
        if config.torch_compile:
            self.prefill_fn = self._build_compiled_prefill()

    @torch.inference_mode()
    def run(self, sched: ScheduledBatch) -> tuple[int, int]:
        d = self._run_decode(sched.decodes) if sched.decodes else None
        p = self._run_prefill(sched.prefills) if sched.prefills else None
        return self._stage_samples(p, d)

    def _stage_samples(
        self, p: torch.Tensor | None, d: torch.Tensor | None
    ) -> tuple[int, int]:
        """把采样结果异步拷入 pinned buffer，返回 (prefill 数, decode 数)。"""
        n_p = 0 if p is None else p.numel()
        n_d = 0 if d is None else d.numel()
        cur = torch.cuda.current_stream()
        self._copy_stream.wait_stream(cur)
        with torch.cuda.stream(self._copy_stream):
            if d is not None:
                self._sample_cpu[:n_d].copy_(d, non_blocking=True)
            if p is not None:
                self._sample_cpu[n_d : n_d + n_p].copy_(p, non_blocking=True)
        self._copy_event.record(self._copy_stream)
        return n_p, n_d

    def read_samples(self, n_p: int, n_d: int) -> tuple[list[int], list[int]]:
        """同步读回采样结果。"""
        self._copy_event.synchronize()
        p_ids = self._sample_cpu[n_d : n_d + n_p].tolist() if n_p else []
        d_ids = self._sample_cpu[:n_d].tolist() if n_d else []
        return p_ids, d_ids

    def _run_decode(self, seqs: list[Sequence]) -> torch.Tensor:
        self.batch.build_decode(seqs)
        bs = len(seqs)
        runner = self.cuda_graph_runner
        if runner is not None and runner.can_use(bs):
            hidden = runner.replay(bs)
        else:
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
        ):
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

    def _build_compiled_prefill(self):
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
        with attention_context(self._meta), torch.inference_mode():
            fn(ids, pos, sm, cu, n)
        torch.cuda.synchronize()
        self.kv_cache.zero_()
        return fn

    def _prefill_forward(
        self, input_ids, positions, slot_mapping, cu_seqlens, max_seqlen
    ):
        attn = AttnInputs(
            slot_mapping=slot_mapping,
            is_prefill=True,
            cu_seqlens_q=cu_seqlens,
            cu_seqlens_k=cu_seqlens,
            max_seqlen_q=max_seqlen,
            max_seqlen_k=max_seqlen,
        )
        return self.model(input_ids, positions, attn)

    def destroy(self) -> None:
        if self.cuda_graph_runner is not None:
            self.cuda_graph_runner.destroy()
            self.cuda_graph_runner = None
        self.prefill_fn = None
