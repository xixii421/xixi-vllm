from __future__ import annotations

import torch

from ..config import EngineConfig
from ..data import Batch, KVManager, SchedulerOutput, SeqSchedule
from ..distributed import (
    IntermediateTensors,
    get_pp_group,
    get_tp_group,
    recv_tensor,
    send_tensor,
)
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
        self.hidden_size = config.hf_config.hidden_size
        self._meta = AttnMetadata(kv_cache=self.kv_cache)

        pp = get_pp_group()
        tp = get_tp_group()
        self.is_first_pp = pp.is_first_rank
        self.is_last_pp = pp.is_last_rank
        # 只有 last PP stage 的 tp rank 0 采样并回报，避免各 tp rank 随机采样不一致。
        self.is_driver = self.is_last_pp and tp.group_rank == 0

        max_bs = min(config.max_num_seqs, 256)
        max_blocks = (config.context_len + self.block_size - 1) // self.block_size
        self.max_bs = max_bs
        self.batch = Batch(
            max_bs=max_bs,
            max_tokens=max(config.max_num_batched_tokens, max_bs),
            max_blocks=max_blocks,
            vocab_size=config.hf_config.vocab_size,
            device=device,
            block_size=self.block_size,
        )

        # PP 静态中间张量 buffer：decode CUDA Graph 要求固定地址，非首 stage 才需要。
        self._pp_recv = self._make_pp_buffer(max_bs) if not self.is_first_pp else None
        # 末 stage 的 decode 输出写到静态 buffer，图重放后切片读取。
        self._decode_out = (
            torch.zeros(max_bs, self.hidden_size, dtype=self.dtype, device=device)
            if self.is_last_pp
            else None
        )

        # 采样结果异步 D2H：独立 copy stream + pinned buffer，采样 kernel 与拷贝解耦。
        self._copy_stream = torch.cuda.Stream(device=device)
        self._copy_event = torch.cuda.Event()
        self._sample_cpu = torch.empty(2 * max_bs, dtype=torch.int64, pin_memory=True)

        # decode full-pipeline CUDA Graph：recv → model(含 TP 通信) → send 整段捕获。
        self.cuda_graph_runner: CUDAGraphRunner | None = None
        if not config.enforce_eager:
            self.cuda_graph_runner = CUDAGraphRunner(
                CUDAGraphRunner.make_bs_list(max_bs), device
            )
            self.cuda_graph_runner.capture(self._decode_body)

        # prefill torch.compile：仅包 model forward，recv/send 保持 eager。
        self.prefill_fn = None
        if config.torch_compile:
            self.prefill_fn = self._build_compiled_prefill()

    # ------------------------------------------------------------------ #
    # PP 中间张量
    # ------------------------------------------------------------------ #

    def _make_pp_buffer(self, n: int) -> IntermediateTensors:
        return IntermediateTensors(
            hidden_states=torch.zeros(
                n, self.hidden_size, dtype=self.dtype, device=self.device
            ),
            residual=torch.zeros(
                n, self.hidden_size, dtype=self.dtype, device=self.device
            ),
        )

    def _pp_dst(self) -> int:
        return (get_pp_group().group_rank + 1) % get_pp_group().size

    def _pp_src(self) -> int:
        return (get_pp_group().group_rank - 1) % get_pp_group().size

    def _send_intermediate(self, out: IntermediateTensors) -> None:
        group = get_pp_group()
        send_tensor(out["hidden_states"], self._pp_dst(), group)
        send_tensor(out["residual"], self._pp_dst(), group)

    def _recv_intermediate(self, n: int) -> IntermediateTensors:
        """eager 路径：接收中间张量到新建张量（形状随 token 数变化）。"""
        group = get_pp_group()
        buf = self._make_pp_buffer(n)
        recv_tensor(buf["hidden_states"], self._pp_src(), group)
        recv_tensor(buf["residual"], self._pp_src(), group)
        return buf

    def _recv_intermediate_static(self, bs: int) -> IntermediateTensors:
        """CG 路径：接收进静态 buffer 的切片（固定地址供图重放）。"""
        group = get_pp_group()
        buf = self._pp_recv
        recv_tensor(buf["hidden_states"][:bs], self._pp_src(), group)
        recv_tensor(buf["residual"][:bs], self._pp_src(), group)
        return IntermediateTensors(
            hidden_states=buf["hidden_states"][:bs],
            residual=buf["residual"][:bs],
        )

    # ------------------------------------------------------------------ #
    # 一步执行
    # ------------------------------------------------------------------ #

    @torch.inference_mode()
    def run_pipeline(self, sched: SchedulerOutput):
        """执行一步前向（含 PP 编排）。

        入参为最小增量 SchedulerOutput（wire 格式），由 executor 在边界处
        从 ScheduledBatch 转换而来；单卡与多卡走同一消费路径。

        返回：
            - 非末 PP rank：None（中间张量已在图内 / eager 发出）。
            - 末 PP rank：(prefill 数, decode 数)，配合 read_samples 读取结果。
        """
        d = self._run_decode(sched.decodes) if sched.decodes else None
        p = self._run_prefill(sched.prefills) if sched.prefills else None
        if not self.is_last_pp:
            return None
        return self._stage_samples(p, d)

    # ------------------------------------------------------------------ #
    # decode
    # ------------------------------------------------------------------ #

    def _decode_attn(self, bs: int) -> AttnInputs:
        return AttnInputs(
            slot_mapping=self.batch.slot_mapping[:bs],
            is_prefill=False,
            cache_seqlens=self.batch.cache_seqlens[:bs],
            block_tables=self.batch.block_tables[:bs],
        )

    def _decode_body(self, bs: int):
        """decode 图体：recv → model → send，末 stage 写静态 out buffer。"""
        it = self._recv_intermediate_static(bs) if not self.is_first_pp else None
        with attention_context(self._meta):
            out = self.model(
                self.batch.input_ids[:bs],
                self.batch.positions[:bs],
                self._decode_attn(bs),
                it,
            )
        if self.is_last_pp:
            self._decode_out[:bs] = out
            return self._decode_out[:bs]
        self._send_intermediate(out)
        return out

    def _run_decode(self, seqs: list[SeqSchedule]):
        if not seqs:
            return None
        self.batch.build_decode(seqs)
        bs = len(seqs)

        if self.cuda_graph_runner is not None and bs <= self.max_bs:
            hidden, _ = self.cuda_graph_runner.replay(bs)
            if not self.is_last_pp:
                return None
            hidden = hidden[:bs]
        else:
            it = self._recv_intermediate(bs) if not self.is_first_pp else None
            with attention_context(self._meta):
                hidden = self.model(
                    self.batch.input_ids[:bs],
                    self.batch.positions[:bs],
                    self._decode_attn(bs),
                    it,
                )
            if not self.is_last_pp:
                self._send_intermediate(hidden)
                return None

        # compute_logits 含 all-gather，所有 tp rank 都必须调用完成集合通信。
        # all_greedy 走 argmax 快路径：只聚合局部最大/索引，跳过全量 logits 通信。
        if self.batch.all_greedy:
            ids = self.model.argmax(hidden)
            if not self.is_driver:
                return None
            return ids
        logits = self.model.compute_logits(hidden)
        if not self.is_driver:
            return None
        return self._sample(bs, logits)

    # ------------------------------------------------------------------ #
    # prefill
    # ------------------------------------------------------------------ #

    def _prefill_attn(
        self,
        bs: int,
        n_tokens: int,
        max_q: int,
        max_k: int,
        has_cache: bool,
    ) -> AttnInputs:
        return AttnInputs(
            slot_mapping=self.batch.slot_mapping[:n_tokens],
            is_prefill=True,
            cu_seqlens_q=self.batch.cu_seqlens_q[: bs + 1],
            cu_seqlens_k=self.batch.cu_seqlens_k[: bs + 1],
            max_seqlen_q=max_q,
            max_seqlen_k=max_k,
            block_tables=self.batch.block_tables[:bs] if has_cache else None,
        )

    def _run_prefill(self, seqs: list[SeqSchedule]):
        if not seqs:
            return None
        n_tokens, max_q, max_k, has_cache = self.batch.build_prefill(seqs)
        bs = len(seqs)
        it = self._recv_intermediate(n_tokens) if not self.is_first_pp else None

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
                    it,
                )
        else:
            with attention_context(self._meta):
                hidden = self.model(
                    self.batch.input_ids[:n_tokens],
                    self.batch.positions[:n_tokens],
                    self._prefill_attn(bs, n_tokens, max_q, max_k, has_cache),
                    it,
                )

        if not self.is_last_pp:
            self._send_intermediate(hidden)
            return None
        if not any(s.need_sample for s in seqs):
            if not self.is_driver:
                return None
            return torch.zeros(bs, dtype=torch.int64, device=self.device)
        last = self.batch.cu_seqlens_q[1 : bs + 1] - 1
        # 与 decode 一致：all_greedy 走 argmax 快路径。
        if self.batch.all_greedy:
            ids = self.model.argmax(hidden[last])
            if not self.is_driver:
                return None
            return ids
        logits = self.model.compute_logits(hidden[last])
        if not self.is_driver:
            return None
        return self._sample(bs, logits)

    # ------------------------------------------------------------------ #
    # 采样
    # ------------------------------------------------------------------ #

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

    def _stage_samples(
        self, p: torch.Tensor | None, d: torch.Tensor | None
    ) -> tuple[int, int]:
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
        self._copy_event.synchronize()
        p_ids = self._sample_cpu[n_d : n_d + n_p].tolist() if n_p else []
        d_ids = self._sample_cpu[:n_d].tolist() if n_d else []
        return p_ids, d_ids

    # ------------------------------------------------------------------ #
    # prefill torch.compile
    # ------------------------------------------------------------------ #

    def _resolve_compile_mode(self) -> str:
        # 分布式下禁用 inductor 自带 cudagraph，避免与 NCCL 集合通信冲突；
        # 单卡保持 config 指定的 mode（如 reduce-overhead）。
        if get_tp_group().size == 1 and get_pp_group().size == 1:
            return self.config.compile_mode
        return "max-autotune-no-cudagraphs"

    def _prefill_forward(
        self,
        input_ids,
        positions,
        slot_mapping,
        cu_seqlens,
        max_seqlen,
        intermediate_tensors,
    ):
        attn = AttnInputs(
            slot_mapping=slot_mapping,
            is_prefill=True,
            cu_seqlens_q=cu_seqlens,
            cu_seqlens_k=cu_seqlens,
            max_seqlen_q=max_seqlen,
            max_seqlen_k=max_seqlen,
        )
        return self.model(input_ids, positions, attn, intermediate_tensors)

    def _build_compiled_prefill(self):
        fn = torch.compile(
            self._prefill_forward,
            mode=self._resolve_compile_mode(),
            dynamic=self.config.compile_dynamic,
        )
        n = min(256, self.config.context_len)
        ids = torch.arange(1, n + 1, dtype=torch.int64, device=self.device)
        pos = torch.arange(n, dtype=torch.int64, device=self.device)
        cu = torch.tensor([0, n], dtype=torch.int32, device=self.device)
        sm = torch.arange(n, dtype=torch.int32, device=self.device)
        it = self._make_pp_buffer(n) if not self.is_first_pp else None
        with attention_context(self._meta), torch.inference_mode():
            fn(ids, pos, sm, cu, n, it)
        torch.cuda.synchronize()
        self.kv_cache.zero_()
        return fn

    def destroy(self) -> None:
        if self.cuda_graph_runner is not None:
            self.cuda_graph_runner.destroy()
            self.cuda_graph_runner = None
        self.prefill_fn = None
