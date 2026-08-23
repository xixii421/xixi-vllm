"""Engine：面向用户的推理门面（对齐 vLLM V1 的 ``LLMEngine``）。

负责 tokenizer、输入预处理（文本 → token ids、截断、max_tokens 收敛）与输出
组装（token ids → 文本、增量流式）。真正的调度与执行委托给 ``EngineCoreClient``。
"""

from __future__ import annotations

import warnings
from collections.abc import Iterator
from dataclasses import replace

import torch
from transformers import AutoTokenizer

from ..config import EngineConfig
from ..data import Sequence, SequenceStatus
from ..sampling_params import SamplingParams
from .client import InprocClient, make_client


class Engine:
    """LLM 推理引擎门面。

    对外主要接口：
        - add_request: 添加一条生成请求。
        - step: 执行一次调度、前向计算和采样。
        - generate_batch: 批量同步生成。
        - generate: 单条流式生成。
        - destroy: 释放引擎资源。
    """

    def __init__(self, config: EngineConfig, executor=None) -> None:
        self.config = config
        self.device = torch.device("cuda:0")
        self.dtype = self._resolve_dtype()

        # 禁用 BF16 matmul 的低精度 reduction，保证矩阵乘累加精度。
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False

        self.tokenizer = self._load_tokenizer()
        self.eos_token_id = int(self.tokenizer.eos_token_id)
        self.context_len = config.context_len
        self.block_size = config.block_size

        # executor 注入用于 torchrun 多机路径（已初始化的 rank0 执行器）。
        self._client = make_client(
            config,
            eos_token_id=self.eos_token_id,
            device=self.device,
            dtype=self.dtype,
            executor=executor,
        )
        # 门面侧持有的请求镜像：dp=1 时即调度器内活对象；dp>1 时按输出增量更新。
        self._seqs: dict[int, Sequence] = {}

    # ------------------------------------------------------------------ #
    # 初始化
    # ------------------------------------------------------------------ #

    def _resolve_dtype(self) -> torch.dtype:
        if self.config.dtype != "auto":
            return getattr(torch, self.config.dtype)
        dt = getattr(self.config.hf_config, "torch_dtype", None)
        return dt if isinstance(dt, torch.dtype) else torch.float16

    def _load_tokenizer(self):
        tokenizer = AutoTokenizer.from_pretrained(
            self.config.model, trust_remote_code=True
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        return tokenizer

    @property
    def scheduler(self):
        """暴露调度器（仅 dp=1 的 Inproc 客户端可用）。"""
        if not isinstance(self._client, InprocClient):
            raise NotImplementedError(
                "scheduler is only reachable for the in-process (dp=1) engine"
            )
        return self._client.engine_core.scheduler

    @property
    def last_num_cached_tokens(self) -> int:
        if not isinstance(self._client, InprocClient):
            return 0
        return self._client.engine_core.last_num_cached_tokens

    def reset_kv(self, prefix_backend: str | None = None) -> None:
        if not isinstance(self._client, InprocClient):
            raise NotImplementedError("reset_kv is only supported for dp=1")
        if prefix_backend is not None and prefix_backend not in ("none", "hash", "radix"):
            raise ValueError(f"invalid prefix_backend: {prefix_backend!r}")
        core = self._client.engine_core
        core.kv = core.executor.reset_kv(prefix_backend)
        core.scheduler = core._build_scheduler()
        core.last_num_cached_tokens = 0
        torch.cuda.empty_cache()

    # ------------------------------------------------------------------ #
    # 请求入队
    # ------------------------------------------------------------------ #

    def add_request(
        self,
        prompt: str | list[int],
        sampling_params: SamplingParams | None = None,
    ) -> Sequence:
        """添加一条生成请求。

        Returns:
            已加入调度器的 Sequence 对象（dp=1 时即调度器内活对象）。
        """
        sp = sampling_params or SamplingParams()
        token_ids = (
            list(prompt) if isinstance(prompt, list) else self.tokenizer.encode(prompt)
        )
        if not token_ids:
            raise ValueError("empty prompt")

        if len(token_ids) > self.context_len:
            warnings.warn(
                f"prompt length {len(token_ids)} > context_len={self.context_len}; "
                f"truncating to last {self.context_len} tokens",
                stacklevel=2,
            )
            token_ids = token_ids[-self.context_len :]

        max_ok = max(0, self.context_len - len(token_ids))
        if sp.max_tokens > max_ok:
            warnings.warn(
                f"max_tokens={sp.max_tokens} + prompt={len(token_ids)} exceeds "
                f"context_len={self.context_len}; clamping max_tokens to {max_ok}",
                stacklevel=2,
            )
            sp = replace(sp, max_tokens=max_ok)

        seq = Sequence.from_prompt(token_ids, block_size=self.block_size, sampling=sp)
        self._seqs[seq.seq_id] = seq
        self._client.add_request(seq)
        return seq

    # ------------------------------------------------------------------ #
    # 执行
    # ------------------------------------------------------------------ #

    def step(self) -> tuple[list[tuple[int, list[int]]], int]:
        """执行一步调度、前向计算和采样。

        Returns:
            (finished, metric)：本步完成的 [(seq_id, completion token ids)] + token 指标。
        """
        outputs, metric = self._client.step()

        # dp>1 时门面镜像需按输出增量推进；dp=1 的活对象已由调度器原地更新。
        if not isinstance(self._client, InprocClient):
            for out in outputs:
                seq = self._seqs.get(out.seq_id)
                if seq is None:
                    continue
                for tid in out.new_token_ids:
                    seq.append_token(tid)
                if out.is_finished:
                    seq.status = SequenceStatus.FINISHED

        finished: list[tuple[int, list[int]]] = []
        for out in outputs:
            if out.is_finished:
                seq = self._seqs.get(out.seq_id)
                if seq is not None:
                    finished.append(
                        (out.seq_id, seq.token_ids[seq.num_prompt_tokens :])
                    )
        return finished, metric

    def generate_batch(
        self,
        prompts: list[str | list[int]],
        sampling_params: SamplingParams | list[SamplingParams] | None = None,
    ) -> list[str]:
        """批量同步生成文本。"""
        sps = self._normalize_sampling_params(prompts, sampling_params)
        seqs = [self.add_request(p, sp) for p, sp in zip(prompts, sps)]
        finished_text: dict[int, str] = {}
        while self.has_unfinished():
            outputs, _ = self.step()
            for seq_id, completion_ids in outputs:
                finished_text[seq_id] = self.tokenizer.decode(completion_ids)
        return [finished_text[seq.seq_id] for seq in seqs]

    def generate(
        self,
        prompt: str | list[int],
        max_new_tokens: int = 128,
        temperature: float = 0.0,
        top_p: float | None = None,
        top_k: int | None = None,
        min_p: float | None = None,
    ) -> Iterator[str]:
        """单条流式生成。"""
        sp = SamplingParams(
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            min_p=min_p,
            max_tokens=max_new_tokens,
        )
        seq = self.add_request(prompt, sp)
        emitted = 0
        while not seq.is_finished:
            self.step()
            while emitted < seq.num_completion_tokens:
                tid = seq.token_ids[seq.num_prompt_tokens + emitted]
                emitted += 1
                yield self.tokenizer.decode([tid])

    def has_unfinished(self) -> bool:
        return self._client.has_unfinished()

    @staticmethod
    def _normalize_sampling_params(
        prompts: list[str | list[int]],
        sampling_params: SamplingParams | list[SamplingParams] | None,
    ) -> list[SamplingParams]:
        if sampling_params is None:
            return [SamplingParams() for _ in prompts]
        if isinstance(sampling_params, SamplingParams):
            return [sampling_params for _ in prompts]
        if len(sampling_params) != len(prompts):
            raise ValueError("sampling_params length must match prompts")
        return list(sampling_params)

    def destroy(self) -> None:
        self._client.destroy()
        self._seqs.clear()
