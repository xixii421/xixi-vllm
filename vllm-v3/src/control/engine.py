from __future__ import annotations
import gc
import warnings
from collections.abc import Iterator
from dataclasses import replace
import torch
from transformers import AutoTokenizer

from ..compute.executor import ModelExecutor
from ..config import EngineConfig
from ..data import KVManager, Sequence
from ..sampling_params import SamplingParams
from .scheduler import Scheduler


class Engine:
    """LLM 推理引擎（控制面）。

    负责 tokenizer 初始化、请求调度与采样后处理；模型加载、KV 池分配与
    前向执行由 ModelExecutor 承担。

    对外主要接口：
        - add_request: 添加一条生成请求。
        - step: 执行一次调度、前向计算和采样。
        - generate_batch: 批量同步生成。
        - generate: 单条流式生成。
        - destroy: 释放引擎资源。
    """

    # 仍有请求但调度器持续返回空批时，超过该步数判定异常。
    # 常见原因是 KV cache 池耗尽，导致请求无法继续调度。
    _MAX_IDLE_STEPS = 1024

    def __init__(self, config: EngineConfig):
        self.config = config
        self.device = torch.device("cuda:0")
        self.dtype = self._resolve_dtype()

        # 禁用 BF16 matmul 的低精度 reduction，保证矩阵乘累加精度。
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False

        self.tokenizer = self._load_tokenizer()
        self.eos_token_id = int(self.tokenizer.eos_token_id)
        self.context_len = config.context_len
        self.block_size = config.block_size

        self.executor = ModelExecutor(config, self.device, self.dtype)
        self._idle_steps = 0
        self._init_kv()

    # ------------------------------------------------------------------ #
    # 初始化
    # ------------------------------------------------------------------ #

    def _resolve_dtype(self) -> torch.dtype:
        """解析模型计算 dtype。

        优先级：
            1. EngineConfig.dtype 显式指定。
            2. HuggingFace config 中的 torch_dtype。
            3. 默认 float16。
        """
        if self.config.dtype != "auto":
            return getattr(torch, self.config.dtype)
        dt = getattr(self.config.hf_config, "torch_dtype", None)
        return dt if isinstance(dt, torch.dtype) else torch.float16

    def _load_tokenizer(self):
        """加载 tokenizer，并确保 pad_token 可用。"""
        tokenizer = AutoTokenizer.from_pretrained(
            self.config.model, trust_remote_code=True
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        return tokenizer

    def _build_scheduler(self) -> Scheduler:
        return Scheduler(
            self.kv,
            max_num_seqs=self.config.max_num_seqs,
            max_num_batched_tokens=self.config.max_num_batched_tokens,
            eos_token_id=self.eos_token_id,
            mix_prefill_decode=self.config.mix_prefill_decode,
        )

    def _init_kv(self) -> None:
        """初始化 KV 池与调度器（KV 池与 runner 由 executor 装配）。"""
        self.kv = self.executor.init_kv()
        self.scheduler = self._build_scheduler()
        self.last_num_cached_tokens = 0

    def reset_kv(self, prefix_backend: str | None = None) -> None:
        """重建 KV cache 池、调度器和 ModelRunner。

        Args:
            prefix_backend: 可选的新前缀缓存后端，仅支持 "none"、"hash"、"radix"。

        Raises:
            ValueError: prefix_backend 非法。
        """
        if prefix_backend is not None and prefix_backend not in ("none", "hash", "radix"):
            raise ValueError(f"invalid prefix_backend: {prefix_backend!r}")
        self.kv = self.executor.reset_kv(prefix_backend)
        self.scheduler = self._build_scheduler()
        self.last_num_cached_tokens = 0
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

        Args:
            prompt: 文本 prompt 或 token id 列表。
            sampling_params: 采样参数；为 None 时使用默认采样参数。

        Returns:
            已加入调度器的 Sequence 对象。

        Raises:
            ValueError: prompt 为空。
        """
        sp = sampling_params or SamplingParams()
        # 将输入统一转换为 token id 列表。
        token_ids = (
            list(prompt) if isinstance(prompt, list) else self.tokenizer.encode(prompt)
        )
        if not token_ids:
            raise ValueError("empty prompt")

        # prompt 超长时保留末尾 token，以尽量保留最近上下文。
        if len(token_ids) > self.context_len:
            warnings.warn(
                f"prompt length {len(token_ids)} > context_len={self.context_len}; "
                f"truncating to last {self.context_len} tokens",
                stacklevel=2,
            )
            token_ids = token_ids[-self.context_len :]

        # 限制 max_tokens，确保 prompt + completion 不超过 context_len。
        max_ok = max(0, self.context_len - len(token_ids))
        if sp.max_tokens > max_ok:
            warnings.warn(
                f"max_tokens={sp.max_tokens} + prompt={len(token_ids)} exceeds "
                f"context_len={self.context_len}; clamping max_tokens to {max_ok}",
                stacklevel=2,
            )
            sp = replace(sp, max_tokens=max_ok)

        seq = Sequence.from_prompt(token_ids, block_size=self.block_size, sampling=sp)
        self.scheduler.add(seq)
        return seq

    # ------------------------------------------------------------------ #
    # 执行
    # ------------------------------------------------------------------ #

    def step(self) -> tuple[list[tuple[int, list[int]]], int]:
        """执行一步调度、前向计算和采样。

        Returns:
            tuple:
                finished: [(seq_id, 本步新生成的 completion token ids)]。
                metric: 本步处理 token 数，用于吞吐统计。
        """
        # 调度可执行的 prefill 与 decode 请求。
        sched = self.scheduler.schedule()
        if sched.is_empty():
            # 仍有请求但调度为空：累计空转次数；超过阈值报错，通常表示 KV cache 池耗尽。
            if not self.scheduler.is_finished():
                self._idle_steps += 1
                if self._idle_steps >= self._MAX_IDLE_STEPS:
                    raise RuntimeError(
                        "scheduler returned empty for too long while requests remain "
                        f"(idle_steps={self._idle_steps}). Likely KV pool exhaustion "
                        f"(free_blocks={self.kv.num_free})."
                    )
            return [], 0
        self._idle_steps = 0

        # 记录最近一次 prefix cache 命中 token 数，供外部基准脚本读取。
        self.last_num_cached_tokens = self.kv.last_num_cached_tokens

        # 本步 token 指标：prefill token 数 + decode 序列数。
        prefill_toks = sum(s.num_scheduled_tokens for s in sched.prefills)
        metric = prefill_toks + len(sched.decodes)

        n_p, n_d = self.executor.run(sched)
        p_ids, d_ids = self.executor.read_samples(n_p, n_d)

        # 分别处理 decode 与 prefill 的后处理，并收集已完成序列。
        finished: list[Sequence] = []
        if sched.decodes:
            finished.extend(self.scheduler.postprocess(sched.decodes, d_ids, is_prefill=False))
        if sched.prefills:
            finished.extend(self.scheduler.postprocess(sched.prefills, p_ids, is_prefill=True))
        # 构造对外输出：仅返回 prompt 之后新生成的 token ids。
        outputs = [
            (seq.seq_id, seq.token_ids[seq.num_prompt_tokens :]) for seq in finished
        ]
        return outputs, metric

    def generate_batch(
        self,
        prompts: list[str | list[int]],
        sampling_params: SamplingParams | list[SamplingParams] | None = None,
    ) -> list[str]:
        """批量同步生成文本。

        Args:
            prompts: 文本 prompt 或 token id 列表组成的 batch。
            sampling_params: 单个采样参数，或与 prompts 长度一致的采样参数列表；
                None 使用默认采样参数。

        Returns:
            与 prompts 顺序一致的生成文本列表。
        """
        # 归一化采样参数，并将所有请求加入调度器。
        sps = self._normalize_sampling_params(prompts, sampling_params)
        seqs = [self.add_request(p, sp) for p, sp in zip(prompts, sps)]
        finished_text: dict[int, str] = {}
        # 持续执行调度步骤，直到所有请求完成。
        while not self.scheduler.is_finished():
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
        """单条流式生成。

        Args:
            prompt: 文本 prompt 或 token id 列表。
            max_new_tokens: 最大新生成 token 数。
            temperature: 采样温度；0 表示贪心采样。
            top_p: nucleus sampling 阈值。
            top_k: top-k sampling 阈值。
            min_p: min-p sampling 阈值。

        Yields:
            每次新生成一个 token 对应的解码文本。
        """
        # 构造采样参数并提交请求。
        sp = SamplingParams(
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            min_p=min_p,
            max_tokens=max_new_tokens,
        )
        seq = self.add_request(prompt, sp)
        emitted = 0
        # 每次 step 后增量解码已经新生成的 token。
        while not seq.is_finished:
            self.step()
            while emitted < seq.num_completion_tokens:
                tid = seq.token_ids[seq.num_prompt_tokens + emitted]
                emitted += 1
                yield self.tokenizer.decode([tid])

    @staticmethod
    def _normalize_sampling_params(
        prompts: list[str | list[int]],
        sampling_params: SamplingParams | list[SamplingParams] | None,
    ) -> list[SamplingParams]:
        """将 sampling_params 归一化为与 prompts 长度一致的列表。"""
        if sampling_params is None:
            return [SamplingParams() for _ in prompts]
        if isinstance(sampling_params, SamplingParams):
            return [sampling_params for _ in prompts]
        if len(sampling_params) != len(prompts):
            raise ValueError("sampling_params length must match prompts")
        return list(sampling_params)

    def destroy(self) -> None:
        """释放 executor（模型、KV 池、runner）与调度器占用的显存。"""
        executor = getattr(self, "executor", None)
        if executor is not None:
            executor.destroy()
        self.kv = None
        self.scheduler = None
        gc.collect()
        torch.cuda.empty_cache()