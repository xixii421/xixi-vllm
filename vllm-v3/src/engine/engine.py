from __future__ import annotations
import copy
import gc
import io
import warnings
from collections.abc import Iterator
from contextlib import redirect_stdout
from dataclasses import replace
import torch
from transformers import AutoTokenizer

from ..config import EngineConfig
from ..memory import KVManager, Sequence
from ..models.loader import default_dtype_context, load_model, skip_param_init
from ..models.qwen3 import Qwen3ForCausalLM
from ..sampling_params import SamplingParams
from .model_runner import ModelRunner
from .sampler import Sampler
from .scheduler import Scheduler


class Engine:
    """LLM 推理引擎。

    负责模型加载、tokenizer 初始化、KV cache 池管理、请求调度、模型执行和采样。

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

        self.model = self._load_model()
        torch.cuda.synchronize()
        torch.cuda.empty_cache()

        # 记录显存基线，用于后续测量 CUDA Graph 探针的显存占用。
        self._mem_baseline = torch.cuda.memory_reserved()

        self.sampler = Sampler()
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

    def _load_model(self) -> torch.nn.Module:
        """构建并加载 Qwen3 模型。

        在目标 dtype 下构建模型，跳过随机初始化，参数权重随后由 checkpoint 覆盖。
        """
        with default_dtype_context(self.dtype):
            with skip_param_init():
                model = Qwen3ForCausalLM(self.config.hf_config)
        load_model(model, self.config.model)
        return model.to(dtype=self.dtype, device=self.device).eval()

    def _make_kv(self, num_blocks: int) -> KVManager:
        """创建 KV cache 管理器。

        Args:
            num_blocks: KV cache 物理块数量。

        Returns:
            初始化完成的 KVManager 实例。

        Note:
            decode CUDA Graph 模式下会额外预留 dummy page，用于图捕获时的稳定占位。
        """
        return KVManager(
            self.config.hf_config,
            num_blocks=num_blocks,
            block_size=self.block_size,
            device=self.device,
            dtype=self.dtype,
            prefix_backend=self.config.prefix_backend,
            reserve_dummy_page=not self.config.enforce_eager,
        )

    def _resolve_num_blocks(self, reserve_bytes: int) -> int:
        """根据显存预算自动计算可分配的 KV cache 块数。

        Args:
            reserve_bytes: 需要预先预留的显存字节数，通常用于 decode CUDA Graph 足迹。

        Returns:
            可安全分配的 KV cache block 数量。
        """
        return KVManager.auto_num_blocks(
            self.config.hf_config,
            block_size=self.block_size,
            dtype=self.dtype,
            device=self.device,
            gpu_memory_utilization=self.config.gpu_memory_utilization,
            reserve_bytes=reserve_bytes,
        )

    def _probe_cg_footprint(self) -> int:
        """探针捕获 decode CUDA Graph，测量其真实显存足迹。

        该足迹包括 FlashInfer workspace、CUDA Graph 静态 buffer 和 graph pool。
        测量完成后会销毁探针资源，最终 KV cache 池按剩余显存进行分配。

        Returns:
            decode CUDA Graph 相关的显存占用字节数。
        """
        probe_config = copy.copy(self.config)
        probe_config.torch_compile = False
        kv = self._make_kv(num_blocks=8)
        with redirect_stdout(io.StringIO()):
            runner = ModelRunner(
                self.model, kv, self.sampler, config=probe_config, device=self.device
            )
        torch.cuda.synchronize()
        footprint = torch.cuda.memory_reserved() - self._mem_baseline
        runner.destroy()
        del runner, kv
        gc.collect()
        torch.cuda.empty_cache()
        return int(footprint)

    def _init_kv(self) -> None:
        """初始化或重建 KV cache 池、Scheduler 和 ModelRunner。"""
        use_cg = not self.config.enforce_eager
        if self.config.num_kvcache_blocks > 0:
            num_blocks = self.config.num_kvcache_blocks
        else:
            # 自动计算 KV cache 块数；CUDA Graph 模式下先测量其显存足迹作为预留。
            reserve = self._probe_cg_footprint() if use_cg else 0
            num_blocks = self._resolve_num_blocks(reserve)
            self.config.num_kvcache_blocks = num_blocks  # 缓存结果，供 reset_kv 复用
            if use_cg:
                print(
                    f"CG footprint measured: {reserve / 2**30:.3f} GiB; "
                    f"KV pool sized to remainder ({num_blocks} blocks)"
                )
        self.kv = self._make_kv(num_blocks)
        self.scheduler = Scheduler(
            self.kv,
            max_num_seqs=self.config.max_num_seqs,
            max_num_batched_tokens=self.config.max_num_batched_tokens,
            eos_token_id=self.eos_token_id,
            mix_prefill_decode=self.config.mix_prefill_decode,
        )
        self.model_runner = ModelRunner(
            self.model, self.kv, self.sampler, config=self.config, device=self.device
        )
        self.last_num_cached_tokens = 0

    def reset_kv(self, prefix_backend: str | None = None) -> None:
        """重建 KV cache 池、调度器和 ModelRunner。

        Args:
            prefix_backend: 可选的新前缀缓存后端，仅支持 "none"、"hash"、"radix"。

        Raises:
            ValueError: prefix_backend 非法。
        """
        self.model_runner.destroy()
        if prefix_backend is not None:
            if prefix_backend not in ("none", "hash", "radix"):
                raise ValueError(f"invalid prefix_backend: {prefix_backend!r}")
            self.config.prefix_backend = prefix_backend
        self._init_kv()
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
        prefills, decodes = self.scheduler.schedule()
        if not prefills and not decodes:
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
        prefill_toks = sum(s.num_scheduled_tokens for s in prefills)
        metric = prefill_toks + len(decodes)

        p_ids, d_ids = self.model_runner.run(prefills, decodes)

        # 分别处理 decode 与 prefill 的后处理，并收集已完成序列。
        finished: list[Sequence] = []
        if decodes:
            finished.extend(self.scheduler.postprocess(decodes, d_ids, is_prefill=False))
        if prefills:
            finished.extend(self.scheduler.postprocess(prefills, p_ids, is_prefill=True))
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
        """释放 ModelRunner 及其持有的 CUDA Graph、buffer 等资源。"""
        self.model_runner.destroy()