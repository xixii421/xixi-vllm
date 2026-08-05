from __future__ import annotations

import warnings
from collections.abc import Iterator

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
    def __init__(self, config: EngineConfig):
        self.config = config
        self.device = torch.device("cuda:0")
        self.dtype = self._resolve_dtype()

        self.tokenizer = AutoTokenizer.from_pretrained(config.model, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.eos_token_id = int(self.tokenizer.eos_token_id)
        self.context_len = config.context_len
        self.block_size = config.block_size

        # 目标 dtype 下构建（免 fp32 中间态）+ 跳过随机初始化（权重随后覆盖）。
        # 权重用 load_file 整分片读 + 预取；上 GPU 只搬数据不转 dtype。
        with default_dtype_context(self.dtype):
            with skip_param_init():
                self.model = Qwen3ForCausalLM(config.hf_config)
        load_model(self.model, config.model)
        self.model = self.model.to(dtype=self.dtype, device=self.device).eval()

        self.sampler = Sampler()
        self._init_kv()
        self.last_num_cached_tokens = 0
        self._idle_steps = 0
        self._max_idle_steps = 1024

    def _resolve_dtype(self) -> torch.dtype:
        if self.config.dtype != "auto":
            return getattr(torch, self.config.dtype)
        dt = getattr(self.config.hf_config, "torch_dtype", None)
        return dt if isinstance(dt, torch.dtype) else torch.float16

    def _resolve_num_blocks(self) -> int:
        if self.config.num_kvcache_blocks > 0:
            return self.config.num_kvcache_blocks
        n = KVManager.auto_num_blocks(
            self.config.hf_config,
            block_size=self.block_size,
            dtype=self.dtype,
            device=self.device,
            gpu_memory_utilization=self.config.gpu_memory_utilization,
            reserve_cg_headroom=not self.config.enforce_eager,
        )
        self.config.num_kvcache_blocks = n
        return n

    def _init_kv(self) -> None:
        # decode CG / piece prefill CG 都需要预留 dummy page
        use_cg = (not self.config.enforce_eager) or self.config.prefill_cuda_graph
        num_blocks = self._resolve_num_blocks()
        self.kv = KVManager(
            self.config.hf_config,
            num_blocks=num_blocks,
            block_size=self.block_size,
            device=self.device,
            dtype=self.dtype,
            prefix_backend=self.config.prefix_backend,  # type: ignore[arg-type]
            reserve_dummy_page=use_cg,
        )
        self.scheduler = Scheduler(
            self.kv,
            max_num_seqs=self.config.max_num_seqs,
            max_num_batched_tokens=self.config.max_num_batched_tokens,
            eos_token_id=self.eos_token_id,
            mix_prefill_decode=self.config.mix_prefill_decode,
        )
        self.model_runner = ModelRunner(
            self.model,
            self.kv,
            self.sampler,
            config=self.config,
            device=self.device,
        )

    def reset_kv(self, prefix_backend: str | None = None) -> None:
        destroy = getattr(self.model_runner, "destroy", None)
        if callable(destroy):
            destroy()
        if prefix_backend is not None:
            if prefix_backend not in ("none", "hash", "radix"):
                raise ValueError(f"invalid prefix_backend: {prefix_backend!r}")
            self.config.prefix_backend = prefix_backend
        self._init_kv()
        self.last_num_cached_tokens = 0
        torch.cuda.empty_cache()

    @property
    def kv_cache(self) -> torch.Tensor:
        return self.kv.kv_cache

    def add_request(
        self,
        prompt: str | list[int],
        sampling_params: SamplingParams | None = None,
    ) -> Sequence:
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
        if sp.max_tokens < 0:
            raise ValueError(f"max_tokens must be >= 0, got {sp.max_tokens}")
        if sp.max_tokens > max_ok:
            warnings.warn(
                f"max_tokens={sp.max_tokens} + prompt={len(token_ids)} exceeds "
                f"context_len={self.context_len}; clamping max_tokens to {max_ok}",
                stacklevel=2,
            )
            sp = SamplingParams(
                temperature=sp.temperature,
                top_p=sp.top_p,
                top_k=sp.top_k,
                min_p=sp.min_p,
                max_tokens=max_ok,
                ignore_eos=sp.ignore_eos,
            )

        seq = Sequence.from_prompt(
            token_ids, block_size=self.block_size, sampling=sp
        )
        self.scheduler.add(seq)
        return seq

    def step(self) -> tuple[list[tuple[int, list[int]]], int]:
        prefills, decodes = self.scheduler.schedule()
        if not prefills and not decodes:
            if not self.scheduler.is_finished():
                self._idle_steps += 1
                if self._idle_steps >= self._max_idle_steps:
                    raise RuntimeError(
                        "scheduler returned empty for too long while requests remain "
                        f"(idle_steps={self._idle_steps}). Likely KV pool exhaustion "
                        f"(free_blocks={self.kv.num_free})."
                    )
            return [], 0
        self._idle_steps = 0

        # Prefix-hit size from allocate() (not post-prefill num_cached).
        self.last_num_cached_tokens = self.kv.last_num_cached_tokens

        prefill_toks = sum(s.num_scheduled_tokens for s in prefills)
        metric = prefill_toks + len(decodes)

        p_ids, d_ids = self.model_runner.run(prefills, decodes)

        finished: list[Sequence] = []
        if decodes:
            finished.extend(
                self.scheduler.postprocess(decodes, d_ids, is_prefill=False)
            )
        if prefills:
            finished.extend(
                self.scheduler.postprocess(prefills, p_ids, is_prefill=True)
            )
        outputs = [
            (seq.seq_id, seq.token_ids[seq.num_prompt_tokens :]) for seq in finished
        ]
        return outputs, metric

    def generate_batch(
        self,
        prompts: list[str | list[int]],
        sampling_params: SamplingParams | list[SamplingParams] | None = None,
    ) -> list[str]:
        if sampling_params is None:
            sps = [SamplingParams() for _ in prompts]
        elif isinstance(sampling_params, SamplingParams):
            sps = [sampling_params for _ in prompts]
        else:
            sps = list(sampling_params)
            if len(sps) != len(prompts):
                raise ValueError("sampling_params length must match prompts")

        seqs = [self.add_request(p, sp) for p, sp in zip(prompts, sps)]
        finished_text: dict[int, str] = {}
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

    def destroy(self) -> None:
        self.model_runner.destroy()
