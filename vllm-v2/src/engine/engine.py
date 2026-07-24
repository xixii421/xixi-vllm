from __future__ import annotations

from collections.abc import Iterator
from typing import Literal

import torch
from transformers import AutoTokenizer

from ..config import EngineConfig
from ..kv import KVManager, Sequence
from ..model import Qwen3ForCausalLM
from ..model_loader import load_model
from ..sampling_params import SamplingParams
from ..sampler import Sampler
from .model_runner import ModelRunner
from .scheduler import Scheduler


class Engine:
    def __init__(self, config: EngineConfig) -> None:
        self.config = config
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.dtype = self._resolve_dtype()

        self.tokenizer = AutoTokenizer.from_pretrained(
            config.model, trust_remote_code=True
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.eos_token_id = int(self.tokenizer.eos_token_id)
        self.context_len = config.context_len
        self.block_size = config.block_size

        self.model = Qwen3ForCausalLM(config.hf_config)
        load_model(self.model, config.model)
        self.model = self.model.to(dtype=self.dtype, device=self.device).eval()
        if not config.enforce_eager and self.device.type == "cuda":
            self.model = torch.compile(self.model, mode="reduce-overhead")

        self.sampler = Sampler()
        self._init_kv()
        self.last_num_cached_tokens = 0

    def _init_kv(self) -> None:
        self.kv = KVManager(
            self.model,
            num_blocks=self.config.num_kv_blocks,
            block_size=self.config.block_size,
            prefix_backend=self.config.prefix_backend,
            device=self.device,
            dtype=self.dtype,
        )
        self.scheduler = Scheduler(
            self.kv,
            max_num_seqs=self.config.max_num_seqs,
            max_num_batched_tokens=self.config.max_num_batched_tokens,
            eos_token_id=self.eos_token_id,
            mix_prefill_decode=self.config.mix_prefill_decode,
        )
        self.model_runner = ModelRunner(
            self.model, self.kv, self.sampler, device=self.device
        )

    def reset_kv(
        self, prefix_backend: Literal["none", "hash", "radix"] | None = None
    ) -> None:
        """清空 KV / 调度队列；可选切换 prefix_backend。供测试复用同一模型。"""
        if prefix_backend is not None:
            if prefix_backend not in ("none", "hash", "radix"):
                raise ValueError(f"invalid prefix_backend: {prefix_backend!r}")
            self.config.prefix_backend = prefix_backend
        self._init_kv()
        self.last_num_cached_tokens = 0
        if self.device.type == "cuda":
            torch.cuda.empty_cache()

    def _resolve_dtype(self) -> torch.dtype:
        if self.config.dtype != "auto":
            return getattr(torch, self.config.dtype)
        cfg = self.config.hf_config
        dt = getattr(cfg, "dtype", None) or getattr(cfg, "torch_dtype", None)
        return dt if isinstance(dt, torch.dtype) else torch.float16

    def add_request(
        self,
        prompt: str | list[int],
        sampling_params: SamplingParams | None = None,
    ) -> Sequence:
        token_ids = (
            list(prompt) if isinstance(prompt, list) else self.tokenizer.encode(prompt)
        )
        if len(token_ids) > self.context_len:
            token_ids = token_ids[-self.context_len :]
        seq = Sequence.from_prompt(
            token_ids, block_size=self.block_size, sampling=sampling_params
        )
        self.scheduler.add(seq)
        return seq

    def step(self) -> tuple[list[tuple[int, list[int]]], int]:
        prefills, decodes = self.scheduler.schedule()
        if not prefills and not decodes:
            return [], 0

        prefill_toks = sum(s.num_scheduled_tokens for s in prefills)
        metric = prefill_toks + len(decodes)

        # Decode first so in-flight requests keep moving; then prefill chunks.
        p_ids, d_ids = self.model_runner.run(prefills, decodes)

        finished: list = []
        if decodes:
            finished.extend(
                self.scheduler.postprocess(decodes, d_ids, is_prefill=False)
            )
        if prefills:
            finished.extend(
                self.scheduler.postprocess(prefills, p_ids, is_prefill=True)
            )
            self.last_num_cached_tokens = max(
                (s.num_cached_tokens for s in prefills), default=0
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
        elif isinstance(sampling_params, (list, tuple)):
            sps = list(sampling_params)
            if len(sps) != len(prompts):
                raise ValueError("sampling_params length must match prompts")
        else:
            sps = [sampling_params for _ in prompts]

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
