from __future__ import annotations

from collections.abc import Iterator

import torch
from transformers import AutoTokenizer

from ..config import EngineConfig
from ..memory import PagedKVCache, Sequence
from ..models.loader import load_model
from ..models.qwen3 import Qwen3ForCausalLM
from ..sampling_params import SamplingParams
from .model_runner import ModelRunner
from .sampler import Sampler
from .scheduler import Scheduler


class Engine:
    def __init__(self, config: EngineConfig):
        self.config = config
        self.tokenizer = AutoTokenizer.from_pretrained(config.model, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.eos_token_id = self.tokenizer.eos_token_id
        self.context_len = config.context_len
        self.block_size = config.block_size

        self.device = torch.device("cuda:0")
        self.dtype = self._resolve_dtype()

        self.model = Qwen3ForCausalLM(config.hf_config)
        load_model(self.model, config.model)
        self.model = self.model.to(dtype=self.dtype, device=self.device)
        self.model.eval()

        if not config.enforce_eager:
            self.model = torch.compile(self.model, mode="reduce-overhead")

        self.sampler = Sampler().to(device=self.device)
        assert config.num_gpu_blocks is not None
        self.kv_cache = PagedKVCache.allocate(
            self.model,
            num_blocks=config.num_gpu_blocks,
            block_size=config.block_size,
            device=self.device,
            dtype=self.dtype,
        )

        self.scheduler = Scheduler(
            max_num_seqs=config.max_num_seqs,
            max_num_batched_tokens=config.max_num_batched_tokens,
            num_blocks=config.num_gpu_blocks,
            block_size=config.block_size,
            eos_token_id=int(self.eos_token_id),
        )
        self.model_runner = ModelRunner(
            self.model,
            self.kv_cache,
            self.sampler,
            block_size=config.block_size,
            device=self.device,
        )

    def _resolve_dtype(self):
        if self.config.dtype != "auto":
            return getattr(torch, self.config.dtype)
        dt = self.config.hf_config.torch_dtype
        return dt if isinstance(dt, torch.dtype) else torch.float16

    def add_request(
        self,
        prompt: str | list[int],
        sampling_params: SamplingParams | None = None,
    ) -> Sequence:
        sp = sampling_params or SamplingParams()
        token_ids = (
            list(prompt) if isinstance(prompt, list) else self.tokenizer.encode(prompt)
        )
        if len(token_ids) > self.context_len:
            token_ids = token_ids[-self.context_len :]
        if not token_ids:
            raise ValueError("empty prompt")
        seq = Sequence(token_ids, sp)
        self.scheduler.add(seq)
        return seq

    def step(self) -> tuple[list[tuple[int, list[int]]], int]:
        seqs, is_prefill = self.scheduler.schedule()
        if not seqs:
            return [], 0
        metric = (
            sum(seq.num_scheduled_tokens for seq in seqs)
            if is_prefill
            else -len(seqs)
        )
        token_ids = self.model_runner.run(seqs, is_prefill)
        finished = self.scheduler.postprocess(seqs, token_ids, is_prefill)
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
