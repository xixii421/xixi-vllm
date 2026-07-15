from __future__ import annotations

import hashlib
import warnings
from collections.abc import Iterator
from typing import TYPE_CHECKING, Optional

import torch
from transformers import AutoTokenizer

from ..config import EngineConfig
from ..memory import MultiModalFeature, Sequence
from ..sampling_params import SamplingParams
from .model_runner import ModelRunner
from .scheduler import Scheduler

if TYPE_CHECKING:
    from PIL import Image


class Engine:

    def __init__(self, config: EngineConfig):
        self.config = config
        self.device = torch.device("cuda:0")

        self.tokenizer = AutoTokenizer.from_pretrained(config.model, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.eos_token_id = self.tokenizer.eos_token_id
        self.context_len = config.context_len
        self.block_size = config.kvcache_block_size

        Sequence.block_size = self.block_size

        self.mm_processor = None
        if config.is_multimodal:
            from ..multimodal import MultiModalProcessor

            self.mm_processor = MultiModalProcessor(config.model)
            print(f"Multimodal mode: Qwen3-VL processor loaded ({config.model})")

        self.model_runner = ModelRunner(config, self.device)

        num_blocks = config.num_kvcache_blocks
        if not config.enforce_eager:
            if num_blocks < 2:
                raise RuntimeError(
                    f"CUDA Graph requires >= 2 KV blocks (got {num_blocks}): "
                    "one page is reserved as pad/dummy and cannot be scheduled"
                )
            num_blocks = num_blocks - 1
        self.scheduler = Scheduler(
            max_num_seqs=config.max_num_seqs,
            max_num_batched_tokens=config.max_num_batched_tokens,
            num_blocks=num_blocks,
            block_size=self.block_size,
            eos_token_id=int(self.eos_token_id),
        )
        self._idle_steps = 0
        self._max_idle_steps = 1024

    @property
    def model(self):
        return self.model_runner.model

    @property
    def kv_cache(self) -> torch.Tensor:
        return self.model_runner.kv_cache

    def add_request(
        self,
        prompt: str | list[int],
        sampling_params: SamplingParams | None = None,
        images: Optional[list["Image.Image"]] = None,
    ) -> Sequence:
        sp = sampling_params or SamplingParams()
        mm_features: list[MultiModalFeature] = []
        if images:
            if self.mm_processor is None:
                raise ValueError(
                    "images passed but model is not multimodal "
                    f"(is_multimodal={self.config.is_multimodal})"
                )
            mm_features, token_ids = self._build_mm_features(prompt, images)
        else:
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
            token_ids, mm_features = self._truncate_left(
                token_ids, mm_features, self.context_len
            )

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

        seq = Sequence(token_ids, sp, mm_features=mm_features or None)
        self.scheduler.add(seq)
        return seq

    def _build_mm_features(
        self, prompt: str | list[int], images: list["Image.Image"]
    ) -> tuple[list[MultiModalFeature], list[int]]:
        assert self.mm_processor is not None
        text = prompt if isinstance(prompt, str) else self.tokenizer.decode(prompt)
        input_ids, pixel_values, mm_info = self.mm_processor.process(text, images)
        if pixel_values is None or not mm_info:
            return [], list(input_ids)

        features: list[MultiModalFeature] = []

        patch_offsets = [0]
        for info in mm_info:
            g = info.get("grid_thw")
            if g is None:
                patch_offsets.append(patch_offsets[-1])
                continue
            n_patches = int(g.reshape(-1).prod().item())
            patch_offsets.append(patch_offsets[-1] + n_patches)

        for i, info in enumerate(mm_info):
            g_thw = info.get("grid_thw")
            # Processor already stores per-image ``grid_thw`` as shape [1, 3].
            if g_thw is not None and g_thw.dim() == 1:
                g_thw = g_thw.view(1, -1)
            start_p, end_p = patch_offsets[i], patch_offsets[i + 1]
            pv = (
                pixel_values[start_p:end_p].contiguous()
                if end_p > start_p
                else pixel_values
            )
            raw = pv.detach().float().cpu().numpy().tobytes()
            uid = hashlib.sha1(
                raw[:4096] + str(tuple(pv.shape)).encode() + f"_{i}".encode()
            ).hexdigest()
            features.append(
                MultiModalFeature(
                    modality="image",
                    data=pv.cpu(),
                    start=info["start"],
                    end=info["end"],
                    num_encoder_tokens=info["num_encoder_tokens"],
                    uid=uid,
                    grid_thw=g_thw.cpu() if g_thw is not None else None,
                )
            )
        return features, list(input_ids)

    @staticmethod
    def _truncate_left(
        token_ids: list[int],
        mm_features: list[MultiModalFeature],
        context_len: int,
    ) -> tuple[list[int], list[MultiModalFeature]]:
        """Keep the last ``context_len`` tokens and remap fully in-window MM spans.

        Features that would be partially cut by left-truncation are dropped
        (partial vision placeholders cannot be merged safely).
        """
        keep_from = len(token_ids) - context_len
        kept: list[MultiModalFeature] = []
        dropped = 0
        for mf in mm_features:
            if mf.start >= keep_from and mf.end <= len(token_ids):
                mf.start -= keep_from
                mf.end -= keep_from
                mf.mrope_origin = None
                kept.append(mf)
            else:
                dropped += 1
        if dropped:
            warnings.warn(
                f"left-truncation dropped {dropped} multimodal feature(s) that "
                "were not fully inside the kept window",
                stacklevel=3,
            )
        return token_ids[-context_len:], kept

    def step(self) -> tuple[list[tuple[int, list[int]]], int]:
        seqs, is_prefill = self.scheduler.schedule()
        if not seqs:
            if not self.scheduler.is_finished():
                self._idle_steps += 1
                if self._idle_steps >= self._max_idle_steps:
                    raise RuntimeError(
                        "scheduler returned empty for too long while requests remain "
                        f"(idle_steps={self._idle_steps}). Likely KV pool exhaustion "
                        f"(free_blocks={self.scheduler.block_manager.num_free})."
                    )
            return [], 0
        self._idle_steps = 0
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
        images: list["Image.Image"] | list[list["Image.Image"]] | None = None,
    ) -> list[str]:
        if sampling_params is None:
            sps = [SamplingParams() for _ in prompts]
        elif isinstance(sampling_params, SamplingParams):
            sps = [sampling_params for _ in prompts]
        else:
            sps = list(sampling_params)
            if len(sps) != len(prompts):
                raise ValueError("sampling_params length must match prompts")

        seqs = []
        for i, (p, sp) in enumerate(zip(prompts, sps)):
            imgs = None
            if images is not None and i < len(images):
                raw = images[i]
                imgs = raw if isinstance(raw, list) else [raw]
            seqs.append(self.add_request(p, sp, images=imgs))

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
        images: Optional[list["Image.Image"]] = None,
    ) -> Iterator[str]:
        sp = SamplingParams(
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            min_p=min_p,
            max_tokens=max_new_tokens,
        )
        seq = self.add_request(prompt, sp, images=images)
        emitted = 0
        while not seq.is_finished:
            self.step()
            while emitted < seq.num_completion_tokens:
                tid = seq.token_ids[seq.num_prompt_tokens + emitted]
                emitted += 1
                yield self.tokenizer.decode([tid])
