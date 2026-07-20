"""Control-plane loop: add_request → schedule → execute → postprocess.

Pipeline stages live in subpackages (call order):
  ``input/``   request preprocess (CPU)
  ``execute/`` distributed RPC / workers
  ``runner/``  GPU model step + mRoPE sync back to Engine
  ``vision/``  VL encode/merge (worker, prefill only)
"""

from __future__ import annotations

import warnings
from collections.abc import Iterator
from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

import torch
from transformers import AutoTokenizer

from ..config import EngineConfig
from ..sampling import SamplingParams
from ..scheduling import MultiModalFeature, Sequence, Scheduler
from ..utils.profiler import Timer

if TYPE_CHECKING:
    from PIL import Image

    from .execute.executor import Executor
    from .execute.local import LocalExecutor
    from .runner import ModelRunner

    ExecutorLike = Executor | LocalExecutor


@dataclass(slots=True)
class PendingEngineStep:
    """Immutable ownership record for one scheduled model execution."""

    seqs: list[Sequence]
    is_prefill: bool
    metric: int
    rpc_future: object | None = None


class Engine:

    def __init__(self, config: EngineConfig):
        self.config = config
        device_ids = config.resolved_device_ids()
        # Parent only needs FlashInfer for LocalExecutor (ws==1). Multiproc
        # workers pin their *own* GPU arch in worker_main (hetero-safe).
        if config.world_size == 1:
            from ..utils.cuda_env import apply_flashinfer_arch_list

            apply_flashinfer_arch_list(device_ids)
        # Import after arch pin — runner/worker pull FlashInfer at import time.
        from .execute.executor import Executor
        from .execute.local import LocalExecutor

        self.device = torch.device(f"cuda:{device_ids[0]}")
        self.tokenizer = AutoTokenizer.from_pretrained(
            config.model, trust_remote_code=True
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.eos_token_id = self.tokenizer.eos_token_id
        self.context_len = config.context_len
        self.block_size = config.kvcache_block_size
        self.mm_processor = None
        if config.is_multimodal:
            from .input import MultiModalProcessor

            self.mm_processor = MultiModalProcessor(config.model)
            print(
                f"Multimodal mode: Qwen3-VL processor loaded ({config.model}) "
                f"tp={config.tp_size} pp={config.pp_size}"
            )
        self._validate_multimodal_pp(config)
        # Always an Executor-like: Local (ws==1, zero IPC) or multiproc.
        self.executor: ExecutorLike
        if config.world_size > 1:
            self.executor = Executor(config)
        else:
            self.executor = LocalExecutor(config, self.device)
        num_blocks = self.executor.get_num_kvcache_blocks()
        config.num_kvcache_blocks = num_blocks
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
            enable_radix=config.enable_radix,
        )
        self._idle_steps = 0
        self._max_idle_steps = 1024

    @staticmethod
    def _validate_multimodal_pp(config: EngineConfig) -> None:
        """DeepStack needs first PP stage to own early vision-injection layers."""
        if not (config.is_multimodal and config.pp_size > 1):
            return
        from ..distributed.pp import get_pp_indices

        text_cfg = config.text_config
        vc = getattr(config.hf_config, "vision_config", None)
        n_ds = len(getattr(vc, "deepstack_visual_indexes", None) or []) if vc else 0
        if n_ds <= 0:
            return
        _s, end0 = get_pp_indices(
            int(text_cfg.num_hidden_layers), pp_rank=0, pp_size=config.pp_size
        )
        if end0 < n_ds:
            raise ValueError(
                f"pp_size={config.pp_size} gives first-stage layers [0,{end0}), "
                f"but DeepStack needs layers 0..{n_ds - 1}; reduce pp_size"
            )

    @property
    def model_runner(self) -> ModelRunner | None:
        """In-process ModelRunner when ``world_size==1``; else None."""
        from .execute.local import LocalExecutor

        if isinstance(self.executor, LocalExecutor):
            return self.executor.model_runner
        return None

    @property
    def model(self):
        mr = self.model_runner
        if mr is None:
            raise RuntimeError("model only available on world_size==1 path")
        return mr.model

    @property
    def kv_cache(self) -> torch.Tensor:
        mr = self.model_runner
        if mr is None:
            raise RuntimeError("kv_cache only available on world_size==1 path")
        return mr.kv_cache

    def abort(self, seq: Sequence) -> None:
        """Abort one sequence and free its KV pages."""
        self.scheduler.abort(seq)

    def is_finished(self) -> bool:
        """True when the scheduler has no waiting / running work."""
        return self.scheduler.is_finished()

    def decode_token(self, token_id: int) -> str:
        return self.tokenizer.decode([token_id])

    def decode_tokens(self, token_ids: list[int]) -> str:
        return self.tokenizer.decode(token_ids)

    def add_request(
        self,
        prompt: str | list[int] | None = None,
        sampling_params: SamplingParams | None = None,
        images: Optional[list["Image.Image"]] = None,
        *,
        chat_messages: list[dict] | None = None,
    ) -> Sequence:
        """Enqueue one sequence.

        Prefer ``chat_messages`` (HF-style) from OpenAI Serving so multi-turn
        multimodal uses the same ``apply_chat_template`` path as the processor.
        """
        sp = sampling_params or SamplingParams()
        mm_features: list[MultiModalFeature] = []
        if chat_messages is not None:
            mm_features, token_ids = self._build_from_chat_messages(
                chat_messages, images
            )
        elif images:
            if prompt is None:
                raise ValueError("prompt is required when images are set without chat_messages")
            if self.mm_processor is None:
                raise ValueError(
                    f"images passed but model is not multimodal "
                    f"(is_multimodal={self.config.is_multimodal})"
                )
            mm_features, token_ids = self._build_mm_features(prompt, images)
        else:
            if prompt is None:
                raise ValueError("prompt or chat_messages is required")
            token_ids = (
                list(prompt)
                if isinstance(prompt, list)
                else self.tokenizer.encode(prompt)
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
        seq = Sequence(
            token_ids,
            sp,
            mm_features=mm_features or None,
            block_size=self.block_size,
        )
        self.scheduler.add(seq)
        return seq

    def _build_mm_features(
        self, prompt: str | list[int], images: list["Image.Image"]
    ) -> tuple[list[MultiModalFeature], list[int]]:
        from .input import build_mm_features

        assert self.mm_processor is not None
        text = prompt if isinstance(prompt, str) else self.tokenizer.decode(prompt)
        input_ids, pixel_values, spans = self.mm_processor.process(text, images)
        if pixel_values is None or not spans:
            return [], list(input_ids)
        return build_mm_features(pixel_values, spans), list(input_ids)

    def _build_from_chat_messages(
        self,
        chat_messages: list[dict],
        images: Optional[list["Image.Image"]],
    ) -> tuple[list[MultiModalFeature], list[int]]:
        from .input import build_mm_features

        if images and self.mm_processor is None:
            raise ValueError(
                f"images in chat but model is not multimodal "
                f"(is_multimodal={self.config.is_multimodal})"
            )
        if self.mm_processor is not None:
            input_ids, pixel_values, spans = self.mm_processor.process_messages(
                chat_messages, images or None
            )
            if pixel_values is None or not spans:
                return [], list(input_ids)
            return build_mm_features(pixel_values, spans), list(input_ids)

        # Text-only: tokenizer chat template when available.
        tok = self.tokenizer
        if hasattr(tok, "apply_chat_template"):
            # Strip PIL objects from content for text-only template.
            plain = []
            for m in chat_messages:
                c = m.get("content")
                if isinstance(c, list):
                    text = "".join(
                        str(p.get("text", ""))
                        for p in c
                        if isinstance(p, dict) and p.get("type") == "text"
                    )
                    plain.append({"role": m["role"], "content": text})
                else:
                    plain.append({"role": m["role"], "content": c})
            prompt = tok.apply_chat_template(
                plain, tokenize=False, add_generation_prompt=True
            )
            return [], list(tok.encode(prompt))
        raise ValueError("chat_messages require a chat template tokenizer or mm_processor")

    @staticmethod
    def _truncate_left(
        token_ids: list[int],
        mm_features: list[MultiModalFeature],
        context_len: int,
    ) -> tuple[list[int], list[MultiModalFeature]]:
        from .input import truncate_mm_features

        before = len(mm_features)
        token_ids, kept = truncate_mm_features(token_ids, mm_features, context_len)
        dropped = before - len(kept)
        if dropped:
            warnings.warn(
                f"left-truncation dropped {dropped} multimodal feature(s) that were "
                "not fully inside the kept window",
                stacklevel=3,
            )
        return token_ids, kept

    def begin_step(self) -> PendingEngineStep | None:
        """Schedule and submit one step without waiting for worker replies."""
        with Timer("engine.schedule"):
            sched_out = self.scheduler.schedule()
        if not sched_out:
            if not self.scheduler.is_finished():
                self._idle_steps += 1
                if self._idle_steps >= self._max_idle_steps:
                    raise RuntimeError(
                        f"scheduler returned empty for too long while requests remain "
                        f"(idle_steps={self._idle_steps}). Likely KV pool exhaustion "
                        f"(free_blocks={self.scheduler.block_manager.num_free})."
                    )
            return None
        self._idle_steps = 0
        seqs = list(sched_out.seqs)
        future = self.executor.submit_execute_model(seqs, sched_out.is_prefill)
        return PendingEngineStep(
            seqs, sched_out.is_prefill, sched_out.metric, future
        )

    def wait_step(
        self, pending: PendingEngineStep
    ) -> tuple[list[int], list | None]:
        """Wait outside the engine-state critical section."""
        with Timer("engine.execute"):
            assert pending.rpc_future is not None
            result = pending.rpc_future.result()
            assert isinstance(result, tuple)
            return result

    def finish_step(
        self,
        pending: PendingEngineStep,
        result: tuple[list[int], list | None],
    ) -> tuple[list[tuple[int, list[int]]], int]:
        """Commit one completed step while holding engine-state ownership."""
        token_ids, mrope_updates = result
        if mrope_updates:
            from .runner import apply_mrope_updates

            apply_mrope_updates(pending.seqs, mrope_updates)
        with Timer("engine.postprocess"):
            finished = self.scheduler.postprocess(
                pending.seqs, token_ids, pending.is_prefill
            )
        outputs = [
            (seq.seq_id, seq.token_ids[seq.num_prompt_tokens :]) for seq in finished
        ]
        return outputs, pending.metric

    def step(self) -> tuple[list[tuple[int, list[int]]], int]:
        pending = self.begin_step()
        if pending is None:
            return [], 0
        return self.finish_step(pending, self.wait_step(pending))

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
        while not self.is_finished():
            outputs, _ = self.step()
            for seq_id, completion_ids in outputs:
                finished_text[seq_id] = self.decode_tokens(completion_ids)
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
        from .emit import iter_new_token_ids

        seq = self.add_request(prompt, sp, images=images)
        emitted = 0
        while not seq.is_finished:
            self.step()
            for emitted, tid in iter_new_token_ids(seq, emitted):
                yield self.decode_token(tid)

    def destroy(self) -> None:
        self.executor.shutdown()
