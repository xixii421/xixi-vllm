
import warnings
from collections.abc import Iterator
from typing import Optional

import torch
from transformers import AutoTokenizer

from ..config import EngineConfig
from ..memory import KVManager, MultiModalFeature, Sequence
from ..models.loader import load_model
from ..models.qwen3 import Qwen3ForCausalLM
from ..sampling_params import SamplingParams
from .model_runner import ModelRunner
from .sampler import Sampler
from .scheduler import Scheduler

from PIL import Image


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

        self.mm_processor = None
        if config.is_multimodal:
            from ..multimodal import MultiModalProcessor

            self.mm_processor = MultiModalProcessor(config.model)
            print(f"Multimodal mode: Qwen3-VL processor loaded ({config.model})")
            from ..models.qwen3_vl import Qwen3VLForCausalLM

            self.model = Qwen3VLForCausalLM(config.hf_config)
            self.model.load_weights(config.model, self.device, self.dtype)
            self.model = self.model.eval()
        else:
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
        cfg = self.config.text_config
        dt = getattr(cfg, "torch_dtype", None) or getattr(
            self.config.hf_config, "torch_dtype", None
        )
        return dt if isinstance(dt, torch.dtype) else torch.float16

    def _resolve_num_blocks(self) -> int:
        if self.config.num_kvcache_blocks > 0:
            return self.config.num_kvcache_blocks
        n = KVManager.auto_num_blocks(
            self.config.text_config,
            block_size=self.block_size,
            dtype=self.dtype,
            device=self.device,
            gpu_memory_utilization=self.config.gpu_memory_utilization,
            reserve_cg_headroom=not self.config.enforce_eager,
        )
        self.config.num_kvcache_blocks = n
        return n

    def _init_kv(self) -> None:
        use_cg = not self.config.enforce_eager
        num_blocks = self._resolve_num_blocks()
        self.kv = KVManager(
            self.config.text_config,
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
            encoder_compute_budget=self.config.encoder_compute_budget,
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

        seq = Sequence.from_prompt(
            token_ids,
            block_size=self.block_size,
            sampling=sp,
            mm_features=mm_features or None,
        )
        self.scheduler.add(seq)
        return seq

    def _build_mm_features(
        self, prompt: str | list[int], images: list["Image.Image"]
    ) -> tuple[list[MultiModalFeature], list[int]]:
        assert self.mm_processor is not None
        text = prompt if isinstance(prompt, str) else self.tokenizer.decode(prompt)
        input_ids, pixel_values, mm_info, img_uids = self.mm_processor.process(
            text, images
        )
        if pixel_values is None or not mm_info:
            return [], list(input_ids)

        features: list[MultiModalFeature] = []
        patch_offsets = [0]
        for info in mm_info:
            g_tuple = info.get("grid_thw_tuple")
            if g_tuple is None:
                g = info.get("grid_thw")
                if g is None:
                    patch_offsets.append(patch_offsets[-1])
                    continue
                flat = g.detach().view(-1, 3)[0].tolist()
                g_tuple = (int(flat[0]), int(flat[1]), int(flat[2]))
            n_patches = int(g_tuple[0] * g_tuple[1] * g_tuple[2])
            patch_offsets.append(patch_offsets[-1] + n_patches)

        for i, info in enumerate(mm_info):
            g_thw = info.get("grid_thw")
            g_tuple = info.get("grid_thw_tuple")
            if g_tuple is None and g_thw is not None:
                flat = g_thw.detach().view(-1, 3)[0].tolist()
                g_tuple = (int(flat[0]), int(flat[1]), int(flat[2]))
            if g_thw is not None and g_thw.dim() == 1:
                g_thw = g_thw.view(1, -1)
            start_p, end_p = patch_offsets[i], patch_offsets[i + 1]
            pv = (
                pixel_values[start_p:end_p].contiguous()
                if end_p > start_p
                else pixel_values
            )
            if not pv.is_pinned:
                pv = pv.pin_memory()
            # uid from raw image hash (cheap); index disambiguates multi-image.
            uid = img_uids[i] if i < len(img_uids) else f"img-{i}"
            if len(img_uids) > 1:
                uid = f"{uid}_{i}"
            features.append(
                MultiModalFeature(
                    modality="image",
                    data=pv,
                    start=info["start"],
                    end=info["end"],
                    num_encoder_tokens=info["num_encoder_tokens"],
                    uid=uid,
                    grid_thw=g_thw.cpu() if g_thw is not None else None,
                    grid_thw_tuple=g_tuple,
                )
            )
        return features, list(input_ids)

    @staticmethod
    def _truncate_left(
        token_ids: list[int],
        mm_features: list[MultiModalFeature],
        context_len: int,
    ) -> tuple[list[int], list[MultiModalFeature]]:
        """Keep the last ``context_len`` tokens; remap fully in-window MM spans.

        Features partially cut by left-truncation are dropped (a partial vision
        placeholder run cannot be merged safely).
        """
        if len(token_ids) <= context_len:
            return token_ids, mm_features
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
                f"left-truncation dropped {dropped} multimodal feature(s) not "
                "fully inside the kept window",
                stacklevel=3,
            )
        return token_ids[-context_len:], kept

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

    def destroy(self) -> None:
        self.model_runner.destroy()
