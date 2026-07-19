from __future__ import annotations

import torch
from transformers import AutoTokenizer

from .config import EngineConfig
from .kv import ForwardContext, KVManager, Sequence, set_forward_context
from .model import Qwen3ForCausalLM
from .model_loader import load_model
from .sampler import Sampler


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
        self.eos_token_id = self.tokenizer.eos_token_id
        self.context_len = config.context_len

        self.model = Qwen3ForCausalLM(config.hf_config)
        load_model(self.model, config.model)
        self.model = self.model.to(dtype=self.dtype, device=self.device).eval()
        if not config.enforce_eager and self.device.type == "cuda":
            self.model = torch.compile(self.model, mode="reduce-overhead")

        self.sampler = Sampler()
        self.kv = KVManager(
            self.model,
            num_blocks=config.num_kv_blocks,
            block_size=config.block_size,
            prefix_backend=config.prefix_backend,
            device=self.device,
            dtype=self.dtype,
        )
        self.last_num_cached_tokens = 0

    def _resolve_dtype(self) -> torch.dtype:
        if self.config.dtype != "auto":
            return getattr(torch, self.config.dtype)
        cfg = self.config.hf_config
        dt = getattr(cfg, "dtype", None) or getattr(cfg, "torch_dtype", None)
        return dt if isinstance(dt, torch.dtype) else torch.float16

    def _sample(
        self,
        logits: torch.Tensor,
        temp_t,
        min_p_t,
        top_p_t,
        top_k_t,
    ) -> int:
        return int(self.sampler(logits, temp_t, min_p_t, top_p_t, top_k_t).item())

    @torch.inference_mode()
    def generate(
        self,
        prompt: str | list[int],
        max_new_tokens: int = 128,
        temperature: float = 0.0,
        top_p: float | None = None,
        top_k: int | None = None,
        min_p: float | None = None,
    ):
        input_ids = (
            list(prompt) if isinstance(prompt, list) else self.tokenizer.encode(prompt)
        )
        if len(input_ids) > self.context_len:
            input_ids = input_ids[-self.context_len :]

        def pack(v, dtype):
            return None if v is None else torch.tensor([v], device=self.device, dtype=dtype)

        temp_t = None if temperature <= 0 else pack(temperature, torch.float)
        top_p_t = pack(top_p, torch.float)
        top_k_t = pack(top_k, torch.int)
        min_p_t = pack(min_p, torch.float)

        seq: Sequence = self.kv.new_sequence(input_ids)
        self.kv.allocate(seq)
        self.last_num_cached_tokens = seq.num_cached_tokens
        cached = seq.num_cached_tokens
        prompt_len = len(input_ids)

        try:
            # Prefill only uncached tokens (full prompt if cache miss).
            if cached < prompt_len:
                set_forward_context(
                    ForwardContext(
                        is_prefill=True,
                        cache_len=cached,
                        manager=self.kv,
                        seq=seq,
                    )
                )
                new_ids = input_ids[cached:]
                ids = torch.tensor([new_ids], device=self.device, dtype=torch.long)
                positions = torch.arange(
                    cached, prompt_len, device=self.device
                ).unsqueeze(0)
                h = self.model(ids, positions)
                logits = self.model.compute_logits(h[:, -1:]).squeeze(1)
            else:
                # Entire prompt was prefix-cached — still need a decode-style step
                # on the last cached token to get next-token logits is wrong;
                # recompute last position: feed last token with full context.
                set_forward_context(
                    ForwardContext(
                        is_prefill=False,
                        cache_len=prompt_len - 1,
                        manager=self.kv,
                        seq=seq,
                    )
                )
                # Overwrite last slot's K/V by re-forwarding last token (rare path).
                last = input_ids[-1]
                ids = torch.tensor([[last]], device=self.device, dtype=torch.long)
                positions = torch.tensor(
                    [[prompt_len - 1]], device=self.device, dtype=torch.long
                )
                h = self.model(ids, positions)
                logits = self.model.compute_logits(h[:, -1:]).squeeze(1)

            next_id = self._sample(logits, temp_t, min_p_t, top_p_t, top_k_t)
            yield self.tokenizer.decode([next_id])
            if next_id == self.eos_token_id or prompt_len >= self.context_len:
                return

            seq.append_token(next_id)
            self.kv.ensure_blocks(seq)
            cache_len = prompt_len

            for _ in range(max_new_tokens - 1):
                if cache_len >= self.context_len:
                    break
                set_forward_context(
                    ForwardContext(
                        is_prefill=False,
                        cache_len=cache_len,
                        manager=self.kv,
                        seq=seq,
                    )
                )
                ids = torch.tensor([[next_id]], device=self.device, dtype=torch.long)
                positions = torch.tensor(
                    [[cache_len]], device=self.device, dtype=torch.long
                )
                h = self.model(ids, positions)
                logits = self.model.compute_logits(h[:, -1:]).squeeze(1)
                next_id = self._sample(logits, temp_t, min_p_t, top_p_t, top_k_t)
                cache_len += 1
                seq.append_token(next_id)
                self.kv.ensure_blocks(seq)

                yield self.tokenizer.decode([next_id])
                if next_id == self.eos_token_id:
                    break
        finally:
            set_forward_context(None)
            self.kv.finish(seq)
