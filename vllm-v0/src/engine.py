import torch
from transformers import AutoTokenizer

from .config import EngineConfig
from .model import Qwen3ForCausalLM
from .model_loader import load_model
from .sampler import Sampler


class Engine:
    def __init__(self, config: EngineConfig):
        self.config = config
        self.tokenizer = AutoTokenizer.from_pretrained(config.model, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.eos_token_id = self.tokenizer.eos_token_id
        self.context_len = config.context_len

        self.device = torch.device("cuda:0")
        self.dtype = self._resolve_dtype()

        self.model = Qwen3ForCausalLM(config.hf_config)
        load_model(self.model, config.model)
        self.model = self.model.to(dtype=self.dtype, device=self.device)

        if not config.enforce_eager:
            self.model = torch.compile(self.model, mode="reduce-overhead")

        self.sampler = Sampler().to(device=self.device)

    def _resolve_dtype(self):
        if self.config.dtype != "auto":
            return getattr(torch, self.config.dtype)
        dt = self.config.hf_config.torch_dtype
        return dt if isinstance(dt, torch.dtype) else torch.float16

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
        input_ids = (list(prompt) if isinstance(prompt, list)
                     else self.tokenizer.encode(prompt))
        if len(input_ids) > self.context_len:
            input_ids = input_ids[-self.context_len:]

        def _t(v, dtype):
            return None if v is None else torch.tensor([v], device=self.device, dtype=dtype)

        temp_t = None if temperature <= 0 else _t(temperature, torch.float)
        top_p_t = _t(top_p, torch.float)
        top_k_t = _t(top_k, torch.int)
        min_p_t = _t(min_p, torch.float)

        for _ in range(max_new_tokens):
            ids = torch.tensor([input_ids], device=self.device, dtype=torch.long)
            positions = torch.arange(len(input_ids), device=self.device).unsqueeze(0)

            h = self.model(ids, positions)
            logits = self.model.compute_logits(h[:, -1:]).squeeze(1)
            next_id = self.sampler(logits, temp_t, min_p_t, top_p_t, top_k_t).item()

            yield self.tokenizer.decode([next_id])
            input_ids.append(next_id)

            if next_id == self.eos_token_id or len(input_ids) >= self.context_len:
                break
