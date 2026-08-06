from __future__ import annotations

import torch
from transformers import AutoTokenizer

from .config import EngineConfig
from .model import Qwen3ForCausalLM
from .model_loader import load_model
from .sampler import Sampler


class Engine:
    def __init__(self, config: EngineConfig) -> None:
        self.config = config
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.dtype = self._resolve_dtype()

        # 分词器：部分模型无 pad_token，回退到 eos，避免后续填充报错
        self.tokenizer = AutoTokenizer.from_pretrained(
            config.model, trust_remote_code=True
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.eos_token_id = self.tokenizer.eos_token_id

        # 先建空骨架，再把 safetensors 灌进参数，最后上设备
        self.model = Qwen3ForCausalLM(config.hf_config)
        load_model(self.model, config.model)
        self.model = self.model.to(dtype=self.dtype, device=self.device).eval()

        # 采样器无训练参数；放在同一设备上即可
        self.sampler = Sampler()

    def _resolve_dtype(self) -> torch.dtype:
        if self.config.dtype != "auto":
            return getattr(torch, self.config.dtype)
        cfg = self.config.hf_config
        dt = getattr(cfg, "dtype", None) or getattr(cfg, "torch_dtype", None)
        return dt if isinstance(dt, torch.dtype) else torch.float16

    @torch.inference_mode()
    def generate(
        self,
        prompt: str | list[int],
        max_new_tokens: int = 128,
        temperature: float = 0.0,
        top_p: float | None = None,
        top_k: int | None = None,
    ):
        ids = (
            list(prompt)
            if isinstance(prompt, list)
            else self.tokenizer.encode(prompt)
        )
        ctx = self.config.context_len

        if len(ids) > ctx:
            ids = ids[-ctx:]

        def pack(v, dtype):

            return None if v is None else torch.tensor([v], device=self.device, dtype=dtype)

        # temperature<=0 → 传 None，Sampler 走 argmax 快路径
        temp_t = None if temperature <= 0 else pack(temperature, torch.float)
        top_p_t = pack(top_p, torch.float)
        top_k_t = pack(top_k, torch.int)

        for _ in range(max_new_tokens):
            # ids: Python list 长度 N → 张量 [1, N]；positions 为 0..N-1
            input_ids = torch.tensor([ids], device=self.device, dtype=torch.long)
            positions = torch.arange(len(ids), device=self.device).unsqueeze(0)

            h = self.model(input_ids, positions)  # [1, N, hidden]
            # 只投影最后一个时间步 → [1, vocab]
            logits = self.model.compute_logits(h[:, -1:]).squeeze(1)
            # 传给 Sampler 的参数顺序：temp, top_p, top_k
            next_id = int(self.sampler(logits, temp_t, top_p_t, top_k_t).item())

            yield self.tokenizer.decode([next_id])
            ids.append(next_id)
            if next_id == self.eos_token_id or len(ids) >= ctx:
                break
