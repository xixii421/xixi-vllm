"""推理引擎：组装 tokenizer / 模型 / 采样器，驱动自回归 generate。"""

from __future__ import annotations

import torch
from transformers import AutoTokenizer

from .config import EngineConfig
from .model import Qwen3ForCausalLM
from .model_loader import load_model
from .sampler import Sampler


class Engine:
    """单请求、同步、流式吐字的编排层。

    把 prompt 变成 GPU 张量，调模型前向，再经 Sampler 得到 token。
    """

    def __init__(self, config: EngineConfig) -> None:
        self.config = config
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.dtype = self._resolve_dtype()

        # ① 分词器：部分模型无 pad_token，回退到 eos，避免后续填充报错
        self.tokenizer = AutoTokenizer.from_pretrained(
            config.model, trust_remote_code=True
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.eos_token_id = self.tokenizer.eos_token_id

        # ② 先建空骨架，再把 safetensors 灌进参数，最后上设备
        self.model = Qwen3ForCausalLM(config.hf_config)
        load_model(self.model, config.model)
        self.model = self.model.to(dtype=self.dtype, device=self.device).eval()
        if not config.enforce_eager and self.device.type == "cuda":
            # reduce-overhead：降低反复进图时的 Python / 调度开销
            self.model = torch.compile(self.model, mode="reduce-overhead")

        # ③ 采样器无训练参数；放在同一设备上即可
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
        min_p: float | None = None,
    ):
        """自回归生成；每产出一个 token 的文本就 yield 一次。

        每一步把「当前整段 ids」送入模型，再只取**最后一个位置**的 logits 采样。
        中间位置的 hidden 也算过，但本循环只用末位决定下一个字。
        """
        ids = (
            list(prompt)
            if isinstance(prompt, list)
            else self.tokenizer.encode(prompt)
        )
        ctx = self.config.context_len
        # 超长 prompt：只保留右侧 context_len（硬截断）
        if len(ids) > ctx:
            ids = ids[-ctx:]

        def pack(v, dtype):
            # 采样超参做成 [1] 张量，循环内复用，避免每步 new tensor
            return None if v is None else torch.tensor([v], device=self.device, dtype=dtype)

        # temperature<=0 → 传 None，Sampler 走 argmax 快路径
        temp_t = None if temperature <= 0 else pack(temperature, torch.float)
        top_p_t = pack(top_p, torch.float)
        top_k_t = pack(top_k, torch.int)
        min_p_t = pack(min_p, torch.float)

        for _ in range(max_new_tokens):
            # ids: Python list 长度 N → 张量 [1, N]；positions 为 0..N-1
            input_ids = torch.tensor([ids], device=self.device, dtype=torch.long)
            positions = torch.arange(len(ids), device=self.device).unsqueeze(0)

            h = self.model(input_ids, positions)  # [1, N, hidden]
            # 只投影最后一个时间步 → [1, vocab]
            logits = self.model.compute_logits(h[:, -1:]).squeeze(1)
            # 注意传给 Sampler 的参数顺序：temp, min_p, top_p, top_k
            next_id = int(self.sampler(logits, temp_t, min_p_t, top_p_t, top_k_t).item())

            yield self.tokenizer.decode([next_id])
            ids.append(next_id)
            if next_id == self.eos_token_id or len(ids) >= ctx:
                break
