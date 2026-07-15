import torch
from transformers import AutoTokenizer

from ..config import EngineConfig
from ..memory import BlockManager, PagedKVCache, slots_for_range
from ..models.loader import load_model
from ..models.qwen3 import Qwen3ForCausalLM
from ..utils import ForwardContext, set_forward_context
from .sampler import Sampler


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
        self.block_manager = BlockManager(
            num_blocks=config.num_gpu_blocks,
            block_size=config.block_size,
        )
        self._block_table: list[int] | None = None

    def _resolve_dtype(self):
        if self.config.dtype != "auto":
            return getattr(torch, self.config.dtype)
        dt = self.config.hf_config.torch_dtype
        return dt if isinstance(dt, torch.dtype) else torch.float16

    def _sample(
        self,
        logits: torch.Tensor,
        temp_t,
        min_p_t,
        top_p_t,
        top_k_t,
    ) -> int:
        return self.sampler(logits, temp_t, min_p_t, top_p_t, top_k_t).item()

    def _slot_tensor(self, slots: list[int]) -> torch.Tensor:
        return torch.tensor(slots, device=self.device, dtype=torch.long)

    def free_active_blocks(self) -> None:
        if self._block_table is not None:
            self.block_manager.free(self._block_table)
            self._block_table = None

    def allocate_prompt_blocks(self, prompt_len: int) -> list[int]:
        self.free_active_blocks()
        self._block_table = self.block_manager.allocate(prompt_len)
        return self._block_table

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

        def _t(v, dtype):
            return None if v is None else torch.tensor([v], device=self.device, dtype=dtype)

        temp_t = None if temperature <= 0 else _t(temperature, torch.float)
        top_p_t = _t(top_p, torch.float)
        top_k_t = _t(top_k, torch.int)
        min_p_t = _t(min_p, torch.float)

        prompt_len = len(input_ids)
        block_table = self.allocate_prompt_blocks(prompt_len)
        try:
            # --- Prefill ---
            slots = slots_for_range(block_table, 0, prompt_len, self.block_size)
            set_forward_context(
                ForwardContext(
                    is_prefill=True,
                    seq_len=prompt_len,
                    kv_cache=self.kv_cache,
                    block_table=block_table,
                    slot_mapping=self._slot_tensor(slots),
                )
            )
            ids = torch.tensor([input_ids], device=self.device, dtype=torch.long)
            positions = torch.arange(prompt_len, device=self.device).unsqueeze(0)
            h = self.model(ids, positions)
            logits = self.model.compute_logits(h[:, -1:]).squeeze(1)
            next_id = self._sample(logits, temp_t, min_p_t, top_p_t, top_k_t)

            yield self.tokenizer.decode([next_id])
            if next_id == self.eos_token_id or prompt_len >= self.context_len:
                return

            num_tokens = prompt_len

            # --- Decode ---
            for _ in range(max_new_tokens - 1):
                if num_tokens >= self.context_len:
                    break
                num_tokens += 1
                self.block_manager.may_append(block_table, num_tokens)
                pos = num_tokens - 1
                slots = slots_for_range(block_table, pos, pos + 1, self.block_size)
                set_forward_context(
                    ForwardContext(
                        is_prefill=False,
                        seq_len=num_tokens,
                        kv_cache=self.kv_cache,
                        block_table=block_table,
                        slot_mapping=self._slot_tensor(slots),
                    )
                )
                ids = torch.tensor([[next_id]], device=self.device, dtype=torch.long)
                positions = torch.tensor([[pos]], device=self.device, dtype=torch.long)
                h = self.model(ids, positions)
                logits = self.model.compute_logits(h[:, -1:]).squeeze(1)
                next_id = self._sample(logits, temp_t, min_p_t, top_p_t, top_k_t)

                yield self.tokenizer.decode([next_id])
                if next_id == self.eos_token_id:
                    break
        finally:
            set_forward_context(None)
            self.free_active_blocks()
