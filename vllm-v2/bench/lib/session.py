"""Benchmark session and model forward helpers (v2 PagedAttention)."""

from __future__ import annotations

import gc
from dataclasses import dataclass

import torch

from src import Engine, EngineConfig
from src.memory import slots_for_range
from src.utils import ForwardContext, set_forward_context

from .config import BenchConfig
from .timing import CudaTimer, sync


@dataclass(frozen=True)
class BenchMeta:
    model: str
    gpu: str
    vram_gb: float
    torch_version: str
    cuda_version: str
    enforce_eager: bool
    warmup_iters: int
    measure_iters: int
    block_size: int = 16
    num_gpu_blocks: int = 0


def collect_meta(cfg: BenchConfig) -> BenchMeta:
    gpu = "cpu"
    vram = 0.0
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        gpu = torch.cuda.get_device_name(0)
        vram = props.total_memory / (1024**3)
    return BenchMeta(
        model=cfg.model,
        gpu=gpu,
        vram_gb=round(vram, 2),
        torch_version=torch.__version__,
        cuda_version=torch.version.cuda or "n/a",
        enforce_eager=cfg.enforce_eager,
        warmup_iters=cfg.warmup_iters,
        measure_iters=cfg.measure_iters,
        block_size=cfg.block_size,
        num_gpu_blocks=cfg.num_gpu_blocks or 0,
    )


class BenchSession:
    """Owns a loaded engine and reusable CUDA buffers for stable timing."""

    def __init__(self, cfg: BenchConfig) -> None:
        if not cfg.model_path.is_dir():
            raise FileNotFoundError(
                f"Model not found at {cfg.model_path}. Run: python run.py --download"
            )
        self.cfg = cfg
        self.engine = Engine(
            EngineConfig(
                model=str(cfg.model_path),
                context_len=cfg.context_len,
                block_size=cfg.block_size,
                max_num_seqs=cfg.max_num_seqs,
                num_gpu_blocks=cfg.num_gpu_blocks,
                enforce_eager=cfg.enforce_eager,
            )
        )
        self.device = self.engine.device
        self._bench_table: list[int] | None = None

    @property
    def tokenizer(self):
        return self.engine.tokenizer

    def global_warmup(self, steps: int = 3) -> None:
        prompt_ids = self.make_prompt_tokens(64)
        for _ in range(steps):
            self.prefill_forward_only(prompt_ids)
        self.clear_kv_state()
        sync()

    def make_prompt_tokens(self, target_len: int) -> list[int]:
        chunk = "The quick brown fox jumps over the lazy dog. "
        ids: list[int] = []
        while len(ids) < target_len:
            ids.extend(self.tokenizer.encode(chunk, add_special_tokens=False))
        return ids[:target_len]

    def _prepare_tensors(self, input_ids: list[int]) -> tuple[torch.Tensor, torch.Tensor]:
        ids = torch.tensor([input_ids], device=self.device, dtype=torch.long)
        positions = torch.arange(len(input_ids), device=self.device).unsqueeze(0)
        return ids, positions

    def _slot_tensor(self, slots: list[int]) -> torch.Tensor:
        return torch.tensor(slots, device=self.device, dtype=torch.long)

    def _free_bench_table(self) -> None:
        if self._bench_table is not None:
            self.engine.block_manager.free(self._bench_table)
            self._bench_table = None
        self.engine.free_active_blocks()

    @torch.inference_mode()
    def prefill_forward_only(self, input_ids: list[int]) -> None:
        self._free_bench_table()
        prompt_len = len(input_ids)
        table = self.engine.block_manager.allocate(prompt_len)
        self._bench_table = table
        slots = slots_for_range(table, 0, prompt_len, self.engine.block_size)
        set_forward_context(
            ForwardContext(
                is_prefill=True,
                seq_len=prompt_len,
                kv_cache=self.engine.kv_cache,
                block_table=table,
                slot_mapping=self._slot_tensor(slots),
            )
        )
        ids, positions = self._prepare_tensors(input_ids)
        h = self.engine.model(ids, positions)
        self.engine.model.compute_logits(h[:, -1:])
        set_forward_context(None)

    @torch.inference_mode()
    def decode_forward_only(self, cache_len: int, next_token_id: int = 1) -> None:
        if self._bench_table is None:
            raise RuntimeError("decode_forward_only requires fill_kv_cache / prefill first")
        table = self._bench_table
        num_tokens = cache_len + 1
        self.engine.block_manager.may_append(table, num_tokens)
        pos = cache_len
        slots = slots_for_range(table, pos, pos + 1, self.engine.block_size)
        set_forward_context(
            ForwardContext(
                is_prefill=False,
                seq_len=num_tokens,
                kv_cache=self.engine.kv_cache,
                block_table=table,
                slot_mapping=self._slot_tensor(slots),
            )
        )
        ids = torch.tensor([[next_token_id]], device=self.device, dtype=torch.long)
        positions = torch.tensor([[pos]], device=self.device, dtype=torch.long)
        h = self.engine.model(ids, positions)
        self.engine.model.compute_logits(h[:, -1:])
        set_forward_context(None)

    def fill_kv_cache(self, prompt_ids: list[int]) -> None:
        self.prefill_forward_only(prompt_ids)

    def clear_kv_state(self) -> None:
        self._free_bench_table()
        # 多请求场景可能留下多张表。
        for table in getattr(self, "_held_tables", []) or []:
            if table:
                self.engine.block_manager.free(table)
        self._held_tables = []
        set_forward_context(None)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
        gc.collect()
        sync()

    @torch.inference_mode()
    def prefill_on_table(self, input_ids: list[int], block_table: list[int]) -> None:
        """在已有 block_table 上 Prefill，不释放其它请求的页。"""
        prompt_len = len(input_ids)
        slots = slots_for_range(block_table, 0, prompt_len, self.engine.block_size)
        set_forward_context(
            ForwardContext(
                is_prefill=True,
                seq_len=prompt_len,
                kv_cache=self.engine.kv_cache,
                block_table=block_table,
                slot_mapping=self._slot_tensor(slots),
            )
        )
        ids, positions = self._prepare_tensors(input_ids)
        h = self.engine.model(ids, positions)
        self.engine.model.compute_logits(h[:, -1:])
        set_forward_context(None)

    @torch.inference_mode()
    def decode_on_table(
        self, block_table: list[int], cache_len: int, next_token_id: int = 1
    ) -> None:
        num_tokens = cache_len + 1
        self.engine.block_manager.may_append(block_table, num_tokens)
        pos = cache_len
        slots = slots_for_range(block_table, pos, pos + 1, self.engine.block_size)
        set_forward_context(
            ForwardContext(
                is_prefill=False,
                seq_len=num_tokens,
                kv_cache=self.engine.kv_cache,
                block_table=block_table,
                slot_mapping=self._slot_tensor(slots),
            )
        )
        ids = torch.tensor([[next_token_id]], device=self.device, dtype=torch.long)
        positions = torch.tensor([[pos]], device=self.device, dtype=torch.long)
        h = self.engine.model(ids, positions)
        self.engine.model.compute_logits(h[:, -1:])
        set_forward_context(None)

    def multi_request_prefill(
        self, lengths: list[int], *, decode_steps: int = 1
    ) -> dict:
        """R 路并发占用：各自 allocate + Prefill（+可选 Decode），页表同时存活。"""
        self.clear_kv_state()
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        tables: list[list[int]] = []
        self._held_tables = tables
        try:
            for t_len in lengths:
                ids = self.make_prompt_tokens(t_len)
                table = self.engine.block_manager.allocate(len(ids))
                tables.append(table)
                self.prefill_on_table(ids, table)
                if decode_steps > 0:
                    self.decode_on_table(table, cache_len=len(ids), next_token_id=1)
            sync()
            used_blocks = sum(len(t) for t in tables)
            return {
                "num_requests": len(lengths),
                "used_blocks": used_blocks,
                "free_blocks": self.engine.block_manager.num_free,
                "peak_gb": round(self.peak_memory_gb(), 4),
                "allocated_gb": round(self.allocated_memory_gb(), 4),
            }
        finally:
            pass

    def forward_cuda_ms(self, input_ids: list[int]) -> float:
        with CudaTimer() as timer:
            self.prefill_forward_only(input_ids)
        return timer.ms

    def forward_only(self, input_ids: list[int]) -> None:
        self.prefill_forward_only(input_ids)

    @torch.inference_mode()
    def greedy_next_token(self, input_ids: list[int]) -> int:
        self.fill_kv_cache(input_ids)
        cache_len = len(input_ids)
        assert self._bench_table is not None
        self.engine.block_manager.may_append(self._bench_table, cache_len + 1)
        slots = slots_for_range(
            self._bench_table, cache_len, cache_len + 1, self.engine.block_size
        )
        set_forward_context(
            ForwardContext(
                is_prefill=False,
                seq_len=cache_len + 1,
                kv_cache=self.engine.kv_cache,
                block_table=self._bench_table,
                slot_mapping=self._slot_tensor(slots),
            )
        )
        ids = torch.tensor([[0]], device=self.device, dtype=torch.long)
        positions = torch.tensor([[cache_len]], device=self.device, dtype=torch.long)
        h = self.engine.model(ids, positions)
        logits = self.engine.model.compute_logits(h[:, -1:]).squeeze(1)
        set_forward_context(None)
        return self.engine.sampler(logits, None, None, None, None).item()

    def greedy_step_cuda_ms(self, input_ids: list[int]) -> float:
        self.fill_kv_cache(input_ids)
        cache_len = len(input_ids)

        def _run() -> None:
            self.decode_forward_only(cache_len)

        with CudaTimer() as timer:
            _run()
        return timer.ms

    def reset_peak_memory(self) -> None:
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

    def peak_memory_gb(self) -> float:
        sync()
        return torch.cuda.max_memory_allocated() / (1024**3)

    def allocated_memory_gb(self) -> float:
        sync()
        return torch.cuda.memory_allocated() / (1024**3)
