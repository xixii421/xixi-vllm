"""Benchmark session and model forward helpers."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from src.config import EngineConfig
from src.engine import Engine

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
                enforce_eager=cfg.enforce_eager,
            )
        )
        self.device = self.engine.device

    @property
    def tokenizer(self):
        return self.engine.tokenizer

    def global_warmup(self, steps: int = 3) -> None:
        prompt_ids = self.make_prompt_tokens(64)
        for _ in range(steps):
            self.forward_cuda_ms(prompt_ids)
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

    def forward_cuda_ms(self, input_ids: list[int]) -> float:
        ids, positions = self._prepare_tensors(input_ids)
        with CudaTimer() as timer:
            h = self.engine.model(ids, positions)
            self.engine.model.compute_logits(h[:, -1:])
        return timer.ms

    def forward_only(self, input_ids: list[int]) -> None:
        ids, positions = self._prepare_tensors(input_ids)
        h = self.engine.model(ids, positions)
        self.engine.model.compute_logits(h[:, -1:])

    def greedy_next_token(self, input_ids: list[int]) -> int:
        ids, positions = self._prepare_tensors(input_ids)
        h = self.engine.model(ids, positions)
        logits = self.engine.model.compute_logits(h[:, -1:]).squeeze(1)
        return self.engine.sampler(logits, None, None, None, None).item()

    def greedy_step_cuda_ms(self, input_ids: list[int]) -> float:
        ids, positions = self._prepare_tensors(input_ids)

        def _run() -> None:
            h = self.engine.model(ids, positions)
            logits = self.engine.model.compute_logits(h[:, -1:]).squeeze(1)
            self.engine.sampler(logits, None, None, None, None)

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
