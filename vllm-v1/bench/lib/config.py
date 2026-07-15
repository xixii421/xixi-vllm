"""Benchmark configuration."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

LIB_ROOT = Path(__file__).resolve().parent
BENCH_ROOT = LIB_ROOT.parent
RESULTS_ROOT = BENCH_ROOT / "results"

DEFAULT_MODEL = "Qwen3-0.6B"
DEFAULT_PROMPT = "Hello, my name is"
DEFAULT_MAX_SEQ_LEN = 8192

# Log-spaced sweep from short prompts to 8K context.
DEFAULT_LENGTH_SWEEP = [128, 512, 1024, 2048, 4096, 8192]


@dataclass
class BenchConfig:
    model: str = DEFAULT_MODEL
    prompt: str = DEFAULT_PROMPT
    max_new_tokens: int = 128
    context_len: int = DEFAULT_MAX_SEQ_LEN
    enforce_eager: bool = True
    warmup_iters: int = 10
    measure_iters: int = 25
    prefill_lengths: list[int] = field(default_factory=lambda: list(DEFAULT_LENGTH_SWEEP))
    latency_lengths: list[int] = field(default_factory=lambda: list(DEFAULT_LENGTH_SWEEP))
    latency_prompt_len: int = 128
    # growing decode isolated points from prompt_len up to context_len
    latency_decode_steps: int = DEFAULT_MAX_SEQ_LEN - 128
    latency_sample_every: int = 512

    @property
    def model_path(self) -> Path:
        path = Path(self.model).expanduser()
        if path.exists():
            return path
        return Path.home() / "huggingface" / self.model

    def iters_for_seq_len(self, seq_len: int) -> int:
        """Use fewer timed iterations on very long sequences to keep runtime reasonable."""
        if seq_len >= 8192:
            return max(5, self.measure_iters // 5)
        if seq_len >= 4096:
            return max(8, self.measure_iters // 2)
        return self.measure_iters

    def warmup_for_seq_len(self, seq_len: int) -> int:
        if seq_len >= 4096:
            return max(3, self.warmup_iters // 2)
        return self.warmup_iters
