"""Shared helpers for vllm-v0 / vllm-v1 benchmarks."""

from __future__ import annotations

import importlib
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterator, Literal, Sequence

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
V0_ROOT = REPO_ROOT / "vllm-v0"
V1_ROOT = REPO_ROOT / "vllm-v1"

Version = Literal["v0", "v1"]

_DEFAULT_HF = Path("/mnt/nfs/zhangli/huggingface")
DEFAULT_CACHE_DIR = Path(
    os.environ.get("HF_HOME")
    or os.environ.get("HUGGINGFACE_HUB_CACHE")
    or (_DEFAULT_HF if _DEFAULT_HF.is_dir() else Path.home() / "huggingface")
)


def sync_device() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def resolve_model(name: str, cache_dir: Path | None = None) -> Path:
    path = Path(name).expanduser()
    if path.exists():
        return path.resolve()
    root = cache_dir or DEFAULT_CACHE_DIR
    cand = root / name
    if cand.exists():
        return cand.resolve()
    raise FileNotFoundError(f"model not found: {name} (tried {cand})")


def _purge_src_modules() -> None:
    for key in list(sys.modules):
        if key == "src" or key.startswith("src."):
            del sys.modules[key]


def _import_src(root: Path) -> None:
    """Import ``root/src`` as package ``src`` (purge first when switching trees)."""
    _purge_src_modules()
    root_s = str(root)
    while root_s in sys.path:
        sys.path.remove(root_s)
    sys.path.insert(0, root_s)
    importlib.import_module("src")


@dataclass
class BenchEngine:
    """Version-agnostic handle; keeps the real Engine alive."""

    label: str
    version: Version
    _engine: Any

    def encode(self, text: str) -> list[int]:
        return self._engine.tokenizer.encode(text)

    def generate(
        self,
        prompt: str,
        *,
        max_new_tokens: int,
        temperature: float = 0.0,
    ) -> Iterator[str]:
        return self._engine.generate(
            prompt, max_new_tokens=max_new_tokens, temperature=temperature
        )

    @property
    def last_cached_tokens(self) -> int:
        return int(getattr(self._engine, "last_num_cached_tokens", 0) or 0)

    def reset_kv(self, prefix_backend: str | None = None) -> None:
        reset = getattr(self._engine, "reset_kv", None)
        if reset is None:
            return
        if prefix_backend is None:
            reset()
        else:
            reset(prefix_backend)


def load_engine(
    version: Version,
    model: str | Path,
    *,
    enforce_eager: bool = True,
    context_len: int = 2048,
    prefix_backend: str = "none",
    block_size: int = 16,
) -> BenchEngine:
    root = V0_ROOT if version == "v0" else V1_ROOT
    if not root.is_dir():
        raise FileNotFoundError(f"missing tree: {root}")

    _import_src(root)
    from src.config import EngineConfig  # type: ignore
    from src.engine import Engine  # type: ignore

    model_s = str(model)
    if version == "v0":
        cfg = EngineConfig(model=model_s, context_len=context_len, enforce_eager=enforce_eager)
    else:
        cfg = EngineConfig(
            model=model_s,
            context_len=context_len,
            enforce_eager=enforce_eager,
            prefix_backend=prefix_backend,  # type: ignore[arg-type]
            block_size=block_size,
        )
    eng = Engine(cfg)
    label = "v0" if version == "v0" else f"v1/{prefix_backend}"
    return BenchEngine(label=label, version=version, _engine=eng)


@dataclass(frozen=True)
class RunMetrics:
    label: str
    prompt_tokens: int
    new_tokens: int
    e2e_s: float
    ttft_s: float
    cached_tokens: int = 0

    @property
    def e2e_tps(self) -> float:
        return self.new_tokens / self.e2e_s if self.e2e_s > 0 else float("nan")

    @property
    def decode_tps(self) -> float:
        if self.new_tokens <= 1:
            return float("nan")
        decode_s = self.e2e_s - self.ttft_s
        return (self.new_tokens - 1) / decode_s if decode_s > 0 else float("nan")


def timed_run(
    engine: BenchEngine,
    prompt: str,
    max_new_tokens: int,
) -> RunMetrics:
    prompt_tokens = len(engine.encode(prompt))
    sync_device()
    t0 = time.perf_counter()
    ttft_s: float | None = None
    n = 0
    for _ in engine.generate(prompt, max_new_tokens=max_new_tokens, temperature=0.0):
        if ttft_s is None:
            sync_device()
            ttft_s = time.perf_counter() - t0
        n += 1
    sync_device()
    e2e_s = time.perf_counter() - t0
    if ttft_s is None:
        ttft_s = e2e_s
    return RunMetrics(
        label=engine.label,
        prompt_tokens=prompt_tokens,
        new_tokens=n,
        e2e_s=e2e_s,
        ttft_s=ttft_s,
        cached_tokens=engine.last_cached_tokens,
    )


def measure(
    engine: BenchEngine,
    prompt: str,
    max_new_tokens: int,
    *,
    warmup: int = 1,
    repeats: int = 3,
) -> RunMetrics:
    """Warmup then take the median-e2e run (stable for small teaching benches)."""
    for _ in range(max(0, warmup)):
        for _tok in engine.generate(prompt, max_new_tokens=max_new_tokens, temperature=0.0):
            pass
        sync_device()

    runs = [timed_run(engine, prompt, max_new_tokens) for _ in range(max(1, repeats))]
    # Median by e2e; break ties with decode_tps.
    runs.sort(key=lambda r: (r.e2e_s, -r.decode_tps))
    mid = runs[len(runs) // 2]
    return mid


def make_prompt(*, target_tokens: int, tokenizer_encode: Callable[[str], list[int]]) -> str:
    """Grow a fixed sentence until encoded length >= target_tokens."""
    unit = "The quick brown fox jumps over the lazy dog. "
    text = unit
    while len(tokenizer_encode(text)) < target_tokens:
        text += unit
    # Trim by binary-search-ish shrink of repeats for tighter fit.
    parts = max(1, target_tokens // max(1, len(tokenizer_encode(unit))))
    text = unit * (parts + 2)
    while len(tokenizer_encode(text)) > target_tokens + 8 and len(text) > len(unit):
        text = text[: -len(unit)]
    return text.strip()


def format_table(rows: Sequence[RunMetrics], *, title: str) -> str:
    headers = (
        "label",
        "prompt",
        "new",
        "cached",
        "TTFT_s",
        "E2E_s",
        "E2E_tps",
        "Dec_tps",
    )
    lines = [f"### {title}", "", "| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |"]
    for r in rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    r.label,
                    str(r.prompt_tokens),
                    str(r.new_tokens),
                    str(r.cached_tokens),
                    f"{r.ttft_s:.3f}",
                    f"{r.e2e_s:.3f}",
                    f"{r.e2e_tps:.1f}",
                    f"{r.decode_tps:.1f}",
                ]
            )
            + " |"
        )
    return "\n".join(lines)


def speedup(baseline: RunMetrics, other: RunMetrics) -> str:
    if baseline.e2e_s <= 0:
        return "n/a"
    return f"{baseline.e2e_s / other.e2e_s:.2f}x"
