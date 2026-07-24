"""Shared helpers for v0 / v1 / v2 benchmarks."""

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
V2_ROOT = REPO_ROOT / "vllm-v2"

Version = Literal["v0", "v1", "v2"]

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
    _purge_src_modules()
    root_s = str(root)
    while root_s in sys.path:
        sys.path.remove(root_s)
    sys.path.insert(0, root_s)
    importlib.import_module("src")


@dataclass
class BenchEngine:
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

    def generate_batch(
        self,
        prompts: list[str],
        *,
        max_new_tokens: int,
        temperature: float = 0.0,
    ) -> list[str]:
        """v2: true CB. v0/v1: sequential generate (same wall-clock API)."""
        if self.version == "v2":
            from src.sampling_params import SamplingParams  # type: ignore

            sp = SamplingParams(temperature=temperature, max_tokens=max_new_tokens)
            return self._engine.generate_batch(prompts, sp)

        outs: list[str] = []
        for p in prompts:
            outs.append(
                "".join(
                    self.generate(p, max_new_tokens=max_new_tokens, temperature=temperature)
                )
            )
        return outs

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
    max_num_seqs: int = 8,
    max_num_batched_tokens: int = 4096,
    mix_prefill_decode: bool = True,
    reuse_src: bool = False,
) -> BenchEngine:
    root = {"v0": V0_ROOT, "v1": V1_ROOT, "v2": V2_ROOT}[version]
    if not root.is_dir():
        raise FileNotFoundError(f"missing tree: {root}")

    if not reuse_src:
        _import_src(root)
    from src.config import EngineConfig  # type: ignore
    from src.engine import Engine  # type: ignore

    model_s = str(model)
    if version == "v0":
        cfg = EngineConfig(model=model_s, context_len=context_len, enforce_eager=enforce_eager)
        label = "v0"
    elif version == "v1":
        cfg = EngineConfig(
            model=model_s,
            context_len=context_len,
            enforce_eager=enforce_eager,
            prefix_backend=prefix_backend,  # type: ignore[arg-type]
            block_size=block_size,
        )
        label = f"v1/{prefix_backend}"
    else:
        cfg = EngineConfig(
            model=model_s,
            context_len=context_len,
            enforce_eager=enforce_eager,
            prefix_backend=prefix_backend,  # type: ignore[arg-type]
            block_size=block_size,
            max_num_seqs=max_num_seqs,
            max_num_batched_tokens=max_num_batched_tokens,
            mix_prefill_decode=mix_prefill_decode,
        )
        tag = "mix" if mix_prefill_decode else "pfirst"
        label = f"v2/{tag}"

    return BenchEngine(label=label, version=version, _engine=Engine(cfg))


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


@dataclass(frozen=True)
class BatchMetrics:
    label: str
    num_reqs: int
    new_tokens: int
    wall_s: float

    @property
    def tok_per_s(self) -> float:
        return self.new_tokens / self.wall_s if self.wall_s > 0 else float("nan")


def timed_run(engine: BenchEngine, prompt: str, max_new_tokens: int) -> RunMetrics:
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
    for _ in range(max(0, warmup)):
        for _tok in engine.generate(prompt, max_new_tokens=max_new_tokens, temperature=0.0):
            pass
        sync_device()

    runs = [timed_run(engine, prompt, max_new_tokens) for _ in range(max(1, repeats))]
    runs.sort(key=lambda r: (r.e2e_s, -r.decode_tps))
    return runs[len(runs) // 2]


def timed_batch(
    engine: BenchEngine, prompts: list[str], max_new_tokens: int
) -> BatchMetrics:
    sync_device()
    t0 = time.perf_counter()
    outs = engine.generate_batch(prompts, max_new_tokens=max_new_tokens, temperature=0.0)
    sync_device()
    wall_s = time.perf_counter() - t0
    # Approximate new tokens via encode of completions (greedy, no EOS early stop assumed).
    new_tokens = sum(len(engine.encode(o)) for o in outs)
    return BatchMetrics(
        label=engine.label,
        num_reqs=len(prompts),
        new_tokens=new_tokens,
        wall_s=wall_s,
    )


def measure_batch(
    engine: BenchEngine,
    prompts: list[str],
    max_new_tokens: int,
    *,
    warmup: int = 1,
    repeats: int = 3,
) -> BatchMetrics:
    for _ in range(max(0, warmup)):
        engine.generate_batch(prompts, max_new_tokens=max_new_tokens, temperature=0.0)
        sync_device()
        if engine.version != "v0":
            engine.reset_kv()

    runs = [timed_batch(engine, prompts, max_new_tokens) for _ in range(max(1, repeats))]
    runs.sort(key=lambda r: r.wall_s)
    return runs[len(runs) // 2]


def make_prompt(*, target_tokens: int, tokenizer_encode: Callable[[str], list[int]]) -> str:
    unit = "The quick brown fox jumps over the lazy dog. "
    parts = max(1, target_tokens // max(1, len(tokenizer_encode(unit))))
    text = unit * (parts + 2)
    while len(tokenizer_encode(text)) > target_tokens + 8 and len(text) > len(unit):
        text = text[: -len(unit)]
    return text.strip()


def speedup(baseline_s: float, other_s: float) -> str:
    if baseline_s <= 0 or other_s <= 0:
        return "n/a"
    return f"{baseline_s / other_s:.2f}x"


def format_rows(rows: Sequence[RunMetrics], *, title: str) -> str:
    headers = ("label", "prompt", "new", "cached", "TTFT_s", "E2E_s", "Dec_tps")
    lines = [
        f"### {title}",
        "",
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
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
                    f"{r.decode_tps:.1f}",
                ]
            )
            + " |"
        )
    return "\n".join(lines)
