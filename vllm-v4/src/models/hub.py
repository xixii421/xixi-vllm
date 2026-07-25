import os
import time
from dataclasses import dataclass
from pathlib import Path

import torch
from huggingface_hub import snapshot_download

from ..config import EngineConfig


DEFAULT_CACHE_DIR = Path.home() / "huggingface"
RTX_4090_VRAM_GB = 24.0


@dataclass(frozen=True)
class ModelSpec:
    local_name: str
    repo_id: str
    fp16_gb: float

    @property
    def local_path(self) -> Path:
        return DEFAULT_CACHE_DIR / self.local_name


QWEN3_REGISTRY: tuple[ModelSpec, ...] = (
    ModelSpec("Qwen3-0.6B", "Qwen/Qwen3-0.6B", 1.2),
    ModelSpec("Qwen3-1.7B", "Qwen/Qwen3-1.7B", 3.4),
    ModelSpec("Qwen3-4B", "Qwen/Qwen3-4B", 8.0),
)


def get_cache_dir() -> Path:
    return DEFAULT_CACHE_DIR


def is_downloaded(spec: ModelSpec) -> bool:
    path = spec.local_path
    if not path.is_dir():
        return False
    return any(path.glob("*.safetensors"))


def download_model(spec: ModelSpec, cache_dir: Path | None = None) -> Path:
    root = cache_dir or DEFAULT_CACHE_DIR
    root.mkdir(parents=True, exist_ok=True)
    target = root / spec.local_name

    if is_downloaded(spec):
        print(f"[skip] {spec.local_name} already at {target}")
        return target

    print(f"[download] {spec.repo_id} -> {target}")
    snapshot_download(
        repo_id=spec.repo_id,
        local_dir=str(target),
    )
    return target


def select_models_for_vram(
    vram_gb: float = RTX_4090_VRAM_GB,
    reserve_gb: float = 4.0,
    registry: tuple[ModelSpec, ...] = QWEN3_REGISTRY,
) -> list[ModelSpec]:
    """Pick the largest set of models whose weights fit in VRAM together."""
    budget = vram_gb - reserve_gb
    selected: list[ModelSpec] = []
    used = 0.0
    for spec in registry:
        if used + spec.fp16_gb <= budget:
            selected.append(spec)
            used += spec.fp16_gb
    return selected


def download_models(
    specs: list[ModelSpec] | None = None,
    cache_dir: Path | None = None,
) -> list[Path]:
    specs = specs or list(QWEN3_REGISTRY)
    paths = []
    for spec in specs:
        paths.append(download_model(spec, cache_dir=cache_dir))
    return paths


def gpu_memory_gb() -> float:
    if not torch.cuda.is_available():
        return 0.0
    props = torch.cuda.get_device_properties(0)
    return props.total_memory / (1024**3)


def run_smoke_test(
    model_path: str | Path,
    *,
    prompt: str = "Hello, my name is",
    max_new_tokens: int = 32,
    enforce_eager: bool = True,
) -> dict:
    # Lazy import: avoids models.hub ↔ engine circular load.
    from ..engine.engine import Engine

    path = Path(os.path.expanduser(str(model_path)))
    config = EngineConfig(model=str(path), enforce_eager=enforce_eager)

    torch.cuda.reset_peak_memory_stats()
    mem_before = torch.cuda.memory_allocated() / (1024**3)

    t0 = time.perf_counter()
    engine = Engine(config)

    mem_after_load = torch.cuda.memory_allocated() / (1024**3)
    output_parts: list[str] = []
    n = 0
    for tok in engine.generate(
        prompt,
        max_new_tokens=max_new_tokens,
        temperature=0.0,
    ):
        output_parts.append(tok)
        n += 1

    elapsed = time.perf_counter() - t0
    peak_gb = torch.cuda.max_memory_allocated() / (1024**3)

    del engine
    torch.cuda.empty_cache()

    return {
        "model": path.name,
        "path": str(path),
        "tokens": n,
        "seconds": elapsed,
        "tok_per_s": n / elapsed if elapsed > 0 else 0.0,
        "mem_before_gb": mem_before,
        "mem_after_load_gb": mem_after_load,
        "peak_gb": peak_gb,
        "output": prompt + "".join(output_parts),
    }


def verify_models(
    specs: list[ModelSpec] | None = None,
    *,
    download: bool = True,
    vram_gb: float | None = None,
) -> list[dict]:
    if vram_gb is None:
        vram_gb = gpu_memory_gb() or RTX_4090_VRAM_GB

    specs = specs or select_models_for_vram(vram_gb=vram_gb)
    if download:
        download_models(specs)

    results = []
    print(f"\n{'=' * 60}")
    print(f"Verifying {len(specs)} model(s) on {vram_gb:.1f} GB VRAM budget")
    print(f"{'=' * 60}")

    for spec in specs:
        print(f"\n--- {spec.local_name} ({spec.repo_id}) ---")
        if not is_downloaded(spec):
            raise FileNotFoundError(f"Missing weights at {spec.local_path}")

        result = run_smoke_test(spec.local_path)
        results.append(result)
        print(f"  load VRAM  {result['mem_after_load_gb']:.2f} GB")
        print(f"  peak VRAM  {result['peak_gb']:.2f} GB")
        print(f"  speed      {result['tok_per_s']:.1f} tok/s ({result['tokens']} tokens)")
        print(f"  output     {result['output'][:120]}...")

    total_weight_gb = sum(s.fp16_gb for s in specs)
    print(f"\n{'=' * 60}")
    print(f"All {len(results)} model(s) passed. Est. weights total ~{total_weight_gb:.1f} GB")
    print(f"{'=' * 60}")
    return results
