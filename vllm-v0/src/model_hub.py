"""Qwen3 权重注册、下载与冒烟测试辅助。"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from pathlib import Path

import torch
from huggingface_hub import snapshot_download
from huggingface_hub.errors import HfHubHTTPError, RepositoryNotFoundError

from .config import EngineConfig
from .engine import Engine

# 缓存根目录：优先环境变量，其次本机公共路径，再退回 ~/huggingface
_DEFAULT = Path("/mnt/nfs/zhangli/huggingface")
DEFAULT_CACHE_DIR = Path(
    os.environ.get("HF_HOME")
    or os.environ.get("HUGGINGFACE_HUB_CACHE")
    or (_DEFAULT if _DEFAULT.is_dir() else Path.home() / "huggingface")
)


@dataclass(frozen=True)
class ModelSpec:
    """注册表条目：本地目录名、HF repo、约估计的 fp16 体积（GB）。"""

    local_name: str
    repo_id: str
    fp16_gb: float

    def path_under(self, root: Path) -> Path:
        return root / self.local_name

    @property
    def local_path(self) -> Path:
        return self.path_under(DEFAULT_CACHE_DIR)


QWEN3_REGISTRY: tuple[ModelSpec, ...] = (
    ModelSpec("Qwen3-0.6B", "Qwen/Qwen3-0.6B", 1.2),
    ModelSpec("Qwen3-1.7B", "Qwen/Qwen3-1.7B", 3.4),
    ModelSpec("Qwen3-4B", "Qwen/Qwen3-4B", 8.0),
)


def is_downloaded(spec: ModelSpec, cache_dir: Path | None = None) -> bool:
    path = spec.path_under(cache_dir or DEFAULT_CACHE_DIR)
    return path.is_dir() and any(path.glob("*.safetensors"))


def _snapshot_download(repo_id: str, local_dir: str, *, token) -> None:
    kwargs = dict(repo_id=repo_id, local_dir=local_dir, token=token)
    endpoint = os.environ.get("HF_ENDPOINT", "").strip()
    if endpoint:
        kwargs["endpoint"] = endpoint
    snapshot_download(**kwargs)


def download_model(spec: ModelSpec, cache_dir: Path | None = None) -> Path:
    """下载公开快照到本地目录。

    损坏的 HF_TOKEN / 缓存登录常导致公开仓库也 401：先试环境 token，
    失败再匿名（token=False）重试。
    """
    root = cache_dir or DEFAULT_CACHE_DIR
    root.mkdir(parents=True, exist_ok=True)
    target = spec.path_under(root)
    if is_downloaded(spec, root):
        print(f"[skip] {spec.local_name} already at {target}")
        return target

    print(f"[download] {spec.repo_id} -> {target}")
    env_tok = (
        os.environ.get("HF_TOKEN")
        or os.environ.get("HUGGING_FACE_HUB_TOKEN")
        or True  # 让 hub 自行读取 ~/.cache/huggingface/token
    )
    attempts: list = [env_tok, False]
    last_err: Exception | None = None
    for i, tok in enumerate(attempts):
        try:
            _snapshot_download(spec.repo_id, str(target), token=tok)
            if not is_downloaded(spec, root):
                raise FileNotFoundError(
                    f"download finished but no *.safetensors under {target}"
                )
            return target
        except (RepositoryNotFoundError, HfHubHTTPError, OSError) as e:
            last_err = e
            msg = str(e)
            authish = any(
                s in msg
                for s in ("401", "403", "Unauthorized", "signature verification", "gated")
            )
            if i == 0 and authish:
                print(
                    "[warn] HuggingFace auth failed "
                    "(invalid/expired HF_TOKEN or cached login). "
                    "Retrying anonymous download for public model …"
                )
                continue
            break

    hint = (
        "\nHints:\n"
        "  - unset broken tokens:  unset HF_TOKEN HUGGING_FACE_HUB_TOKEN\n"
        "  - or:  huggingface-cli logout\n"
        "  - mirror:  export HF_ENDPOINT=https://hf-mirror.com\n"
        "  - or point at an existing folder:  python run.py --model /path/to/Qwen3-0.6B"
    )
    raise RuntimeError(f"failed to download {spec.repo_id}: {last_err}{hint}") from last_err


def download_models(
    specs: list[ModelSpec] | None = None,
    cache_dir: Path | None = None,
) -> list[Path]:
    return [download_model(s, cache_dir) for s in (specs or list(QWEN3_REGISTRY))]


def gpu_memory_gb() -> float:
    if not torch.cuda.is_available():
        return 0.0
    return torch.cuda.get_device_properties(0).total_memory / (1024**3)


def select_models_for_vram(
    vram_gb: float | None = None,
    reserve_gb: float = 4.0,
    registry: tuple[ModelSpec, ...] = QWEN3_REGISTRY,
) -> list[ModelSpec]:
    """按「单模型可装下」粗筛；校验时仍是一个一个加载。"""
    budget = (vram_gb if vram_gb is not None else gpu_memory_gb() or 24.0) - reserve_gb
    return [s for s in registry if s.fp16_gb <= budget]


def run_smoke_test(
    model_path: str | Path,
    *,
    prompt: str = "Hello, my name is",
    max_new_tokens: int = 32,
) -> dict:
    """短序列贪心生成，返回吞吐与峰值显存。"""
    path = Path(model_path).expanduser()
    if not path.is_dir() or not any(path.glob("*.safetensors")):
        raise FileNotFoundError(f"incomplete model dir: {path}")
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    engine = Engine(EngineConfig(model=str(path), enforce_eager=True))
    parts = list(
        engine.generate(prompt, max_new_tokens=max_new_tokens, temperature=0.0)
    )
    elapsed = time.perf_counter() - t0
    peak = (
        torch.cuda.max_memory_allocated() / (1024**3) if torch.cuda.is_available() else 0.0
    )
    n = len(parts)
    del engine
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return {
        "model": path.name,
        "tokens": n,
        "seconds": elapsed,
        "tok_per_s": n / elapsed if elapsed > 0 else 0.0,
        "peak_gb": peak,
        "output": prompt + "".join(parts),
    }


def verify_models(
    specs: list[ModelSpec] | None = None,
    *,
    download: bool = True,
    vram_gb: float | None = None,
) -> list[dict]:
    """按显存预算下载（可选）并冒烟跑通一批模型。"""
    vram = vram_gb if vram_gb is not None else (gpu_memory_gb() or 24.0)
    specs = specs or select_models_for_vram(vram_gb=vram)
    if download:
        download_models(specs)

    print(f"\nVerifying {len(specs)} model(s) on ~{vram:.1f} GB VRAM")
    print(f"Cache: {DEFAULT_CACHE_DIR}")
    results = []
    for spec in specs:
        print(f"\n--- {spec.local_name} ---")
        if not is_downloaded(spec):
            raise FileNotFoundError(f"Missing weights at {spec.local_path}")
        r = run_smoke_test(spec.local_path)
        results.append(r)
        print(f"  peak {r['peak_gb']:.2f} GB | {r['tok_per_s']:.1f} tok/s")
        print(f"  out  {r['output'][:100]}...")
    print(f"\nPassed {len(results)} model(s).")
    return results
