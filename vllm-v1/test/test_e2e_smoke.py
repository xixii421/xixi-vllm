from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import EngineConfig
from src.engine import Engine
from src.model_hub import DEFAULT_CACHE_DIR


def _resolve_model(name: str) -> str:
    path = Path(name).expanduser()
    if path.exists():
        return str(path)
    cand = DEFAULT_CACHE_DIR / name
    if cand.exists():
        return str(cand)
    raise FileNotFoundError(f"model not found: {name} (tried {cand})")


def run(model: str, backend: str, max_new: int) -> str:
    cfg = EngineConfig(
        model=model,
        context_len=512,
        enforce_eager=True,
        prefix_backend=backend,  # type: ignore[arg-type]
        block_size=16,
    )
    eng = Engine(cfg)
    chunks = list(
        eng.generate("Hello, say hi briefly.", max_new_tokens=max_new, temperature=0.0)
    )
    text = "".join(chunks)
    print(f"backend={backend} cached={eng.last_num_cached_tokens} text={text!r}")
    return text


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen3-0.6B")
    ap.add_argument("--max-new-tokens", type=int, default=16)
    args = ap.parse_args()
    model = _resolve_model(args.model)

    t0 = run(model, "none", args.max_new_tokens)
    t1 = run(model, "hash", args.max_new_tokens)
    eng = Engine(
        EngineConfig(
            model=model,
            context_len=512,
            enforce_eager=True,
            prefix_backend="hash",
            block_size=16,
        )
    )
    _ = "".join(
        eng.generate("Hello, say hi briefly.", max_new_tokens=8, temperature=0.0)
    )
    t2 = "".join(
        eng.generate("Hello, say hi briefly.", max_new_tokens=8, temperature=0.0)
    )
    print(f"hash second-run cached={eng.last_num_cached_tokens} text={t2!r}")
    assert t0
    assert t1
    assert t2
    print("SMOKE_OK")


if __name__ == "__main__":
    main()
