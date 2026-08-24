"""bench 共享常量与结果写入。

统一 jsonl 字段，供 bench_v4 / bench_vllm / report 共用。
"""

from __future__ import annotations

import json
from pathlib import Path

BENCH_DIR = Path(__file__).resolve().parent
OUT_DIR = BENCH_DIR / "out"
JSONL_PATH = OUT_DIR / "bench.jsonl"

PROMPT_LEN = 256
MAX_TOKENS = 128
BATCH_SIZE = 8
DP_N_REQS = 32
DP_PROMPT_LEN = 128

DEFAULT_MODEL = str(Path.home() / "huggingface" / "Qwen3-0.6B")


def dummy_ids(n: int, salt: int = 0) -> list[int]:
    """生成可复现的假 token 序列，避免依赖真实文本。"""
    n = max(1, n)
    return [((i * 7919 + 10007 + salt * 97) % 150000) + 100 for i in range(n)]


def mode_name(*, cg: bool, compile: bool) -> str:
    if compile:
        return "cg+compile" if cg else "compile"
    return "cg" if cg else "eager"


def parse_int_list(text: str | None) -> list[int] | None:
    if text is None or text.strip() == "":
        return None
    return [int(x) for x in text.split(",")]


def ensure_out() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)


def append_jsonl(record: dict) -> None:
    """追加一条结构化结果。失败时也写入 ok=false。"""
    ensure_out()
    with JSONL_PATH.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def base_record(
    *,
    engine: str,
    model: str,
    tp: int,
    pp: int,
    dp: int,
    mode: str,
    tag: str,
) -> dict:
    return {
        "engine": engine,
        "model": Path(model).name,
        "tp": tp,
        "pp": pp,
        "dp": dp,
        "mode": mode,
        "tag": tag,
        "ttft_ms": None,
        "decode_tps": None,
        "batch_tps": None,
        "throughput_tps": None,
        "n_reqs": None,
        "ok": True,
        "error": None,
    }
