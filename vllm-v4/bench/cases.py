"""矩阵用例表：smoke / full / vs-vllm。

每个 case 是一份可交给 run_matrix 执行的参数字典。
"""

from __future__ import annotations

from pathlib import Path

HOME = Path.home()
M06 = str(HOME / "huggingface" / "Qwen3-0.6B")
M4B = str(HOME / "huggingface" / "Qwen3-4B")
M8B = str(HOME / "shared" / "models" / "Qwen3-8B")
M14B = str(HOME / "shared" / "models" / "Qwen3-14B")

ALL_MODES = ("eager", "cg", "compile", "cg+compile")
SAFE_DP_MODES = ("eager", "compile")  # 12GB 卡上 CG 易 OOM


def _modes(names: tuple[str, ...]) -> list[dict]:
    out = []
    for name in names:
        out.append(
            {
                "cg": name in ("cg", "cg+compile"),
                "compile": name in ("compile", "cg+compile"),
                "mode": name,
            }
        )
    return out


def expand(cases: list[dict]) -> list[dict]:
    """把含 modes 字段的条目展开成可执行 case 列表。"""
    rows: list[dict] = []
    for c in cases:
        modes = c.get("modes", ALL_MODES)
        for m in _modes(tuple(modes)):
            row = {k: v for k, v in c.items() if k != "modes"}
            row.update(m)
            tag = row.get("tag_prefix", "case")
            row["tag"] = f"{tag}-{m['mode']}"
            rows.append(row)
    return rows


def suite_smoke() -> list[dict]:
    """日常冒烟：0.6B 单卡/TP/PP，eager+cg。"""
    base = [
        {
            "engine": "v4",
            "model": M06,
            "tp": 1,
            "pp": 1,
            "dp": 1,
            "tag_prefix": "0.6b-single",
            "modes": ("eager", "cg"),
            "requires_gpus": 1,
        },
        {
            "engine": "v4",
            "model": M06,
            "tp": 2,
            "pp": 1,
            "dp": 1,
            "tag_prefix": "0.6b-tp2",
            "modes": ("eager", "cg"),
            "requires_gpus": 2,
        },
        {
            "engine": "v4",
            "model": M06,
            "tp": 1,
            "pp": 2,
            "dp": 1,
            "tag_prefix": "0.6b-pp2",
            "modes": ("eager", "cg"),
            "requires_gpus": 2,
        },
    ]
    return expand(base)


def suite_full() -> list[dict]:
    """对齐 docs/03-性能对比.md 的主矩阵（本机 2 卡可跑的部分）。"""
    base = [
        {
            "engine": "v4",
            "model": M06,
            "tp": 1,
            "pp": 1,
            "dp": 1,
            "tag_prefix": "0.6b-single",
            "requires_gpus": 1,
        },
        {
            "engine": "v4",
            "model": M06,
            "tp": 2,
            "pp": 1,
            "dp": 1,
            "tag_prefix": "0.6b-tp2",
            "requires_gpus": 2,
        },
        {
            "engine": "v4",
            "model": M06,
            "tp": 1,
            "pp": 2,
            "dp": 1,
            "tag_prefix": "0.6b-pp2",
            "requires_gpus": 2,
        },
        {
            "engine": "v4",
            "model": M8B,
            "tp": 1,
            "pp": 1,
            "dp": 1,
            "device_ids": "0",
            "tag_prefix": "8b-single",
            "requires_gpus": 1,
        },
        {
            "engine": "v4",
            "model": M8B,
            "tp": 2,
            "pp": 1,
            "dp": 1,
            "tag_prefix": "8b-tp2",
            "requires_gpus": 2,
        },
        {
            "engine": "v4",
            "model": M8B,
            "tp": 1,
            "pp": 2,
            "dp": 1,
            "tag_prefix": "8b-pp2",
            "requires_gpus": 2,
        },
        {
            "engine": "v4",
            "model": M14B,
            "tp": 1,
            "pp": 2,
            "dp": 1,
            "pp_layer_counts": "12,28",
            "device_ids": "1,0",
            "num_kvcache_blocks": 32,
            "tag_prefix": "14b-pp2",
            "requires_gpus": 2,
        },
        {
            "engine": "v4",
            "model": M4B,
            "tp": 1,
            "pp": 1,
            "dp": 1,
            "device_ids": "0",
            "n_reqs": 32,
            "num_kvcache_blocks": 48,
            "tag_prefix": "4b-dp1",
            "requires_gpus": 1,
        },
        {
            "engine": "v4",
            "model": M4B,
            "tp": 1,
            "pp": 1,
            "dp": 2,
            "n_reqs": 32,
            "num_kvcache_blocks": 48,
            "modes": SAFE_DP_MODES,
            "tag_prefix": "4b-dp2",
            "requires_gpus": 2,
        },
    ]
    return expand(base)


def suite_vs_vllm() -> list[dict]:
    """v4 与官方对照：8B 单卡与 TP=2。"""
    v4 = [
        {
            "engine": "v4",
            "model": M8B,
            "tp": 1,
            "pp": 1,
            "dp": 1,
            "device_ids": "0",
            "modes": ("eager", "cg", "compile"),
            "tag_prefix": "8b-single",
            "requires_gpus": 1,
        },
        {
            "engine": "v4",
            "model": M8B,
            "tp": 2,
            "pp": 1,
            "dp": 1,
            "modes": ("eager", "cg", "compile"),
            "tag_prefix": "8b-tp2",
            "requires_gpus": 2,
        },
    ]
    vllm = [
        {
            "engine": "vllm",
            "model": M8B,
            "tp": 1,
            "pp": 1,
            "dp": 1,
            "cuda_devices": "0",
            "modes": ("eager", "cg", "compile"),
            "tag_prefix": "vllm-8b-tp1",
            "requires_gpus": 1,
        },
        {
            "engine": "vllm",
            "model": M8B,
            "tp": 2,
            "pp": 1,
            "dp": 1,
            "cuda_devices": "0,1",
            "modes": ("eager", "cg", "compile"),
            "tag_prefix": "vllm-8b-tp2",
            "requires_gpus": 2,
        },
    ]
    return expand(v4 + vllm)


SUITES = {
    "smoke": suite_smoke,
    "full": suite_full,
    "vs-vllm": suite_vs_vllm,
}
