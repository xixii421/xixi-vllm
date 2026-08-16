"""清空 bench/out 后按序跑 01-07。

v3 / nano 用仓库 .venv2；官方 vLLM 用 .venv-vllm。
默认钉在本机 NVIDIA GeForce RTX 4090 D（nvidia-smi PCI 序）。

用法（在 vllm-v3/ 下）:
    ../.venv2/bin/python bench/run_all.py
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

BENCH = Path(__file__).resolve().parent
ROOT = BENCH.parent
REPO = ROOT.parent
PY = REPO / ".venv2/bin/python"
PYV = REPO / ".venv-vllm/bin/python"
OUT = BENCH / "out"


def pick_4090() -> str:
    raw = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=index,name", "--format=csv,noheader"],
        text=True,
    )
    for line in raw.strip().splitlines():
        idx, name = [x.strip() for x in line.split(",", 1)]
        if "4090" in name:
            return idx
    return "0"


def run(py: Path, *args: str, env: dict[str, str]) -> None:
    cmd = [str(py), *args]
    print(f"\n{'=' * 64}\n$ {' '.join(cmd)}\n{'=' * 64}\n", flush=True)
    r = subprocess.run(cmd, cwd=str(ROOT), env=env)
    if r.returncode != 0:
        sys.exit(r.returncode)


def main() -> None:
    if not PY.is_file():
        sys.exit(f"missing {PY}")
    if not PYV.is_file():
        sys.exit(f"missing {PYV}")

    env = os.environ.copy()
    env["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    env["CUDA_VISIBLE_DEVICES"] = pick_4090()
    print(
        f"GPU CUDA_VISIBLE_DEVICES={env['CUDA_VISIBLE_DEVICES']} "
        f"(PCI_BUS_ID, 4090 D if present)",
        flush=True,
    )

    shutil.rmtree(OUT, ignore_errors=True)
    OUT.mkdir(parents=True, exist_ok=True)

    for name in (
        "01_ops.py",
        "02_correctness.py",
        "03_prefill.py",
        "04_prefix.py",
        "05_scheduler.py",
        "06_e2e.py",
    ):
        run(PY, f"bench/{name}", env=env)

    run(PY, "bench/07_compare.py", "--engine", "v3", env=env)
    run(PY, "bench/07_compare.py", "--engine", "nano", env=env)
    run(PYV, "bench/07_compare.py", "--engine", "vllm", env=env)

    print("\n======== bench/out ========")
    for p in sorted(OUT.iterdir()):
        print(f"  {p.name:24s} {p.stat().st_size:6d} B")


if __name__ == "__main__":
    main()
