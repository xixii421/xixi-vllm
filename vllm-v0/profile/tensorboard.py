"""启动 TensorBoard 可视化 profiling trace JSON。

用法:  python -m vllm-v0.profile.tensorboard [--logdir ./traces/v0] [--port 6006]
"""
from __future__ import annotations

import argparse, subprocess, sys
from pathlib import Path

def run(logdir: str = "./traces/v0", port: int = 6006, host: str = "0.0.0.0") -> None:
    logdir = str(Path(logdir).resolve())
    if not Path(logdir).exists():
        print(f"[!] logdir 不存在: {logdir}")
        sys.exit(1)

    # 查找匹配的 trace json 文件
    jsons = list(Path(logdir).rglob("*.pt.trace.json"))
    if jsons:
        print(f"  找到 {len(jsons)} 个 trace 文件:")
        for j in jsons:
            print(f"    {j}")
    else:
        print(f"[!] {logdir} 中未找到 .pt.trace.json 文件")
        print("    请先运行 trace.py 生成 trace")

    cmd = [
        sys.executable, "-m", "tensorboard.main",
        "--logdir", logdir,
        "--port", str(port),
        "--host", host,
        "--load_fast=false",
    ]
    print(f"\n  tensorboard → http://{host}:{port}")
    print(f"  打开后选择左侧 PYTORCH_PROFILER 标签\n")
    subprocess.run(cmd)


def main():
    p = argparse.ArgumentParser(description="TensorBoard 可视化 trace")
    p.add_argument("--logdir", default="./traces/v0")
    p.add_argument("--port", type=int, default=6006)
    p.add_argument("--host", default="0.0.0.0")
    args = p.parse_args()
    run(args.logdir, args.port, args.host)


if __name__ == "__main__":
    main()
