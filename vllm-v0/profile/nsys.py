"""一键 nsys profile —— NVIDIA Nsight Systems 内核级分析。

nsys 比 torch.profiler 更深层: 可以看到每个 CUDA kernel 的
    - 启动/执行时间线
    - 显存传输 (H2D/D2H)
    - GPU 利用率/SM 占用率
    - CPU-GPU 同步点

用法:
  python vllm-v0/profile/nsys.py --model ~/huggingface/Qwen3-0.6B

生成 .nsys-rep 文件后:
  nsys-ui v0_profile.nsys-rep     # GUI 打开 (需要 Nsight Systems GUI)
或
  nsys stats v0_profile.nsys-rep  # 命令行统计
"""

from __future__ import annotations

import argparse, subprocess, sys, tempfile
from pathlib import Path


_WRAPPER = r'''
"""nsys 内执行脚本 —— 与 trace.py 相同逻辑，但输出精简。"""
import sys, time
sys.path.append("{proj}")
sys.path.append("{proj}/vllm-v0")
import torch
from src.config import EngineConfig
from src.engine import Engine

eng = Engine(EngineConfig(model="{model}", enforce_eager=True, context_len={ctx}))
text = "The capital of France is Paris. " * 10
ids = eng.tokenizer.encode(text)[: {ctx} - {max_new} - 10]

for _ in range(3):
    h = eng.model(torch.tensor([ids], device=eng.device, dtype=torch.long),
                  torch.arange(len(ids), device=eng.device).unsqueeze(0))
    ids.append(int(eng.sampler(eng.model.compute_logits(h[:, -1:]).squeeze(1),
                               None, None, None, None).item()))
torch.cuda.synchronize()

t0 = time.perf_counter()
for _ in range({max_new}):
    sl = len(ids)
    h = eng.model(torch.tensor([ids], device=eng.device, dtype=torch.long),
                  torch.arange(sl, device=eng.device).unsqueeze(0))
    ids.append(int(eng.sampler(eng.model.compute_logits(h[:, -1:]).squeeze(1),
                               None, None, None, None).item()))
    if ids[-1] == eng.eos_token_id or sl >= {ctx}:
        break
torch.cuda.synchronize()
print(f"nsys done: {len(ids)-sl} steps in {time.perf_counter()-t0:.2f}s")
'''


def run(model: str, max_new: int = 30, ctx: int = 512,
        output: str = "./traces/v0_nsys", python: str = "") -> None:
    """生成 nsys 临时脚本并调用 nsys profile。"""

    proj = str(Path(__file__).resolve().parent.parent.parent)
    script = _WRAPPER.format(proj=proj, model=model, ctx=ctx, max_new=max_new)
    python = python or sys.executable

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".py", delete=False, prefix="nsys_v0_"
    ) as f:
        f.write(script)
        tmp = f.name

    Path(output).parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "nsys", "profile",
        "--trace=cuda,nvtx,osrt",  # CUDA kernel + NVTX 标注 + OS runtime
        "--output", output,
        "--force-overwrite=true",
        "--cuda-memory-usage=true",  # 显存使用情况
        python, tmp,
    ]
    print(f"\n  {subprocess.list2cmdline(cmd)}\n")
    result = subprocess.run(cmd)
    Path(tmp).unlink(missing_ok=True)

    if result.returncode == 0:
        rep = f"{output}.nsys-rep"
        print(f"\n[nsys] {rep}")
        print(f"  打开 GUI:  nsys-ui {rep}")
        print(f"  命令行:   nsys stats {rep}")
    else:
        print(f"\n[!] nsys 失败 (code={result.returncode})，"
              f"请确认 nsys 已安装: nvidia-smi 同目录下的 nsys")


def main():
    p = argparse.ArgumentParser(description="nsys profile v0")
    p.add_argument("--model", default="~/huggingface/Qwen3-0.6B")
    p.add_argument("--max-new", type=int, default=30)
    p.add_argument("--ctx", type=int, default=512)
    p.add_argument("--output", default="./traces/v0_nsys")
    p.add_argument("--python", default="")
    args = p.parse_args()
    run(args.model, args.max_new, args.ctx, args.output, args.python)


if __name__ == "__main__":
    main()
