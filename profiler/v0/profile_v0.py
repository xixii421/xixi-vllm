"""vllm-v0 系统性 profile：TTFT / TPOT / Throughput + 算子级 trace。

用法:
    python profile_v0.py --seqs 64,256,512 --decode-steps 20          # eager
    python profile_v0.py --compile reduce-overhead --seqs 512         # torch.compile

输出:
    控制台: 各序列长度的 prefill/decode 指标汇总
    traces/v0/seq<N>_<mode>/{trace.json, table.txt}                  # Chrome trace + 算子表
"""

from __future__ import annotations

import argparse
import statistics
import sys
import time
from pathlib import Path

import torch
from torch.profiler import ProfilerActivity, profile, record_function

V0_DIR = Path(__file__).resolve().parents[2] / "vllm-v0"
sys.path.insert(0, str(V0_DIR))
from src.model import Qwen3ForCausalLM          # noqa: E402
from src.model_loader import load_model          # noqa: E402

MODEL = "~/huggingface/Qwen3-0.6B"
VOCAB = 151936


def load(model_path: str, compile_mode: str) -> torch.nn.Module:
    from transformers import AutoConfig

    model_path = str(Path(model_path).expanduser())
    cfg = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    model = Qwen3ForCausalLM(cfg)
    load_model(model, model_path)
    dtype = getattr(cfg, "torch_dtype", torch.bfloat16)
    model = model.to(dtype=dtype, device="cuda").eval()
    if compile_mode != "none":
        model = torch.compile(model, mode=compile_mode)
    return model


def make_inputs(seq: int) -> tuple[torch.Tensor, torch.Tensor]:
    """随机 token 序列 + 绝对位置 0..seq-1（模拟 engine 的输入形态）。"""
    ids = torch.randint(0, VOCAB, (1, seq), device="cuda", dtype=torch.long)
    pos = torch.arange(seq, device="cuda").unsqueeze(0)
    return ids, pos


@torch.inference_mode()
def forward_once(model: torch.nn.Module, ids: torch.Tensor, pos: torch.Tensor) -> torch.Tensor:
    h = model(ids, pos)
    return model.compute_logits(h[:, -1:])


def timed(fn) -> float:
    """返回一次调用在 GPU 上的耗时（ms）。"""
    s, e = torch.cuda.Event(True), torch.cuda.Event(True)
    s.record()
    fn()
    e.record()
    torch.cuda.synchronize()
    return s.elapsed_time(e)


def bench_seq(model: torch.nn.Module, seq: int, steps: int) -> dict:
    """对单个序列长度：测 TTFT（prefill）与每步 TPOT（decode，序列递增）。"""
    ids, pos = make_inputs(seq)
    forward_once(model, ids, pos)
    torch.cuda.synchronize()

    # prefill：完整 prompt 一次 forward → 第一个 token
    ttft = timed(lambda: forward_once(model, ids, pos))

    # decode：每步 append 新 token，序列变长（v0 无 KV Cache，每步重算全序列）
    step_ms: list[float] = []
    for i in range(steps):
        step_ms.append(timed(lambda: forward_once(model, ids, pos)))
        nxt = torch.randint(0, VOCAB, (1, 1), device="cuda", dtype=torch.long)
        ids = torch.cat([ids, nxt], dim=1)
        pos = torch.arange(ids.shape[1], device="cuda").unsqueeze(0)  # 位置同步增长

    return {
        "ttft": ttft,
        "tpot_med": statistics.median(step_ms),
        "tpot_last": step_ms[-1],
        # 吞吐按中位 TPOT 估算（compile 下个别步会触发 CUDA Graph 重录，均值被污染）
        "throughput": 1000 / statistics.median(step_ms),
    }


def profile_trace(model: torch.nn.Module, seq: int, steps: int, compile_mode: str, out_dir: Path) -> None:
    """对指定序列跑 torch.profiler，输出 chrome trace + 算子汇总表。"""
    ids, pos = make_inputs(seq)
    for _ in range(3):  # 预热，避开首轮分配/编译噪声
        forward_once(model, ids, pos)
    torch.cuda.synchronize()

    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        with record_function(f"prefill_seq{seq}"):
            forward_once(model, ids, pos)
        with record_function(f"decode_x{steps}"):
            for _ in range(steps):
                forward_once(model, ids, pos)

    tag = f"seq{seq}_{compile_mode}"
    out_dir.mkdir(parents=True, exist_ok=True)
    prof.export_chrome_trace(str(out_dir / f"{tag}.json"))
    with open(out_dir / f"{tag}.txt", "w") as f:
        f.write(prof.key_averages().table(sort_by="cuda_time_total", row_limit=35))
    print(f"trace → {out_dir / tag}.{{json,txt}}")


def main() -> None:
    p = argparse.ArgumentParser(description="vllm-v0 系统性 profile")
    p.add_argument("--seqs", default="64,256,512", help="逗号分隔的 prompt token 长度")
    p.add_argument("--decode-steps", type=int, default=20, help="decode 步数")
    p.add_argument("--compile", default="none",
                   choices=["none", "reduce-overhead"], help="torch.compile 模式")
    p.add_argument("--model", default=MODEL)
    p.add_argument("--trace-dir", default="./traces/v0")
    args = p.parse_args()

    seqs = [int(s) for s in args.seqs.split(",")]
    model = load(args.model, args.compile)
    print(f"model={args.model} compile={args.compile} seqs={seqs}")

    print(f"\n{'seq':>5} {'TTFT(ms)':>10} {'TPOT med':>10} {'TPOT last':>10} {'tok/s':>8}")
    print("-" * 48)
    for seq in seqs:
        r = bench_seq(model, seq, args.decode_steps)
        print(f"{seq:>5} {r['ttft']:>10.2f} {r['tpot_med']:>10.2f} "
              f"{r['tpot_last']:>10.2f} {r['throughput']:>8.1f}")

    profile_trace(model, seqs[-1], args.decode_steps, args.compile,
                  Path(args.trace_dir))


if __name__ == "__main__":
    main()
