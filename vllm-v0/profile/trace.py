"""v0 算子画像 —— prefill + decode 同 trace，展示 S 变化导致的时间变化。

用法:  python -m vllm-v0.profile.trace --model ~/huggingface/Qwen3-0.6B
"""

from __future__ import annotations

import argparse, sys, time
from pathlib import Path

_PROJ = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(_PROJ / "vllm-v0"))
sys.path.append(str(_PROJ))

import torch
from src.config import EngineConfig
from src.engine import Engine
from profiler.profiler import Profiler

_PROMPT = ("The capital of France is Paris. Deep neural networks "
           "have many layers. Transformers use self-attention. " * 5)


def _fwd(eng, ids):
    """一次前向 + 采样，返回 (new_token, seq_len)。"""
    s = len(ids)
    h = eng.model(torch.tensor([ids], device=eng.device, dtype=torch.long),
                  torch.arange(s, device=eng.device).unsqueeze(0))
    nid = int(eng.sampler(eng.model.compute_logits(h[:, -1:]).squeeze(1),
                          None, None, None, None).item())
    return nid, s


def run(model: str, max_new: int = 30, ctx: int = 512,
        trace_dir: str = "./traces/v0") -> None:
    eng = Engine(EngineConfig(model=model, enforce_eager=True, context_len=ctx))
    ids = eng.tokenizer.encode(_PROMPT)[:ctx - max_new - 10]
    prompt_len = len(ids)

    # 1. 独立 warmup: 消除 CUDA JIT，逐 token 增长 (不在 profiler 内)
    for _ in range(5):
        nid, _ = _fwd(eng, ids)
        ids.append(nid)
    torch.cuda.synchronize()
    prefill_S = len(ids)  # warmup 后的 S

    # 2. prefill + decode 全量 trace
    pf = Profiler(trace_dir=trace_dir, label="v0")
    pf.start()
    # prefill: 一次全量提示词前向
    with pf.annotate("PREFILL"):
        nid, _ = _fwd(eng, ids)
    ids.append(nid)
    # decode: 逐步生成
    for i in range(max_new):
        with pf.annotate(f"DECODE_S={len(ids)}"):
            nid, sl = _fwd(eng, ids)
        ids.append(nid)
        if nid == eng.eos_token_id or sl >= ctx:
            break
    pf.stop()

    # 3. 输出
    print(f"\n  {'='*60}")
    print(f"  PREFILL  S={prefill_S}    (全量提示词一次前向)")
    print(f"  DECODE   步数={len(ids)-prefill_S-1}  (每步 S 增长: {prefill_S}→{len(ids)-1})")
    print(f"  {'='*60}")

    # 4. 算子 TOP-20 明细
    print(f"\n  算子 TOP-20 (按 CUDA 时间排序):")
    pf.print_full_table(sort_by="cuda_time_total", top=20)

    # 5. 分类汇总: CPU vs CUDA 分别占比
    query = (
        "  → TensorBoard: 找到 PREFILL/DECODE_S=* 标注\n"
        "  → 观察: S 变大时 scaled_dot_product_attention 时间膨胀\n"
        "  → CPU/GPU>3 说明 Python 开销大或数据搬运是瓶颈"
    )
    pf.print_category_summary()
    print(f"\n[trace] {Path(trace_dir).resolve()}/v0*.pt.trace.json")
    print(query)


def main():
    p = argparse.ArgumentParser(description="v0 算子画像")
    p.add_argument("--model", default="~/huggingface/Qwen3-0.6B")
    p.add_argument("--max-new", type=int, default=30)
    p.add_argument("--ctx", type=int, default=512)
    args = p.parse_args()
    run(args.model, args.max_new, args.ctx)


if __name__ == "__main__":
    main()
