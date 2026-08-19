"""本实现 vs vLLM 官方 VocabParallelEmbedding 前向的公平对比。

vLLM 官方（main 分支，去掉 LoRA 分支后的核心路径）：
    1. org_mask = (x >= start) & (x < end)
    2. local = org_mask * (x - start)          # 越界 token 归 0，会命中 weight[0] 真实词
    3. out = F.embedding(local, weight)        # weight 形状 [per, H]，无哑元行
    4. out.masked_fill_(~org_mask, 0)          # 输出侧 [B,S,H] 写回，必须做

本实现：
    1. mask = (x < start) | (x >= end)
    2. local = (x - start + 1).masked_fill_(mask, 0)   # [B,S] 整数维
    3. out = F.embedding(local, weight)        # weight 形状 [per+1, H]，第 0 行恒 0
    （越界 token 查表即得 0，无输出侧写回）

另对比 vLLM 官方用 torch.compile(dynamic=True) 融合 pointwise 的效果，
以及两方案的解码（小 token 数）与预填充（大 token 数）两种场景。

用法（在 vllm-v4/ 下）:
    ../.venv2/bin/python bench/bench_vs_vllm.py
"""

from __future__ import annotations

import torch

VOCAB = 151936
DIM = 4096


def _time_fn(fn, *args, warmup: int = 30, iters: int = 300) -> float:
    for _ in range(warmup):
        fn(*args)
    torch.cuda.synchronize()
    s = torch.cuda.Event(enable_timing=True)
    e = torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(iters):
        fn(*args)
    e.record()
    torch.cuda.synchronize()
    return s.elapsed_time(e) / iters  # ms


# --------------------------------------------------------------------------- #
# 两种 Embedding forward（eager，tp>1 的本地部分，不含通信）
# --------------------------------------------------------------------------- #


def vllm_forward(x, weight, start, end):
    """vLLM 官方方案：输出侧 [B,S,H] masked_fill_。"""
    org_mask = (x >= start) & (x < end)
    local = org_mask * (x - start)
    out = torch.nn.functional.embedding(local, weight)
    out.masked_fill_(~org_mask.unsqueeze(-1), 0)
    return out


def ours_forward(x, weight_plus1, start, end):
    """本实现：哑元行 + [B,S] 整数掩码，无输出侧写回。"""
    mask = (x < start) | (x >= end)
    local = x - start + 1
    local.masked_fill_(mask, 0)
    return torch.nn.functional.embedding(local, weight_plus1)


def vllm_forward_compiled(x, weight, start, end):
    """vLLM 官方方案，但把 pointwise 部分用 torch.compile 融合（官方默认做法）。"""
    mask = (x >= start) & (x < end)
    local = mask * (x - start)
    out = torch.nn.functional.embedding(local, weight)
    out.masked_fill_(~mask.unsqueeze(-1), 0)
    return out


def main() -> None:
    if not torch.cuda.is_available():
        print("无 GPU，跳过性能对比")
        return
    torch.manual_seed(0)

    per = VOCAB // 8  # tp=8
    start, end = per * 3, per * 4

    print("=" * 72)
    print("Embedding forward 对比  vLLM官方(输出掩码) vs 本实现(哑元行)")
    print("=" * 72)
    print(f"vocab={VOCAB} dim={DIM} per_partition={per}（tp=8 中间 rank）")

    vllm_compiled = torch.compile(vllm_forward_compiled, dynamic=True)

    for name, bs in [("decode bs=256", 256), ("prefill n=8192", 8192)]:
        x = torch.randint(0, VOCAB, (bs,), device="cuda")
        w = torch.randn(per, DIM, device="cuda")
        w1 = torch.zeros(per + 1, DIM, device="cuda")
        w1[1:] = w

        # 正确性对照：两方案应逐位一致
        a = vllm_forward(x, w, start, end)
        b = ours_forward(x, w1, start, end)
        ok = torch.equal(a, b)

        t_v = _time_fn(vllm_forward, x, w, start, end)
        t_o = _time_fn(ours_forward, x, w1, start, end)
        t_c = _time_fn(vllm_compiled, x, w, start, end)
        print(f"\n{name}: 结果一致={ok}")
        print(f"  vLLM 官方 eager       : {t_v:8.4f} ms")
        print(f"  vLLM 官方 +compile     : {t_c:8.4f} ms   ({t_c / t_v:.2f}x vs eager)")
        print(f"  本实现(哑元行)         : {t_o:8.4f} ms   "
              f"({t_o / t_v:.2f}x vs vLLM eager)")

    print()


if __name__ == "__main__":
    main()
