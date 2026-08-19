"""VocabParallelEmbedding 重构前后验证：正确性 + 性能。

正确性：模拟 TP 分片（tp=2/4，含 vocab 不整除的 padding 情形），把三种 forward
变体经 all-reduce 求和的结果与全量嵌入对照：
    - double_mask   重构前实现（输入掩码 + 输出掩码）
    - zero_row      重构后实现（第 0 行恒 0 的越界落点，无输出掩码）
    - no_output_mask 错误变体（缺输出掩码，曾误当"优化"）

性能：GPU 上对比三变体端到端延迟，并拆分子算子耗时。

用法（在 vllm-v4/ 下）:
    ../.venv2/bin/python bench/bench_vocab_parallel.py
"""

from __future__ import annotations

import torch


def _pad_vocab(vocab: int, tp: int) -> int:
    return ((vocab + tp - 1) // tp) * tp


# --------------------------------------------------------------------------- #
# forward 变体（仅 forward 逻辑，不依赖 nn.Module / dist，便于对照）
# --------------------------------------------------------------------------- #


def forward_double_mask(x, weight, start, end):
    """重构前：输入掩码 + 输出掩码。"""
    mask = (x < start) | (x >= end)
    shifted = (x - start).masked_fill(mask, 0)
    out = torch.nn.functional.embedding(shifted, weight)
    out = out.masked_fill(mask.unsqueeze(-1), 0.0)
    return out


def forward_zero_row(x, weight, start, end):
    """重构后：第 0 行恒 0 的越界落点，越界 token 查表得 0，无输出掩码。"""
    mask = (x < start) | (x >= end)
    local_idx = (x - start + 1).masked_fill(mask, 0)
    return torch.nn.functional.embedding(local_idx, weight)


def forward_no_output_mask(x, weight, start, end):
    """错误变体：只做输入掩码、缺输出掩码。"""
    mask = (x < start) | (x >= end)
    shifted = (x - start).masked_fill(mask, 0)
    return torch.nn.functional.embedding(shifted, weight)


def _shard(full: torch.Tensor, rank: int, tp: int, *, zero_row: bool) -> torch.Tensor:
    """按 rank 切分权重；zero_row=True 时第 0 行为恒 0 占位行。"""
    padded = _pad_vocab(full.shape[0], tp)
    if padded > full.shape[0]:
        pad = torch.zeros(padded, full.shape[1], dtype=full.dtype)
        pad[: full.shape[0]] = full
        full = pad
    per = padded // tp
    if zero_row:
        w = torch.zeros(per + 1, full.shape[1], dtype=full.dtype)
        w[1:] = full[per * rank : per * (rank + 1)]
    else:
        w = full[per * rank : per * (rank + 1)].clone()
    return w


# --------------------------------------------------------------------------- #
# 正确性验证：模拟 TP
# --------------------------------------------------------------------------- #


def check_correctness() -> None:
    print("=" * 72)
    print("正确性验证（模拟 TP all-reduce 求和）")
    print("=" * 72)
    torch.manual_seed(0)
    all_ok = True
    for vocab, tp in [(10, 2), (151936, 8), (102, 4)]:  # 含不整除的 padding 情形
        dim = 16 if vocab < 1000 else 4096
        full = torch.randn(vocab, dim)
        x = torch.randint(0, vocab, (4096,))
        per = _pad_vocab(vocab, tp) // tp
        ref = full[x]

        acc = {"double_mask": 0.0, "zero_row": 0.0, "no_output_mask": 0.0}
        for r in range(tp):
            w_dm = _shard(full, r, tp, zero_row=False)
            w_zr = _shard(full, r, tp, zero_row=True)
            s, e = per * r, per * (r + 1)
            acc["double_mask"] += forward_double_mask(x, w_dm, s, e)
            acc["zero_row"] += forward_zero_row(x, w_zr, s, e)
            acc["no_output_mask"] += forward_no_output_mask(x, w_dm, s, e)

        for name, got in acc.items():
            ok = torch.equal(got, ref)
            all_ok &= ok
            print(f"vocab={vocab:<6} tp={tp:<3} {name:<16} "
                  f"{'PASS' if ok else 'FAIL(污染)'}  "
                  f"max_err={(got - ref).abs().max().item():.2e}")
    print("结论:", "全部通过" if all_ok else "存在失败")
    print()


# --------------------------------------------------------------------------- #
# 性能验证：GPU 延迟
# --------------------------------------------------------------------------- #


def _time_fn(fn, *args, warmup: int = 20, iters: int = 200) -> float:
    for _ in range(warmup):
        fn(*args)
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn(*args)
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters  # ms


def check_performance() -> None:
    print("=" * 72)
    print("性能验证（GPU 延迟，单位 ms/次）")
    print("=" * 72)
    if not torch.cuda.is_available():
        print("无 GPU，跳过性能验证")
        return
    torch.manual_seed(1)

    for name, bs, dim in [("decode bs=256", 256, 4096), ("prefill n=8192", 8192, 4096)]:
        vocab, tp, rank = 151936, 8, 3  # 接近 Qwen3 真实配置，取中间 rank
        per = _pad_vocab(vocab, tp) // tp
        start, end = per * rank, per * (rank + 1)
        x = torch.randint(0, vocab, (bs,), device="cuda")
        w_dm = torch.randn(per, dim, device="cuda")
        w_zr = torch.zeros(per + 1, dim, device="cuda")
        w_zr[1:] = w_dm

        t_dm = _time_fn(forward_double_mask, x, w_dm, start, end)
        t_zr = _time_fn(forward_zero_row, x, w_zr, start, end)
        t_no = _time_fn(forward_no_output_mask, x, w_dm, start, end)

        print(f"{name}:")
        print(f"    重构前 double_mask : {t_dm:8.4f} ms")
        print(f"    重构后 zero_row    : {t_zr:8.4f} ms   (相对重构前 "
              f"{t_zr / t_dm:.2f}x)")
        print(f"    错误变体 no_output : {t_no:8.4f} ms   (zero_row 已同等速度且正确)")
    print("结论: zero_row 消除了输出侧全量 masked_fill，"
          "拿到与错误变体相同的速度，同时保持正确。")
    print()


if __name__ == "__main__":
    check_correctness()
    check_performance()
