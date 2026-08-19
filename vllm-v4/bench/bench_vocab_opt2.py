"""AI 评审提出的优化点逐项实测验证。

覆盖：
  优化1  Tensor Core 对齐（per_partition 是否应为 16/64 倍数）       GEMM 实测
  优化2  _load_weight 零分配（直接拷贝+zero_  vs  cat+new_zeros）     实测
  优化3  forward 用 masked_fill_ in-place                            实测
  优化4  greedy decode 只 all_gather argmax（vs 全量 logits）         NCCL 实测

用法（在 vllm-v4/ 下，需 >=2 张卡测优化4）:
    ../.venv2/bin/python bench/bench_vocab_opt2.py
"""

from __future__ import annotations

import time

import torch

VOCAB = 151937
DIM = 4096
TP = 8


def _pad_vocab(vocab: int, tp: int) -> int:
    return ((vocab + tp - 1) // tp) * tp


def _pad_vocab_tc(vocab: int, tp: int, multiple: int = 64) -> int:
    per = (vocab + tp - 1) // tp
    per = ((per + multiple - 1) // multiple) * multiple
    return per * tp


def _time_cuda(fn, *args, warmup: int = 30, iters: int = 200) -> float:
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


def _time_cpu(fn, *args, iters: int = 5, **kwargs) -> float:
    t0 = time.perf_counter()
    for _ in range(iters):
        fn(*args, **kwargs)
    return (time.perf_counter() - t0) / iters * 1e3  # ms


# --------------------------------------------------------------------------- #
# 优化1 Tensor Core 对齐
# --------------------------------------------------------------------------- #


def bench_gemm_padding() -> None:
    print("=" * 72)
    print("优化1: Tensor Core 对齐（LM head GEMM 输出维）")
    print("=" * 72)
    if not torch.cuda.is_available():
        print("无 GPU，跳过")
        return
    dev = "cuda"
    torch.manual_seed(0)
    bs = 256
    cases = [
        ("Qwen3真实", VOCAB - 1),  # 151936, per=18992
        ("非对齐", VOCAB),          # 151937, per=18993
    ]
    for label, vocab in cases:
        padded = _pad_vocab(vocab, TP)
        per = padded // TP
        per_tc64 = _pad_vocab_tc(vocab, TP, 64) // TP
        print(f"[{label}] vocab={vocab}: per={per} (16-mult={per % 16 == 0}, "
              f"64-mult={per % 64 == 0}), per_tc64={per_tc64}")
        x = torch.randn(bs, DIM, device=dev, dtype=torch.bfloat16)
        for name, n in [("per(原)", per), ("per_tc64", per_tc64)]:
            w = torch.randn(n, DIM, device=dev, dtype=torch.bfloat16)
            t = _time_cuda(lambda: torch.nn.functional.linear(x, w))
            print(f"  {name}: out_dim={n:<6} linear={t:.4f} ms")
    print("结论: 由实测判定 16/64 对齐是否带来 GEMM 收益")
    print()


# --------------------------------------------------------------------------- #
# 优化2 零分配加载
# --------------------------------------------------------------------------- #


def load_cat(param, loaded_weight, *, padded_vocab, rank=0, row_offset=0):
    tp = TP
    per = padded_vocab // tp
    start = per * rank
    seg = loaded_weight[start : start + per]
    if seg.shape[0] < per:
        seg = torch.cat([seg, seg.new_zeros(per - seg.shape[0], seg.shape[1])], dim=0)
    param[row_offset : row_offset + per].copy_(seg)


def load_zeroalloc(param, loaded_weight, *, padded_vocab, rank=0, row_offset=0):
    tp = TP
    per = padded_vocab // tp
    start = per * rank
    total = loaded_weight.shape[0]
    seg_len = max(0, min(per, total - start))
    if seg_len > 0:
        param[row_offset : row_offset + seg_len].copy_(
            loaded_weight[start : start + seg_len]
        )
    if seg_len < per:
        param[row_offset + seg_len : row_offset + per].zero_()


def bench_load() -> None:
    print("=" * 72)
    print("优化2: _load_weight 零分配")
    print("=" * 72)
    padded = _pad_vocab(VOCAB, TP)
    per = padded // TP
    loaded = torch.randn(VOCAB, DIM)  # CPU 加载场景
    for rank in (0, TP - 1):
        p1 = torch.zeros(per, DIM)
        p2 = torch.zeros(per, DIM)
        t_cat = _time_cpu(load_cat, p1, loaded, padded_vocab=padded, rank=rank)
        t_za = _time_cpu(load_zeroalloc, p2, loaded, padded_vocab=padded, rank=rank)
        same = torch.equal(p1, p2)
        print(f"rank={rank}: 结果一致={same}  cat={t_cat:.2f}ms  "
              f"零分配={t_za:.2f}ms  ({t_cat / max(t_za, 1e-9):.1f}x)")
    print()


# --------------------------------------------------------------------------- #
# 优化3 masked_fill in-place
# --------------------------------------------------------------------------- #


def bench_mask() -> None:
    print("=" * 72)
    print("优化3: masked_fill vs masked_fill_")
    print("=" * 72)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(2)
    n = 1 << 16
    x = torch.randint(0, VOCAB, (n,), device=dev)
    start, end = 1000, 2000

    def mf():
        mask = (x < start) | (x >= end)
        return (x - start + 1).masked_fill(mask, 0)

    def mf_inplace():
        mask = (x < start) | (x >= end)
        y = x - start + 1
        y.masked_fill_(mask, 0)
        return y

    a, b = mf(), mf_inplace()
    t1 = _time_cuda(mf) if dev == "cuda" else _time_cpu(mf)
    t2 = _time_cuda(mf_inplace) if dev == "cuda" else _time_cpu(mf_inplace)
    print(f"n={n}: masked_fill={t1:.4f}ms  masked_fill_={t2:.4f}ms  "
          f"等价={torch.equal(a, b)}")
    print()


# --------------------------------------------------------------------------- #
# 优化4 greedy argmax 通信（需 2 卡）
# --------------------------------------------------------------------------- #


def _greedy_worker(rank: int, port: int) -> None:
    import sys
    from pathlib import Path

    root = Path(__file__).resolve().parent.parent
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

    from src.distributed import (
        destroy_distributed_environment,
        destroy_model_parallel,
        get_tp_group,
        init_distributed_environment,
        initialize_model_parallel,
        tensor_model_parallel_all_gather,
    )

    init_distributed_environment(
        world_size=2, rank=rank, local_rank=rank, backend="nccl",
        init_method=f"tcp://127.0.0.1:{port}",
    )
    initialize_model_parallel(2, 1, rank, 0)
    torch.manual_seed(rank)
    bs = 256
    padded = _pad_vocab(VOCAB, 2)
    per = padded // 2
    local = torch.randn(bs, per, device="cuda", dtype=torch.bfloat16)

    def full():
        return tensor_model_parallel_all_gather(local, dim=-1)

    def greedy():
        vals, idx = local.max(dim=-1)  # 均为 [bs]
        idx = idx + per * get_tp_group().group_rank  # 本地行号 -> 全局词表号
        bs = vals.shape[0]
        vals = tensor_model_parallel_all_gather(vals, dim=-1)  # [bs*tp] 按 rank 拼接
        idx = tensor_model_parallel_all_gather(idx, dim=-1)    # int64 保精度
        tp = vals.numel() // bs
        vals = vals.reshape(tp, bs).t()  # [bs, tp]
        idx = idx.reshape(tp, bs).t()
        best = vals.argmax(dim=-1)  # 平局取第一个最大 = 最低 vocab 序号，与全量 argmax 一致
        return idx.gather(-1, best.unsqueeze(-1)).squeeze(-1)

    # 正确性对照：全量 gather 后 argmax 与 局部 argmax 再 gather 应一致
    full_id = full().argmax(dim=-1)
    fast_id = greedy()
    ok = torch.equal(full_id, fast_id)
    if not ok:
        ne = (full_id != fast_id)
        diff_idx = ne.nonzero()[:5].flatten().tolist()
        print(f"rank{rank}: dtype full_id={full_id.dtype} fast_id={fast_id.dtype} "
              f"diff={diff_idx}", flush=True)
    print(f"rank{rank}: argmax 一致={ok}", flush=True)

    t_full = _time_cuda(full)
    t_gr = _time_cuda(greedy)
    print(f"rank{rank}: 全量 all_gather={t_full:.4f}ms  "
          f"argmax all_gather={t_gr:.4f}ms  ({t_full / max(t_gr, 1e-9):.1f}x)",
          flush=True)
    destroy_model_parallel()
    destroy_distributed_environment()


def bench_greedy_gather() -> None:
    print("=" * 72)
    print("优化4: greedy 只 all_gather argmax vs 全量 logits")
    print("=" * 72)
    if torch.cuda.device_count() < 2:
        print("不足 2 卡，跳过 NCCL 实测（仅打印理论通信量）")
        bs = 256
        full_bytes = bs * _pad_vocab(VOCAB, TP) * 2  # bf16
        argmax_bytes = bs * 2 * 8
        print(f"  decode bs={bs}: 全量 logits 通信={full_bytes/2**20:.1f}MB, "
              f"argmax 通信={argmax_bytes/2**10:.1f}KB "
              f"({full_bytes/argmax_bytes:.0f}x)")
        print()
        return

    import os
    import torch.multiprocessing as mp

    port = 29500 + (os.getpid() % 2000)
    mp.spawn(_greedy_worker, args=(port,), nprocs=2, join=True)
    print()


if __name__ == "__main__":
    bench_gemm_padding()
    bench_load()
    bench_mask()
    bench_greedy_gather()
