"""Decode Attention 微基准：v2 gather+SDPA vs v3 FlashAttention paged（含 torch.profiler）。

对照 docs/01-v2性能瓶颈.md §3.5，隔离测量单层 decode attention 路径：
不含 qkv 投影 / MLP / RMSNorm / 采样，只有「取历史 KV → 算 attention」这一段。

被测路径（忠实复刻源码，非近似）：
- v2 gather     —— vllm-v2/src/pagedattention/store.py::gather 的 advanced indexing
- v2 decode     —— vllm-v2/src/layers/attention.py::_decode：逐序列 gather + SDPA(enable_gqa)
- v3 decode     —— vllm-v3/src/compute/layers/attention.py decode 分支：flash_attn_with_kvcache

两阶段输出：
1. 细粒度 L 扫描：v2/v3 耗时随 L 增长，展示「v2 近常数（launch 主导）vs v3 线性（带宽主导）」。
2. torch.profiler：L=2048 时 v2/v3 的 kernel 数、总 GPU 时间、top kernel 分布。

形状对齐理论：B=8, Hkv=8, D=128, bf16（Qwen3-0.6B 单层）。
单序列单层有效访存 = 2*L*Hkv*D*2 bytes。

用法:
    python bench/bench_decode_attention.py
    python bench/bench_decode_attention.py --batch 8 --lengths 256,512,1024,2048,4096,8192
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from flash_attn import flash_attn_with_kvcache
from torch.profiler import DeviceType, ProfilerActivity, profile

BENCH = Path(__file__).resolve().parent
V3_ROOT = BENCH.parent
if str(V3_ROOT) not in sys.path:
    sys.path.insert(0, str(V3_ROOT))

NHEADS_Q = 16   # Qwen3-0.6B num_attention_heads
NHEADS_KV = 8   # num_key_value_heads（GQA=2）
HEAD_DIM = 128
BLOCK_SIZE = 256


# --------------------------------------------------------------------------- #
# 输入构造：v2 / v3 共用的分页 KV 池 + block_table
# --------------------------------------------------------------------------- #
def build_inputs(B: int, L: int, *, seed: int = 0):
    """构造解码所需输入。v2 与 v3 共用同一份 KV 池，保证对比公平。

    k_cache/v_cache 形状 [num_blocks, BLOCK_SIZE, Hkv, D] 同时是 v2 pool.k[layer_id]
    与 v3 pool[0/1, layer_id] 的逐层切片；block_table 连续分配，物理布局不影响访存量。
    注意 q 布局：v2 内部 [B, H, 1, D]，v3 是 [B, 1, H, D]（互为转置）。
    """
    g = torch.Generator(device="cuda").manual_seed(seed)
    max_blocks = (L + BLOCK_SIZE - 1) // BLOCK_SIZE
    num_blocks = B * max_blocks

    q_v2 = torch.randn(B, NHEADS_Q, 1, HEAD_DIM, dtype=torch.bfloat16, device="cuda", generator=g)
    q_v3 = q_v2.transpose(1, 2).contiguous()  # [B, 1, H, D]
    k_cache = torch.randn(num_blocks, BLOCK_SIZE, NHEADS_KV, HEAD_DIM, dtype=torch.bfloat16, device="cuda", generator=g)
    v_cache = torch.randn(num_blocks, BLOCK_SIZE, NHEADS_KV, HEAD_DIM, dtype=torch.bfloat16, device="cuda", generator=g)

    block_table = torch.arange(num_blocks, device="cuda").reshape(B, max_blocks).to(torch.int32)
    cache_seqlens = torch.full((B,), L, dtype=torch.int32, device="cuda")

    # v2 gather 的索引元数据：每步不变，可预计算（不含数据搬动）
    positions = torch.arange(L, device="cuda", dtype=torch.long)
    physical = block_table.to(torch.long)[:, positions // BLOCK_SIZE]  # [B, L] 物理块号
    offsets = positions % BLOCK_SIZE                                   # [L]    块内偏移
    return q_v2, q_v3, k_cache, v_cache, block_table, cache_seqlens, physical, offsets


# --------------------------------------------------------------------------- #
# v2 路径
# --------------------------------------------------------------------------- #
def v2_gather(k_cache, v_cache, physical, offsets, b: int):
    """store.py::gather：读分页池并物化连续 [L, Hkv, D]。读 + 写各一次。"""
    return k_cache[physical[b], offsets], v_cache[physical[b], offsets]


def v2_decode(q_v2, k_cache, v_cache, physical, offsets):
    """attention.py::_decode：逐序列 gather + SDPA(enable_gqa)，再 cat + 转置回 [B, 1, H, D]。"""
    outs = []
    for b in range(q_v2.shape[0]):
        k, v = v2_gather(k_cache, v_cache, physical, offsets, b)
        k4 = k.unsqueeze(0).transpose(1, 2)  # [1, Hkv, L, D]
        v4 = v.unsqueeze(0).transpose(1, 2)
        outs.append(F.scaled_dot_product_attention(q_v2[b : b + 1], k4, v4, enable_gqa=True, is_causal=False))
    return torch.cat(outs, dim=0).transpose(1, 2).contiguous()  # [B, 1, H, D]


# --------------------------------------------------------------------------- #
# v3 路径
# --------------------------------------------------------------------------- #
def v3_decode(q_v3, k_cache, v_cache, block_table, cache_seqlens):
    """attention.py decode 分支：flash_attn_with_kvcache 直接读分页池，无 gather。"""
    return flash_attn_with_kvcache(
        q_v3, k_cache, v_cache, cache_seqlens=cache_seqlens, block_table=block_table, causal=True
    )


# --------------------------------------------------------------------------- #
# 正确性对拍：两路径应输出相同的 attention 结果（验证基准忠实）
# --------------------------------------------------------------------------- #
def check_correctness() -> bool:
    q_v2, q_v3, k_cache, v_cache, block_table, cache_seqlens, physical, offsets = build_inputs(8, 2048, seed=7)
    o_v2 = v2_decode(q_v2, k_cache, v_cache, physical, offsets)
    o_v3 = v3_decode(q_v3, k_cache, v_cache, block_table, cache_seqlens)
    diff = (o_v2 - o_v3).abs().max().item()
    print(f"  max|v2-v3| = {diff:.4f}  -> {'ok' if diff < 5e-2 else 'FAIL'}")
    return diff < 5e-2


# --------------------------------------------------------------------------- #
# 计时
# --------------------------------------------------------------------------- #
def bench_fn(fn, iters: int, warmup: int = 25) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters  # ms


# --------------------------------------------------------------------------- #
# torch.profiler：kernel 数 / 总 GPU 时间 / top kernel
# --------------------------------------------------------------------------- #
_KERNEL_ALIAS = {
    "{lambda": "gather(高级索引)",
    "aten::index": "gather(高级索引)",
    "aten::_scaled_dot_product_attention": "SDPA",
    "scaled_dot_product_attention": "SDPA",
    "Flash_fwd_params": "SDPA flash kernel",
    "flash_attn": "flash_attn paged",
    "aten::cat": "cat",
    "aten::transpose": "transpose",
}


def _short(key: str) -> str:
    for pat, alias in _KERNEL_ALIAS.items():
        if pat in key:
            return alias
    return key.split("::")[-1].split("(")[0][:24]


def profile_decode(fn, iters: int = 50, warmup: int = 10) -> dict:
    """跑一次 profiler，返回 {kernel数, gpu_us, top列表}。"""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        for _ in range(iters):
            fn()
        torch.cuda.synchronize()

    cuda = [e for e in prof.key_averages() if e.device_type == DeviceType.CUDA]
    n_kernels = sum(e.count for e in cuda)
    gpu_us = sum(e.self_device_time_total for e in cuda)
    top = sorted(cuda, key=lambda e: -e.self_device_time_total)[:6]
    return {
        "n_kernels": n_kernels,
        "gpu_us": gpu_us,
        "top": [(e.key, e.count, e.self_device_time_total) for e in top],
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--lengths", default="256,512,1024,2048,4096,8192")
    p.add_argument("--iters", type=int, default=100)
    p.add_argument("--profile-len", type=int, default=2048)
    args = p.parse_args()
    torch.cuda.set_device(args.device)

    B = args.batch
    lengths = [int(x) for x in args.lengths.split(",")]

    print("== Decode attention 正确性对拍（v2 vs v3 应一致）==")
    if not check_correctness():
        sys.exit("正确性校验失败，终止。")
    print()

    # ---- 阶段 1：细粒度 L 扫描 ----
    hdr = "| L | v2 gather(us) | v2 decode(us) | v3 decode(us) | v2/v3 加速比 |"
    sep = "| ---: | ---: | ---: | ---: | ---: |"
    lines = [hdr, sep]
    print(hdr)
    print(sep)

    for L in lengths:
        q_v2, q_v3, k_cache, v_cache, block_table, cache_seqlens, physical, offsets = build_inputs(B, L, seed=3)
        t_gather = bench_fn(
            lambda: [v2_gather(k_cache, v_cache, physical, offsets, b) for b in range(B)],
            args.iters,
        )
        t_v2 = bench_fn(lambda: v2_decode(q_v2, k_cache, v_cache, physical, offsets), args.iters)
        t_v3 = bench_fn(lambda: v3_decode(q_v3, k_cache, v_cache, block_table, cache_seqlens), args.iters)
        row = f"| {L} | {t_gather * 1e3:.1f} | {t_v2 * 1e3:.1f} | {t_v3 * 1e3:.1f} | {t_v2 / t_v3:.2f}x |"
        print(row)
        lines.append(row)

    # ---- 阶段 2：torch.profiler 对比 ----
    Lp = args.profile_len
    q_v2, q_v3, k_cache, v_cache, block_table, cache_seqlens, physical, offsets = build_inputs(B, Lp, seed=3)
    pv2 = profile_decode(lambda: v2_decode(q_v2, k_cache, v_cache, physical, offsets))
    pv3 = profile_decode(lambda: v3_decode(q_v3, k_cache, v_cache, block_table, cache_seqlens))

    print(f"\n== torch.profiler（L={Lp}, B={B}, 单层 decode attention）==")
    print(f"v2: kernel 数={pv2['n_kernels']}, 总 GPU 时间={pv2['gpu_us']:.0f} us")
    for key, cnt, t in pv2["top"]:
        print(f"    {t:9.0f} us x{cnt:4d}  {_short(key)}")
    print(f"v3: kernel 数={pv3['n_kernels']}, 总 GPU 时间={pv3['gpu_us']:.0f} us")
    for key, cnt, t in pv3["top"]:
        print(f"    {t:9.0f} us x{cnt:4d}  {_short(key)}")

    lines += [
        "",
        f"## torch.profiler（L={Lp}, B={B}, 单层 decode attention）",
        "",
        f"- v2：{pv2['n_kernels']} 个 kernel，总 GPU 时间 {pv2['gpu_us']:.0f} us",
    ]
    for key, cnt, t in pv2["top"]:
        lines.append(f"    - {t:9.0f} us x{cnt:4d}  {_short(key)}")
    lines += [f"- v3：{pv3['n_kernels']} 个 kernel，总 GPU 时间 {pv3['gpu_us']:.0f} us"]
    for key, cnt, t in pv3["top"]:
        lines.append(f"    - {t:9.0f} us x{cnt:4d}  {_short(key)}")

    out = BENCH / "out"
    out.mkdir(parents=True, exist_ok=True)
    path = out / "bench_decode_attention.txt"
    path.write_text("\n".join(lines) + "\n")
    print(f"\n→ {path}")


if __name__ == "__main__":
    main()
