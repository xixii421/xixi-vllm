"""KV Cache 写入微基准：index_copy_ vs Triton store_kvcache。

回答「v3 的 Triton store_kvcache 相对 v2 的 index_copy_ 到底快多少」，并据此微调。

对比项（各自忠实还原 v2 / v3 的真实调用）：
1. ``index_copy_``    —— v2 ``PagedStore.write``：k/v 各一次，slot_mapping 为 int64。
2. ``store_kvcache``  —— 当前 src/data/kv_ops.py：1 token / program，slot_mapping 为 int32。
3. ``store_kvcache`` + num_warps 8 —— 对当前 kernel 的唯一调参点。
4. ``store_tiled``    —— 调优尝试：BLOCK_N 个 token / program + 2D 访存（结果见下）。

结论（实测，4090 D，d=1024 bf16）：

- 小 n（decode，1~64）：发射开销主导。Triton ~15us，index_copy_ ~25us（非向量化 scatter 的固定开销），**Triton 约 1.6x**。
- 大 n（prefill，>=16384）：HBM 带宽主导，两边都贴近 ~900 GB/s（4090 D 峰值 ~1008 GB/s），**Triton 约 1.1x**。
- 调参无效：num_warps=8 与 BLOCK_N 分块（tiled）都比默认（1 token/program, 4 warps）更慢，当前实现已接近最优。

即：这是一次零计算、纯访存的 scatter，物理上限是 HBM 带宽；Triton 的收益来自
① 单 kernel 融合 k+v、② 128-bit 向量化访存、③ 更低的固定开销，而非「减少冗余索引」。

口径：先正确性对拍（含 -1 padding），再用 ``torch.cuda.Event`` warmup 后取均值。
有效带宽 = 4 * n * d * itemsize / 时间（读 key+value、写 k_cache+v_cache）。

用法:
    python bench/bench_kvcache_write.py
    python bench/bench_kvcache_write.py --tokens 1,8,64,1024,8192 --d 1024 --iters 200
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import triton
import triton.language as tl

BENCH = Path(__file__).resolve().parent
V3_ROOT = BENCH.parent
if str(V3_ROOT) not in sys.path:
    sys.path.insert(0, str(V3_ROOT))

from src.data.kv_ops import _store_kvcache_kernel, store_kvcache  # noqa: E402


# --------------------------------------------------------------------------- #
# 参考实现：v2 的 index_copy_ 写入（slot_mapping 为 int64）
# --------------------------------------------------------------------------- #
def index_copy_write(
    key: torch.Tensor,
    value: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    slots_i64: torch.Tensor,
) -> None:
    n, h, d = key.shape
    k_cache.reshape(-1, h * d).index_copy_(0, slots_i64, key.reshape(n, h * d).detach())
    v_cache.reshape(-1, h * d).index_copy_(0, slots_i64, value.reshape(n, h * d).detach())


# --------------------------------------------------------------------------- #
# Triton 变体：当前 kernel 加 num_warps 参数
# --------------------------------------------------------------------------- #
def store_kvcache_warps(
    key: torch.Tensor,
    value: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
    *,
    num_warps: int,
) -> None:
    if not key.is_contiguous():
        key = key.contiguous()
    if not value.is_contiguous():
        value = value.contiguous()
    n, num_heads, head_dim = key.shape
    d = num_heads * head_dim
    block = triton.next_power_of_2(d)
    _store_kvcache_kernel[(n,)](
        key,
        key.stride(0),
        value,
        value.stride(0),
        k_cache.reshape(-1, num_heads, head_dim),
        v_cache.reshape(-1, num_heads, head_dim),
        slot_mapping,
        d,
        block,
        num_warps=num_warps,
    )


# --------------------------------------------------------------------------- #
# Triton 调优变体：BLOCK_N 个 token / program + 2D 访存
# --------------------------------------------------------------------------- #
@triton.jit
def _store_kvcache_tiled(
    key_ptr,
    key_stride,
    value_ptr,
    value_stride,
    k_cache_ptr,
    v_cache_ptr,
    slot_mapping_ptr,
    n_tokens,
    D: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid = tl.program_id(0)
    rows = pid * BLOCK_N + tl.arange(0, BLOCK_N)
    row_mask = rows < n_tokens
    slots = tl.load(slot_mapping_ptr + rows, mask=row_mask, other=-1)
    valid = row_mask & (slots >= 0)

    cols = tl.arange(0, BLOCK_D)
    col_mask = cols < D
    load_mask = valid[:, None] & col_mask[None, :]

    key = tl.load(
        key_ptr + rows[:, None] * key_stride + cols[None, :],
        mask=load_mask,
        other=0.0,
    )
    value = tl.load(
        value_ptr + rows[:, None] * value_stride + cols[None, :],
        mask=load_mask,
        other=0.0,
    )

    cache_off = slots[:, None] * D + cols[None, :]
    tl.store(k_cache_ptr + cache_off, key, mask=load_mask)
    tl.store(v_cache_ptr + cache_off, value, mask=load_mask)


def store_tiled(
    key: torch.Tensor,
    value: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
    *,
    block_n: int,
) -> None:
    if not key.is_contiguous():
        key = key.contiguous()
    if not value.is_contiguous():
        value = value.contiguous()
    n, num_heads, head_dim = key.shape
    d = num_heads * head_dim
    block_d = triton.next_power_of_2(d)
    grid = (triton.cdiv(n, block_n),)
    _store_kvcache_tiled[grid](
        key,
        key.stride(0),
        value,
        value.stride(0),
        k_cache.reshape(-1, num_heads, head_dim),
        v_cache.reshape(-1, num_heads, head_dim),
        slot_mapping,
        n,
        d,
        block_d,
        block_n,
    )


# --------------------------------------------------------------------------- #
# 工具：输入构造 / 正确性 / 计时
# --------------------------------------------------------------------------- #
def make_inputs(n: int, d: int, *, blocks: int = 2048, seed: int = 0, pad: float = 0.0):
    """构造 key/value/cache/slot_mapping。pad > 0 时随机把部分 slot 置 -1。"""
    g = torch.Generator(device="cuda").manual_seed(seed)
    head_dim = 128
    num_heads = d // head_dim
    assert d % head_dim == 0, "d 需为 head_dim=128 的整数倍"

    key = torch.randn(n, num_heads, head_dim, dtype=torch.bfloat16, device="cuda", generator=g)
    value = torch.randn(n, num_heads, head_dim, dtype=torch.bfloat16, device="cuda", generator=g)

    num_slots = blocks * 128
    slots_i32 = torch.randint(0, num_slots, (n,), dtype=torch.int32, device="cuda", generator=g)
    if pad > 0:
        mask = torch.rand(n, device="cuda", generator=g) < pad
        slots_i32 = torch.where(mask, torch.full_like(slots_i32, -1), slots_i32)

    k_cache = torch.zeros(num_slots, num_heads, head_dim, dtype=torch.bfloat16, device="cuda")
    v_cache = torch.zeros(num_slots, num_heads, head_dim, dtype=torch.bfloat16, device="cuda")
    return key, value, k_cache, v_cache, slots_i32


def _ref_with_pad(key, value, k_cache, v_cache, slots):
    """正确处理 -1 的参考实现（仅用于小 n 对拍）。"""
    n, h, d = key.shape
    dd = h * d
    for i in range(n):
        s = int(slots[i])
        if s < 0:
            continue
        k_cache.view(-1, dd)[s].copy_(key.view(n, dd)[i])
        v_cache.view(-1, dd)[s].copy_(value.view(n, dd)[i])


def check_correctness() -> bool:
    ok = True
    for n, pad in [(37, 0.0), (100, 0.3)]:
        key, value, k_cache, v_cache, slots_i32 = make_inputs(n, 1024, seed=1, pad=pad)

        ref_k, ref_v = k_cache.clone(), v_cache.clone()
        if pad > 0:
            _ref_with_pad(key, value, ref_k, ref_v, slots_i32)
        else:
            index_copy_write(key, value, ref_k, ref_v, slots_i32.to(torch.long))

        for name, fn in [
            ("store_kvcache", lambda kc, vc: store_kvcache(key, value, kc, vc, slots_i32)),
            ("store_kvcache(w8)", lambda kc, vc: store_kvcache_warps(key, value, kc, vc, slots_i32, num_warps=8)),
            ("store_tiled(16)", lambda kc, vc: store_tiled(key, value, kc, vc, slots_i32, block_n=16)),
        ]:
            a_k, a_v = k_cache.clone(), v_cache.clone()
            fn(a_k, a_v)
            good = torch.equal(a_k, ref_k) and torch.equal(a_v, ref_v)
            if not good:
                diff = (a_k != ref_k).sum().item() + (a_v != ref_v).sum().item()
                print(f"  [FAIL] {name} (n={n}, pad={pad}) mismatches={diff}")
                ok = False
            else:
                print(f"  [ok]   {name} (n={n}, pad={pad})")
    return ok


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


def _bytes(n: int, d: int) -> float:
    return 4.0 * n * d * 2  # bf16：读 key+value，写 k_cache+v_cache


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--tokens", default="1,8,32,64,256,1024,4096,8192")
    p.add_argument("--d", default="1024")
    p.add_argument("--iters", type=int, default=200)
    args = p.parse_args()
    torch.cuda.set_device(args.device)

    tokens = [int(x) for x in args.tokens.split(",")]
    ds = [int(x) for x in args.d.split(",")]

    print("== KV Cache 写入正确性对拍 ==")
    if not check_correctness():
        sys.exit("正确性校验失败，终止。")
    print()

    hdr = (
        "| n | d | index_copy(us) | triton(us) | triton_w8(us) | "
        "tiled16(us) | triton 加速比 | GB/s(triton) |"
    )
    sep = "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"
    lines = [hdr, sep]
    print(hdr)
    print(sep)

    for d in ds:
        for n in tokens:
            key, value, k_cache, v_cache, slots_i32 = make_inputs(n, d, seed=2)
            slots_i64 = slots_i32.to(torch.long)

            # 全部 view / 类型转换在计时前完成，计时只含真实写操作
            t_idx = bench_fn(lambda: index_copy_write(key, value, k_cache, v_cache, slots_i64), args.iters)
            t_cur = bench_fn(lambda: store_kvcache(key, value, k_cache, v_cache, slots_i32), args.iters)
            t_w8 = bench_fn(lambda: store_kvcache_warps(key, value, k_cache, v_cache, slots_i32, num_warps=8), args.iters)
            t_t16 = bench_fn(lambda: store_tiled(key, value, k_cache, v_cache, slots_i32, block_n=16), args.iters)

            spd = t_idx / t_cur
            gbps = _bytes(n, d) / (t_cur * 1e-3) / 1e9

            row = (
                f"| {n} | {d} | {t_idx * 1e3:.2f} | {t_cur * 1e3:.2f} | "
                f"{t_w8 * 1e3:.2f} | {t_t16 * 1e3:.2f} | {spd:.2f}x | {gbps:.0f} |"
            )
            print(row)
            lines.append(row)

    out = BENCH / "out"
    out.mkdir(parents=True, exist_ok=True)
    path = out / "bench_kvcache_write.txt"
    path.write_text("\n".join(lines) + "\n")
    print(f"\n→ {path}")


if __name__ == "__main__":
    main()
