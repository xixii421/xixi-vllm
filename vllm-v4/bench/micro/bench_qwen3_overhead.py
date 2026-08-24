"""Qwen3 解码层热路径开销剖析（单卡，模拟 Qwen3-0.6B）。

对 decode 与 prefill 两种场景，逐算子计时（CUDA event），量化：
  1. store_kvcache（独立 Triton kernel）相对 attention 的开销
  2. q_norm/k_norm + rope 两个小算子的开销
  3. rotary/attention 里的 .contiguous() 是否真的零拷贝（无则无感）

用法（在 vllm-v4/ 下）:
    ../.venv2/bin/python bench/bench_qwen3_overhead.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.compute.layers.attention import attention_context, AttnMetadata  # noqa: E402
from src.compute.layers.ops import (  # noqa: E402
    fused_add_rmsnorm,
    rmsnorm,
    rotary_embedding,
    silu_and_mul,
)
from src.compute.kv_ops import store_kvcache  # noqa: E402
from flash_attn import flash_attn_with_kvcache  # noqa: E402

H = 1024          # hidden
NQ = 16           # q heads
NKV = 8           # kv heads
HD = 128          # head_dim
INTER = 3072      # mlp inter
LAYERS = 28
QKV_OUT = (NQ + 2 * NKV) * HD


def timed(fn, *args, warmup=20, iters=200):
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
    return s.elapsed_time(e) / iters


def bench_decode(bs: int, dtype=torch.bfloat16):
    torch.manual_seed(0)
    dev = "cuda"
    x = torch.randn(bs, H, device=dev, dtype=dtype)
    w_qkv = torch.randn(QKV_OUT, H, device=dev, dtype=dtype) * 0.02
    w_o = torch.randn(H, NQ * HD, device=dev, dtype=dtype) * 0.02
    w_gu = torch.randn(2 * INTER, H, device=dev, dtype=dtype) * 0.02
    w_down = torch.randn(H, INTER, device=dev, dtype=dtype) * 0.02
    w_norm = torch.ones(H, device=dev)
    w_qk = torch.ones(HD, device=dev, dtype=dtype)
    cos_sin = torch.randn(32768, HD, device=dev) * 0.001

    # paged kv cache：布局 [2, pages, page_size(256), NKV, HD]，与 engine 的 pool 一致
    BSZ = 256
    PAGES = 16
    kv_pool = torch.randn(2, PAGES, BSZ, NKV, HD, device=dev, dtype=dtype) * 0.01
    slot = torch.arange(bs, device=dev, dtype=torch.long)
    block_table = torch.arange(PAGES, device=dev, dtype=torch.int32).repeat(bs, 1)
    cache_seqlens = (torch.arange(bs, device=dev) % (BSZ - 16) + 16).to(torch.int32)
    positions = cache_seqlens.clone().long()

    def layer():
        qkv = torch.nn.functional.linear(x, w_qkv)
        q, k, v = qkv.split([NQ * HD, NKV * HD, NKV * HD], dim=-1)
        q = q.view(bs, NQ, HD)
        k = k.view(bs, NKV, HD)
        v = v.view(bs, NKV, HD)
        q = rmsnorm(q, w_qk, 1e-6)
        k = rmsnorm(k, w_qk, 1e-6)
        q, k = rotary_embedding(q, k, positions, cos_sin, HD)
        store_kvcache(
            k, v,
            kv_pool[0].reshape(-1, NKV, HD),
            kv_pool[1].reshape(-1, NKV, HD),
            slot,
        )
        o = flash_attn_with_kvcache(
            q.unsqueeze(1), kv_pool[0], kv_pool[1],
            cache_seqlens=cache_seqlens, block_table=block_table, causal=True,
        ).view(bs, -1)
        o = torch.nn.functional.linear(o, w_o)
        up = silu_and_mul(torch.nn.functional.linear(x, w_gu))
        return torch.nn.functional.linear(up, w_down) + o

    t_layer = timed(layer)
    # 纯 GPU 时间：每次迭代同步，测单次 layer 的 kernel 串行时间
    for _ in range(10):
        layer()
    torch.cuda.synchronize()
    s = torch.cuda.Event(enable_timing=True)
    e = torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(30):
        layer()
        torch.cuda.synchronize()
    e.record()
    torch.cuda.synchronize()
    t_layer_sync = (s.elapsed_time(e) / 30) * 1000  # us
    t_store = timed(store_kvcache,
                    torch.randn(bs, NKV, HD, device=dev, dtype=dtype),
                    torch.randn(bs, NKV, HD, device=dev, dtype=dtype),
                    kv_pool[0].reshape(-1, NKV, HD),
                    kv_pool[1].reshape(-1, NKV, HD), slot)
    t_qkv = timed(lambda: torch.nn.functional.linear(x, w_qkv))
    xo = torch.randn(bs, NQ * HD, device=dev, dtype=dtype)
    t_o = timed(lambda: torch.nn.functional.linear(xo, w_o))
    t_gu = timed(lambda: torch.nn.functional.linear(x, w_gu))
    t_attn = timed(lambda: flash_attn_with_kvcache(
        torch.randn(bs, 1, NQ, HD, device=dev, dtype=dtype), kv_pool[0], kv_pool[1],
        cache_seqlens=cache_seqlens, block_table=block_table, causal=True))
    t_silu = timed(lambda: silu_and_mul(x))
    print(f"[decode bs={bs:>3}] layer(pipelined) {t_layer * 1000:7.1f} us | "
          f"layer(sync GPU) {t_layer_sync:7.1f} us | "
          f"store {t_store * 1000:5.1f} | attn {t_attn * 1000:6.1f} | "
          f"qkv {t_qkv * 1000:6.1f} | o {t_o * 1000:6.1f} | "
          f"gu {t_gu * 1000:6.1f} | silu {t_silu * 1000:5.1f}")


def bench_prefill(n: int, dtype=torch.bfloat16):
    torch.manual_seed(0)
    dev = "cuda"
    x = torch.randn(n, H, device=dev, dtype=dtype)
    w_qkv = torch.randn(QKV_OUT, H, device=dev, dtype=dtype) * 0.02
    w_qk = torch.ones(HD, device=dev, dtype=dtype)
    cos_sin = torch.randn(32768, HD, device=dev) * 0.001
    positions = torch.arange(n, device=dev, dtype=torch.long)
    kv_pool = torch.randn(2, n + 16, NKV, HD, device=dev, dtype=dtype) * 0.01
    slot = torch.arange(n, device=dev, dtype=torch.long)
    cu = torch.tensor([0, n], device=dev, dtype=torch.int32)

    def qkv_rope_store():
        qkv = torch.nn.functional.linear(x, w_qkv)
        q, k, v = qkv.split([NQ * HD, NKV * HD, NKV * HD], dim=-1)
        q = rmsnorm(q.view(n, NQ, HD), w_qk, 1e-6)
        k = rmsnorm(k.view(n, NKV, HD), w_qk, 1e-6)
        q, k = rotary_embedding(q, k, positions, cos_sin, HD)
        store_kvcache(k, v, kv_pool[0].reshape(-1, NKV, HD),
                      kv_pool[1].reshape(-1, NKV, HD), slot)
        return q, k, v

    t_qkv = timed(qkv_rope_store)
    t_linear = timed(lambda: torch.nn.functional.linear(x, w_qkv))
    qv = torch.randn(n, NQ, HD, device=dev, dtype=dtype)
    kv_ = torch.randn(n, NKV, HD, device=dev, dtype=dtype)
    t_norm = timed(lambda: rmsnorm(qv, w_qk, 1e-6))
    t_rope = timed(lambda: rotary_embedding(qv, kv_, positions, cos_sin, HD))
    t_store = timed(store_kvcache, kv_, kv_,
                    kv_pool[0].reshape(-1, NKV, HD), kv_pool[1].reshape(-1, NKV, HD), slot)
    print(f"[prefill n={n:>5}] qkv总 {t_qkv * 1000:7.1f} us = "
          f"linear {t_linear * 1000:6.1f} + rmsnorm {t_norm * 1000:6.1f}x2 + "
          f"rope {t_rope * 1000:6.1f} + store {t_store * 1000:6.1f}")


def main() -> None:
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    for bs in (1, 8, 16, 64):
        bench_decode(bs)
    for n in (256, 2048, 8192):
        bench_prefill(n)


if __name__ == "__main__":
    main()
