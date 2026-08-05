#!/usr/bin/env python3
"""prefill attention 对比微基准：FlashInfer BatchPrefill vs FlashAttention-2。

0.6B 几何（16 q-heads, 8 kv-heads, head_dim=128），单 seq causal prefill。
对比 M ∈ {512, 1024, 2048, 4096, 8192} 的耗时。

结论将决定 prefill 的 attention 部分是否还有优化空间。
"""
from __future__ import annotations

import sys
import torch

sys.path.insert(0, "/home/xingyongwang/code/mycuda/dzyy-vllm/vllm-v3")

from flashinfer import BatchPrefillWithPagedKVCacheWrapper
from flash_attn import flash_attn_func

N_QH, N_KVH, HD = 16, 8, 128
BLOCK = 256
D = torch.bfloat16


def bench(fn, iters=50):
    for _ in range(5):
        fn()
    torch.cuda.synchronize()
    t0, t1 = torch.cuda.Event(True), torch.cuda.Event(True)
    t0.record()
    for _ in range(iters):
        fn()
    t1.record()
    torch.cuda.synchronize()
    return t0.elapsed_time(t1) / iters


def main() -> None:
    torch.manual_seed(0)
    dev = torch.device("cuda")
    ws = torch.empty(256 << 20, dtype=torch.uint8, device=dev)

    # FlashInfer 用与 v3 相同的 paged KV wrapper（无 graph capture）
    wrapper = BatchPrefillWithPagedKVCacheWrapper(ws, "NHD", use_cuda_graph=False)

    print(f"{'M':>6s} | {'FlashInfer':>11s} | {'FA2':>9s} | {'FI/FA2':>8s}")
    print("-" * 45)
    for M in (512, 1024, 2048, 4096, 8192):
        q = torch.randn(M, N_QH, HD, dtype=D, device=dev)
        k = torch.randn(M, N_KVH, HD, dtype=D, device=dev)
        v = torch.randn(M, N_KVH, HD, dtype=D, device=dev)
        pages = (M + BLOCK - 1) // BLOCK
        indptr = torch.tensor([0, pages], dtype=torch.int32, device="cpu")
        last = torch.tensor([M % BLOCK or BLOCK], dtype=torch.int32, device="cpu")
        idx = torch.arange(pages, dtype=torch.int32, device=dev)
        qo = torch.tensor([0, M], dtype=torch.int32, device="cpu")

        def fi():
            wrapper.plan(
                qo_indptr=qo, paged_kv_indptr=indptr, paged_kv_indices=idx,
                paged_kv_last_page_len=last, num_qo_heads=N_QH, num_kv_heads=N_KVH,
                head_dim_qk=HD, head_dim_vo=HD, page_size=BLOCK, causal=True,
                sm_scale=HD**-0.5, q_data_type=D, kv_data_type=D,
            )
            kv = torch.cat([k.reshape(M, 1, N_KVH, HD),
                            v.reshape(M, 1, N_KVH, HD)], dim=1)  # [M,2,KVH,HD]
            kv_flat = kv.reshape(-1, 2 * N_KVH * HD)
            kv_cache = torch.zeros(pages * BLOCK, 2, N_KVH, HD, dtype=D, device=dev)
            kv_cache[:M] = kv
            o = wrapper.run(q, (kv_cache[:, 0], kv_cache[:, 1]))
            return o

        def fa2():
            return flash_attn_func(
                q.unsqueeze(0), k.unsqueeze(0), v.unsqueeze(0),
                causal=True,
            )[0]

        tf = bench(fi)
        tfa = bench(fa2)
        print(f"{M:>6d} | {tf:>10.3f}ms | {tfa:>8.3f}ms | {tf/tfa:>7.2f}x")


if __name__ == "__main__":
    main()
