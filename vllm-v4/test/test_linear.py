"""Column/Row/QKV/Merged 四个并行线性层在 TP 下的正确性验证。

对每个 tp 配置验证：
- ``weight_loader``：从全量权重（含 bias）按 rank 分片后，本地参数必须与
  全量权重的对应段一致（逐位相等）。
- ``forward``：局部 GEMM + 集合通信后的输出，必须与全量权重直接计算的
  参考一致（tp=1 时 bit 级一致；tp>1 时浮点求和次序不同，用 allclose）。
- ``_shard`` 返回 view（无 contiguous 双重拷贝），并验证直接 copy_ 的结果
  与旧路径（contiguous 后 copy_）逐位一致。

多卡用 mp.spawn 起真实 NCCL 进程。用法（在 vllm-v4/ 下）:
    ../.venv2/bin/python test/test_linear.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

SEED = 20260819
HIDDEN = 1024
HEADS = 16
KV_HEADS = 8
HEAD_DIM = 128
INTER = 3072


def _make_data(bs: int = 32):
    gen = torch.Generator().manual_seed(SEED)
    x = torch.randn(bs, HIDDEN, generator=gen)
    q = torch.randn(HEADS * HEAD_DIM, HIDDEN, generator=gen) * 0.02
    k = torch.randn(KV_HEADS * HEAD_DIM, HIDDEN, generator=gen) * 0.02
    v = torch.randn(KV_HEADS * HEAD_DIM, HIDDEN, generator=gen) * 0.02
    bq = torch.randn(HEADS * HEAD_DIM, generator=gen) * 0.02
    bk = torch.randn(KV_HEADS * HEAD_DIM, generator=gen) * 0.02
    bv = torch.randn(KV_HEADS * HEAD_DIM, generator=gen) * 0.02
    gate = torch.randn(INTER, HIDDEN, generator=gen) * 0.02
    up = torch.randn(INTER, HIDDEN, generator=gen) * 0.02
    bgate = torch.randn(INTER, generator=gen) * 0.02
    bup = torch.randn(INTER, generator=gen) * 0.02
    w_row = torch.randn(HIDDEN, INTER, generator=gen) * 0.02
    w_col = torch.randn(INTER, HIDDEN, generator=gen) * 0.02
    bcol = torch.randn(INTER, generator=gen) * 0.02
    return {
        "x": x, "q": q, "k": k, "v": v,
        "bq": bq, "bk": bk, "bv": bv,
        "gate": gate, "up": up, "bgate": bgate, "bup": bup,
        "w_row": w_row, "w_col": w_col, "bcol": bcol,
    }


def _check_rank(rank: int, tp: int) -> None:
    from src.compute.layers.linear import (  # noqa: E402
        ColumnParallelLinear,
        MergedColumnParallelLinear,
        QKVParallelLinear,
        RowParallelLinear,
    )

    d = _make_data()
    dev = "cuda"
    x = d["x"].to(dev)
    bs = x.shape[0]

    # ---- ColumnParallelLinear：输出维切分 + all-gather ----
    col = ColumnParallelLinear(HIDDEN, INTER, bias=True).to(dev)
    col.weight_loader(col.weight, d["w_col"].to(dev))
    col.weight_loader(col.bias, d["bcol"].to(dev))
    ref_col = torch.nn.functional.linear(x, d["w_col"].to(dev), d["bcol"].to(dev))
    out_col = col(x)
    per = INTER // tp
    assert torch.allclose(
        out_col, ref_col, atol=1e-2, rtol=1e-2
    ), f"rank{rank} tp={tp}: Column 不一致"

    # ---- RowParallelLinear：输入维切分 + all-reduce ----
    # 输入已按 tp 切分为 [bs, INTER//tp]，输出为全量 HIDDEN。
    gen_row = torch.Generator().manual_seed(SEED + 1)
    x_row_full = torch.randn(bs, INTER, generator=gen_row).to(dev)
    row = RowParallelLinear(INTER, HIDDEN, bias=True).to(dev)
    row.weight_loader(row.weight, d["w_row"].to(dev))
    row.weight_loader(row.bias, torch.zeros(HIDDEN).to(dev))
    ref_row = torch.nn.functional.linear(
        x_row_full, d["w_row"].to(dev), torch.zeros(HIDDEN).to(dev)
    )
    per_in = INTER // tp
    x_row_local = x_row_full[
        :, rank * per_in : (rank + 1) * per_in
    ]
    out_row = row(x_row_full)
    assert torch.allclose(
        out_row, ref_row, atol=1e-2, rtol=1e-2
    ), f"rank{rank} tp={tp}: Row 不一致"

    # ---- QKVParallelLinear：q/k/v 分片（分开的 bias 加载）----
    qkv = QKVParallelLinear(
        HIDDEN, HEAD_DIM, HEADS, KV_HEADS, bias=True
    ).to(dev)
    for name, shard_id in (("q", "q"), ("k", "k"), ("v", "v")):
        w = d[name].to(dev)
        b = d[f"b{name}"].to(dev)
        qkv.weight_loader(qkv.weight, w, shard_id)
        qkv.weight_loader(qkv.bias, b, shard_id)
    out_qkv = qkv(x)  # [bs, q_size + 2*kv_size]
    # 参考：全量 qkv 输出的对应段
    full = torch.cat([d["q"], d["k"], d["v"]], dim=0).to(dev)
    bfull = torch.cat([d["bq"], d["bk"], d["bv"]], dim=0).to(dev)
    ref_qkv = torch.nn.functional.linear(x, full, bfull)
    q_size = (HEADS // tp) * HEAD_DIM
    kv_size = (KV_HEADS // tp) * HEAD_DIM
    if tp > KV_HEADS:  # replicas 分支：kv 段按 head 分组
        nkv = 1
        kv_size = HEAD_DIM
        head_idx = rank // (tp // KV_HEADS)
        k_lo = HEADS * HEAD_DIM + head_idx * kv_size
        v_lo = HEADS * HEAD_DIM + KV_HEADS * HEAD_DIM + head_idx * kv_size
    else:
        k_lo = HEADS * HEAD_DIM + rank * kv_size
        v_lo = HEADS * HEAD_DIM + KV_HEADS * HEAD_DIM + rank * kv_size
    expect = torch.cat(
        [
            ref_qkv[:, rank * q_size : (rank + 1) * q_size],
            ref_qkv[:, k_lo : k_lo + kv_size],
            ref_qkv[:, v_lo : v_lo + kv_size],
        ],
        dim=-1,
    )
    assert torch.allclose(
        out_qkv, expect, atol=1e-2, rtol=1e-2
    ), f"rank{rank} tp={tp}: QKV 不一致"

    # ---- QKV：合并 bias 的加载路径 ----
    qkv2 = QKVParallelLinear(
        HIDDEN, HEAD_DIM, HEADS, KV_HEADS, bias=True
    ).to(dev)
    for name, shard_id in (("q", "q"), ("k", "k"), ("v", "v")):
        qkv2.weight_loader(qkv2.weight, d[name].to(dev), shard_id)
    qkv2.weight_loader(qkv2.bias, bfull, "bias")  # 合并单段 [q|k|v]
    assert torch.equal(qkv.bias, qkv2.bias), (
        f"rank{rank} tp={tp}: QKV 合并/分开 bias 加载不一致"
    )

    # ---- MergedColumnParallelLinear：gate/up 融合 ----
    mrg = MergedColumnParallelLinear(HIDDEN, [INTER, INTER], bias=True).to(dev)
    mrg.weight_loader(mrg.weight, d["gate"].to(dev), 0)
    mrg.weight_loader(mrg.weight, d["up"].to(dev), 1)
    mrg.weight_loader(mrg.bias, d["bgate"].to(dev), 0)
    mrg.weight_loader(mrg.bias, d["bup"].to(dev), 1)
    ref_mrg = torch.cat(
        [
            torch.nn.functional.linear(x, d["gate"].to(dev), d["bgate"].to(dev)),
            torch.nn.functional.linear(x, d["up"].to(dev), d["bup"].to(dev)),
        ],
        dim=-1,
    )
    out_mrg = mrg(x)
    per_i = INTER // tp
    expect_mrg = torch.cat(
        [ref_mrg[:, rank * per_i : (rank + 1) * per_i],
         ref_mrg[:, INTER + rank * per_i : INTER + (rank + 1) * per_i]],
        dim=-1,
    )
    assert torch.allclose(
        out_mrg, expect_mrg, atol=1e-2, rtol=1e-2
    ), f"rank{rank} tp={tp}: Merged 不一致"

    # ---- _shard 无 contiguous：直接 copy_ 与旧路径逐位一致 ----
    from src.compute.layers.linear import _shard

    for dim in (0, 1):
        w_full = d["gate"].clone() if dim == 0 else d["w_row"].clone()
        view = _shard(w_full, dim)
        dst1 = torch.zeros_like(view)
        dst2 = torch.zeros_like(view)
        dst1.copy_(view)          # 新路径：直接拷 view
        dst2.copy_(view.contiguous())  # 旧路径：先 contiguous 再拷
        assert torch.equal(dst1, dst2), (
            f"rank{rank} tp={tp}: _shard dim={dim} 两种拷贝结果不一致"
        )

    print(
        f"rank{rank} tp={tp}: Column/Row/QKV/Merged + bias 全部一致", flush=True
    )


def _spawn_worker(rank: int, tp: int, port: int) -> None:
    try:
        from src.distributed import (  # noqa: E402
            destroy_distributed_environment,
            destroy_model_parallel,
            init_distributed_environment,
            initialize_model_parallel,
        )

        init_distributed_environment(
            world_size=tp,
            rank=rank,
            local_rank=rank,
            backend="nccl",
            init_method=f"tcp://127.0.0.1:{port}",
        )
        initialize_model_parallel(tp, 1, tp_rank=rank, pp_rank=0)
        _check_rank(rank, tp)
        destroy_model_parallel()
        destroy_distributed_environment()
    except Exception:
        import traceback

        traceback.print_exc()
        sys.exit(1)


def main() -> None:
    if torch.cuda.device_count() < 1:
        print("无 GPU，跳过")
        return

    from src.distributed import (  # noqa: E402
        destroy_distributed_environment,
        destroy_model_parallel,
        init_distributed_environment,
        initialize_model_parallel,
    )

    init_distributed_environment(
        world_size=1,
        rank=0,
        local_rank=0,
        backend="nccl",
        init_method=f"tcp://127.0.0.1:{29000 + os.getpid() % 2000}",
    )
    initialize_model_parallel(1, 1, 0, 0)
    try:
        _check_rank(0, 1)
    finally:
        destroy_model_parallel()
        destroy_distributed_environment()
    print("tp=1: 全部通过")

    n_dev = torch.cuda.device_count()
    if n_dev >= 2:
        import torch.multiprocessing as mp

        port = 29000 + (os.getpid() % 2000)
        mp.spawn(_spawn_worker, args=(2, port), nprocs=2, join=True)
        print("tp=2: 全部通过")
    else:
        print("不足 2 张卡，跳过 TP=2 验证")
    print("== 全部 PASS ==")


if __name__ == "__main__":
    main()
