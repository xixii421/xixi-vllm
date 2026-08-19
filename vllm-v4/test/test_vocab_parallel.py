"""VocabParallelEmbedding / ParallelLMHead 单卡 & 多卡（TP）正确性验证。

对每个 tp 配置，验证：
- ``VocabParallelEmbedding.forward``：局部查表 + all-reduce 求和 的结果
  必须与"全量权重直接查表"逐位相等（torch.equal）。
- ``ParallelLMHead.forward``：局部 GEMM + all-gather 拼接 的结果
  必须与"全量权重直接线性层"一致（torch.allclose）。

覆盖 vocab 能被/不能被 tp 整除两种情形（padding 分支）。
多卡用 mp.spawn 起真实进程 + NCCL 集合通信，与推理路径完全一致。

用法（在 vllm-v4/ 下）:
    ../.venv2/bin/python test/test_vocab_parallel.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

SEED = 20260818


def _make_data(vocab: int, dim: int, bs: int):
    """用独立生成器构造数据，保证跨进程逐位一致（不依赖全局 RNG）。"""
    gen = torch.Generator().manual_seed(SEED)
    full = torch.randn(vocab, dim, generator=gen)
    x = torch.randint(0, vocab, (bs,), generator=gen)
    h = torch.randn(bs, dim, generator=gen)
    return full, x, h


def _check_rank(rank: int, tp: int, vocab: int, dim: int, bs: int) -> None:
    from src.compute.layers.vocab_parallel import (  # noqa: E402
        ParallelLMHead,
        VocabParallelEmbedding,
    )

    full, x, h = _make_data(vocab, dim, bs)

    # --- 嵌入：局部查表 + all-reduce 求和 ---
    emb = VocabParallelEmbedding(vocab, dim).cuda()
    emb.weight_loader(emb.weight, full)
    out_emb = emb(x.cuda()).cpu()
    ref_emb = full[x]  # 全量权重直接查表
    assert torch.equal(out_emb, ref_emb), (
        f"rank{rank} tp={tp} vocab={vocab}: embedding 逐位不一致 "
        f"max_err={(out_emb - ref_emb).abs().max().item():.2e}"
    )

    # --- LM head：局部 GEMM + all-gather 拼接 ---
    head = ParallelLMHead(dim, vocab).cuda()
    head.weight_loader(head.weight, full)
    out_logits = head(h.cuda()).cpu()
    ref_logits = h @ full.t()  # 全量权重直接线性层
    max_err = (out_logits - ref_logits).abs().max().item()
    assert torch.allclose(out_logits, ref_logits, atol=1e-4, rtol=1e-4), (
        f"rank{rank} tp={tp} vocab={vocab}: lm_head 不一致 max_err={max_err:.2e}"
    )

    # --- argmax 贪心快路径：必须与全量 logits 的 argmax 逐位一致 ---
    fast_id = head.argmax(h.cuda()).cpu()
    ref_id = head(h.cuda()).argmax(-1).cpu()
    assert torch.equal(fast_id, ref_id), (
        f"rank{rank} tp={tp} vocab={vocab}: argmax 快路径不一致"
    )

    print(
        f"rank{rank} tp={tp} vocab={vocab}: embedding 逐位相等, "
        f"lm_head max_err={max_err:.2e}, argmax 快路径一致",
        flush=True,
    )


def _spawn_worker(
    rank: int,
    tp: int,
    configs: list[tuple[int, int, int]],
    port: int,
) -> None:
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
        initialize_model_parallel(tp, 1, tp_rank=rank % tp, pp_rank=rank // tp)
        for vocab, dim, bs in configs:
            _check_rank(rank, tp, vocab, dim, bs)
        destroy_model_parallel()
        destroy_distributed_environment()
    except Exception:
        import traceback

        traceback.print_exc()
        sys.exit(1)


def run_tp(tp: int, configs: list[tuple[int, int, int]]) -> None:
    """单卡（tp=1）直接在本进程跑；多卡用 spawn 起真实 NCCL 进程。"""
    if tp == 1:
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
            for vocab, dim, bs in configs:
                _check_rank(0, 1, vocab, dim, bs)
        finally:
            destroy_model_parallel()
            destroy_distributed_environment()
        print(f"tp=1: 全部通过")
        return

    import torch.multiprocessing as mp

    world = tp
    port = 29000 + (os.getpid() % 2000)

    mp.spawn(_spawn_worker, args=(tp, configs, port), nprocs=world, join=True)
    print(f"tp={tp}: 全部通过")


def main() -> None:
    # 每种配置都要覆盖 vocab 能被/不能被 tp 整除的 padding 分支
    configs = [
        (151936, 128, 256),  # Qwen3 真实 vocab，能被 2 整除
        (151937, 128, 256),  # 不能被 2 整除 → 触发 padding 补零分支
        (10, 32, 64),        # 小 vocab 边界
    ]
    run_tp(1, configs)
    n_dev = torch.cuda.device_count()
    print(f"检测到 {n_dev} 张卡，运行 TP 多卡验证...")
    if n_dev >= 2:
        run_tp(2, configs)
    else:
        print("不足 2 张卡，跳过 TP=2 验证")
    print("== 全部 PASS ==")


if __name__ == "__main__":
    main()
