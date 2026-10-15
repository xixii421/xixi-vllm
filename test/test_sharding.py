"""TP 分片层与 PP 划分的 CPU 正确性验证（gloo 后端，无需多卡）。

用法：python test/test_sharding.py

以 world_size=2 在单机 CPU 上验证：
  - ColumnParallelLinear / RowParallelLinear / MergedColumnParallelLinear 分片前后向等价
  - QKVParallelLinear 的 head 切分与 KV head 复制
  - VocabParallelEmbedding / ParallelLMHead 的 vocab 切分（含 padding）
  - get_pp_indices 层划分与 PPMissingLayer 透传（纯函数，无需分布式）
"""

from __future__ import annotations

import os
import sys
import traceback

import torch
import torch.multiprocessing as mp

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.compute.layers.linear import (  # noqa: E402
    ColumnParallelLinear,
    MergedColumnParallelLinear,
    QKVParallelLinear,
    RowParallelLinear,
)
from src.compute.layers.vocab_parallel import (  # noqa: E402
    ParallelLMHead,
    VocabParallelEmbedding,
)
from src.distributed import (  # noqa: E402
    destroy_distributed_environment,
    destroy_model_parallel,
    get_pp_indices,
    get_tp_group,
    init_distributed_environment,
    initialize_model_parallel,
)
from src.distributed.utils import PPMissingLayer  # noqa: E402

TP = 2
INIT_METHOD = "tcp://127.0.0.1:29509"


def _init(rank: int) -> None:
    init_distributed_environment(
        world_size=TP, rank=rank, backend="gloo", init_method=INIT_METHOD
    )
    initialize_model_parallel(TP, 1, tp_rank=rank, pp_rank=0)


def _cleanup() -> None:
    destroy_model_parallel()
    destroy_distributed_environment()


def _check(name: str, cond: bool) -> bool:
    print(f"  [{name}] {'PASS' if cond else 'FAIL'}")
    return cond


def _rank() -> int:
    return get_tp_group().group_rank


def test_column_parallel() -> bool:
    torch.manual_seed(0)
    in_size, out_size = 6, 8
    x = torch.randn(2, in_size, dtype=torch.float64)
    w = torch.randn(out_size, in_size, dtype=torch.float64)

    layer = ColumnParallelLinear(in_size, out_size, gather_output=True)
    layer.weight_loader(layer.weight, w)

    ref = torch.nn.functional.linear(x, w)
    got = layer(x)
    return _check("ColumnParallelLinear", torch.allclose(got, ref, atol=1e-10))


def test_row_parallel() -> bool:
    torch.manual_seed(1)
    in_size, out_size = 8, 3
    x = torch.randn(2, in_size, dtype=torch.float64)
    w = torch.randn(out_size, in_size, dtype=torch.float64)

    layer = RowParallelLinear(in_size, out_size, bias=False)
    layer.weight_loader(layer.weight, w)

    ref = torch.nn.functional.linear(x, w)
    # input_is_parallel=False：forward 内 scatter 全量输入。
    got = layer(x)
    return _check("RowParallelLinear", torch.allclose(got, ref, atol=1e-10))


def test_merged_column() -> bool:
    torch.manual_seed(2)
    in_size, inter = 6, 8
    x = torch.randn(2, in_size, dtype=torch.float64)
    gate = torch.randn(inter, in_size, dtype=torch.float64)
    up = torch.randn(inter, in_size, dtype=torch.float64)

    layer = MergedColumnParallelLinear(in_size, [inter, inter], bias=False)
    layer.weight_loader(layer.weight, gate, 0)
    layer.weight_loader(layer.weight, up, 1)

    gate_ref = torch.nn.functional.linear(x, gate)
    up_ref = torch.nn.functional.linear(x, up)
    per = inter // TP
    r = _rank()
    ref_local = torch.cat(
        [gate_ref[:, r * per : (r + 1) * per], up_ref[:, r * per : (r + 1) * per]],
        dim=-1,
    )
    got = layer(x)
    return _check("MergedColumnParallelLinear", torch.allclose(got, ref_local, atol=1e-10))


def test_qkv_no_replica() -> bool:
    torch.manual_seed(3)
    hidden, head_size, nq, nkv = 6, 4, 4, 4
    x = torch.randn(2, hidden, dtype=torch.float64)
    qw = torch.randn(nq * head_size, hidden, dtype=torch.float64)
    kw = torch.randn(nkv * head_size, hidden, dtype=torch.float64)
    vw = torch.randn(nkv * head_size, hidden, dtype=torch.float64)

    layer = QKVParallelLinear(hidden, head_size, nq, nkv, bias=False)
    layer.weight_loader(layer.weight, qw, "q")
    layer.weight_loader(layer.weight, kw, "k")
    layer.weight_loader(layer.weight, vw, "v")

    q_ref = torch.nn.functional.linear(x, qw)
    k_ref = torch.nn.functional.linear(x, kw)
    v_ref = torch.nn.functional.linear(x, vw)
    q_per = nq * head_size // TP
    kv_per = nkv * head_size // TP
    r = _rank()
    ref_local = torch.cat(
        [
            q_ref[:, r * q_per : (r + 1) * q_per],
            k_ref[:, r * kv_per : (r + 1) * kv_per],
            v_ref[:, r * kv_per : (r + 1) * kv_per],
        ],
        dim=-1,
    )
    got = layer(x)
    return _check("QKVParallelLinear(no-replica)", torch.allclose(got, ref_local, atol=1e-10))


def test_qkv_replica() -> bool:
    torch.manual_seed(4)
    hidden, head_size, nq, nkv = 6, 4, 4, 1
    x = torch.randn(2, hidden, dtype=torch.float64)
    qw = torch.randn(nq * head_size, hidden, dtype=torch.float64)
    kw = torch.randn(nkv * head_size, hidden, dtype=torch.float64)
    vw = torch.randn(nkv * head_size, hidden, dtype=torch.float64)

    layer = QKVParallelLinear(hidden, head_size, nq, nkv, bias=False)
    layer.weight_loader(layer.weight, qw, "q")
    layer.weight_loader(layer.weight, kw, "k")
    layer.weight_loader(layer.weight, vw, "v")

    q_ref = torch.nn.functional.linear(x, qw)
    k_ref = torch.nn.functional.linear(x, kw)
    v_ref = torch.nn.functional.linear(x, vw)
    q_per = nq * head_size // TP
    r = _rank()
    # 每 rank 完整复制同一 KV head，q 按 rank 切分
    ref_local = torch.cat(
        [q_ref[:, r * q_per : (r + 1) * q_per], k_ref, v_ref], dim=-1
    )
    got = layer(x)
    return _check("QKVParallelLinear(replica)", torch.allclose(got, ref_local, atol=1e-10))


def test_vocab_embedding() -> bool:
    torch.manual_seed(5)
    vocab, dim = 7, 4  # 非 2 整除，顺带验证 padding
    w = torch.randn(vocab, dim, dtype=torch.float64)
    x = torch.tensor([0, 1, 3, 4, 5, 6], dtype=torch.long)

    layer = VocabParallelEmbedding(vocab, dim)
    layer.weight_loader(layer.weight, w)

    ref = torch.nn.functional.embedding(x, w)
    got = layer(x)
    return _check("VocabParallelEmbedding", torch.allclose(got, ref, atol=1e-10))


def test_lm_head() -> bool:
    torch.manual_seed(6)
    hidden, vocab = 4, 7
    h = torch.randn(2, hidden, dtype=torch.float64)
    w = torch.randn(vocab, hidden, dtype=torch.float64)

    layer = ParallelLMHead(hidden, vocab, bias=False)
    layer.weight_loader(layer.weight, w)

    ref = torch.nn.functional.linear(h, w)
    got = layer(h)
    return _check("ParallelLMHead", torch.allclose(got, ref, atol=1e-10))


def test_pp_indices() -> bool:
    ok = True
    ok &= get_pp_indices(28, 0, 2) == (0, 14)
    ok &= get_pp_indices(28, 1, 2) == (14, 28)
    ok &= get_pp_indices(28, 0, 3) == (0, 9)
    ok &= get_pp_indices(28, 1, 3) == (9, 18)
    ok &= get_pp_indices(28, 2, 3) == (18, 28)
    ok &= get_pp_indices(28, 0, 4) == (0, 7)
    ok &= get_pp_indices(28, 3, 4) == (21, 28)
    return _check("get_pp_indices", ok)


def test_pp_missing_layer() -> bool:
    layer = PPMissingLayer()
    x = torch.tensor([1.0, 2.0])
    ok = torch.equal(layer(x), x)
    ok &= layer(x, "residual") is x
    return _check("PPMissingLayer", ok)


def _worker(rank: int, world_size: int, q: mp.Queue) -> None:
    torch.set_default_dtype(torch.float64)
    torch.set_grad_enabled(False)  # 推理路径无梯度，避免 gloo 集合通信对 autograd 视图报错
    _init(rank)
    try:
        tests = [
            test_column_parallel,
            test_row_parallel,
            test_merged_column,
            test_qkv_no_replica,
            test_qkv_replica,
            test_vocab_embedding,
            test_lm_head,
        ]
        ok = True
        for t in tests:
            ok = t() and ok
        q.put(ok)
    except Exception:  # noqa: BLE001
        traceback.print_exc()
        q.put(False)
    finally:
        _cleanup()


def main() -> int:
    print("== PP 划分（纯函数） ==")
    ok_pp = test_pp_indices()
    ok_pp = test_pp_missing_layer() and ok_pp

    print(f"== TP 分片（cpu/gloo, world={TP}） ==")
    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    procs = [ctx.Process(target=_worker, args=(r, TP, q)) for r in range(TP)]
    for p in procs:
        p.start()
    results = [q.get() for _ in range(TP)]
    for p in procs:
        p.join()

    ok_tp = all(results)
    print("\n" + ("ALL PASS" if (ok_pp and ok_tp) else "FAIL"))
    return 0 if (ok_pp and ok_tp) else 1


if __name__ == "__main__":
    sys.exit(main())
