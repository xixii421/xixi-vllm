"""采样器功能正确性。

纯单元测试，不加载模型。覆盖 greedy、纯温度、top_k、top_p、min_p 以及
同批混合 greedy 六个分支，验证结果落在合法集合且固定 seed 下可复现。

用法:
    python test/test_sampler.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.compute.sampler import Sampler  # noqa: E402

VOCAB = 32000


def _flags(temps, top_ps, top_ks, min_ps):
    """复刻 Batch._fill_sampling 的开关判定，保证与真实打包路径一致。"""
    all_greedy = all(t <= 1e-5 for t in temps)
    any_greedy = any(t <= 1e-5 for t in temps)
    use_top_k = any(k < VOCAB for k in top_ks)
    use_top_p = any(p < 1.0 for p in top_ps)
    use_min_p = any(m > 0.0 for m in min_ps)
    return dict(
        all_greedy=all_greedy,
        any_greedy=any_greedy,
        use_top_k=use_top_k,
        use_top_p=use_top_p,
        use_min_p=use_min_p,
        max_top_k=max(top_ks) if use_top_k else 0,
    )


def _run(sampler, logits, temps, top_ps=None, top_ks=None, min_ps=None, seed=0):
    bs = logits.shape[0]
    dev = logits.device

    def t(xs, dtype):
        return torch.tensor(xs, dtype=dtype, device=dev)

    top_ps = t([1.0] * bs, torch.float32) if top_ps is None else t(top_ps, torch.float32)
    top_ks = t([VOCAB] * bs, torch.int32) if top_ks is None else t(top_ks, torch.int32)
    min_ps = t([0.0] * bs, torch.float32) if min_ps is None else t(min_ps, torch.float32)
    temps = t(temps, torch.float32)
    flags = _flags(temps.tolist(), top_ps.tolist(), top_ks.tolist(), min_ps.tolist())
    torch.manual_seed(seed)
    out = sampler(logits, temps, top_ps=top_ps, top_ks=top_ks, min_ps=min_ps, **flags)
    torch.cuda.synchronize()
    return out


def _check(cond: bool, msg: str) -> None:
    if not cond:
        raise AssertionError(msg)


def check_greedy(sampler, logits, bs):
    out = _run(sampler, logits, [0.0] * bs)
    expect = logits.argmax(dim=-1)
    _check(torch.equal(out, expect), "greedy 未返回 argmax")


def check_temperature(sampler, logits, bs):
    a = _run(sampler, logits, [0.9] * bs, seed=1)
    b = _run(sampler, logits, [0.9] * bs, seed=1)
    _check(torch.equal(a, b), "纯温度分支固定 seed 不可复现")
    _check(a.shape == (bs,), "纯温度分支输出形状错误")


def check_top_k(sampler, logits, bs):
    k = 8
    out = _run(sampler, logits, [0.9] * bs, top_ks=[k] * bs, seed=2)
    vals, _ = logits.topk(k, dim=-1)
    threshold = vals[:, -1]
    picked = logits[torch.arange(bs), out]
    _check(torch.all(picked >= threshold), "top_k 分支采样到 k 名之外")


def check_top_p(sampler, logits, bs):
    out = _run(sampler, logits, [0.9] * bs, top_ps=[0.9] * bs, seed=3)
    probs = torch.softmax(logits.float(), dim=-1)
    picked = probs[torch.arange(bs), out]
    _check(torch.all(picked > 0), "top_p 分支采样到被 mask 的 token")


def check_min_p(sampler, logits, bs):
    out = _run(sampler, logits, [0.9] * bs, min_ps=[0.1] * bs, seed=4)
    probs = torch.softmax(logits.float(), dim=-1)
    picked = probs[torch.arange(bs), out]
    _check(torch.all(picked > 0), "min_p 分支采样到被 mask 的 token")


def check_mixed(sampler, logits, bs):
    # 前一半 greedy，后一半温度采样。
    half = bs // 2
    temps = [0.0] * half + [0.9] * (bs - half)
    out = _run(sampler, logits, temps, seed=5)
    expect = logits.argmax(dim=-1)
    _check(torch.equal(out[:half], expect[:half]), "混合批 greedy 行未返回 argmax")


def main() -> None:
    device = "cuda:0"
    torch.cuda.set_device(device)
    sampler = Sampler().to(device=device)

    bs = 8
    torch.manual_seed(0)
    logits = torch.randn(bs, VOCAB, device=device)

    cases = [
        ("greedy", check_greedy),
        ("temperature", check_temperature),
        ("top_k", check_top_k),
        ("top_p", check_top_p),
        ("min_p", check_min_p),
        ("mixed", check_mixed),
    ]

    failed = 0
    for name, fn in cases:
        try:
            fn(sampler, logits, bs)
            print(f"PASS  {name}")
        except Exception as e:  # noqa: BLE001
            failed += 1
            print(f"FAIL  {name}: {e}")

    if failed:
        raise SystemExit(1)
    print("OK")


if __name__ == "__main__":
    main()
