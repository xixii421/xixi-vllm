"""Functional + micro-perf: hash page-LRU vs radix tree-evict under KV pressure.

CPU-only (no model). Measures prefix hit survival after forced reclaim.

Usage::

    cd vllm-v3 && PYTHONPATH=. python -m bench.prefix_evict_compare
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.memory.manager import KVManager
from src.memory.sequence import Sequence
from src.sampling_params import SamplingParams


class _Cfg:
    num_key_value_heads = 1
    num_attention_heads = 1
    hidden_size = 16
    head_dim = 16
    num_hidden_layers = 1


def _seq(tokens: list[int], bs: int) -> Sequence:
    return Sequence.from_prompt(
        tokens, block_size=bs, sampling=SamplingParams(max_tokens=1)
    )


def _publish(kv: KVManager, tokens: list[int]) -> None:
    s = _seq(tokens, kv.block_size)
    kv.allocate(s)
    kv.deallocate(s, publish=True)


def _hit_pages(kv: KVManager, tokens: list[int]) -> int:
    return len(kv.index.match(_seq(tokens, kv.block_size), kv.blocks))


def run_backend(backend: str, *, n_branches: int = 8, pages_per: int = 2) -> dict:
    bs = 8
    # Tight pool: exactly fill with all branches, then force one reclaim.
    num_blocks = n_branches * pages_per
    kv = KVManager(
        _Cfg(),
        num_blocks=num_blocks,
        block_size=bs,
        device=torch.device("cpu"),
        dtype=torch.float16,
        prefix_backend=backend,  # type: ignore[arg-type]
    )
    branches = [
        list(range(i * 1000, i * 1000 + pages_per * bs)) for i in range(n_branches)
    ]
    hot = branches[0]
    cold = branches[-1]

    t0 = time.perf_counter()
    # Publish cold first so it is oldest unless touched; hot last + explicit touch.
    _publish(kv, cold)
    for b in branches[1:-1]:
        _publish(kv, b)
    _publish(kv, hot)
    publish_s = time.perf_counter() - t0

    # Touch hot repeatedly (radix bumps tree access_time; hash bumps page LRU).
    t0 = time.perf_counter()
    for _ in range(5):
        h = _hit_pages(kv, hot)
        assert h == pages_per, (backend, h)
        for bid in kv.index.match(_seq(hot, bs), kv.blocks):
            if bid in kv.blocks.cached_ids:
                kv.blocks.touch_cached(bid)
    touch_s = time.perf_counter() - t0

    assert kv.blocks.num_free == 0
    assert kv.blocks.num_cached == num_blocks

    # Force reclaim of ``pages_per`` pages (one whole cold leaf for radix).
    t0 = time.perf_counter()
    for _ in range(pages_per):
        kv.blocks.allocate_fresh()
    reclaim_s = time.perf_counter() - t0

    hot_hit = _hit_pages(kv, hot)
    cold_hit = _hit_pages(kv, cold)
    # How many other branches still fully intact?
    intact = sum(1 for b in branches if _hit_pages(kv, b) == pages_per)

    return {
        "backend": backend,
        "reclaim_fn": kv.blocks.reclaim_fn is not None,
        "hot_hit": hot_hit,
        "cold_hit": cold_hit,
        "intact_branches": intact,
        "n_branches": n_branches,
        "pages_per": pages_per,
        "publish_ms": publish_s * 1e3,
        "touch_ms": touch_s * 1e3,
        "reclaim_ms": reclaim_s * 1e3,
    }


def run_fragment_case() -> None:
    """Hash may punch the first page of a 2-page hot prefix; radix drops a cold leaf whole."""
    print()
    print("Fragment case (2-page hot + 2-page cold, no touch after publish order):")
    bs = 8
    hot = list(range(0, 16))
    cold = list(range(1000, 1016))

    def one(backend: str) -> tuple[int, int]:
        kv = KVManager(
            _Cfg(),
            num_blocks=4,
            block_size=bs,
            device=torch.device("cpu"),
            dtype=torch.float16,
            prefix_backend=backend,  # type: ignore[arg-type]
        )
        # Release order into cached: hot pages first (older), then cold (newer).
        _publish(kv, hot)
        _publish(kv, cold)
        # Only radix: bump hot leaf access_time via match (hash index has no tree time).
        if backend == "radix":
            assert _hit_pages(kv, hot) == 2
        # Need 1 page → hash LRU pops oldest = hot's first page (fragments hot).
        # radix tree-evicts colder leaf = cold (keeps hot intact).
        kv.blocks.allocate_fresh()
        return _hit_pages(kv, hot), _hit_pages(kv, cold)

    hh, hc = one("hash")
    rh, rc = one("radix")
    print(f"  hash  after 1 reclaim: hot_pages={hh} cold_pages={hc}  (expect hot fragmented <2)")
    print(f"  radix after 1 reclaim: hot_pages={rh} cold_pages={rc}  (expect hot=2 cold=0)")
    ok = hh < 2 and rh == 2 and rc == 0
    print(f"FRAGMENT_RESULT: {'OK' if ok else 'CHECK'}")


def main() -> None:
    print("Prefix eviction policy compare (CPU)")
    print("hash = page-pool LRU | radix = tree leaf LRU → allocate")
    print()
    rows = [run_backend("hash"), run_backend("radix")]
    print(
        f"{'backend':<8}{'reclaim':>10}{'hot':>6}{'cold':>6}"
        f"{'intact':>8}{'pub_ms':>10}{'rcl_ms':>10}"
    )
    print("-" * 60)
    for r in rows:
        print(
            f"{r['backend']:<8}"
            f"{'tree' if r['reclaim_fn'] else 'pageLRU':>10}"
            f"{r['hot_hit']:>6}{r['cold_hit']:>6}"
            f"{r['intact_branches']:>5}/{r['n_branches']}"
            f"{r['publish_ms']:>10.3f}{r['reclaim_ms']:>10.3f}"
        )
    print()
    h, x = rows[0], rows[1]
    print("Functional expect:")
    print(f"  hash:  reclaim_fn=False (page LRU)")
    print(f"  radix: reclaim_fn=True; hot full, cold gone when cold is older")
    ok = (
        not h["reclaim_fn"]
        and x["reclaim_fn"]
        and x["hot_hit"] == x["pages_per"]
        and x["cold_hit"] == 0
    )
    print(f"RESULT: {'OK' if ok else 'CHECK'} radix_hot={x['hot_hit']} radix_cold={x['cold_hit']}")
    run_fragment_case()
    print("BENCH_DONE")


if __name__ == "__main__":
    main()
