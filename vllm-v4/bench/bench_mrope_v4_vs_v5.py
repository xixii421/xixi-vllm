#!/usr/bin/env python3
"""Microbench: v4 vs v5 mrope_positions.

Usage::

    cd vllm-v4 && PYTHONPATH=. python bench/bench_mrope_v4_vs_v5.py
"""

from __future__ import annotations

import copy
import importlib.util
import statistics
import sys
import time
import types
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.memory.sequence import MultiModalFeature, Sequence  # noqa: E402
from src.multimodal._mrope import mrope_positions as mrope_v4  # noqa: E402


def _load_v5():
    v5_path = ROOT.parent / "vllm-v5" / "src" / "runtime" / "vision" / "positions.py"
    features_stub = types.ModuleType("v5_runtime.input.features")

    def overlaps_window(mf, chunk_start, chunk_end):
        end = int(mf.start + mf.num_encoder_tokens)
        return max(mf.start, chunk_start) < min(end, chunk_end)

    def span_end(mf):
        return int(mf.start + mf.num_encoder_tokens)

    features_stub.overlaps_window = overlaps_window
    features_stub.span_end = span_end
    pkg_rt = types.ModuleType("v5_runtime")
    pkg_in = types.ModuleType("v5_runtime.input")
    pkg_in.features = features_stub
    pkg_rt.input = pkg_in
    sys.modules["v5_runtime"] = pkg_rt
    sys.modules["v5_runtime.input"] = pkg_in
    sys.modules["v5_runtime.input.features"] = features_stub

    spec = importlib.util.spec_from_file_location("v5_mrope_bench", v5_path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules["v5_mrope_bench"] = mod
    src = v5_path.read_text().replace(
        "from ..input.features import overlaps_window, span_end",
        "from v5_runtime.input.features import overlaps_window, span_end",
    )
    exec(compile(src, str(v5_path), "exec"), mod.__dict__)
    return mod.mrope_positions, mod.compute_mrope_positions, mod.apply_mrope_updates


mrope_v5, compute_v5, apply_v5 = _load_v5()


@dataclass
class Case:
    name: str
    seq_len: int
    img_start: int
    grid: tuple[int, int, int]  # pre-merge
    merge: int = 2
    chunk: int | None = None
    batch: int = 1
    use_tuple: bool = True  # v4 fast path; v5 ignores


def _n_vis(grid: tuple[int, int, int], merge: int) -> int:
    return (grid[0] * grid[1] * grid[2]) // (merge * merge)


def _make_seqs(case: Case) -> list[Sequence]:
    n_vis = _n_vis(case.grid, case.merge)
    seqs = []
    for b in range(case.batch):
        feat = MultiModalFeature(
            modality="image",
            data=torch.zeros(1),
            start=case.img_start,
            end=case.img_start + n_vis,
            num_encoder_tokens=n_vis,
            uid=f"img-{b}",
            grid_thw=torch.tensor([list(case.grid)], dtype=torch.int64),
            grid_thw_tuple=case.grid if case.use_tuple else None,
        )
        if not case.use_tuple:
            feat.grid_thw_tuple = None
        seq = Sequence(
            token_ids=list(range(case.seq_len)),
            block_size=max(case.seq_len, 8),
            mm_features=[feat],
        )
        seq.seq_id = b + 1
        seqs.append(seq)
    return seqs


def _reset(seqs: list[Sequence], cached: int, sched: int) -> None:
    for s in seqs:
        s.num_cached_tokens = cached
        s.num_scheduled_tokens = sched
        s.mrope_next = None
        s.rope_delta = 0
        for mf in s.mm_features:
            mf.mrope_origin = None


def _run_impl(impl: str, case: Case, seqs: list[Sequence], device: torch.device) -> torch.Tensor:
    chunk = case.chunk or case.seq_len
    cached = 0
    last = None
    while cached < case.seq_len:
        m = min(chunk, case.seq_len - cached)
        # For batch, each seq contributes m tokens → total batch*m
        _reset(seqs, cached, m)
        # After first chunk, preserve state across chunks like real engine:
        if cached > 0:
            # restore from previous iteration's mutated state — re-run needs care.
            pass
        ids = torch.arange(m * case.batch)
        if impl == "v4":
            last = mrope_v4(ids, seqs, case.merge, device)
        elif impl == "v5":
            last = mrope_v5(ids, seqs, case.merge, device)
        elif impl == "v5_compute":
            last, updates = compute_v5(ids, seqs, case.merge, device)
            apply_v5(seqs, updates)
        else:
            raise ValueError(impl)
        cached += m
    return last  # type: ignore[return-value]


def _run_chunked(impl: str, case: Case, device: torch.device) -> torch.Tensor:
    """Proper chunked run that keeps mrope state between chunks."""
    seqs = _make_seqs(case)
    chunk = case.chunk or case.seq_len
    cached = 0
    last = None
    while cached < case.seq_len:
        m = min(chunk, case.seq_len - cached)
        for s in seqs:
            s.num_cached_tokens = cached
            s.num_scheduled_tokens = m
        ids = torch.arange(m * case.batch)
        if impl == "v4":
            last = mrope_v4(ids, seqs, case.merge, device)
        elif impl == "v5":
            last = mrope_v5(ids, seqs, case.merge, device)
        else:
            last, updates = compute_v5(ids, seqs, case.merge, device)
            apply_v5(seqs, updates)
        cached += m
    return last  # type: ignore[return-value]


def bench_once(impl: str, case: Case, device: torch.device, repeats: int, warmup: int) -> list[float]:
    # correctness warmup + timing on fresh deep copies each repeat
    for _ in range(warmup):
        _run_chunked(impl, case, device)

    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        _run_chunked(impl, case, device)
        times.append((time.perf_counter() - t0) * 1e3)
    return times


def main() -> None:
    device = torch.device("cpu")
    # Realistic grids: small unit, medium, cat.jpg-like (~15808 vision tokens)
    cases = [
        Case("tiny_2x2", seq_len=16, img_start=2, grid=(1, 4, 4), chunk=None),
        Case("med_16x16", seq_len=3 + 256 + 8, img_start=3, grid=(1, 32, 32), chunk=None),
        Case("large_catlike", seq_len=8 + 15808 + 8, img_start=8, grid=(1, 304, 208), chunk=None),
        Case("large_chunk256", seq_len=8 + 15808 + 8, img_start=8, grid=(1, 304, 208), chunk=256),
        Case("batch8_med", seq_len=3 + 256 + 8, img_start=3, grid=(1, 32, 32), batch=8),
        Case("v4_no_tuple_med", seq_len=3 + 256 + 8, img_start=3, grid=(1, 32, 32), use_tuple=False),
    ]

    # Correctness: v4 vs v5 on tiny
    a = _run_chunked("v4", cases[0], device).numpy()
    b = _run_chunked("v5", cases[0], device).numpy()
    assert np.array_equal(a, b), "v4/v5 positions diverge on tiny — abort bench"

    print(f"{'case':<18} {'impl':<12} {'median_ms':>10} {'mean_ms':>10} {'min_ms':>10} {'speedup':>10}")
    print("-" * 72)

    for case in cases:
        # Large cases: fewer repeats
        large = case.seq_len > 5000
        repeats = 15 if not large else 8
        warmup = 3 if not large else 2

        # Force use_tuple for named cases except the no_tuple one
        results = {}
        for impl in ("v4", "v5"):
            c = copy.copy(case)
            times = bench_once(impl, c, device, repeats=repeats, warmup=warmup)
            results[impl] = times
            med = statistics.median(times)
            print(
                f"{case.name:<18} {impl:<12} {med:10.3f} {statistics.mean(times):10.3f} "
                f"{min(times):10.3f} {'':>10}"
            )

        v4_med = statistics.median(results["v4"])
        v5_med = statistics.median(results["v5"])
        ratio = v5_med / v4_med if v4_med > 0 else float("inf")
        winner = "v4 faster" if ratio > 1 else ("v5 faster" if ratio < 1 else "tie")
        print(
            f"{'':18} {'→':<12} {'':>10} {'':>10} {'':>10} "
            f"{ratio:9.2f}x {winner}"
        )
        print()

    print("Notes:")
    print("  - v4: sliced grid coords + optional grid_thw_tuple (no tensor.tolist)")
    print("  - v5: full np.indices(t,gh,gw) + grid_thw tensor .tolist() every call")
    print("  - Both include state mutation (v5 via apply_mrope_updates)")


if __name__ == "__main__":
    main()
