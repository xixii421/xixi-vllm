#!/usr/bin/env python3
"""Compare mrope.py vs mrope_plus.py: correctness, latency, GPU memory.

  source ../.venv/bin/activate && source ../.env.cuda.sh
  export CUDA_VISIBLE_DEVICES=1
  PYTHONPATH=. python -m tests.t
"""

from __future__ import annotations

import gc

import torch

from src.layers.mrope import MRotaryEmbedding as MRopeA
from src.layers.mrope_plus import MRotaryEmbedding as MRopeB
from src.layers.rotary import apply_rotary_emb


def _mb(n: int) -> float:
    return n / (1024**2)


def bench_us(fn, warmup: int = 50, iters: int = 200) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start, end = torch.cuda.Event(True), torch.cuda.Event(True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters * 1000.0


def peak_delta_mib(fn) -> float:
    torch.cuda.synchronize()
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    base = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    fn()
    torch.cuda.synchronize()
    return _mb(torch.cuda.max_memory_allocated() - base)


def make_module(cls, *, interleaved: bool, max_pos: int = 8192, device="cuda"):
    # Qwen3-VL: head_dim=128, mrope_section typically (24,20,20)
    m = cls(
        head_size=128,
        rotary_dim=128,
        max_position_embeddings=max_pos,
        base=1_000_000.0,
        mrope_section=(24, 20, 20),
        mrope_interleaved=interleaved,
    )
    return m.to(device)


def sync_cache(a: MRopeA, b: MRopeB) -> None:
    """Align caches for numeric A/B (truncate/pad to common length)."""
    n = min(a.cos_sin_cache.shape[0], b.cos_sin_cache.shape[0])
    b.cos_sin_cache[:n].copy_(a.cos_sin_cache[:n])


def positions_1d(n: int, device) -> torch.Tensor:
    return torch.arange(n, device=device, dtype=torch.long)


def positions_3d(n: int, device, *, offset: int = 0) -> torch.Tensor:
    # Simple synthetic 3-axis positions (t/h/w cursors).
    t = torch.arange(n, device=device, dtype=torch.long) + offset
    h = (torch.arange(n, device=device, dtype=torch.long) // 4) + offset
    w = (torch.arange(n, device=device, dtype=torch.long) % 4) + offset
    return torch.stack([t, h, w], dim=0)


def ref_from_module(m, positions, q, k):
    """Out-of-place reference using the module's _cos_sin + native apply."""
    if positions.ndim == 1 or positions.shape[0] < 3:
        pos = positions if positions.ndim == 1 else positions[0]
        cos, sin = m._get_cos_sin(pos) if hasattr(m, "_get_cos_sin") else m._cos_sin(pos)
        # _get_cos_sin returns (N,1,half); _cos_sin for 1d expands inside
        if positions.ndim == 1:
            cos, sin = m._get_cos_sin(positions)
        return apply_rotary_emb(q, cos, sin), apply_rotary_emb(k, cos, sin)
    cos, sin = m._cos_sin(positions)
    return apply_rotary_emb(q, cos, sin), apply_rotary_emb(k, cos, sin)


def test_correctness(device: torch.device, dtype: torch.dtype) -> None:
    print("\n" + "=" * 78)
    print(f"Correctness  dtype={dtype}")
    print("=" * 78)
    atol = 2e-2 if dtype == torch.bfloat16 else 1e-3
    n_qh, n_kh, hd = 32, 8, 128

    cases = [
        ("1d-decode", False, lambda n: positions_1d(n, device), 1),
        ("1d-prefill", False, lambda n: positions_1d(n, device), 256),
        ("3d-section", False, lambda n: positions_3d(n, device), 128),
        ("3d-interleaved", True, lambda n: positions_3d(n, device), 128),
        ("3d-long", False, lambda n: positions_3d(n, device, offset=100), 512),
    ]

    for name, interleaved, pos_fn, n_tok in cases:
        a = make_module(MRopeA, interleaved=interleaved, device=device)
        b = make_module(MRopeB, interleaved=interleaved, device=device)
        sync_cache(a, b)

        positions = pos_fn(n_tok)
        # Grow cache if needed (both APIs)
        need = int(positions.max().item()) + 1
        a.ensure_cache_size(need)
        b.ensure_cache_size(need)
        sync_cache(a, b)

        q0 = torch.randn(n_tok, n_qh, hd, device=device, dtype=dtype)
        k0 = torch.randn(n_tok, n_kh, hd, device=device, dtype=dtype)

        # Reference: A's native-style path via _cos_sin / _get_cos_sin
        if positions.ndim == 1:
            cos, sin = a._get_cos_sin(positions)
            q_ref = apply_rotary_emb(q0.clone(), cos, sin)
            k_ref = apply_rotary_emb(k0.clone(), cos, sin)
        else:
            cos, sin = a._cos_sin(positions)
            q_ref = apply_rotary_emb(q0.clone(), cos, sin)
            k_ref = apply_rotary_emb(k0.clone(), cos, sin)

        qa, ka = a(positions, q0.clone(), k0.clone())
        qb, kb = b(positions, q0.clone(), k0.clone())

        d_a = max(
            (qa.float() - q_ref.float()).abs().max().item(),
            (ka.float() - k_ref.float()).abs().max().item(),
        )
        d_b = max(
            (qb.float() - q_ref.float()).abs().max().item(),
            (kb.float() - k_ref.float()).abs().max().item(),
        )
        d_ab = max(
            (qa.float() - qb.float()).abs().max().item(),
            (ka.float() - kb.float()).abs().max().item(),
        )
        ok = (
            torch.allclose(qa, q_ref, atol=atol, rtol=atol)
            and torch.allclose(qb, q_ref, atol=atol, rtol=atol)
            and torch.allclose(qa, qb, atol=atol, rtol=atol)
        )
        print(
            f"  [{name}] N={n_tok} intl={interleaved}: "
            f"{'OK' if ok else 'FAIL'}  |A-ref|={d_a:.3e} |B-ref|={d_b:.3e} |A-B|={d_ab:.3e}"
        )
        assert ok, f"correctness failed: {name}"


def test_static_memory(device: torch.device) -> None:
    print("\n" + "=" * 78)
    print("Static cos_sin_cache size (same max_pos ctor arg)")
    print("=" * 78)
    for max_pos in (8192, 32768, 131072):
        a = make_module(MRopeA, interleaved=False, max_pos=max_pos, device=device)
        b = make_module(MRopeB, interleaved=False, max_pos=max_pos, device=device)
        sa = a.cos_sin_cache.nelement() * a.cos_sin_cache.element_size()
        sb = b.cos_sin_cache.nelement() * b.cos_sin_cache.element_size()
        print(
            f"  max_pos={max_pos}: A={_mb(sa):.2f} MiB (len={a.cos_sin_cache.shape[0]})  "
            f"B={_mb(sb):.2f} MiB (len={b.cos_sin_cache.shape[0]})"
        )


def run_perf() -> None:
    print("\n" + "=" * 78)
    print("Latency + activation peak  bf16  (A=mrope, B=mrope_plus)")
    print("=" * 78)
    device = torch.device("cuda")
    dtype = torch.bfloat16
    n_qh, n_kh, hd = 32, 8, 128

    configs = [
        ("1d-decode", False, 1, True),
        ("1d-prefill512", False, 512, True),
        ("3d-128", False, 128, False),
        ("3d-512", False, 512, False),
        ("3d-2048", False, 2048, False),
        ("3d-intl-512", True, 512, False),
        ("3d-intl-2048", True, 2048, False),
    ]

    print(
        f"{'case':<14} {'A_us':>8} {'B_us':>8} {'A/B':>6} "
        f"{'A_peak':>8} {'B_peak':>8}  (peak MiB)"
    )
    for name, interleaved, n_tok, is_1d in configs:
        a = make_module(MRopeA, interleaved=interleaved, device=device)
        b = make_module(MRopeB, interleaved=interleaved, device=device)
        sync_cache(a, b)
        positions = (
            positions_1d(n_tok, device) if is_1d else positions_3d(n_tok, device)
        )
        need = int(positions.max().item()) + 1
        a.ensure_cache_size(need)
        b.ensure_cache_size(need)
        sync_cache(a, b)

        q = torch.randn(n_tok, n_qh, hd, device=device, dtype=dtype)
        k = torch.randn(n_tok, n_kh, hd, device=device, dtype=dtype)

        def run_a():
            return a(positions, q.clone(), k.clone())

        def run_b():
            return b(positions, q.clone(), k.clone())

        _ = run_a()
        _ = run_b()
        torch.cuda.synchronize()

        t_a, t_b = bench_us(run_a), bench_us(run_b)
        m_a, m_b = peak_delta_mib(run_a), peak_delta_mib(run_b)
        print(
            f"{name:<14} {t_a:8.2f} {t_b:8.2f} {t_a/max(t_b,1e-9):6.2f}x "
            f"{m_a:8.3f} {m_b:8.3f}"
        )


if __name__ == "__main__":
    if not torch.cuda.is_available():
        raise SystemExit("CUDA required")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print("A = src.layers.mrope.MRotaryEmbedding")
    print("B = src.layers.mrope_plus.MRotaryEmbedding")
    test_correctness(torch.device("cuda"), torch.bfloat16)
    test_correctness(torch.device("cuda"), torch.float16)
    test_static_memory(torch.device("cuda"))
    run_perf()
    print("\nDone. A/B<1 → plus faster; A/B≈1 → no win.")
