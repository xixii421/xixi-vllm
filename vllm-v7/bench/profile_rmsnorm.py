#!/usr/bin/env python3
"""RMSNorm micro-bench: Triton kernel vs torch.compile(native).

Compares numerical agreement and CUDA latency across decode / prefill /
fused-residual / strided-QK shapes.

Usage::

    CUDA_VISIBLE_DEVICES=5 PYTHONPATH=. python -m bench.profile_rmsnorm
    CUDA_VISIBLE_DEVICES=5 PYTHONPATH=. python -m bench.profile_rmsnorm \\
        --dtype float16 --warmup 50 --iters 200 --output bench/results/rmsnorm_bench.json
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bench.lib.timing import TimingStats, benchmark_cuda, sync
from src.layers.layernorm import RMSNorm


def rms_norm_native_plain(
    x: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    """Reference math matching ``RMSNorm._forward_native`` without residual."""
    dtype = x.dtype
    x_float = x.float()
    x_float = x_float * torch.rsqrt(x_float.square().mean(dim=-1, keepdim=True) + eps)
    return (x_float * weight.float()).to(dtype)


def rms_norm_native_residual(
    x: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Reference math matching fused residual + RMSNorm."""
    dtype = x.dtype
    x_float = x.float() + residual.float()
    new_residual = x_float.to(dtype)
    x_float = x_float * torch.rsqrt(x_float.square().mean(dim=-1, keepdim=True) + eps)
    out = (x_float * weight.float()).to(dtype)
    return out, new_residual


def rms_norm_native(
    x: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
    residual: torch.Tensor | None = None,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    if residual is None:
        return rms_norm_native_plain(x, weight, eps)
    return rms_norm_native_residual(x, residual, weight, eps)


@dataclass(frozen=True)
class Case:
    name: str
    shape: tuple[int, ...]
    with_residual: bool
    strided_heads: bool = False


CASES: list[Case] = [
    # Decode-ish: small token count, full hidden.
    Case("decode_1x4096", (1, 4096), False),
    Case("decode_8x4096", (8, 4096), False),
    Case("decode_32x4096", (32, 4096), False),
    # Prefill-ish.
    Case("prefill_512x4096", (512, 4096), False),
    Case("prefill_2048x4096", (2048, 4096), False),
    # Fused residual (pre-attn / post-attn LN).
    Case("decode_resid_8x4096", (8, 4096), True),
    Case("prefill_resid_512x4096", (512, 4096), True),
    Case("prefill_resid_2048x4096", (2048, 4096), True),
    # Q/K RMSNorm on packed heads [T, H, D].
    Case("qk_norm_8x32x128", (8, 32, 128), False, strided_heads=True),
    Case("qk_norm_512x32x128", (512, 32, 128), False, strided_heads=True),
]


def _make_inputs(
    case: Case,
    *,
    device: torch.device,
    dtype: torch.dtype,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    g = torch.Generator(device=device)
    g.manual_seed(seed)
    if case.strided_heads:
        # Simulate packed QKV slice: allocate [T, 3H, D] then take Q heads.
        tokens, heads, dim = case.shape
        packed = torch.randn(tokens, 3 * heads, dim, device=device, dtype=dtype, generator=g)
        x = packed[:, :heads, :]  # non-contiguous in dim-1
        assert not x.is_contiguous()
        return x, None
    x = torch.randn(*case.shape, device=device, dtype=dtype, generator=g)
    residual = None
    if case.with_residual:
        residual = torch.randn(*case.shape, device=device, dtype=dtype, generator=g)
    return x, residual


def _max_abs(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a.float() - b.float()).abs().max().item())


def _rel_l2(a: torch.Tensor, b: torch.Tensor) -> float:
    diff = (a.float() - b.float()).norm()
    denom = b.float().norm().clamp_min(1e-12)
    return float((diff / denom).item())


def _unpack(
    y: torch.Tensor | tuple[torch.Tensor, torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor | None]:
    if isinstance(y, tuple):
        return y[0], y[1]
    return y, None


def _compile_for_case(case: Case, mode: str):
    """Compile a fixed signature so shape/residual switches do not thrash the cache."""
    if case.with_residual:
        return torch.compile(rms_norm_native_residual, mode=mode, fullgraph=True)
    return torch.compile(rms_norm_native_plain, mode=mode, fullgraph=True)


def _run_compiled(compiled_fn, x, weight, eps, residual):
    if residual is None:
        return compiled_fn(x, weight, eps)
    return compiled_fn(x, residual, weight, eps)


def check_accuracy(
    *,
    triton_mod: RMSNorm,
    compiled_fn,
    case: Case,
    device: torch.device,
    dtype: torch.dtype,
    atol: float,
    rtol: float,
) -> dict:
    x, residual = _make_inputs(case, device=device, dtype=dtype, seed=0)
    weight = triton_mod.weight.detach()

    out_t, res_t = _unpack(triton_mod(x, residual))
    out_e, res_e = _unpack(rms_norm_native(x, weight, triton_mod.eps, residual))
    out_c, res_c = _unpack(_run_compiled(compiled_fn, x, weight, triton_mod.eps, residual))
    sync()

    rows = {
        "case": case.name,
        "shape": list(case.shape),
        "with_residual": case.with_residual,
        "strided_heads": case.strided_heads,
        "triton_vs_eager_max_abs": _max_abs(out_t, out_e),
        "triton_vs_eager_rel_l2": _rel_l2(out_t, out_e),
        "compile_vs_eager_max_abs": _max_abs(out_c, out_e),
        "compile_vs_eager_rel_l2": _rel_l2(out_c, out_e),
        "triton_vs_compile_max_abs": _max_abs(out_t, out_c),
        "triton_vs_compile_rel_l2": _rel_l2(out_t, out_c),
    }
    if case.with_residual:
        assert res_t is not None and res_e is not None and res_c is not None
        rows.update(
            {
                "resid_triton_vs_eager_max_abs": _max_abs(res_t, res_e),
                "resid_compile_vs_eager_max_abs": _max_abs(res_c, res_e),
                "resid_triton_vs_compile_max_abs": _max_abs(res_t, res_c),
            }
        )

    # torch.compile should match eager exactly (same graph). Triton matches fp32 ref within tol.
    ok_compile = torch.allclose(out_c, out_e, atol=0.0, rtol=0.0) or torch.allclose(
        out_c, out_e, atol=atol, rtol=rtol
    )
    ok_triton = torch.allclose(out_t, out_e, atol=atol, rtol=rtol)
    if case.with_residual:
        ok_compile = ok_compile and torch.allclose(res_c, res_e, atol=atol, rtol=rtol)
        ok_triton = ok_triton and torch.allclose(res_t, res_e, atol=atol, rtol=rtol)
    rows["pass_compile_vs_eager"] = bool(ok_compile)
    rows["pass_triton_vs_eager"] = bool(ok_triton)
    rows["pass"] = bool(ok_compile and ok_triton)
    return rows


def _stats_dict(stats: TimingStats) -> dict:
    return asdict(stats)


def bench_case(
    *,
    triton_mod: RMSNorm,
    compiled_fn,
    case: Case,
    device: torch.device,
    dtype: torch.dtype,
    warmup: int,
    iters: int,
) -> dict:
    x, residual = _make_inputs(case, device=device, dtype=dtype, seed=1)
    weight = triton_mod.weight.detach()

    def run_triton():
        return triton_mod(x, residual)

    def run_eager():
        return rms_norm_native(x, weight, triton_mod.eps, residual)

    def run_compile():
        return _run_compiled(compiled_fn, x, weight, triton_mod.eps, residual)

    # Warm compile / Triton autotune outside measured path.
    for _ in range(max(3, warmup // 5)):
        run_triton()
        run_eager()
        run_compile()
    sync()

    t_tri = benchmark_cuda(run_triton, warmup=warmup, iters=iters)
    t_eag = benchmark_cuda(run_eager, warmup=warmup, iters=iters)
    t_cmp = benchmark_cuda(run_compile, warmup=warmup, iters=iters)

    return {
        "case": case.name,
        "shape": list(case.shape),
        "with_residual": case.with_residual,
        "strided_heads": case.strided_heads,
        "triton_ms": _stats_dict(t_tri),
        "eager_ms": _stats_dict(t_eag),
        "compile_ms": _stats_dict(t_cmp),
        "speedup_triton_vs_eager": round(t_eag.median_ms / max(t_tri.median_ms, 1e-9), 3),
        "speedup_triton_vs_compile": round(t_cmp.median_ms / max(t_tri.median_ms, 1e-9), 3),
        "speedup_compile_vs_eager": round(t_eag.median_ms / max(t_cmp.median_ms, 1e-9), 3),
    }


def _print_accuracy(rows: list[dict]) -> None:
    print("\n=== Accuracy (vs eager native fp32-accum) ===")
    hdr = f"{'case':<28} {'tri↔eag':>10} {'cmp↔eag':>10} {'tri↔cmp':>10} {'ok':>4}"
    print(hdr)
    for r in rows:
        print(
            f"{r['case']:<28} "
            f"{r['triton_vs_eager_max_abs']:10.3e} "
            f"{r['compile_vs_eager_max_abs']:10.3e} "
            f"{r['triton_vs_compile_max_abs']:10.3e} "
            f"{'Y' if r['pass'] else 'N':>4}"
        )


def _print_perf(rows: list[dict]) -> None:
    print("\n=== Latency median (ms) ===")
    hdr = f"{'case':<28} {'triton':>10} {'eager':>10} {'compile':>10} {'tri/eag':>8} {'tri/cmp':>8}"
    print(hdr)
    for r in rows:
        print(
            f"{r['case']:<28} "
            f"{r['triton_ms']['median_ms']:10.4f} "
            f"{r['eager_ms']['median_ms']:10.4f} "
            f"{r['compile_ms']['median_ms']:10.4f} "
            f"{r['speedup_triton_vs_eager']:8.2f} "
            f"{r['speedup_triton_vs_compile']:8.2f}"
        )


def main() -> None:
    # Keep logical ids aligned with nvidia-smi (otherwise CUDA may reorder by size/CC).
    import os

    os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--device", type=int, default=0)
    ap.add_argument("--dtype", choices=("float16", "bfloat16", "float32"), default="float16")
    ap.add_argument("--hidden", type=int, default=4096, help="override contiguous hidden size")
    ap.add_argument("--eps", type=float, default=1e-6)
    ap.add_argument("--warmup", type=int, default=50)
    ap.add_argument("--iters", type=int, default=200)
    ap.add_argument("--atol", type=float, default=None, help="default: dtype-dependent")
    ap.add_argument("--rtol", type=float, default=None)
    ap.add_argument(
        "--output",
        type=Path,
        default=ROOT / "bench" / "results" / "rmsnorm_bench.json",
    )
    ap.add_argument(
        "--mode",
        choices=("default", "reduce-overhead", "max-autotune"),
        default="default",
        help="torch.compile mode",
    )
    args = ap.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA required")

    torch.cuda.set_device(args.device)
    device = torch.device(f"cuda:{args.device}")
    dtype = getattr(torch, args.dtype)
    atol = args.atol if args.atol is not None else (1e-3 if dtype != torch.float32 else 1e-5)
    rtol = args.rtol if args.rtol is not None else (1e-3 if dtype != torch.float32 else 1e-5)

    # Remap contiguous cases to --hidden if requested.
    cases = []
    for c in CASES:
        if c.strided_heads or len(c.shape) != 2:
            cases.append(c)
            continue
        cases.append(
            Case(c.name.replace("4096", str(args.hidden)), (c.shape[0], args.hidden), c.with_residual)
        )

    print(
        f"device={torch.cuda.get_device_name(args.device)} dtype={args.dtype} "
        f"compile_mode={args.mode} warmup={args.warmup} iters={args.iters}"
    )

    # One compiled graph per case so measured latency excludes recompile thrash.
    compiled_cache: dict[str, object] = {}
    mod_cache: dict[int, RMSNorm] = {}

    acc_rows: list[dict] = []
    perf_rows: list[dict] = []
    for case in cases:
        last = case.shape[-1]
        if last not in mod_cache:
            mod_cache[last] = RMSNorm(last, eps=args.eps).to(device=device, dtype=dtype).eval()
        mod = mod_cache[last]
        if case.name not in compiled_cache:
            compiled_cache[case.name] = _compile_for_case(case, args.mode)
        compiled_fn = compiled_cache[case.name]

        acc = check_accuracy(
            triton_mod=mod,
            compiled_fn=compiled_fn,
            case=case,
            device=device,
            dtype=dtype,
            atol=atol,
            rtol=rtol,
        )
        acc_rows.append(acc)
        perf = bench_case(
            triton_mod=mod,
            compiled_fn=compiled_fn,
            case=case,
            device=device,
            dtype=dtype,
            warmup=args.warmup,
            iters=args.iters,
        )
        perf_rows.append(perf)

    _print_accuracy(acc_rows)
    _print_perf(perf_rows)

    all_pass = all(r["pass"] for r in acc_rows)
    report = {
        "device": torch.cuda.get_device_name(args.device),
        "dtype": args.dtype,
        "compile_mode": args.mode,
        "eps": args.eps,
        "atol": atol,
        "rtol": rtol,
        "warmup": args.warmup,
        "iters": args.iters,
        "accuracy_pass": all_pass,
        "accuracy": acc_rows,
        "perf": perf_rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(f"\nWrote {args.output}")
    print(f"accuracy_pass={all_pass}")
    if not all_pass:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
