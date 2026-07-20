#!/usr/bin/env python3
"""MoE parallel matrix: TP + PP with cutlass only (no EP / segment / naive).

Examples
--------
  cd vllm-v7
  CUDA_HOME=/usr/local/cuda-12.9 PYTHONPATH=. ../.venv/bin/python \\
      scripts/run_moe_parallel_matrix.py --tiny --devices 0
  CUDA_HOME=/usr/local/cuda-12.9 PYTHONPATH=. ../.venv/bin/python \\
      scripts/run_moe_parallel_matrix.py --real --devices 0 --max-tokens 32
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config.moe_backend import cutlass_moe_available
from tests.helpers import MOE_MODEL, cfg, greedy_tokens, homo_device_ids, measure
from tests.helpers.devices import gpu_name
from tests.helpers.moe_tiny import build_tiny_moe
from tests.helpers.parallel import layout_ok


def _print(msg: str) -> None:
    print(msg, flush=True)


def _require_cutlass(ids: list[int]) -> None:
    bad = [i for i in ids if not cutlass_moe_available(device=i)]
    if bad:
        raise SystemExit(f"cutlass MoE unavailable on devices {bad}")


def _match(
    tag: str,
    *,
    base: list[int],
    case: list[int],
    devices: list[int],
    wall_s: float,
    prefix: int | None = None,
) -> dict[str, Any]:
    ok = (base[:prefix] == case[:prefix]) if prefix else (base == case)
    row = {
        "case": tag,
        "ok": ok,
        "wall_s": round(wall_s, 2),
        "devices": devices,
        "base_tokens": base,
        "case_tokens": case,
    }
    _print(f"  {'PASS' if ok else 'FAIL'} {tag} wall={row['wall_s']}s devices={devices}")
    if not ok:
        _print(f"  base={base}\n  case={case}")
    return row


def _correct_pair(
    tag: str,
    model,
    prompt,
    *,
    n: int,
    baseline_kw: dict,
    case_kw: dict,
    prefix: int | None = None,
) -> dict[str, Any]:
    _print(f"\n=== CORRECT {tag} ===")
    t0 = time.perf_counter()
    base = greedy_tokens(cfg(model, **baseline_kw), prompt, n=n)
    case = greedy_tokens(cfg(model, **case_kw), prompt, n=n)
    return _match(
        tag,
        base=base,
        case=case,
        devices=list(case_kw.get("device_ids") or []),
        wall_s=time.perf_counter() - t0,
        prefix=prefix,
    )


def _perf(tag: str, model, prompt: str, *, n: int, **kw) -> dict[str, Any]:
    _print(f"\n=== PERF {tag} devices={kw.get('device_ids')} ===")
    t0 = time.perf_counter()
    m = measure(cfg(model, **kw), prompt, n=n, warmup=0, repeats=1)
    row = {
        "case": tag,
        "device_ids": kw.get("device_ids"),
        "decode_tok_s": m.get("decode_tok_s"),
        "ttft_ms": m.get("ttft_ms"),
        "e2e_tok_s": m.get("e2e_tok_s"),
        "wall_s": round(time.perf_counter() - t0, 2),
    }
    _print(
        f"  decode={row['decode_tok_s']:.2f} tok/s ttft={row['ttft_ms']} "
        f"wall={row['wall_s']}s"
    )
    return row


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tiny", action="store_true", help="tiny MoE (fast correctness)")
    ap.add_argument(
        "--real",
        action="store_true",
        help="real MoE weights (prefer Qwen1.5-MoE-A2.7B, else 30B-A3B)",
    )
    ap.add_argument(
        "--devices",
        default="",
        help="comma-separated GPU ids (1 = TP1; 2 = PP2 or TP2)",
    )
    ap.add_argument("--max-tokens", type=int, default=32)
    ap.add_argument(
        "--out",
        type=Path,
        default=ROOT / "bench" / "results" / "moe_parallel_matrix.json",
    )
    args = ap.parse_args()
    if not args.tiny and not args.real:
        args.tiny = True

    if args.devices.strip():
        devices = [int(x) for x in args.devices.split(",")]
    else:
        devices = [i for i in range(8) if cutlass_moe_available(device=i)]
        if not devices:
            raise SystemExit("no cutlass-capable GPU; pass --devices")
        devices = devices[:2] if len(devices) >= 2 else devices[:1]

    _require_cutlass(devices)
    _print(f"devices={devices} names={[gpu_name(i) for i in devices]}")
    results: dict[str, Any] = {"devices": devices, "correct": [], "perf": []}

    if args.tiny:
        with tempfile.TemporaryDirectory(prefix="moe_tiny_") as td:
            model = build_tiny_moe(Path(td))
            prompt = [1, 2, 3, 4, 5, 6, 7, 8]
            common = dict(
                max_num_seqs=1,
                context_len=256,
                num_kvcache_blocks=16,
                gpu_memory_utilization=0.35,
                enable_radix=False,
                dtype="float16",
                enforce_eager=True,
            )

            results["correct"].append(
                _correct_pair(
                    "tiny_tp1",
                    model,
                    prompt,
                    n=8,
                    baseline_kw={**common, "device_ids": [devices[0]]},
                    case_kw={**common, "device_ids": [devices[0]]},
                )
            )

            if len(devices) >= 2:
                try:
                    ids = homo_device_ids(2)
                    _require_cutlass(ids)
                    ok, reason = layout_ok(model, tp=2)
                    if ok:
                        results["correct"].append(
                            _correct_pair(
                                "tiny_tp2",
                                model,
                                prompt,
                                n=8,
                                baseline_kw={**common, "device_ids": [ids[0]]},
                                case_kw={
                                    **common,
                                    "tp_size": 2,
                                    "device_ids": ids,
                                },
                            )
                        )
                    else:
                        _print(f"SKIP tiny_tp2: {reason}")
                except Exception as e:
                    _print(f"SKIP tiny_tp2: {e}")

                ok, reason = layout_ok(model, tp=1, pp=2)
                if ok:
                    results["correct"].append(
                        _correct_pair(
                            "tiny_pp2",
                            model,
                            prompt,
                            n=8,
                            baseline_kw={**common, "device_ids": [devices[0]]},
                            case_kw={
                                **common,
                                "tp_size": 1,
                                "pp_size": 2,
                                "device_ids": devices[:2],
                            },
                        )
                    )
                else:
                    _print(f"SKIP tiny_pp2: {reason}")

    if args.real:
        if not Path(MOE_MODEL).is_dir():
            raise SystemExit(f"missing model {MOE_MODEL}")
        prompt = "The capital of France is"
        common = dict(
            max_num_seqs=1,
            context_len=512,
            num_kvcache_blocks=16,
            gpu_memory_utilization=0.85,
            enable_radix=False,
            enforce_eager=True,
        )
        results["perf"].append(
            _perf(
                "real_tp1",
                MOE_MODEL,
                prompt,
                n=args.max_tokens,
                **{**common, "device_ids": [devices[0]], "num_kvcache_blocks": 24},
            )
        )
        if len(devices) >= 2:
            ok, reason = layout_ok(MOE_MODEL, tp=1, pp=2)
            if ok:
                results["correct"].append(
                    _correct_pair(
                        "real_pp2",
                        MOE_MODEL,
                        prompt,
                        n=8,
                        baseline_kw={**common, "device_ids": [devices[0]]},
                        case_kw={
                            **common,
                            "tp_size": 1,
                            "pp_size": 2,
                            "device_ids": devices[:2],
                        },
                    )
                )
                results["perf"].append(
                    _perf(
                        "real_pp2",
                        MOE_MODEL,
                        prompt,
                        n=args.max_tokens,
                        **{
                            **common,
                            "tp_size": 1,
                            "pp_size": 2,
                            "device_ids": devices[:2],
                            "num_kvcache_blocks": 24,
                        },
                    )
                )
            else:
                _print(f"SKIP real_pp2: {reason}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, indent=2))
    _print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
