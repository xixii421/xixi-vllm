"""TP/PP performance + memory smoke for vllm-v6.

  CUDA_DEVICE_ORDER=PCI_BUS_ID PYTHONPATH=. python bench/bench_tp_pp.py \\
    --model ~/huggingface/Qwen3-0.6B --modes single,tp2,pp2

  CUDA_DEVICE_ORDER=PCI_BUS_ID PYTHONPATH=. python bench/bench_tp_pp.py \\
    --model ~/huggingface/Qwen3-8B --modes single4070,tp2
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch


def _sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _mem_gb(device_ids: list[int]) -> dict[str, float]:
    out = {}
    for d in device_ids:
        free, total = torch.cuda.mem_get_info(d)
        out[f"gpu{d}_used_gb"] = round((total - free) / 1e9, 2)
        out[f"gpu{d}_total_gb"] = round(total / 1e9, 2)
    return out


def _run_mode(
    model: str,
    *,
    tp_size: int,
    pp_size: int,
    device_ids: list[int],
    max_new: int,
    prompt: str,
    warmup: int,
    iters: int,
    gpu_util: float = 0.85,
) -> dict:
    from src.config import EngineConfig
    from src.engine.engine import Engine
    from src.sampling_params import SamplingParams

    ids = list(device_ids[: tp_size * pp_size])
    cfg = EngineConfig(
        model=model,
        dtype="float16",
        enforce_eager=True,
        tp_size=tp_size,
        pp_size=pp_size,
        device_ids=ids,
        gpu_memory_utilization=gpu_util,
        max_num_seqs=2,
        max_num_batched_tokens=4096,
    )
    eng = Engine(cfg)
    sp = SamplingParams(max_tokens=max_new, temperature=0.0, ignore_eos=True)

    for _ in range(warmup):
        seq = eng.add_request(
            prompt, SamplingParams(max_tokens=min(8, max_new), temperature=0.0, ignore_eos=True)
        )
        while not seq.is_finished:
            eng.step()

    rows = []
    for _ in range(iters):
        seq = eng.add_request(prompt, sp)
        t_start = time.perf_counter()
        first = None
        prefill_s = decode_s = 0.0
        n_dec = 0
        while not seq.is_finished:
            before = seq.num_completion_tokens
            _sync()
            t0 = time.perf_counter()
            eng.step()
            _sync()
            dt = time.perf_counter() - t0
            after = seq.num_completion_tokens
            if before == 0:
                prefill_s += dt
                if after > 0 and first is None:
                    first = time.perf_counter() - t_start
            else:
                decode_s += dt
                n_dec += after - before
        total = time.perf_counter() - t_start
        gen = seq.num_completion_tokens
        rows.append(
            {
                "ttft_s": first if first is not None else prefill_s,
                "prefill_s": prefill_s,
                "decode_s": decode_s,
                "total_s": total,
                "gen_tokens": gen,
                "decode_tok_s": (n_dec / decode_s) if decode_s > 0 else 0.0,
                "e2e_tok_s": (gen / total) if total > 0 else 0.0,
                "prompt_tokens": seq.num_prompt_tokens,
            }
        )

    mem = _mem_gb(ids)
    try:
        eng.destroy()
    except Exception:
        pass
    del eng
    torch.cuda.empty_cache()

    avg = {k: sum(r[k] for r in rows) / len(rows) for k in rows[0]}
    avg.update(mem)
    avg["tp_size"] = tp_size
    avg["pp_size"] = pp_size
    avg["device_ids"] = ids
    return avg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--modes", default="single,tp2,pp2")
    ap.add_argument("--device-ids", default="0,1")
    ap.add_argument("--max-new", type=int, default=64)
    ap.add_argument("--warmup", type=int, default=1)
    ap.add_argument("--iters", type=int, default=3)
    ap.add_argument(
        "--out",
        default=str(Path(__file__).resolve().parent / "results/tp_pp_perf.json"),
    )
    ap.add_argument("--prompt", default="Explain tensor parallelism in two short sentences.")
    args = ap.parse_args()

    all_ids = [int(x) for x in args.device_ids.split(",") if x.strip() != ""]
    mode_map = {
        "single": (1, 1, all_ids[:1] or [0]),
        "single4070": (1, 1, [0]),
        "single4090": (1, 1, [1] if len(all_ids) > 1 else [0]),
        "tp2": (2, 1, all_ids[:2]),
        "pp2": (1, 2, all_ids[:2]),
    }

    report = {"model": args.model, "max_new": args.max_new, "modes": {}}
    for name in args.modes.split(","):
        name = name.strip()
        if name not in mode_map:
            print(f"skip unknown mode {name}")
            continue
        tp, pp, ids = mode_map[name]
        if len(ids) < tp * pp:
            print(f"skip {name}: need {tp * pp} devices, got {ids}")
            continue
        print(f"\n======== {name} tp={tp} pp={pp} devices={ids} ========")
        try:
            row = _run_mode(
                args.model,
                tp_size=tp,
                pp_size=pp,
                device_ids=ids,
                max_new=args.max_new,
                prompt=args.prompt,
                warmup=args.warmup,
                iters=args.iters,
            )
        except Exception as e:
            print(f"FAILED {name}: {type(e).__name__}: {e}")
            report["modes"][name] = {"error": f"{type(e).__name__}: {e}"}
            torch.cuda.empty_cache()
            continue
        report["modes"][name] = row
        mem = {k: v for k, v in row.items() if k.endswith("_gb")}
        print(
            f"  TTFT={row['ttft_s']*1000:.1f}ms decode={row['decode_tok_s']:.1f} tok/s "
            f"e2e={row['e2e_tok_s']:.1f} tok/s mem={mem}"
        )

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    md = out.with_suffix(".md")
    lines = [
        "# TP/PP performance",
        "",
        f"- model: `{args.model}`",
        f"- max_new={args.max_new}",
        "",
        "| mode | tp | pp | TTFT_ms | decode_tok/s | e2e_tok/s | mem |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for name, m in report["modes"].items():
        if "error" in m:
            lines.append(f"| {name} | — | — | FAIL | — | — | `{m['error'][:80]}` |")
            continue
        mem = ", ".join(f"{k}={v}" for k, v in m.items() if k.endswith("used_gb"))
        lines.append(
            f"| {name} | {m['tp_size']} | {m['pp_size']} | {m['ttft_s']*1000:.1f} | "
            f"{m['decode_tok_s']:.1f} | {m['e2e_tok_s']:.1f} | {mem} |"
        )
    md.write_text("\n".join(lines) + "\n")
    print(f"\nWrote {out} and {md}")


if __name__ == "__main__":
    main()
