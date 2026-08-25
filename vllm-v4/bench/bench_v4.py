"""v4 引擎基准：单卡、TP、PP、DP 及组合。

用法（在 vllm-v4 目录下）:
  ../.venv2/bin/python bench/bench_v4.py --tp 1 --pp 1 --dp 1 --cg
  ../.venv2/bin/python bench/bench_v4.py --tp 2 --pp 1 --dp 1 --compile
  ../.venv2/bin/python bench/bench_v4.py --tp 1 --pp 2 --dp 1 --pp-layer-counts 12,28
  ../.venv2/bin/python bench/bench_v4.py --tp 1 --pp 1 --dp 2 --n-reqs 32

dp=1 时输出 TTFT、单请求 decode、batch8 吞吐。
dp>1 时输出 N 路请求聚合吞吐。
"""

from __future__ import annotations

import argparse
import socket
import sys
import time
from pathlib import Path


def _free_tcp_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])

ROOT = Path(__file__).resolve().parent.parent
BENCH = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(BENCH) not in sys.path:
    sys.path.insert(0, str(BENCH))

from common import (  # noqa: E402
    DEFAULT_MODEL,
    append_jsonl,
    base_record,
    mode_name,
    parse_int_list,
)
from measure import (  # noqa: E402
    measure_batch,
    measure_dp_throughput,
    measure_ttft_decode,
    warmup_dp,
    warmup_single,
)


def make_engine(
    *,
    model: str,
    tp: int,
    pp: int,
    dp: int,
    cg: bool,
    compile: bool,
    device_ids: list[int] | None,
    pp_layer_counts: list[int] | None,
    num_kvcache_blocks: int,
    max_num_seqs: int,
):
    from src import Engine, EngineConfig

    return Engine(
        EngineConfig(
            model=model,
            tp_size=tp,
            pp_size=pp,
            dp_size=dp,
            enforce_eager=not cg,
            torch_compile=compile,
            dtype="bfloat16",
            context_len=4096,
            max_num_seqs=max_num_seqs,
            max_num_batched_tokens=8192 if dp > 1 else 4096,
            prefix_backend="none",
            gpu_memory_utilization=0.6,
            num_kvcache_blocks=num_kvcache_blocks,
            device_ids=device_ids,
            pp_layer_counts=pp_layer_counts,
            master_addr="127.0.0.1",
            nccl_port=_free_tcp_port(),
            nccl_ifname="lo",
        )
    )


def run_single_replica(engine, record: dict) -> dict:
    """dp=1：测 TTFT、decode、batch8。"""
    warmup_single(engine)
    ttft_ms, decode_tps = measure_ttft_decode(engine)
    batch_tps = measure_batch(engine)
    record["ttft_ms"] = round(ttft_ms, 2)
    record["decode_tps"] = round(decode_tps, 1)
    record["batch_tps"] = round(batch_tps, 1)
    print(
        f"{record['tag']} | ttft={record['ttft_ms']:.2f}ms "
        f"| decode={record['decode_tps']:.1f} "
        f"| batch8={record['batch_tps']:.1f}",
        flush=True,
    )
    return record


def run_dp_replicas(engine, record: dict, n_reqs: int) -> dict:
    """dp>1：测多请求聚合吞吐。"""
    warmup_dp(engine, n_reqs)
    tps, n_tok = measure_dp_throughput(engine, n_reqs)
    record["throughput_tps"] = round(tps, 1)
    record["n_reqs"] = n_reqs
    print(
        f"{record['tag']} | dp={record['dp']} | reqs={n_reqs} "
        f"| {n_tok} tok | {record['throughput_tps']:.1f} tok/s",
        flush=True,
    )
    return record


def run(args: argparse.Namespace) -> int:
    model = str(Path(args.model).expanduser())
    cg = bool(args.cg)
    compile = bool(args.compile)
    mode = mode_name(cg=cg, compile=compile)
    tag = args.tag or (
        f"{Path(model).name}-tp{args.tp}-pp{args.pp}-dp{args.dp}-{mode}"
    )
    record = base_record(
        engine="v4",
        model=model,
        tp=args.tp,
        pp=args.pp,
        dp=args.dp,
        mode=mode,
        tag=tag,
    )
    need = args.tp * args.pp * args.dp
    print(
        f"=== {tag}: tp={args.tp} pp={args.pp} dp={args.dp} "
        f"mode={mode} gpus>={need} ===",
        flush=True,
    )

    engine = None
    last_exc: Exception | None = None
    # NCCL 建连偶发失败，重试几轮换新端口重建。
    for attempt in range(args.retries):
        try:
            engine = make_engine(
                model=model,
                tp=args.tp,
                pp=args.pp,
                dp=args.dp,
                cg=cg,
                compile=compile,
                device_ids=parse_int_list(args.device_ids),
                pp_layer_counts=parse_int_list(args.pp_layer_counts),
                num_kvcache_blocks=args.num_kvcache_blocks,
                max_num_seqs=args.max_num_seqs,
            )
            last_exc = None
            break
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
            print(
                f"engine init attempt {attempt + 1}/{args.retries} failed: "
                f"{type(exc).__name__}: {exc}",
                flush=True,
            )
            if attempt < args.retries - 1:
                time.sleep(3)
    try:
        if engine is None:
            assert last_exc is not None
            raise last_exc
        if args.dp > 1:
            record = run_dp_replicas(engine, record, args.n_reqs)
        else:
            record = run_single_replica(engine, record)
    except Exception as exc:  # noqa: BLE001
        record["ok"] = False
        record["error"] = f"{type(exc).__name__}: {exc}"
        print(f"FAIL {tag}: {record['error']}", flush=True)
        append_jsonl(record)
        return 1
    finally:
        if engine is not None:
            try:
                engine.destroy()
            except Exception:  # noqa: BLE001
                pass

    append_jsonl(record)
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="v4 parallel bench")
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--tp", type=int, default=1)
    p.add_argument("--pp", type=int, default=1)
    p.add_argument("--dp", type=int, default=1)
    p.add_argument("--cg", action="store_true", help="开启 decode CUDA Graph")
    p.add_argument("--compile", action="store_true", help="开启 prefill torch.compile")
    p.add_argument("--tag", default=None)
    p.add_argument("--device-ids", type=str, default=None)
    p.add_argument("--pp-layer-counts", type=str, default=None)
    p.add_argument("--num-kvcache-blocks", type=int, default=64)
    p.add_argument("--max-num-seqs", type=int, default=16)
    p.add_argument("--n-reqs", type=int, default=32, help="仅 dp>1 使用")
    p.add_argument("--retries", type=int, default=6, help="引擎初始化重试次数")
    return p


def main() -> None:
    args = build_parser().parse_args()
    if args.tp < 1 or args.pp < 1 or args.dp < 1:
        raise SystemExit("tp/pp/dp 必须 >= 1")
    if args.dp > 1:
        args.max_num_seqs = max(args.max_num_seqs, 64)
        if args.num_kvcache_blocks == 64:
            args.num_kvcache_blocks = 128
    raise SystemExit(run(args))


if __name__ == "__main__":
    main()
