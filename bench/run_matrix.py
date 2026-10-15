"""矩阵编排：按 suite 依次调用 bench_v4 或 bench_vllm。

用法（在仓库根目录）:
  python bench/run_matrix.py --suite smoke
  python bench/run_matrix.py --suite full
  python bench/run_matrix.py --suite vs-vllm
  python bench/run_matrix.py --suite smoke --only 0.6b-pp2-cg
  python bench/run_matrix.py --suite vs-vllm --py-vllm /path/to/vllm/python
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BENCH = Path(__file__).resolve().parent
if str(BENCH) not in sys.path:
    sys.path.insert(0, str(BENCH))

from cases import SUITES  # noqa: E402
from common import JSONL_PATH, append_jsonl, base_record, ensure_out  # noqa: E402

DEFAULT_PY_V4 = os.environ.get("BENCH_PY_V4", sys.executable)
_VLLM_GUESS = ROOT / ".venv-vllm" / "bin" / "python"
DEFAULT_PY_VLLM = os.environ.get(
    "BENCH_PY_VLLM",
    str(_VLLM_GUESS) if _VLLM_GUESS.is_file() else sys.executable,
)


def detect_gpus() -> int:
    try:
        import torch

        return int(torch.cuda.device_count())
    except Exception:  # noqa: BLE001
        return 0


def build_cmd(case: dict, *, py_v4: str, py_vllm: str) -> list[str]:
    engine = case["engine"]
    if engine == "v4":
        py = py_v4
        script = str(BENCH / "bench_v4.py")
    elif engine == "vllm":
        py = py_vllm
        script = str(BENCH / "bench_vllm.py")
    else:
        raise ValueError(f"unknown engine: {engine}")

    cmd = [
        py,
        script,
        "--model",
        case["model"],
        "--tp",
        str(case.get("tp", 1)),
        "--pp",
        str(case.get("pp", 1)),
        "--dp",
        str(case.get("dp", 1)),
        "--tag",
        case["tag"],
    ]
    if case.get("cg"):
        cmd.append("--cg")
    if case.get("compile"):
        cmd.append("--compile")
    if case.get("device_ids"):
        cmd.extend(["--device-ids", case["device_ids"]])
    if case.get("pp_layer_counts"):
        cmd.extend(["--pp-layer-counts", case["pp_layer_counts"]])
    if case.get("num_kvcache_blocks") is not None:
        cmd.extend(["--num-kvcache-blocks", str(case["num_kvcache_blocks"])])
    if case.get("n_reqs") is not None and case.get("dp", 1) > 1:
        cmd.extend(["--n-reqs", str(case["n_reqs"])])
    if case.get("cuda_devices") and engine == "vllm":
        cmd.extend(["--cuda-devices", case["cuda_devices"]])
    return cmd


def run_one(
    case: dict,
    *,
    py_v4: str,
    py_vllm: str,
    timeout: int,
    n_gpus: int,
) -> int:
    tag = case["tag"]
    need = int(case.get("requires_gpus", case.get("tp", 1) * case.get("pp", 1) * case.get("dp", 1)))
    if need > n_gpus:
        rec = base_record(
            engine=case["engine"],
            model=case["model"],
            tp=case.get("tp", 1),
            pp=case.get("pp", 1),
            dp=case.get("dp", 1),
            mode=case.get("mode", "eager"),
            tag=tag,
        )
        rec["ok"] = False
        rec["error"] = f"SKIPPED need {need} gpus have {n_gpus}"
        append_jsonl(rec)
        print(f"SKIP {tag}: {rec['error']}", flush=True)
        return 0

    cmd = build_cmd(case, py_v4=py_v4, py_vllm=py_vllm)
    print(f"\n>>> {' '.join(cmd)}", flush=True)
    t0 = time.perf_counter()
    try:
        proc = subprocess.run(
            cmd,
            cwd=str(ROOT),
            timeout=timeout,
            env=os.environ.copy(),
        )
        rc = int(proc.returncode)
    except subprocess.TimeoutExpired:
        rec = base_record(
            engine=case["engine"],
            model=case["model"],
            tp=case.get("tp", 1),
            pp=case.get("pp", 1),
            dp=case.get("dp", 1),
            mode=case.get("mode", "eager"),
            tag=tag,
        )
        rec["ok"] = False
        rec["error"] = f"TIMEOUT after {timeout}s"
        append_jsonl(rec)
        print(f"TIMEOUT {tag}", flush=True)
        return 1

    dt = time.perf_counter() - t0
    print(f"<<< {tag} rc={rc} in {dt:.1f}s", flush=True)
    return rc


def main() -> None:
    p = argparse.ArgumentParser(description="run bench matrix")
    p.add_argument("--suite", choices=sorted(SUITES.keys()), default="smoke")
    p.add_argument("--only", type=str, default=None, help="只跑 tag 包含该子串的 case")
    p.add_argument("--timeout", type=int, default=900)
    p.add_argument("--py-v4", default=os.environ.get("BENCH_PY_V4", DEFAULT_PY_V4))
    p.add_argument("--py-vllm", default=os.environ.get("BENCH_PY_VLLM", DEFAULT_PY_VLLM))
    p.add_argument("--reset-jsonl", action="store_true", help="开始前清空 bench.jsonl")
    args = p.parse_args()

    ensure_out()
    if args.reset_jsonl and JSONL_PATH.exists():
        JSONL_PATH.write_text("", encoding="utf-8")

    cases = SUITES[args.suite]()
    if args.only:
        cases = [c for c in cases if args.only in c["tag"]]
        if not cases:
            raise SystemExit(f"no case matched --only {args.only}")

    n_gpus = detect_gpus()
    print(f"suite={args.suite} cases={len(cases)} gpus={n_gpus}", flush=True)

    fails = 0
    for case in cases:
        rc = run_one(
            case,
            py_v4=args.py_v4,
            py_vllm=args.py_vllm,
            timeout=args.timeout,
            n_gpus=n_gpus,
        )
        if rc != 0:
            fails += 1

    print(f"DONE suite={args.suite} fails={fails}/{len(cases)}", flush=True)
    raise SystemExit(1 if fails else 0)


if __name__ == "__main__":
    main()
