"""从 bench.jsonl 生成 Markdown 表，便于贴进 docs/02。

用法:
  ../.venv2/bin/python bench/report.py
  ../.venv2/bin/python bench/report.py --model Qwen3-8B
  ../.venv2/bin/python bench/report.py --engine v4 --ok-only
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

BENCH = Path(__file__).resolve().parent
if str(BENCH) not in sys.path:
    sys.path.insert(0, str(BENCH))

from common import JSONL_PATH  # noqa: E402


def load_records(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows: list[dict] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        rows.append(json.loads(line))
    return rows


def fmt(v, digits: int = 1) -> str:
    if v is None:
        return "-"
    if isinstance(v, float):
        return f"{v:.{digits}f}"
    return str(v)


def print_single_table(rows: list[dict]) -> None:
    """dp=1 的 TTFT/decode/batch 表。"""
    data = [r for r in rows if int(r.get("dp", 1)) == 1]
    if not data:
        return
    print("### 单副本 (tp/pp)")
    print()
    print("| engine | model | tp | pp | mode | TTFT (ms) | decode | batch8 | ok |")
    print("| --- | --- | --- | --- | --- | --- | --- | --- | --- |")
    for r in data:
        print(
            f"| {r.get('engine')} | {r.get('model')} | {r.get('tp')} | {r.get('pp')} "
            f"| {r.get('mode')} | {fmt(r.get('ttft_ms'), 2)} | {fmt(r.get('decode_tps'))} "
            f"| {fmt(r.get('batch_tps'))} | {r.get('ok')} |"
        )
    print()


def print_dp_table(rows: list[dict]) -> None:
    data = [r for r in rows if int(r.get("dp", 1)) > 1]
    if not data:
        return
    print("### 多副本 DP")
    print()
    print("| engine | model | dp | mode | n_reqs | throughput | ok |")
    print("| --- | --- | --- | --- | --- | --- | --- |")
    for r in data:
        print(
            f"| {r.get('engine')} | {r.get('model')} | {r.get('dp')} | {r.get('mode')} "
            f"| {fmt(r.get('n_reqs'), 0)} | {fmt(r.get('throughput_tps'))} | {r.get('ok')} |"
        )
    print()


def print_failures(rows: list[dict]) -> None:
    bad = [r for r in rows if not r.get("ok", True)]
    if not bad:
        return
    print("### 失败 / 跳过")
    print()
    print("| tag | error |")
    print("| --- | --- |")
    for r in bad:
        err = (r.get("error") or "").replace("|", "/")
        print(f"| {r.get('tag')} | {err} |")
    print()


def print_summary(rows: list[dict]) -> None:
    by = defaultdict(int)
    for r in rows:
        key = "ok" if r.get("ok", True) else "fail"
        by[key] += 1
    print(f"records={len(rows)} ok={by['ok']} fail={by['fail']}")
    print()


def main() -> None:
    p = argparse.ArgumentParser(description="bench jsonl report")
    p.add_argument("--jsonl", type=Path, default=JSONL_PATH)
    p.add_argument("--model", default=None, help="按模型名过滤，如 Qwen3-8B")
    p.add_argument("--engine", choices=("v4", "vllm"), default=None)
    p.add_argument("--ok-only", action="store_true")
    args = p.parse_args()

    rows = load_records(args.jsonl)
    if args.model:
        rows = [r for r in rows if r.get("model") == args.model]
    if args.engine:
        rows = [r for r in rows if r.get("engine") == args.engine]
    if args.ok_only:
        rows = [r for r in rows if r.get("ok", True)]

    # 同 tag 保留最后一条
    latest: dict[str, dict] = {}
    for r in rows:
        latest[str(r.get("tag"))] = r
    rows = list(latest.values())
    rows.sort(
        key=lambda r: (
            str(r.get("engine")),
            str(r.get("model")),
            int(r.get("tp", 1)),
            int(r.get("pp", 1)),
            int(r.get("dp", 1)),
            str(r.get("mode")),
        )
    )

    print_summary(rows)
    print_single_table(rows)
    print_dp_table(rows)
    if not args.ok_only:
        print_failures(load_records(args.jsonl) if not args.model and not args.engine else rows)


if __name__ == "__main__":
    main()
