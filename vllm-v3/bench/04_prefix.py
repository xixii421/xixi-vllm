"""04 — 机制：前缀缓存 none / hash / radix。

流程：seed 写入索引 → 完全命中 → 部分命中。指标是 TTFT 与 cached token 数。
v3 block_size=256，部分命中长度须对齐块边界。

用法:
    python bench/04_prefix.py
    python bench/04_prefix.py --prompt-len 1024 --partial 768
"""
from __future__ import annotations

import argparse

import torch

from common import destroy, dummy_ids, make_engine, ttft_until_first, warmup, write_result


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--prompt-len", type=int, default=1024)
    p.add_argument("--partial", type=int, default=768)
    args = p.parse_args()
    torch.cuda.set_device(args.device)

    lines = [
        f"# 04 prefix cache  (prompt={args.prompt_len}, partial={args.partial})",
        "",
        "| backend | seed (ms) | full-hit (ms) | partial (ms) | "
        "full cached | partial cached |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for s in lines:
        print(s)
    print()

    seed = dummy_ids(args.prompt_len, salt=0)
    full = list(seed)
    partial = list(seed)
    for i in range(args.partial, args.prompt_len):
        partial[i] = (partial[i] + 31) % 1000 + 100

    for backend in ("none", "hash", "radix"):
        engine = make_engine(cg=False, compile=False, prefix=backend)
        warmup(engine, 8)
        t_seed = ttft_until_first(engine, seed)
        cached_seed = engine.last_num_cached_tokens
        t_full = ttft_until_first(engine, full)
        cached_full = engine.last_num_cached_tokens
        t_part = ttft_until_first(engine, partial)
        cached_part = engine.last_num_cached_tokens
        row = (
            f"| {backend} | {t_seed:.2f} | {t_full:.2f} | {t_part:.2f} | "
            f"{cached_full} | {cached_part} |"
        )
        print(row)
        lines.append(row)
        _ = cached_seed
        destroy(engine)
    write_result("04_prefix.md", "\n".join(lines))


if __name__ == "__main__":
    main()
