"""生成统一的端到端对比测试数据（多引擎共享，保证公平）。

输出 out/e2e_prompts.json：
    {"prompts": [[token,...], ...], "max_tokens": [n, ...]}
固定 seed，所有引擎用同一份数据。

用法:
    python bench_e2e_data.py [--num-seqs 16] [--min-in 100] [--max-in 400]
                             [--min-out 50] [--max-out 150]
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

OUT = Path(__file__).resolve().parent / "out"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--num-seqs", type=int, default=16)
    p.add_argument("--min-in", type=int, default=100)
    p.add_argument("--max-in", type=int, default=400)
    p.add_argument("--min-out", type=int, default=50)
    p.add_argument("--max-out", type=int, default=150)
    args = p.parse_args()

    rng = random.Random(0)
    prompts = [
        [rng.randrange(0, 10000) for _ in range(rng.randint(args.min_in, args.max_in))]
        for _ in range(args.num_seqs)
    ]
    max_tokens = [rng.randint(args.min_out, args.max_out) for _ in range(args.num_seqs)]

    OUT.mkdir(parents=True, exist_ok=True)
    data = {"prompts": prompts, "max_tokens": max_tokens}
    (OUT / "e2e_prompts.json").write_text(json.dumps(data))
    total_out = sum(max_tokens)
    print(
        f"→ saved {args.num_seqs} seqs to {OUT / 'e2e_prompts.json'} "
        f"(total output tokens ≈ {total_out})"
    )


if __name__ == "__main__":
    main()
