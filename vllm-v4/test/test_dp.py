"""v4 DP 正确性验证：dp_size 个独立 EngineCoreProc 副本并发推理，greedy 对拍。

用法（需 >= dp_size 张卡）:
    python test/test_dp.py --dp 2            # 2 卡各跑一个独立单卡引擎

对拍基准：Qwen3-0.6B 的 bf16 golden（与 test_parallel 一致）。
多条等长请求经 round-robin 分发到不同副本，输出应逐 token 一致。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src import Engine, EngineConfig, SamplingParams  # noqa: E402

MODEL = str(Path.home() / "huggingface" / "Qwen3-0.6B")

GOLDEN = [271, 40, 1184, 311, 1477, 279, 1372, 315, 56015, 315, 279, 3409, 330, 1782, 1, 304]


def make_cfg(dp: int, *, num_blocks: int = 32) -> EngineConfig:
    return EngineConfig(
        model=MODEL,
        tp_size=1,
        pp_size=1,
        dp_size=dp,
        enforce_eager=True,
        torch_compile=False,
        dtype="bfloat16",
        context_len=512,
        max_num_seqs=16,
        max_num_batched_tokens=1024,
        prefix_backend="none",
        gpu_memory_utilization=0.6,
        num_kvcache_blocks=num_blocks,
    )


def dummy_ids(n: int, salt: int = 0) -> list[int]:
    n = max(1, n)
    return [((i * 7919 + 10007 + salt * 97) % 100000) + 100 for i in range(n)]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--dp", type=int, default=2)
    p.add_argument("--n-reqs", type=int, default=4, help="总请求数（round-robin 分发）")
    p.add_argument("--prompt-len", type=int, default=64)
    p.add_argument("--max-tokens", type=int, default=16)
    args = p.parse_args()

    engine = Engine(make_cfg(args.dp))
    try:
        sp = SamplingParams(temperature=0.0, max_tokens=args.max_tokens, ignore_eos=True)
        seqs = [
            engine.add_request(dummy_ids(args.prompt_len, salt=7), sp)
            for _ in range(args.n_reqs)
        ]
        while engine.has_unfinished():
            engine.step()
        golden = GOLDEN[: args.max_tokens]
        ok = all(
            s.token_ids[s.num_prompt_tokens:] == golden for s in seqs
        )
        print(f"dp={args.dp} n_reqs={args.n_reqs} {'PASS' if ok else 'FAIL'}")
        if not ok:
            for s in seqs:
                print(f"  seq {s.seq_id}: {s.token_ids[s.num_prompt_tokens:]}")
            sys.exit(1)
    finally:
        engine.destroy()


if __name__ == "__main__":
    main()
