"""02 — greedy 正确性：eager 两次可复现；compile 只要求能跑完整条请求。

用法:
    python bench/02_correctness.py
"""
from __future__ import annotations

import argparse

import torch

from common import destroy, dummy_ids, make_engine, warmup, write_result

from src import SamplingParams  # noqa: E402


def _greedy(engine, n_prompt: int, max_tokens: int) -> list[int]:
    warmup(engine, n_prompt, max_tokens=2)
    sp = SamplingParams(temperature=0.0, max_tokens=max_tokens, ignore_eos=True)
    seq = engine.add_request(dummy_ids(n_prompt, salt=1), sp)
    while not seq.is_finished:
        engine.step()
    return list(seq.token_ids[seq.num_prompt_tokens :])


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--prompt-len", type=int, default=32)
    p.add_argument("--max-tokens", type=int, default=16)
    args = p.parse_args()
    torch.cuda.set_device(args.device)

    lines: list[str] = []

    def log(msg: str = "") -> None:
        print(msg)
        lines.append(msg)

    log("# 02 correctness (greedy)")
    log()

    eager = make_engine(cg=False, compile=False, prefix="none")
    a = _greedy(eager, args.prompt_len, args.max_tokens)
    b = _greedy(eager, args.prompt_len, args.max_tokens)
    destroy(eager)
    ok_eager = a == b and len(a) == args.max_tokens
    log(f"eager twice match: {ok_eager}  (n={len(a)})")
    if not ok_eager:
        log(f"  run1={a}")
        log(f"  run2={b}")

    compiled = make_engine(cg=True, compile=True, prefix="none")
    c = _greedy(compiled, args.prompt_len, args.max_tokens)
    destroy(compiled)
    ok_compile = len(c) == args.max_tokens
    log(f"compile+cg finished: {ok_compile}  (n={len(c)})")
    log(
        "compile vs eager token match: "
        f"{a == c}  (bf16 融合允许不一致，仅作观察)"
    )
    if ok_eager and ok_compile:
        log()
        log("OK")
    write_result("02_correctness.md", "\n".join(lines))
    if not ok_eager or not ok_compile:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
