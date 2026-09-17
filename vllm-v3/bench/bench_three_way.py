"""Fair three-way throughput comparison: naive v0, optimized v3, official vLLM.

Each engine runs in a separate process because the local versions share the ``src``
package name and official vLLM has its own dependency environment.  The timed
region excludes model loading, tokenization, printing, and warmup.

Examples (from the repository root):
    .venv/bin/python vllm-v3/bench/bench_three_way.py --engine v0
    .venv/bin/python vllm-v3/bench/bench_three_way.py --engine v3
    /path/to/vllm/python vllm-v3/bench/bench_three_way.py --engine vllm
"""
from __future__ import annotations

import argparse
import gc
import statistics
import sys
import time
from pathlib import Path

import torch

BENCH = Path(__file__).resolve().parent
REPO = BENCH.parent.parent
V0_ROOT = REPO / "vllm-v0"
V3_ROOT = REPO / "vllm-v3"
MODEL = str(Path.home() / "huggingface" / "Qwen3-0.6B")
OUT = BENCH / "out" / "bench_three_way.txt"


def dummy_ids(n: int, salt: int = 0) -> list[int]:
    """Return valid, deterministic Qwen3 token IDs without tokenization."""
    return [((i * 7919 + 10007 + salt * 97) % 150000) + 100 for i in range(n)]


def workload(n: int, prompt_len: int, *, salt: int) -> list[list[int]]:
    return [dummy_ids(prompt_len, salt=salt + i) for i in range(n)]


def _v0_generate(engine, prompt: list[int], output_len: int) -> int:
    """Naive greedy decode: recompute the entire growing context every token.

    This is Engine.generate's model path without tokenizer encode/decode or EOS
    early-stop, so all engines perform exactly the requested amount of model work.
    """
    ids = list(prompt)
    for _ in range(output_len):
        input_ids = torch.tensor([ids], device=engine.device, dtype=torch.long)
        positions = torch.arange(len(ids), device=engine.device).unsqueeze(0)
        hidden = engine.model(input_ids, positions)
        logits = engine.model.compute_logits(hidden[:, -1:]).squeeze(1)
        ids.append(int(logits.argmax(dim=-1).item()))
    return output_len


def run_v0(
    num_seqs: int, prompt_len: int, output_len: int, repeats: int
) -> list[tuple[float, int]]:
    sys.path.insert(0, str(V0_ROOT))
    from src.config import EngineConfig
    from src.engine import Engine

    engine = Engine(EngineConfig(model=MODEL, dtype="auto", enforce_eager=True))
    _v0_generate(engine, dummy_ids(prompt_len, salt=9000), 2)
    for prompt in workload(num_seqs, prompt_len, salt=9900):
        _v0_generate(engine, prompt, output_len)
    torch.cuda.synchronize()

    results = []
    for trial in range(repeats):
        prompts = workload(num_seqs, prompt_len, salt=trial * 1000)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        n_out = sum(_v0_generate(engine, p, output_len) for p in prompts)
        torch.cuda.synchronize()
        results.append((time.perf_counter() - t0, n_out))
    del engine
    gc.collect()
    torch.cuda.empty_cache()
    return results


def run_v3(
    num_seqs: int, prompt_len: int, output_len: int, repeats: int
) -> list[tuple[float, int]]:
    sys.path.insert(0, str(BENCH))
    sys.path.insert(0, str(V3_ROOT))
    from common import destroy, drain, make_engine
    from src import SamplingParams

    engine = make_engine(
        cg=True,
        compile=True,
        prefix="none",
        max_num_seqs=max(8, num_seqs),
        gpu_mem=0.85,
    )
    warm = SamplingParams(temperature=0.0, max_tokens=2, ignore_eos=True)
    for i in range(3):
        for prompt in workload(num_seqs, prompt_len, salt=9000 + i * 1000):
            engine.add_request(prompt, warm)
        drain(engine)
    full_warm = SamplingParams(
        temperature=0.0, max_tokens=output_len, ignore_eos=True
    )
    for prompt in workload(num_seqs, prompt_len, salt=9900):
        engine.add_request(prompt, full_warm)
    drain(engine)
    torch.cuda.synchronize()

    results = []
    sp = SamplingParams(temperature=0.0, max_tokens=output_len, ignore_eos=True)
    for trial in range(repeats):
        prompts = workload(num_seqs, prompt_len, salt=trial * 1000)
        seqs = [engine.add_request(prompt, sp) for prompt in prompts]
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        drain(engine)
        torch.cuda.synchronize()
        results.append(
            (time.perf_counter() - t0, sum(s.num_completion_tokens for s in seqs))
        )
    destroy(engine)
    return results


def run_vllm(
    num_seqs: int, prompt_len: int, output_len: int, repeats: int
) -> list[tuple[float, int]]:
    from vllm import LLM, SamplingParams

    try:
        from vllm import TokensPrompt

        pack = lambda ids: TokensPrompt(prompt_token_ids=ids)
    except ImportError:
        pack = lambda ids: {"prompt_token_ids": ids}

    llm = LLM(
        model=MODEL,
        tokenizer=MODEL,
        dtype="bfloat16",
        max_model_len=4096,
        max_num_seqs=max(8, num_seqs),
        gpu_memory_utilization=0.85,
        enforce_eager=False,
        attention_backend="FLASH_ATTN",
        enable_prefix_caching=False,
        disable_log_stats=True,
    )
    warm_sp = SamplingParams(temperature=0.0, max_tokens=2, ignore_eos=True)
    for i in range(3):
        warm_prompts = [
            pack(p) for p in workload(num_seqs, prompt_len, salt=9000 + i * 1000)
        ]
        llm.generate(warm_prompts, warm_sp, use_tqdm=False)
    full_warm_prompts = [
        pack(p) for p in workload(num_seqs, prompt_len, salt=9900)
    ]
    full_warm_sp = SamplingParams(
        temperature=0.0, max_tokens=output_len, ignore_eos=True
    )
    llm.generate(full_warm_prompts, full_warm_sp, use_tqdm=False)
    torch.cuda.synchronize()

    results = []
    sp = SamplingParams(temperature=0.0, max_tokens=output_len, ignore_eos=True)
    for trial in range(repeats):
        prompts = [pack(p) for p in workload(num_seqs, prompt_len, salt=trial * 1000)]
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        outputs = llm.generate(prompts, sp, use_tqdm=False)
        torch.cuda.synchronize()
        n_out = sum(len(o.outputs[0].token_ids) for o in outputs)
        results.append((time.perf_counter() - t0, n_out))
    del llm
    gc.collect()
    torch.cuda.empty_cache()
    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--engine", choices=("v0", "v3", "vllm"), required=True)
    parser.add_argument("--num-seqs", type=int, default=8)
    parser.add_argument("--prompt-len", type=int, default=256)
    parser.add_argument("--output-len", type=int, default=64)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    torch.cuda.set_device(args.device)

    runner = {"v0": run_v0, "v3": run_v3, "vllm": run_vllm}[args.engine]
    results = runner(
        args.num_seqs, args.prompt_len, args.output_len, args.repeats
    )
    throughputs = [n_out / wall for wall, n_out in results]
    median_tps = statistics.median(throughputs)

    print(
        f"engine={args.engine} n={args.num_seqs} prompt={args.prompt_len} "
        f"output={args.output_len} repeats={args.repeats}"
    )
    for i, ((wall, n_out), tps) in enumerate(zip(results, throughputs), 1):
        print(f"trial={i} wall={wall:.4f}s output={n_out} throughput={tps:.2f} tok/s")
    print(f"median={median_tps:.2f} tok/s")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("a") as f:
        for i, ((wall, n_out), tps) in enumerate(zip(results, throughputs), 1):
            f.write(
                f"{args.engine} | {args.num_seqs} | {args.prompt_len} | "
                f"{args.output_len} | {i} | {wall:.4f} | {n_out} | {tps:.2f}\n"
            )


if __name__ == "__main__":
    main()
