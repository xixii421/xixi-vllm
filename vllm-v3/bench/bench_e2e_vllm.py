"""官方 vllm 端到端吞吐实测（对齐 bench_e2e_self，读同一份数据）。

需在装有官方 vllm 的环境运行（如 .venv-vllm）：
    .venv-vllm/bin/python bench_e2e_vllm.py

用法:
    python bench_e2e_data.py            # 先生成数据
    python bench_e2e_vllm.py
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from vllm import LLM, SamplingParams


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--enforce-eager", action="store_true", help="关闭 CUDA Graph")
    args = p.parse_args()
    torch.cuda.set_device(args.device)

    data = json.loads(
        (Path(__file__).resolve().parent / "out" / "e2e_prompts.json").read_text()
    )
    prompts, max_tokens = data["prompts"], data["max_tokens"]
    total_out = sum(max_tokens)

    model = str(Path.home() / "huggingface" / "Qwen3-0.6B")
    llm = LLM(
        model=model,
        enforce_eager=args.enforce_eager,
        max_model_len=4096,
        gpu_memory_utilization=0.6,
        max_num_seqs=16,
    )
    print("[info] official vllm ready")

    llm.generate(["Benchmark: "], SamplingParams(temperature=0.6, max_tokens=4))
    sps = [
        SamplingParams(temperature=0.6, ignore_eos=True, max_tokens=mt)
        for mt in max_tokens
    ]
    reqs = [dict(prompt_token_ids=p) for p in prompts]

    t0 = time.perf_counter()
    llm.generate(reqs, sps, use_tqdm=False)
    wall = time.perf_counter() - t0

    throughput = total_out / wall
    tag = "eager" if args.enforce_eager else "default"
    print(f"[result] engine=vllm({tag}) wall={wall:.2f}s "
          f"output_tokens={total_out}/{total_out} "
          f"throughput={throughput:.1f} tok/s")

    out = Path(__file__).resolve().parent / "out"
    out.mkdir(parents=True, exist_ok=True)
    (out / "e2e_vllm.md").write_text(
        f"# e2e throughput — engine=vllm({tag})\n\n"
        f"- wall: {wall:.2f} s\n"
        f"- output tokens: {total_out}/{total_out}\n"
        f"- throughput: {throughput:.1f} tok/s\n"
    )
    print(f"\n→ saved to {out / 'e2e_vllm.md'}")


if __name__ == "__main__":
    main()
