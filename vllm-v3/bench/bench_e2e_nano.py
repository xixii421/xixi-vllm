"""nano-vllm 端到端吞吐实测（对齐 bench_e2e_self，读同一份数据）。

用法:
    python bench_e2e_data.py
    python bench_e2e_nano.py            # 需在能 import nanovllm 的环境
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT.parent / "nano-vllm"))

from nanovllm import LLM, SamplingParams  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--device", default="cuda:0")
    args = p.parse_args()
    torch.cuda.set_device(args.device)

    data = json.loads(
        (Path(__file__).resolve().parent / "out" / "e2e_prompts.json").read_text()
    )
    prompts, max_tokens = data["prompts"], data["max_tokens"]
    total_out = sum(max_tokens)

    model = str(Path.home() / "huggingface" / "Qwen3-0.6B")
    llm = LLM(model, enforce_eager=False, max_model_len=4096)
    print("[info] nano-vllm ready")

    llm.generate(["Benchmark: "], SamplingParams(temperature=0.6, max_tokens=4))
    sps = [
        SamplingParams(temperature=0.6, ignore_eos=True, max_tokens=mt)
        for mt in max_tokens
    ]

    t0 = time.perf_counter()
    llm.generate(prompts, sps, use_tqdm=False)
    wall = time.perf_counter() - t0

    throughput = total_out / wall
    print(f"[result] engine=nano-vllm wall={wall:.2f}s "
          f"output_tokens={total_out}/{total_out} "
          f"throughput={throughput:.1f} tok/s")

    out = Path(__file__).resolve().parent / "out"
    out.mkdir(parents=True, exist_ok=True)
    (out / "e2e_nano.md").write_text(
        f"# e2e throughput — engine=nano-vllm\n\n"
        f"- wall: {wall:.2f} s\n"
        f"- output tokens: {total_out}/{total_out}\n"
        f"- throughput: {throughput:.1f} tok/s\n"
    )
    print(f"\n→ saved to {out / 'e2e_nano.md'}")


if __name__ == "__main__":
    main()
