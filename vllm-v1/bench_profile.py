"""Profile vllm-v1 — PagedAttention + KV Cache.

Phase annotations: prefill, decode, model.forward
Output: Chrome‑trace JSON + op‑level table + category summary.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_HERE))

import torch
from src.config import EngineConfig
from src.engine import Engine
from src.kv import Sequence, set_forward_context
from profiler import Profiler


def run_profile(model_path: str, max_new: int = 32) -> None:
    cfg = EngineConfig(
        model=model_path, enforce_eager=True, context_len=512, prefix_backend="radix"
    )
    engine = Engine(cfg)

    prompt = "The capital of France is"
    input_ids: list[int] = engine.tokenizer.encode(prompt)[:cfg.context_len]

    prof = Profiler(trace_dir="./traces/v1", label="v1_eager")
    prof.start()

    try:
        seq: Sequence = engine.kv.new_sequence(input_ids)
        engine.kv.allocate(seq)
        cached = seq.num_cached_tokens
        prompt_len = len(input_ids)

        # --- prefill ---
        with prof.annotate("prefill"):
            if cached < prompt_len:
                engine._bind_ctx(seq, is_prefill=True, cache_len=cached, write_end=prompt_len)
                ids = torch.tensor([input_ids[cached:]], device=engine.device, dtype=torch.long)
                positions = torch.arange(cached, prompt_len, device=engine.device).unsqueeze(0)
            else:
                engine._bind_ctx(seq, is_prefill=False, cache_len=prompt_len - 1, write_end=prompt_len)
                ids = torch.tensor([[input_ids[-1]]], device=engine.device, dtype=torch.long)
                positions = torch.tensor([[prompt_len - 1]], device=engine.device, dtype=torch.long)

            with prof.annotate("model.forward"):
                h = engine.model(ids, positions)
            with prof.annotate("sample"):
                logits = engine.model.compute_logits(h[:, -1:]).squeeze(1)
                next_id = int(engine.sampler(logits, None, None, None, None).item())
            seq.append_token(next_id)
            engine.kv.ensure_blocks(seq)
        cache_len = prompt_len

        # --- decode ---
        for _ in range(max_new - 1):
            if cache_len >= cfg.context_len:
                break
            with prof.annotate("decode"):
                engine._bind_ctx(seq, is_prefill=False, cache_len=cache_len, write_end=cache_len + 1)
                ids = torch.tensor([[next_id]], device=engine.device, dtype=torch.long)
                positions = torch.tensor([[cache_len]], device=engine.device, dtype=torch.long)
                with prof.annotate("model.forward"):
                    h = engine.model(ids, positions)
                with prof.annotate("sample"):
                    logits = engine.model.compute_logits(h[:, -1:]).squeeze(1)
                    next_id = int(engine.sampler(logits, None, None, None, None).item())
                cache_len += 1
                seq.append_token(next_id)
                engine.kv.ensure_blocks(seq)
            if next_id == engine.eos_token_id:
                break
    finally:
        set_forward_context(None)
        engine.kv.finish(seq)
        prof.stop()

    prof.print_stats()
    prof.print_category_summary()
    print(f"\n[profiler] traces → {prof.trace_dir}/v1_eager*.json")
    print("           view:  tensorboard --logdir traces/")


def main():
    p = argparse.ArgumentParser(description="Profile vllm-v1")
    p.add_argument("--model", default="~/huggingface/Qwen3-0.6B")
    p.add_argument("--max-new", type=int, default=32)
    args = p.parse_args()
    run_profile(args.model, args.max_new)


if __name__ == "__main__":
    main()
