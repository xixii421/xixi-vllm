#!/usr/bin/env python3
"""验证 prefill CG(FA2) 与 eager(FA2) 数值一致。"""
import os, sys, gc
os.environ.setdefault("CUDA_DEVICE_ORDER", "FASTEST_FIRST")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
sys.path.insert(0, "/home/xingyongwang/code/mycuda/dzyy-vllm/vllm-v3")
import torch
from src import Engine, EngineConfig, SamplingParams

def build_prompt(eng, n):
    chunk = ("The quick brown fox jumps over the lazy dog. "
             "Explain continuous batching for LLM inference. ") * 4
    ids = []
    while len(ids) < n:
        ids.extend(eng.tokenizer.encode(chunk, add_special_tokens=False))
    return ids[:n]

for L in (129, 513, 777, 1024, 2049):
    # CG 引擎
    eng_cg = Engine(EngineConfig(
        model="/home/xingyongwang/huggingface/Qwen3-0.6B", context_len=16384,
        enforce_eager=False, prefill_cuda_graph=True,
        dtype="auto", prefix_backend="none", max_num_seqs=8,
        num_kvcache_blocks=128, gpu_memory_utilization=0.85,
    ))
    ids = build_prompt(eng_cg, L)
    seq = eng_cg.add_request(ids, SamplingParams(max_tokens=1, temperature=0.0))
    while not seq.is_finished:
        eng_cg.step()
    first_cg = seq.token_ids[seq.num_prompt_tokens:][:1]
    eng_cg.destroy()
    del eng_cg, seq
    gc.collect()
    torch.cuda.empty_cache()

    # eager 引擎
    eng_e = Engine(EngineConfig(
        model="/home/xingyongwang/huggingface/Qwen3-0.6B", context_len=16384,
        enforce_eager=True, prefill_cuda_graph=False,
        dtype="auto", prefix_backend="none", max_num_seqs=8,
        num_kvcache_blocks=256, gpu_memory_utilization=0.85,
    ))
    seq = eng_e.add_request(ids, SamplingParams(max_tokens=1, temperature=0.0))
    while not seq.is_finished:
        eng_e.step()
    first_e = seq.token_ids[seq.num_prompt_tokens:][:1]
    eng_e.destroy()
    del eng_e, seq
    gc.collect()
    torch.cuda.empty_cache()

    ok = first_cg == first_e
    print(f"len={L}: CG-FA2 first={first_cg}  eager first={first_e}  {'MATCH' if ok else 'MISMATCH'}")
