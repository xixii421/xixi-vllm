"""prefill 阶段显存开销分解实测。

测量：
  A. eager 引擎初始化的显存基线（模型+KV池+backend workspace）
  B. +piece prefill CG 的增量（per-piece workspace/buffers/graph pool）
  C. 单步 prefill 峰值激活（eager 长文本，max_memory_allocated 增量）
"""
from __future__ import annotations

import gc
import os
import random
import sys

os.environ.setdefault("CUDA_DEVICE_ORDER", "FASTEST_FIRST")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
sys.path.insert(0, "/home/xingyongwang/code/mycuda/dzyy-vllm/vllm-v3")

import torch

from src.config import EngineConfig
from src.engine.engine import Engine

MODEL = os.path.expanduser("~/huggingface/Qwen3-0.6B")
KV_BLOCKS = 96


def count_buffers(d: dict) -> tuple[int, int]:
    """递归统计 dict 中 tensor 的总字节数与张量个数。"""
    total_bytes, n = 0, 0
    for v in d.values():
        if isinstance(v, torch.Tensor):
            total_bytes += v.numel() * v.element_size()
            n += 1
        elif isinstance(v, dict):
            b, c = count_buffers(v)
            total_bytes += b
            n += c
    return total_bytes, n


def engine_mem(cfg_over: dict) -> dict:
    cfg = EngineConfig(model=MODEL, **cfg_over)
    eng = Engine(cfg)
    torch.cuda.synchronize()
    mem = {
        "reserved": torch.cuda.memory_reserved(),
        "allocated": torch.cuda.memory_allocated(),
        "pieces": None,
    }
    detail = {}
    if eng.model_runner.prefill_cg_runner is not None:
        r = eng.model_runner.prefill_cg_runner
        seen_ws = set()
        ws = 0
        other = 0
        for bufs in r._buffers.values():
            w = bufs["workspace"]
            if id(w) not in seen_ws:
                seen_ws.add(id(w))
                ws += w.numel() * w.element_size()
            for k, v in bufs.items():
                if k in ("workspace", "wrapper", "pages") or not isinstance(v, torch.Tensor):
                    continue
                other += v.numel() * v.element_size()
        detail["piece_workspace_MB"] = ws / 1024**2
        detail["piece_other_MB"] = other / 1024**2
        detail["piece_graphs"] = len(r.graphs)

    mem["detail"] = detail

    # 预读 KV 池与模型大小
    mem["kv_pool_gb"] = eng.kv.kv_cache.numel() * eng.kv.kv_cache.element_size() / 1024**3
    param_bytes = sum(p.numel() * p.element_size() for p in eng.model.parameters())
    mem["model_gb"] = param_bytes / 1024**3

    eng.destroy()
    del eng
    gc.collect()
    torch.cuda.empty_cache()
    return mem


def measure_prefill_peak_activation():
    """eager 引擎跑一次长 prefill，测 max_memory_allocated 峰值增量。"""
    cfg = EngineConfig(
        model=MODEL, context_len=8192, max_num_batched_tokens=8192,
        prefix_backend="none", num_kvcache_blocks=KV_BLOCKS,
        enforce_eager=True, mix_prefill_decode=True,
        prefill_cuda_graph=False,
    )
    eng = Engine(cfg)
    torch.cuda.synchronize()
    base = torch.cuda.memory_allocated()

    rng = random.Random(42)
    prompt = [rng.randint(0, 5000) for _ in range(4096)]
    from src.sampling_params import SamplingParams
    seq = eng.add_request(prompt, SamplingParams(temperature=0.0, max_tokens=8))
    # 单步跑完整个 prefill（max_num_batched_tokens=8192 > 4096）
    torch.cuda.reset_peak_memory_stats()
    eng.step()
    torch.cuda.synchronize()
    peak = torch.cuda.max_memory_allocated()
    print(f"  4096-token prefill 单步: 峰值激活增量 { (peak - base)/1024**3:.3f} GiB "
          f"(allocated {base/1024**3:.2f}→{torch.cuda.memory_allocated()/1024**3:.2f} GiB)")
    eng.destroy()
    del eng
    gc.collect()
    torch.cuda.empty_cache()


def main():
    base = dict(
        context_len=4096, max_num_batched_tokens=8192,
        prefix_backend="hash", num_kvcache_blocks=KV_BLOCKS,
        enforce_eager=True, mix_prefill_decode=True,
    )
    print("=== prefill 显存分解（Qwen3-0.6B, 96 KV blocks = 2.62GiB 池） ===")
    m_eager = engine_mem(dict(base, prefill_cuda_graph=False))
    print(f"[eager]            reserved={m_eager['reserved']/1024**3:.2f} GiB "
          f"allocated={m_eager['allocated']/1024**3:.2f} GiB")
    print(f"                    模型={m_eager['model_gb']:.2f} GiB, KV池={m_eager['kv_pool_gb']:.2f} GiB")

    m_piece = engine_mem(dict(base, prefill_cuda_graph=True))
    print(f"[+piece CG]        reserved={m_piece['reserved']/1024**3:.2f} GiB "
          f"(增量 {(m_piece['reserved']-m_eager['reserved'])/1024**3:+.2f} GiB)")
    print(f"                    piece_workspace={m_piece['detail'].get('piece_workspace_MB',0):.0f} MB "
          f"piece_其他buffer={m_piece['detail'].get('piece_other_MB',0):.0f} MB "
          f"graphs={m_piece['detail'].get('piece_graphs')}")

    print("\n=== prefill 峰值激活（eager） ===")
    measure_prefill_peak_activation()


if __name__ == "__main__":
    main()
