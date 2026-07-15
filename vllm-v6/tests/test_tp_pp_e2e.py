"""TP/PP end-to-end: greedy match vs single-GPU (needs >=2 GPUs).

On heterogeneous GPUs (e.g. 4070+4090), PP can drift after a few decode
steps because later layers run on different silicon; we check a short
greedy prefix. TP all-reduces each layer so it stays bit-matched longer.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest
import torch

from src import Engine, EngineConfig, SamplingParams

os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")

MODEL = Path.home() / "huggingface" / "Qwen3-0.6B"

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available()
    or torch.cuda.device_count() < 2
    or not MODEL.is_dir(),
    reason="needs >=2 GPUs + Qwen3-0.6B",
)

PROMPTS = ["The capital of France is", "Hello, my name is"]


def _token_ids(*, tp: int, pp: int, port: int, device_ids: list[int], n: int) -> list[list[int]]:
    eng = Engine(
        EngineConfig(
            model=str(MODEL),
            enforce_eager=True,
            max_num_seqs=4,
            context_len=512,
            num_kvcache_blocks=32,
            tp_size=tp,
            pp_size=pp,
            nccl_port=port,
            device_ids=device_ids,
        )
    )
    try:
        outs: list[list[int]] = []
        for p in PROMPTS:
            seq = eng.add_request(
                p, SamplingParams(temperature=0.0, max_tokens=n, ignore_eos=True)
            )
            while not seq.is_finished:
                eng.step()
            outs.append(seq.token_ids[seq.num_prompt_tokens :])
        return outs
    finally:
        eng.destroy()


def test_tp2_matches_single_gpu():
    n = 8
    single = _token_ids(tp=1, pp=1, port=29610, device_ids=[0], n=n)
    tp2 = _token_ids(tp=2, pp=1, port=29612, device_ids=[0, 1], n=n)
    assert tp2 == single, f"TP=2 mismatch:\n  single={single!r}\n  tp2={tp2!r}"


def test_pp2_matches_single_gpu_prefix():
    n = 8
    single = _token_ids(tp=1, pp=1, port=29620, device_ids=[0], n=n)
    pp2 = _token_ids(tp=1, pp=2, port=29622, device_ids=[0, 1], n=n)
    # Prefill + early decode must agree; later tokens may drift on mixed GPUs.
    for i, (a, b) in enumerate(zip(single, pp2)):
        assert a[:4] == b[:4], (
            f"PP=2 prefix mismatch on prompt {i}:\n  single={a!r}\n  pp2={b!r}"
        )
