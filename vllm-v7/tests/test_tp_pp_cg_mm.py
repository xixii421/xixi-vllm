"""Multi-GPU + CUDA Graph smoke: text TP and VL TP (needs >=2 GPUs)."""

from __future__ import annotations

import os
from pathlib import Path

import pytest
import torch
from PIL import Image

from src import Engine, EngineConfig, SamplingParams

os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")

TEXT = Path.home() / "huggingface" / "Qwen3-0.6B"
VL = Path.home() / "huggingface" / "Qwen3-VL-2B-Instruct"

need_2gpu = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.device_count() < 2,
    reason="needs >=2 GPUs",
)


def _gen(cfg: EngineConfig, prompt: str, *, n: int = 8, images=None) -> list[int]:
    eng = Engine(cfg)
    try:
        seq = eng.add_request(
            prompt,
            SamplingParams(temperature=0.0, max_tokens=n, ignore_eos=True),
            images=images,
        )
        while not seq.is_finished:
            eng.step()
        return seq.token_ids[seq.num_prompt_tokens :]
    finally:
        eng.destroy()


@need_2gpu
@pytest.mark.skipif(not TEXT.is_dir(), reason="Qwen3-0.6B missing")
def test_text_tp2_cuda_graph_matches_eager():
    common = dict(
        model=str(TEXT),
        max_num_seqs=4,
        context_len=512,
        num_kvcache_blocks=32,
        tp_size=2,
        pp_size=1,
        device_ids=[0, 1],
        nccl_port=0,
    )
    prompt = "The capital of France is"
    eager = _gen(EngineConfig(**common, enforce_eager=True), prompt, n=8)
    cg = _gen(EngineConfig(**common, enforce_eager=False), prompt, n=8)
    assert cg == eager, f"CG vs eager mismatch:\n  eager={eager}\n  cg={cg}"


@need_2gpu
@pytest.mark.skipif(not TEXT.is_dir(), reason="Qwen3-0.6B missing")
def test_text_pp2_cuda_graph_prefix():
    common = dict(
        model=str(TEXT),
        max_num_seqs=4,
        context_len=512,
        num_kvcache_blocks=32,
        tp_size=1,
        pp_size=2,
        device_ids=[0, 1],
        nccl_port=0,
    )
    prompt = "Hello, my name is"
    eager = _gen(EngineConfig(**common, enforce_eager=True), prompt, n=6)
    cg = _gen(EngineConfig(**common, enforce_eager=False), prompt, n=6)
    assert cg[:4] == eager[:4], f"PP CG prefix mismatch:\n  eager={eager}\n  cg={cg}"


@need_2gpu
@pytest.mark.skipif(not VL.is_dir(), reason="Qwen3-VL-2B missing")
def test_vl_tp2_cuda_graph_runs():
    img = Image.new("RGB", (64, 64), color=(32, 128, 200))
    cfg = EngineConfig(
        model=str(VL),
        enforce_eager=False,
        max_num_seqs=2,
        context_len=1024,
        num_kvcache_blocks=64,
        tp_size=2,
        pp_size=1,
        device_ids=[0, 1],
        nccl_port=0,
        gpu_memory_utilization=0.85,
        enable_vit_cuda_graph=False,
    )
    # Tiny image + short prompt; just require a successful greedy decode.
    out = _gen(cfg, "Describe the image briefly.", n=4, images=[img])
    assert len(out) == 4
