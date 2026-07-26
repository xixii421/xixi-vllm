"""Functional tests for MultiModalProcessor (per-image cache, not request cache)."""

from __future__ import annotations

import os
from pathlib import Path

import pytest
import torch
from PIL import Image

from src.multimodal.hasher import hash_pil_image
from src.multimodal.processor import MultiModalInputs, MultiModalProcessor

MODEL = os.environ.get(
    "VLLM_VX_MM_MODEL", "/mnt/nfs/zhangli/huggingface/Qwen3-VL-8B-Instruct"
)
CAT = Path(__file__).resolve().parents[1] / "cat.jpg"


def _have_model() -> bool:
    return Path(MODEL).is_dir() and CAT.is_file()


pytestmark = pytest.mark.skipif(not _have_model(), reason="Qwen3-VL model / cat.jpg missing")


@pytest.fixture(scope="module")
def proc() -> MultiModalProcessor:
    p = MultiModalProcessor(MODEL, image_cache_size=8)
    return p


@pytest.fixture
def cat() -> Image.Image:
    return Image.open(CAT).convert("RGB")


def test_hash_stable_and_content_sensitive(cat: Image.Image):
    a = hash_pil_image(cat)
    b = hash_pil_image(cat.copy())
    assert a == b
    other = Image.new("RGB", cat.size, color=(12, 34, 56))
    assert hash_pil_image(other) != a


def test_process_returns_structured_inputs(proc: MultiModalProcessor, cat: Image.Image):
    proc.clear_cache()
    out = proc.process("描述这张图", [cat])
    assert isinstance(out, MultiModalInputs)
    assert out.prompt_token_ids
    assert out.pixel_values is not None and out.pixel_values.ndim >= 2
    assert out.image_grid_thw is not None
    assert len(out.mm_hashes) == 1
    assert len(out.mm_placeholders) == 1
    ph = out.mm_placeholders[0]
    assert ph.length > 0
    assert out.prompt_token_ids[ph.offset : ph.offset + ph.length]
    assert not out.pixel_values.is_cuda
    pinned = out.pixel_values.is_pinned
    assert not (pinned() if callable(pinned) else pinned)


def test_same_image_different_text_reuses_image_cache(
    proc: MultiModalProcessor, cat: Image.Image
):
    proc.clear_cache()
    a = proc.process("描述这张图", [cat])
    assert proc.image_cache_len == 1
    pv_a = a.pixel_values.clone()

    b = proc.process("这张图里有几只猫？", [cat])
    assert proc.image_cache_len == 1  # still one image entry
    assert torch.equal(pv_a, b.pixel_values)
    assert torch.equal(a.image_grid_thw, b.image_grid_thw)
    # Prompt tokens must differ when the user text differs.
    assert a.prompt_token_ids != b.prompt_token_ids
    # Placeholder length (vision tokens) stays the same for the same image.
    assert a.mm_placeholders[0].length == b.mm_placeholders[0].length


def test_cache_hit_matches_full_hf_pixels(proc: MultiModalProcessor, cat: Image.Image):
    """Cached pixels must match a fresh HF image_processor call."""
    proc.clear_cache()
    out = proc.process("prompt-A", [cat])
    ref = proc.p.image_processor(images=[cat], return_tensors="pt")
    assert torch.equal(out.pixel_values, ref["pixel_values"].cpu())
    assert torch.equal(out.image_grid_thw, ref["image_grid_thw"].cpu())

    # Second prompt: image cache hit, pixels unchanged.
    out2 = proc.process("prompt-B-different", [cat])
    assert proc.image_cache_len == 1
    assert torch.equal(out.pixel_values, out2.pixel_values)


def test_matches_full_hf_processor_ids(proc: MultiModalProcessor, cat: Image.Image):
    """Our expand+tokenize path must match HF AutoProcessor(text, images)."""
    proc.clear_cache()
    text = "完整对齐检查"
    ours = proc.process(text, [cat])
    msgs = [
        {
            "role": "user",
            "content": [{"type": "image", "image": cat}, {"type": "text", "text": text}],
        }
    ]
    prompt = proc.p.apply_chat_template(
        msgs, tokenize=False, add_generation_prompt=True
    )
    ref = proc.p(text=prompt, images=[cat], return_tensors="pt")
    assert ours.prompt_token_ids == ref["input_ids"][0].tolist()
    assert torch.equal(ours.pixel_values, ref["pixel_values"].cpu())


def test_warm_ids_match_cold_for_same_text(proc: MultiModalProcessor, cat: Image.Image):
    proc.clear_cache()
    cold = proc.process("同一句提示词", [cat])
    warm = proc.process("同一句提示词", [cat])
    assert cold.prompt_token_ids == warm.prompt_token_ids
    assert cold.mm_placeholders[0].offset == warm.mm_placeholders[0].offset
    assert cold.mm_placeholders[0].length == warm.mm_placeholders[0].length


def test_two_images_independent_cache(proc: MultiModalProcessor, cat: Image.Image):
    proc.clear_cache()
    other = Image.new("RGB", (64, 64), color=(200, 10, 10))
    out = proc.process("两张图", [cat, other])
    assert len(out.mm_hashes) == 2
    assert len(out.mm_placeholders) == 2
    assert out.mm_hashes[0] != out.mm_hashes[1]
    assert proc.image_cache_len == 2

    # Second request: only cat — still one cache entry reused, other remains.
    out2 = proc.process("只要猫", [cat])
    assert proc.image_cache_len == 2
    assert out2.mm_hashes[0] == out.mm_hashes[0]
