"""Unit coverage for MM text-prefix KV, prefill RPC wire, OpenAI multimodal parse."""

from __future__ import annotations

import base64
import io
import pickle

import torch

from src.entrypoints.openai import ChatMessage, parse_chat_messages
from src.entrypoints.openai.serving import OpenAIServing, sampling_from_request
from src.entrypoints.openai.protocol import CompletionRequest
from src.runtime.execute.request import (
    PrefillSequenceWire,
    collect_mm_pixels,
    compact_execute_sequences,
)
from src.sampling import SamplingParams
from src.scheduling.sequence import MultiModalFeature, Sequence


def _mf(start: int, uid: str = "u1", n: int = 4) -> MultiModalFeature:
    return MultiModalFeature(
        modality="image",
        data=torch.randn(8, 3),
        start=start,
        end=start + n,
        num_encoder_tokens=n,
        uid=uid,
        grid_thw=torch.tensor([[1, 4, 4]]),
    )


def test_prefill_wire_excludes_pixels_and_is_smaller():
    mf = _mf(10)
    seq = Sequence(
        [1] * 20,
        SamplingParams(max_tokens=4),
        mm_features=[mf],
        block_size=8,
    )
    seq.num_scheduled_tokens = 16
    seq.block_table = [0, 1, 2]

    fat = pickle.dumps(seq)
    wire = compact_execute_sequences([seq], is_prefill=True)[0]
    assert isinstance(wire, PrefillSequenceWire)
    assert wire.mm_features[0].data is None
    assert wire.mm_features[0].uid == mf.uid
    thin = pickle.dumps(wire)
    assert len(thin) < len(fat)

    pixels = collect_mm_pixels([seq])
    assert mf.uid in pixels
    assert pixels[mf.uid].shape == mf.data.shape


def test_openai_multipart_image_and_text():
    try:
        from PIL import Image
    except ImportError:
        return  # optional in minimal envs

    img = Image.new("RGB", (8, 8), color=(20, 40, 60))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    data_url = f"data:image/png;base64,{b64}"

    messages = [
        ChatMessage(
            role="user",
            content=[
                {"type": "text", "text": "describe"},
                {"type": "image_url", "image_url": {"url": data_url}},
            ],
        )
    ]
    parsed = parse_chat_messages(messages)
    assert "describe" in parsed.text_fallback
    assert "<image>" in parsed.text_fallback
    assert len(parsed.images) == 1
    assert parsed.images[0].size == (8, 8)
    # HF-style parts for chat template
    assert parsed.hf_messages[0]["role"] == "user"
    parts = parsed.hf_messages[0]["content"]
    assert isinstance(parts, list)
    assert any(p.get("type") == "image" for p in parts)
    assert any(p.get("type") == "text" and "describe" in p.get("text", "") for p in parts)


def test_sampling_from_request():
    req = CompletionRequest(model="m", prompt="hi", max_tokens=8, temperature=0.5)
    sp = sampling_from_request(req)
    assert sp.max_tokens == 8
    assert sp.temperature == 0.5


def test_openai_serving_list_models_no_engine_loop():
    """OpenAIServing.list_models is sync and does not need a live GPU engine."""

    class _FakeLLM:
        config = type("C", (), {"model": "demo-model"})()

    serving = OpenAIServing(_FakeLLM(), model_name="demo-model")  # type: ignore[arg-type]
    listing = serving.list_models()
    assert listing.data[0].id == "demo-model"


def test_prefix_cacheable_full_blocks_math():
    seq = Sequence(
        [1] * 24,
        SamplingParams(max_tokens=2),
        mm_features=[_mf(10)],
        block_size=8,
    )
    assert seq.prefix_cacheable_full_blocks() == 1
    seq2 = Sequence(
        [1] * 24,
        SamplingParams(max_tokens=2),
        mm_features=[_mf(0)],
        block_size=8,
    )
    assert seq2.prefix_cacheable_full_blocks() == 0
