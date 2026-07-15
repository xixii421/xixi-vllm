"""Multi-image + concurrent VL correctness (HF ↔ v5).

Checks:
  1. Single request with 2 images — greedy text match vs HF
  2. Concurrent batch: 2 requests × 1 image each — per-req match vs HF
  3. Engine MRoPE remaps correctly for multi spans (positions exact)

    CUDA_VISIBLE_DEVICES=1 PYTHONPATH=. python -m pytest tests/gpu/test_vl_multi_image.py -q
"""

from __future__ import annotations

import pytest
import torch
from PIL import Image, ImageDraw


def _img(color: tuple[int, int, int], size: int = 128) -> Image.Image:
    im = Image.new("RGB", (size, size), color)
    d = ImageDraw.Draw(im)
    d.rectangle([16, 16, size - 16, size - 16], fill=(255, 255, 255))
    return im


@pytest.fixture(scope="module")
def model_path():
    from pathlib import Path

    p = Path.home() / "huggingface/Qwen3-VL-2B-Instruct"
    if not p.is_dir():
        pytest.skip(f"missing {p}")
    return str(p)


@pytest.fixture(scope="module")
def cuda():
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    return torch.device("cuda:0")


@torch.inference_mode()
def test_multi_image_single_request_matches_hf(model_path, cuda):
    from transformers import AutoProcessor, AutoModelForImageTextToText

    from src.config import EngineConfig
    from src.runtime.engine import Engine

    imgs = [_img((30, 80, 220)), _img((220, 40, 40))]
    prompt = "List the main color of each image in order. Answer briefly."

    proc = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    msgs = [
        {
            "role": "user",
            "content": [{"type": "image", "image": im} for im in imgs]
            + [{"type": "text", "text": prompt}],
        }
    ]
    chat = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    batch = proc(text=chat, images=imgs, return_tensors="pt")
    ids = batch["input_ids"].to(cuda)
    hf_kwargs = {k: v.to(cuda) if hasattr(v, "to") else v for k, v in batch.items()}

    hf = (
        AutoModelForImageTextToText.from_pretrained(
            model_path, dtype=torch.float16, trust_remote_code=True
        )
        .to(cuda)
        .eval()
    )
    pos_hf = hf.model.compute_3d_position_ids(
        input_ids=ids,
        image_grid_thw=batch["image_grid_thw"].to(cuda),
        video_grid_thw=None,
        inputs_embeds=None,
        mm_token_type_ids=batch.get("mm_token_type_ids"),
        attention_mask=None,
        past_key_values=None,
    )[:, 0, :].cpu()
    hf_gen = hf.generate(**hf_kwargs, max_new_tokens=24, do_sample=False)
    hf_text = proc.batch_decode(hf_gen[:, ids.shape[1] :], skip_special_tokens=False)[0]
    del hf
    torch.cuda.empty_cache()

    eng = Engine(
        EngineConfig(
            model=model_path,
            dtype="float16",
            enforce_eager=True,
            enable_vit_cuda_graph=False,
            gpu_memory_utilization=0.55,
            max_num_seqs=2,
            max_num_batched_tokens=16384,
        )
    )

    from src.sampling_params import SamplingParams
    from src.runtime.model_inputs import build_model_inputs

    peek = eng.add_request(prompt, SamplingParams(max_tokens=1, temperature=0.0), images=imgs)
    assert len(peek.mm_features) == 2
    seqs, is_pf = eng.scheduler.schedule()
    assert is_pf and seqs
    inputs = build_model_inputs(seqs, True, eng.model_runner.block_size, eng.device)
    _, _, pos = eng.model.encode_and_merge(inputs.input_ids, seqs, eng.device)
    assert torch.equal(pos.cpu().long(), pos_hf.long()), (
        f"MRoPE mismatch max={(pos.cpu().long() - pos_hf.long()).abs().max()}"
    )
    # Drain peek request so the queue is clean for generate.
    while not eng.scheduler.is_finished():
        eng.step()

    text = "".join(
        list(eng.generate(prompt, max_new_tokens=24, temperature=0.0, images=imgs))
    )

    def _norm(s: str) -> str:
        return s.replace("<|im_end|>", "").replace("<|endoftext|>", "").strip()

    assert _norm(text) == _norm(hf_text), f"v5={text!r} hf={hf_text!r}"


@torch.inference_mode()
def test_concurrent_two_requests_match_hf(model_path, cuda):
    from transformers import AutoProcessor, AutoModelForImageTextToText

    from src.config import EngineConfig
    from src.runtime.engine import Engine
    from src.sampling_params import SamplingParams

    prompts = [
        "What is the dominant color? One word.",
        "What is the dominant color? One word.",
    ]
    imgs_batch = [[_img((20, 40, 200))], [_img((200, 30, 30))]]

    proc = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    hf = (
        AutoModelForImageTextToText.from_pretrained(
            model_path, dtype=torch.float16, trust_remote_code=True
        )
        .to(cuda)
        .eval()
    )
    hf_texts = []
    for prompt, imgs in zip(prompts, imgs_batch):
        msgs = [
            {
                "role": "user",
                "content": [{"type": "image", "image": imgs[0]}, {"type": "text", "text": prompt}],
            }
        ]
        chat = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        batch = proc(text=chat, images=imgs, return_tensors="pt")
        ids = batch["input_ids"].to(cuda)
        kw = {k: v.to(cuda) if hasattr(v, "to") else v for k, v in batch.items()}
        gen = hf.generate(**kw, max_new_tokens=8, do_sample=False)
        hf_texts.append(
            proc.batch_decode(gen[:, ids.shape[1] :], skip_special_tokens=False)[0]
        )
    del hf
    torch.cuda.empty_cache()

    eng = Engine(
        EngineConfig(
            model=model_path,
            dtype="float16",
            enforce_eager=True,
            enable_vit_cuda_graph=False,
            gpu_memory_utilization=0.55,
            max_num_seqs=4,
            max_num_batched_tokens=16384,
        )
    )
    outs = eng.generate_batch(
        prompts,
        SamplingParams(max_tokens=8, temperature=0.0),
        images=imgs_batch,
    )

    def _norm(s: str) -> str:
        return s.replace("<|im_end|>", "").replace("<|endoftext|>", "").strip()

    for i, (a, b) in enumerate(zip(outs, hf_texts)):
        assert _norm(a) == _norm(b), f"req[{i}] v5={a!r} hf={b!r}"


def test_truncate_left_remaps_mm_spans():
    from src.runtime.engine import Engine
    from src.scheduling.sequence import MultiModalFeature
    import torch

    mf0 = MultiModalFeature(
        "image", torch.zeros(2, 2), start=10, end=20, num_encoder_tokens=10, uid="a"
    )
    mf1 = MultiModalFeature(
        "image", torch.zeros(2, 2), start=90, end=100, num_encoder_tokens=10, uid="b"
    )
    tokens = list(range(120))
    kept_tok, kept_mm = Engine._truncate_left(tokens, [mf0, mf1], context_len=40)
    assert kept_tok == tokens[-40:]
    # mf0 starts at 10 < keep_from=80 → dropped; mf1 remapped 90-80=10
    assert len(kept_mm) == 1
    assert kept_mm[0].uid == "b"
    assert kept_mm[0].start == 10 and kept_mm[0].end == 20
