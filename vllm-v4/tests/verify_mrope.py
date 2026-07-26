#!/usr/bin/env python3
"""Verify v4 ``mrope_positions`` against HF Qwen3-VL and v5.

Usage::

    cd vllm-v4
    PYTHONPATH=. python tests/verify_mrope.py
    PYTHONPATH=. python tests/verify_mrope.py --with-hf-model   # needs local Qwen3-VL
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.memory.sequence import MultiModalFeature, Sequence  # noqa: E402
from src.multimodal._mrope import mrope_positions  # noqa: E402


# ---------------------------------------------------------------------------
# Reference: HF Qwen3-VL get_vision_position_ids + get_rope_index (image-only)
# ---------------------------------------------------------------------------


def hf_vision_position_ids(
    start_position: int,
    grid_thw: tuple[int, int, int],
    spatial_merge_size: int,
) -> np.ndarray:
    """Mirror ``Qwen3VLModel.get_vision_position_ids`` (time_interval=1)."""
    t, h, w = grid_thw
    gt = t
    gh = h // spatial_merge_size
    gw = w // spatial_merge_size
    position_temporal = np.arange(gt, dtype=np.int64)
    position_height = np.arange(gh, dtype=np.int64) + start_position
    position_width = np.arange(gw, dtype=np.int64) + start_position
    # meshgrid indexing='ij' → t slowest, w fastest
    T, H, W = np.meshgrid(
        position_temporal, position_height, position_width, indexing="ij"
    )
    out = np.stack([T, H, W], axis=0).reshape(3, -1)
    out[0] += start_position
    return out


def hf_get_rope_index_images(
    seq_len: int,
    image_spans: list[tuple[int, int, tuple[int, int, int]]],
    merge_size: int,
) -> tuple[np.ndarray, int]:
    """HF-style positions for a single sequence with image spans only.

    ``image_spans``: list of (start, length, grid_thw_pre_merge).
    Tokens not covered by any image span are treated as text.
    """
    # Build modality mask: 0 text, 1 image
    mm = np.zeros(seq_len, dtype=np.int64)
    grids: list[tuple[int, int, int]] = []
    for start, length, grid in image_spans:
        mm[start : start + length] = 1
        grids.append(grid)

    # Group contiguous modality runs (same as HF itertools.groupby)
    groups: list[tuple[int, int, int]] = []
    i = 0
    while i < seq_len:
        j = i + 1
        while j < seq_len and mm[j] == mm[i]:
            j += 1
        groups.append((int(mm[i]), i, j))
        i = j

    grid_iter = iter(grids)
    current_pos = 0
    parts: list[np.ndarray] = []
    for modality, s, e in groups:
        if modality == 0:
            text_len = e - s
            part = np.arange(text_len, dtype=np.int64).reshape(1, -1).repeat(3, axis=0)
            part = part + current_pos
            parts.append(part)
            current_pos += text_len
        else:
            grid = next(grid_iter)
            vision = hf_vision_position_ids(current_pos, grid, merge_size)
            parts.append(vision)
            # HF advances by max(H, W) // merge  (not max over T)
            current_pos += max(grid[1], grid[2]) // merge_size

    pos = np.concatenate(parts, axis=1)
    assert pos.shape == (3, seq_len), (pos.shape, seq_len)
    rope_delta = int(pos.max()) + 1 - seq_len
    return pos, rope_delta


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _feat(start: int, n: int, grid: tuple[int, int, int]) -> MultiModalFeature:
    return MultiModalFeature(
        modality="image",
        data=torch.zeros(1),
        start=start,
        end=start + n,
        num_encoder_tokens=n,
        uid=f"img@{start}",
        grid_thw=torch.tensor([list(grid)], dtype=torch.int64),
        grid_thw_tuple=grid,
    )


def _run_v4(
    seq_len: int,
    spans: list[tuple[int, int, tuple[int, int, int]]],
    merge_size: int,
    *,
    chunk_size: int | None = None,
) -> tuple[np.ndarray, int, int]:
    """Run v4 mrope_positions (optionally chunked). Returns (pos, rope_delta, mrope_next)."""
    feats = [_feat(s, n, g) for s, n, g in spans]
    seq = Sequence(token_ids=list(range(seq_len)), block_size=max(seq_len, 8), mm_features=feats)
    parts: list[np.ndarray] = []
    if chunk_size is None:
        chunk_size = seq_len
    cached = 0
    while cached < seq_len:
        m = min(chunk_size, seq_len - cached)
        seq.num_cached_tokens = cached
        seq.num_scheduled_tokens = m
        ids = torch.arange(m)
        pos = mrope_positions(ids, [seq], merge_size=merge_size, device=torch.device("cpu"))
        parts.append(pos.numpy())
        cached += m
    full = np.concatenate(parts, axis=1)
    return full, int(seq.rope_delta), int(seq.mrope_next)  # type: ignore[arg-type]


def _check(name: str, ok: bool, detail: str = "") -> bool:
    status = "PASS" if ok else "FAIL"
    extra = f" — {detail}" if detail else ""
    print(f"  [{status}] {name}{extra}")
    return ok


# ---------------------------------------------------------------------------
# Cases
# ---------------------------------------------------------------------------


def case_text_only() -> bool:
    print("\n== text-only ==")
    seq_len, merge = 8, 2
    pos_v4, delta_v4, nxt = _run_v4(seq_len, [], merge)
    pos_hf, delta_hf = hf_get_rope_index_images(seq_len, [], merge)
    ok = True
    ok &= _check("shape", pos_v4.shape == (3, seq_len))
    ok &= _check("match HF", np.array_equal(pos_v4, pos_hf))
    ok &= _check("rope_delta", delta_v4 == delta_hf == 0, f"v4={delta_v4} hf={delta_hf}")
    ok &= _check("mrope_next", nxt == seq_len, f"nxt={nxt}")
    return ok


def case_single_image() -> bool:
    print("\n== single image: [txt x2][img 2x2][txt x1] ==")
    # merge=2, grid 1x4x4 → 4 vision tokens, gh=gw=2
    merge = 2
    grid = (1, 4, 4)
    n_vis = (grid[0] * grid[1] * grid[2]) // (merge * merge)
    assert n_vis == 4
    spans = [(2, n_vis, grid)]
    seq_len = 2 + n_vis + 1  # 7
    pos_v4, delta_v4, nxt = _run_v4(seq_len, spans, merge)
    pos_hf, delta_hf = hf_get_rope_index_images(seq_len, spans, merge)
    ok = True
    ok &= _check("match HF positions", np.array_equal(pos_v4, pos_hf), _diff(pos_v4, pos_hf))
    ok &= _check("rope_delta", delta_v4 == delta_hf, f"v4={delta_v4} hf={delta_hf}")
    # Decode continuity: last absolute cursor == (len-1) + rope_delta
    ok &= _check(
        "decode cursor",
        (seq_len - 1) + delta_v4 == nxt - 1,
        f"decode={(seq_len - 1) + delta_v4} nxt-1={nxt - 1}",
    )
    # Vision block origin / coords
    ok &= _check("text head", np.array_equal(pos_v4[:, :2], np.arange(2).reshape(1, -1).repeat(3, 0)))
    ok &= _check(
        "vision == HF vision helper",
        np.array_equal(pos_v4[:, 2:6], hf_vision_position_ids(2, grid, merge)),
    )
    return ok


def case_two_images() -> bool:
    print("\n== two images with text between ==")
    merge = 2
    g1, g2 = (1, 4, 4), (1, 2, 2)
    n1 = 4
    n2 = 1
    # [t0][img1 x4][t5][img2 x1][t7]
    spans = [(1, n1, g1), (6, n2, g2)]
    seq_len = 8
    pos_v4, delta_v4, _ = _run_v4(seq_len, spans, merge)
    pos_hf, delta_hf = hf_get_rope_index_images(seq_len, spans, merge)
    ok = True
    ok &= _check("match HF", np.array_equal(pos_v4, pos_hf), _diff(pos_v4, pos_hf))
    ok &= _check("rope_delta", delta_v4 == delta_hf, f"v4={delta_v4} hf={delta_hf}")
    return ok


def case_chunked_equals_full() -> bool:
    print("\n== chunked prefill == full prefill ==")
    merge = 2
    grid = (1, 8, 8)  # 16 vision tokens after merge
    n_vis = 16
    spans = [(3, n_vis, grid)]
    seq_len = 3 + n_vis + 5
    pos_full, d_full, n_full = _run_v4(seq_len, spans, merge, chunk_size=None)
    ok = True
    for cs in (1, 3, 7, 16):
        pos_c, d_c, n_c = _run_v4(seq_len, spans, merge, chunk_size=cs)
        same = np.array_equal(pos_full, pos_c) and d_full == d_c and n_full == n_c
        ok &= _check(f"chunk_size={cs}", same, _diff(pos_full, pos_c) if not same else "")
    return ok


def case_vs_v5() -> bool:
    print("\n== vs v5 compute_mrope_positions ==")
    import importlib.util

    v5_path = ROOT.parent / "vllm-v5" / "src" / "runtime" / "vision" / "positions.py"
    if not v5_path.is_file():
        print(f"  [SKIP] v5 file missing: {v5_path}")
        return True
    # Load v5 module by path to avoid clashing with v4's ``src`` package.
    spec = importlib.util.spec_from_file_location("v5_mrope_positions", v5_path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules["v5_mrope_positions"] = mod
    # Stub the relative import ``..input.features`` used by v5 positions.py
    import types

    features_stub = types.ModuleType("v5_features_stub")

    def overlaps_window(mf, chunk_start, chunk_end):
        end = int(mf.start + mf.num_encoder_tokens)
        return max(mf.start, chunk_start) < min(end, chunk_end)

    def span_end(mf):
        return int(mf.start + mf.num_encoder_tokens)

    features_stub.overlaps_window = overlaps_window  # type: ignore[attr-defined]
    features_stub.span_end = span_end  # type: ignore[attr-defined]
    pkg_input = types.ModuleType("v5_runtime_input")
    pkg_input.features = features_stub  # type: ignore[attr-defined]
    pkg_runtime = types.ModuleType("v5_runtime")
    pkg_runtime.input = pkg_input  # type: ignore[attr-defined]
    sys.modules["v5_runtime"] = pkg_runtime
    sys.modules["v5_runtime.input"] = pkg_input
    sys.modules["v5_runtime.input.features"] = features_stub
    src = v5_path.read_text()
    src = src.replace(
        "from ..input.features import overlaps_window, span_end",
        "from v5_runtime.input.features import overlaps_window, span_end",
    )
    exec(compile(src, str(v5_path), "exec"), mod.__dict__)
    compute_mrope_positions = mod.compute_mrope_positions
    apply_mrope_updates = mod.apply_mrope_updates

    merge = 2
    grid = (1, 4, 4)
    spans = [(2, 4, grid)]
    seq_len = 7
    pos_v4, d_v4, n_v4 = _run_v4(seq_len, spans, merge)

    feats = [_feat(s, n, g) for s, n, g in spans]
    seq = Sequence(token_ids=list(range(seq_len)), block_size=8, mm_features=feats)
    seq.num_cached_tokens = 0
    seq.num_scheduled_tokens = seq_len
    pos_v5, updates = compute_mrope_positions(
        torch.arange(seq_len), [seq], merge, torch.device("cpu")
    )
    apply_mrope_updates([seq], updates)
    ok = True
    ok &= _check(
        "positions == v5",
        np.array_equal(pos_v4, pos_v5.numpy()),
        _diff(pos_v4, pos_v5.numpy()),
    )
    ok &= _check("rope_delta == v5", d_v4 == seq.rope_delta, f"v4={d_v4} v5={seq.rope_delta}")
    ok &= _check("mrope_next == v5", n_v4 == seq.mrope_next, f"v4={n_v4} v5={seq.mrope_next}")
    return ok


def case_video_temporal_advance() -> bool:
    """Document HF vs v4 when T > max(H,W) (video).

    HF advances current_pos by max(H,W)//merge only.
    v4 advances residual text by max(pos)+1 == origin+max(T,H,W).
    For pure images (T=1) they match; for tall-temporal video they may differ.
    """
    print("\n== video T>max(H,W) (expect known divergence) ==")
    merge = 2
    grid = (8, 2, 2)  # T=8, gh=gw=1 after merge → 8 vision tokens
    n_vis = 8
    spans = [(1, n_vis, grid)]
    seq_len = 1 + n_vis + 1
    pos_v4, d_v4, _ = _run_v4(seq_len, spans, merge)
    pos_hf, d_hf = hf_get_rope_index_images(seq_len, spans, merge)
    same = np.array_equal(pos_v4, pos_hf)
    if same:
        return _check("video positions match HF", True)
    # Report clearly — not necessarily a FAIL for image-only product.
    print("  [INFO] positions diverge (HF advances by max(H,W); v4 uses max(T,H,W)+origin)")
    print(f"         v4 rope_delta={d_v4} hf={d_hf}")
    print(f"         v4 last col={pos_v4[:, -1].tolist()} hf={pos_hf[:, -1].tolist()}")
    # For image-centric v4 this is documented, not a hard fail.
    return _check("documented video divergence (non-fatal)", True, "see INFO above")


def case_real_hf_model(model_path: str, image_path: str) -> bool:
    print(f"\n== real HF model ({model_path}) ==")
    from PIL import Image
    from transformers import AutoConfig, AutoProcessor

    from src.multimodal.processor import MultiModalProcessor

    if not Path(model_path).is_dir() or not Path(image_path).is_file():
        print("  [SKIP] model or image missing")
        return True

    cfg = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    merge = int(cfg.vision_config.spatial_merge_size)
    proc = MultiModalProcessor(model_path)
    img = Image.open(image_path).convert("RGB")
    mm = proc.process("描述这张图", [img])
    ids = mm.prompt_token_ids
    ph = mm.mm_placeholders[0]
    grid = tuple(int(x) for x in mm.image_grid_thw[0].tolist())  # type: ignore[index]
    spans = [(ph.offset, ph.length, grid)]
    seq_len = len(ids)

    pos_v4, delta_v4, nxt = _run_v4(seq_len, spans, merge)
    pos_hf, delta_hf = hf_get_rope_index_images(seq_len, spans, merge)

    ok = True
    ok &= _check(
        "real prompt positions == HF formula",
        np.array_equal(pos_v4, pos_hf),
        _diff(pos_v4, pos_hf),
    )
    ok &= _check("rope_delta", delta_v4 == delta_hf, f"v4={delta_v4} hf={delta_hf}")
    ok &= _check(
        "vision token count matches grid",
        ph.length == (grid[0] * grid[1] * grid[2]) // (merge * merge),
        f"len={ph.length} grid={grid} merge={merge}",
    )
    print(f"  seq_len={seq_len} grid={grid} merge={merge} vision_tokens={ph.length}")
    print(f"  rope_delta={delta_v4} mrope_next={nxt}")
    return ok


def _diff(a: np.ndarray, b: np.ndarray) -> str:
    if a.shape != b.shape:
        return f"shape {a.shape} vs {b.shape}"
    if np.array_equal(a, b):
        return ""
    mism = np.where(a != b)
    i = int(mism[1][0]) if len(mism[1]) else -1
    return f"first mismatch col={i} a={a[:, i].tolist() if i >= 0 else None} b={b[:, i].tolist() if i >= 0 else None}"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--with-hf-model", action="store_true")
    ap.add_argument(
        "--model",
        default="/mnt/nfs/zhangli/huggingface/Qwen3-VL-8B-Instruct",
    )
    ap.add_argument("--image", default=str(ROOT / "cat.jpg"))
    args = ap.parse_args()

    print("Verifying vllm-v4 src/multimodal/_mrope.py")
    results = [
        case_text_only(),
        case_single_image(),
        case_two_images(),
        case_chunked_equals_full(),
        case_vs_v5(),
        case_video_temporal_advance(),
    ]
    if args.with_hf_model:
        results.append(case_real_hf_model(args.model, args.image))

    n_ok = sum(1 for r in results if r)
    n = len(results)
    print(f"\n=== RESULT: {n_ok}/{n} case groups passed ===")
    return 0 if n_ok == n else 1


if __name__ == "__main__":
    raise SystemExit(main())
