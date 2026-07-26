"""3D mRoPE position construction for Qwen3-VL (aligned with vllm-v4).

Hot-path choices matching v4:
  * prefer ``grid_thw_tuple`` (no GPU/CPU sync via ``.tolist()``)
  * build only the overlapped vision-token slice (no full ``np.indices``)
  * after an image ends, continue text from ``max(vision_pos) + 1``

v5 keeps compute/apply split so Worker can snapshot updates back to Engine
without mutating Engine-owned ``Sequence`` objects during compute.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import torch

from ..input.features import span_end

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class MRopeStateUpdate:
    """CPU-only mRoPE deltas (worker-local apply + Engine sync)."""

    seq_id: int
    rope_delta: int
    mrope_next: int | None
    mrope_origins: tuple[int | None, ...]


# Backward-compatible alias (tests / older imports).
MRopeSeqUpdate = MRopeStateUpdate


def _grid_slice(gh: int, gw: int, start: int, length: int) -> np.ndarray:
    """Flattened (t, h, w) coords for ``[start, start + length)`` (t slowest, w fastest)."""
    idx = np.arange(start, start + length, dtype=np.int64)
    plane = gh * gw
    out = np.empty((3, length), dtype=np.int64)
    out[0] = idx // plane
    rem = idx % plane
    out[1] = rem // gw
    out[2] = rem % gw
    return out


def _grid_thw_tuple(mf) -> tuple[int, int, int] | None:
    g_tuple = getattr(mf, "grid_thw_tuple", None)
    if g_tuple is not None:
        return int(g_tuple[0]), int(g_tuple[1]), int(g_tuple[2])
    g = getattr(mf, "grid_thw", None)
    if g is None:
        return None
    if isinstance(g, torch.Tensor):
        flat = g[0].tolist() if g.ndim == 2 else g.tolist()
    else:
        flat = g[0].tolist() if hasattr(g[0], "tolist") else g
    return int(flat[0]), int(flat[1]), int(flat[2])


def compute_mrope_positions(
    input_ids: torch.Tensor,
    seqs: list,
    merge_size: int,
    device: torch.device,
) -> tuple[torch.Tensor, list[MRopeStateUpdate]]:
    """Build ``positions[3, T]`` and per-seq mRoPE state updates.

    Does **not** mutate ``seqs``; call :func:`apply_mrope_updates` (or the
    Engine sync path) to commit ``rope_delta`` / ``mrope_next`` / origins.
    """
    total_tokens = int(input_ids.shape[0])
    pos = np.zeros((3, total_tokens), dtype=np.int64)
    updates: list[MRopeStateUpdate] = []
    off = 0

    for seq in seqs:
        m = int(seq.num_scheduled_tokens)
        if m <= 0:
            off += m
            continue

        base = (
            int(seq.mrope_next)
            if seq.mrope_next is not None
            else int(seq.num_cached_tokens)
        )
        pos[:, off : off + m] = base + np.arange(m, dtype=np.int64)

        chunk_start = int(seq.num_cached_tokens)
        chunk_end = chunk_start + m
        origins: list[int | None] = []

        for mf in seq.mm_features:
            g_tuple = _grid_thw_tuple(mf)
            if g_tuple is None:
                origins.append(mf.mrope_origin)
                continue

            img_start = int(mf.start)
            img_end = span_end(mf)
            ov_start = max(img_start, chunk_start)
            ov_end = min(img_end, chunk_end)
            if ov_start >= ov_end:
                origins.append(mf.mrope_origin)
                continue

            _t, h, w = g_tuple
            gh = h // merge_size
            gw = w // merge_size
            plane = gh * gw
            an = ov_end - ov_start
            vis_off = ov_start - img_start
            local = off + (ov_start - chunk_start)

            grid_part = _grid_slice(gh, gw, vis_off, an)

            if ov_start == img_start:
                origin = int(pos[0, local])
            elif mf.mrope_origin is not None:
                origin = int(mf.mrope_origin)
            else:
                # Mid-image chunk without a saved origin: back out from t-coord.
                first_t = vis_off // plane
                origin = int(pos[0, local]) - int(first_t)
            origins.append(origin)

            pos[:, local : local + an] = grid_part + origin

            te = local + an
            if ov_end == img_end and te < off + m:
                nxt = int(pos[:, local:te].max()) + 1
                tail_len = off + m - te
                pos[:, te : off + m] = nxt + np.arange(tail_len, dtype=np.int64)

        abs_end = off + m - 1
        token_idx = chunk_start + m - 1
        mrope_next = int(pos[:, abs_end].max()) + 1
        rope_delta = mrope_next - 1 - token_idx
        updates.append(
            MRopeStateUpdate(
                seq_id=seq.seq_id,
                rope_delta=rope_delta,
                mrope_next=mrope_next,
                mrope_origins=tuple(origins),
            )
        )
        off += m

    out = torch.from_numpy(pos).to(device, non_blocking=True)
    out.max_pos = int(pos.max()) if total_tokens > 0 else 0  # type: ignore[attr-defined]
    return out, updates


def apply_mrope_updates(seqs: list, updates: list[MRopeStateUpdate] | None) -> None:
    """Apply mRoPE deltas onto ``seqs`` (worker copy or Engine sequences)."""
    if not updates:
        return
    by_id = {s.seq_id: s for s in seqs}
    for u in updates:
        seq = by_id.get(u.seq_id)
        if seq is None:
            logger.warning("mRoPE update for unknown seq_id=%s ignored", u.seq_id)
            continue
        if len(u.mrope_origins) != len(seq.mm_features):
            logger.warning(
                "mRoPE origin length mismatch for seq_id=%s (%s vs %s); ignored",
                u.seq_id,
                len(u.mrope_origins),
                len(seq.mm_features),
            )
            continue
        seq.rope_delta = u.rope_delta
        seq.mrope_next = u.mrope_next
        for mf, origin in zip(seq.mm_features, u.mrope_origins):
            mf.mrope_origin = origin


# Backward-compatible name used by vision encode path.
apply_mrope_seq_updates = apply_mrope_updates


def mrope_positions(
    input_ids: torch.Tensor, seqs: list, merge_size: int, device: torch.device
) -> torch.Tensor:
    """Compute positions and apply updates onto ``seqs`` (v4-compatible helper)."""
    pos, updates = compute_mrope_positions(input_ids, seqs, merge_size, device)
    apply_mrope_updates(seqs, updates)
    return pos
