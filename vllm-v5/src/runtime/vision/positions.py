"""3D mRoPE position construction for Qwen3-VL."""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import torch

from ..input.features import overlaps_window, span_end

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


def compute_mrope_positions(
    input_ids: torch.Tensor,
    seqs: list,
    merge_size: int,
    device: torch.device,
) -> tuple[torch.Tensor, list[MRopeStateUpdate]]:
    """Build ``positions[3, T]`` and return per-seq mRoPE state updates.

    Does **not** mutate ``seqs``; call :func:`apply_mrope_updates`.
    """
    n = int(input_ids.shape[0])
    pos = np.zeros((3, n), dtype=np.int64)
    updates: list[MRopeStateUpdate] = []
    off = 0

    for seq in seqs:
        base = (
            int(seq.mrope_next)
            if seq.mrope_next is not None
            else int(seq.num_cached_tokens)
        )
        n_tok = int(seq.num_scheduled_tokens)
        if n_tok > 0:
            pos[:, off : off + n_tok] = base + np.arange(n_tok, dtype=np.int64)

        chunk_start = seq.num_cached_tokens
        chunk_end = chunk_start + seq.num_scheduled_tokens
        origins: list[int | None] = []

        for mf in seq.mm_features:
            g = getattr(mf, "grid_thw", None)
            if g is None or not overlaps_window(mf, chunk_start, chunk_end):
                origins.append(mf.mrope_origin)
                continue

            img_start = mf.start
            img_end = span_end(mf)
            ov_start = max(img_start, chunk_start)
            ov_end = min(img_end, chunk_end)
            if isinstance(g, torch.Tensor):
                g_host = g.detach().to("cpu")
                t, h, w = (
                    g_host[0].tolist() if g_host.ndim == 2 else g_host.tolist()
                )
            else:
                t, h, w = g[0].tolist() if hasattr(g[0], "tolist") else g
            gh, gw = h // merge_size, w // merge_size
            grid = np.indices((t, gh, gw), dtype=np.int64).reshape(3, -1)
            vis_off = ov_start - img_start
            an = ov_end - ov_start
            local = off + (ov_start - chunk_start)

            if ov_start == img_start:
                origin = int(pos[0, local])
            else:
                origin = int(
                    mf.mrope_origin
                    if mf.mrope_origin is not None
                    else int(pos[0, local]) - int(grid[0, vis_off])
                )
            origins.append(origin)
            pos[:, local : local + an] = grid[:, vis_off : vis_off + an] + origin

            te = local + an
            if ov_end == img_end and te < off + seq.num_scheduled_tokens:
                nxt = int(pos[:, local:te].max()) + 1
                rem = seq.num_scheduled_tokens - (te - off)
                if rem > 0:
                    pos[:, te : off + seq.num_scheduled_tokens] = nxt + np.arange(
                        rem, dtype=np.int64
                    )

        if n_tok > 0:
            abs_end = off + n_tok - 1
            token_idx = seq.num_cached_tokens + n_tok - 1
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
        off += n_tok

    return torch.from_numpy(pos).to(device, non_blocking=True), updates


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
    """Backward-compatible helper that applies updates onto ``seqs``."""
    pos, updates = compute_mrope_positions(input_ids, seqs, merge_size, device)
    apply_mrope_updates(seqs, updates)
    return pos
