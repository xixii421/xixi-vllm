import numpy as np
import torch


def mrope_positions(
    input_ids: torch.Tensor,
    seqs: list,
    merge_size: int,
    device: torch.device,
) -> torch.Tensor:
    n = int(input_ids.shape[0])
    pos = np.zeros((3, n), dtype=np.int64)
    off = 0
    for seq in seqs:
        base = int(seq.mrope_next) if seq.mrope_next is not None else int(seq.num_cached_tokens)
        for i in range(seq.num_scheduled_tokens):
            pos[:, off + i] = base + i

        chunk_start = seq.num_cached_tokens
        chunk_end = chunk_start + seq.num_scheduled_tokens

        for mf in seq.mm_features:
            g = getattr(mf, "grid_thw", None)
            if g is None:
                continue
            img_start = mf.start
            img_end = mf.start + mf.num_encoder_tokens
            ov_start = max(img_start, chunk_start)
            ov_end = min(img_end, chunk_end)
            if ov_start >= ov_end:
                continue

            t, h, w = g[0].tolist() if g.ndim == 2 else g.tolist()
            gh, gw = h // merge_size, w // merge_size
            grid = np.indices((t, gh, gw), dtype=np.int64).reshape(3, -1)
            vis_off = ov_start - img_start
            an = ov_end - ov_start
            local = off + (ov_start - chunk_start)

            if ov_start == img_start:
                origin = int(pos[0, local])
                mf.mrope_origin = origin
            else:
                origin = int(
                    mf.mrope_origin
                    if mf.mrope_origin is not None
                    else int(pos[0, local]) - int(grid[0, vis_off])
                )

            pos[:, local : local + an] = grid[:, vis_off : vis_off + an] + origin

            te = local + an
            if ov_end == img_end and te < off + seq.num_scheduled_tokens:
                nxt = int(pos[:, local:te].max()) + 1
                for j in range(te - off, seq.num_scheduled_tokens):
                    pos[:, off + j] = nxt + (j - (te - off))

        if seq.num_scheduled_tokens > 0:
            abs_end = off + seq.num_scheduled_tokens - 1
            token_idx = seq.num_cached_tokens + seq.num_scheduled_tokens - 1
            seq.mrope_next = int(pos[:, abs_end].max()) + 1
            seq.rope_delta = seq.mrope_next - 1 - token_idx

        off += seq.num_scheduled_tokens
    # CPU max — callers use this to grow MRoPE cache without GPU .item() sync.
    max_pos = int(pos.max()) if n > 0 else 0
    out = torch.from_numpy(pos).to(device)
    out.max_pos = max_pos  # type: ignore[attr-defined]
    return out
