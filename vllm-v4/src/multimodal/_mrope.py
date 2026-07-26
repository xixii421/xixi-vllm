from __future__ import annotations

import numpy as np
import torch

def mrope_positions(
    input_ids: torch.Tensor,
    seqs: list,
    merge_size: int,
    device: torch.device,
) -> torch.Tensor:
    """
    为当前调度 batch 生成 MRoPE 所需的 3D position ids。
    返回:
        shape = [3, total_scheduled_tokens]
        三行分别对应 temporal / height / width 三个维度的位置。
    参数:
        input_ids:  当前 batch 的 input ids。
        seqs:       当前 batch 中的 sequence 列表。
        merge_size: 视觉 token merge 大小。
    """

    total_tokens = int(input_ids.shape[0])

    pos = np.zeros((3, total_tokens), dtype=np.int64)

    off = 0

    def _grid_slice(
        t: int,
        gh: int,
        gw: int,
        start: int,
        length: int,
    ) -> np.ndarray:
        """
        只生成 flatten 后 [start, start + length) 这一段对应的 3D grid 坐标。

        完整 grid 的 flatten 顺序是:
            t 最慢
            h 次之
            w 最快

        对于一维 index:
            plane = gh * gw
            t_coord = index // plane
            rem     = index % plane
            h_coord = rem // gw
            w_coord = rem % gw

        返回:
            shape = [3, length]
        """
        idx = np.arange(start, start + length, dtype=np.int64)

        plane = gh * gw

        out = np.empty((3, length), dtype=np.int64)
        out[0] = idx // plane
        rem = idx % plane
        out[1] = rem // gw
        out[2] = rem % gw

        return out

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

        # 当前 chunk 在完整 sequence 中的 token 范围。
        chunk_start = int(seq.num_cached_tokens)
        chunk_end = chunk_start + m

        # 2. 遍历当前 sequence 上的多模态特征，修正视觉 token 的 position。
        for mf in seq.mm_features:
            g_tuple = getattr(mf, "grid_thw_tuple", None)

            if g_tuple is None:
                g = getattr(mf, "grid_thw", None)
                if g is None:
                    continue

                flat = g[0].tolist() if g.ndim == 2 else g.tolist()
                g_tuple = (int(flat[0]), int(flat[1]), int(flat[2]))

            t, h, w = g_tuple

            img_start = int(mf.start)
            img_end = img_start + int(mf.num_encoder_tokens)

            ov_start = max(img_start, chunk_start)
            ov_end = min(img_end, chunk_end)

            if ov_start >= ov_end:
                continue

            gh = h // merge_size
            gw = w // merge_size
            plane = gh * gw

            an = ov_end - ov_start
            vis_off = ov_start - img_start

            local = off + (ov_start - chunk_start)
            grid_part = _grid_slice(t, gh, gw, vis_off, an)

            if ov_start == img_start:
                origin = int(pos[0, local])
                mf.mrope_origin = origin
            else:
                if mf.mrope_origin is not None:
                    origin = int(mf.mrope_origin)
                else:
                    first_t = vis_off // plane
                    origin = int(pos[0, local]) - int(first_t)

            pos[:, local : local + an] = grid_part + origin

            te = local + an

            if ov_end == img_end and te < off + m:
                nxt = int(pos[:, local:te].max()) + 1
                tail_len = off + m - te

                pos[:, te : off + m] = nxt + np.arange(
                    tail_len,
                    dtype=np.int64,
                )

        if m > 0:
            abs_end = off + m - 1
            token_idx = chunk_start + m - 1

            seq.mrope_next = int(pos[:, abs_end].max()) + 1
            seq.rope_delta = seq.mrope_next - 1 - token_idx

        off += m

    max_pos = int(pos.max()) if total_tokens > 0 else 0

    out = torch.from_numpy(pos).to(device)

    out.max_pos = max_pos

    return out