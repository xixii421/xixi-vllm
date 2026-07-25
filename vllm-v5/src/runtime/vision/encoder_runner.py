"""Worker-side vision encode + embed merge (orchestration lives here, not in models/)."""

from __future__ import annotations

import logging
from typing import Callable, Optional

import torch
import torch.nn as nn

from ...distributed import get_tp_group
from ...layers.vision import run_dp_sharded_vision_encode
from ..input.features import overlaps_window, span_end
from .encoder_cache import EncoderCache
from .pixel_store import PixelStore
from .positions import apply_mrope_seq_updates, compute_mrope_positions

logger = logging.getLogger(__name__)


class VisionEncoderRunner:

    def __init__(
        self,
        vision_tower: nn.Module,
        *,
        mm_encoder_tp_mode: str = "data",
        num_deepstack: int = 0,
        cache: EncoderCache | None = None,
        pixel_store: PixelStore | None = None,
    ) -> None:
        self.vision_tower = vision_tower
        self.mm_encoder_tp_mode = mm_encoder_tp_mode
        self.num_deepstack = num_deepstack
        self.cache = cache or EncoderCache()
        self.pixel_store = pixel_store or PixelStore()
        self.vit_cg = None  # optional ViTCUDAGraphRunner, set by ModelRunner

    def encode_and_merge(
        self,
        embed_tokens: Callable[[torch.Tensor], torch.Tensor],
        input_ids: torch.Tensor,
        seqs: list,
        device: torch.device,
    ) -> tuple[torch.Tensor, Optional[list[torch.Tensor]], torch.Tensor]:
        embeds = embed_tokens(input_ids)
        visuals, deepstack_by_uid = self.encode_visual(seqs, device)
        self._scatter_visuals(embeds, seqs, visuals, device)
        deepstack = self._build_deepstack(
            seqs, embeds.shape[0], embeds.shape[1], device, embeds.dtype, deepstack_by_uid
        )
        positions, updates = compute_mrope_positions(
            input_ids, seqs, int(self.vision_tower.merge_size), device
        )
        apply_mrope_seq_updates(seqs, updates)
        return embeds, deepstack, positions

    def encode_visual(
        self, seqs: list, device: torch.device
    ) -> tuple[list[torch.Tensor | None], dict[str, list[torch.Tensor] | None]]:
        """Encode images that overlap the current scheduled windows (cache-aware)."""
        jobs: list[tuple[int, object]] = []  # (flat_index, mf) for needed encodes
        flat_mfs = [mf for seq in seqs for mf in seq.mm_features]
        out: list[torch.Tensor | None] = [None] * len(flat_mfs)
        deepstack_by_uid: dict[str, list[torch.Tensor] | None] = {}

        use_dp = self.mm_encoder_tp_mode == "data" and get_tp_group().size > 1
        fi = 0
        for seq in seqs:
            cs = seq.num_cached_tokens
            ce = cs + seq.num_scheduled_tokens
            for mf in seq.mm_features:
                if not overlaps_window(mf, cs, ce):
                    fi += 1
                    continue
                # DP ranks all see the full gathered batch, so cache is shared-safe.
                cached = self.cache.get(mf.uid)
                if cached is not None:
                    emb, ds = cached
                    out[fi] = emb.to(device=device, non_blocking=True)
                    deepstack_by_uid[mf.uid] = (
                        [t.to(device=device, non_blocking=True) for t in ds]
                        if ds is not None
                        else None
                    )
                else:
                    jobs.append((fi, mf))
                fi += 1

        if not jobs:
            return out, deepstack_by_uid

        miss_idx = [i for i, _ in jobs]
        miss_mfs = [mf for _, mf in jobs]
        m = int(self.vision_tower.merge_size)
        pixel_chunks: list[torch.Tensor] = []
        grids: list[torch.Tensor] = []
        sizes: list[int] = []
        for mf in miss_mfs:
            data = mf.data if mf.data is not None else self.pixel_store.get(mf.uid)
            if data is None:
                raise ValueError(
                    f"mm feature uid={mf.uid} has no pixels "
                    "(missing register_mm_pixels or Feature.data)"
                )
            pixel_chunks.append(data)
            g = mf.grid_thw
            if g is None:
                raise ValueError(
                    f"mm feature uid={mf.uid} missing grid_thw; cannot encode"
                )
            g_cpu = g.detach().to("cpu").view(-1, 3)
            t, h, w = (int(x) for x in g_cpu[0].tolist())
            sizes.append((t * h * w) // (m * m))
            grids.append(g_cpu.to(device=device, non_blocking=True))
        pv = torch.cat(pixel_chunks, dim=0).to(device)
        grid = torch.cat(grids, dim=0)

        flat, dps_batch = self._run_vision(pv, grid, use_dp=use_dp)
        cursor = 0
        for idx, mf, sz in zip(miss_idx, miss_mfs, sizes):
            emb = flat[cursor : cursor + sz].contiguous()
            out[idx] = emb
            ds_i = (
                [t[cursor : cursor + sz].contiguous() for t in dps_batch]
                if dps_batch is not None
                else None
            )
            deepstack_by_uid[mf.uid] = ds_i
            self.cache.put(mf.uid, emb, ds_i)
            self.pixel_store.discard(mf.uid)
            cursor += sz
        return out, deepstack_by_uid

    def _run_vision(
        self, pv: torch.Tensor, grid: torch.Tensor, *, use_dp: bool
    ) -> tuple[torch.Tensor, list[torch.Tensor] | None]:
        cg = self.vit_cg
        if not use_dp and cg is not None and cg.can_use(grid):
            try:
                return cg.replay(pv, grid)
            except Exception as exc:
                logger.warning("ViT CUDA Graph replay failed; eager fallback: %s", exc)

        if use_dp:
            return run_dp_sharded_vision_encode(self.vision_tower, pv, grid)

        result = self.vision_tower(pv, grid_thw=grid)
        if isinstance(result, tuple):
            # New API: (merged, deepstack) or legacy split-by-image
            if (
                len(result) == 2
                and torch.is_tensor(result[0])
                and (result[1] is None or isinstance(result[1], list))
            ):
                flat, dps = result
                return flat, dps
            flat = torch.cat(list(result), dim=0)
        else:
            flat = result
        dps = getattr(self.vision_tower, "deepstack_features", None)
        return flat, dps

    @staticmethod
    def _scatter_visuals(
        embeds: torch.Tensor,
        seqs: list,
        visuals: list[torch.Tensor | None],
        device: torch.device,
    ) -> None:
        off = vis_i = 0
        for seq in seqs:
            cs = seq.num_cached_tokens
            ce = cs + seq.num_scheduled_tokens
            for mf in seq.mm_features:
                img_start, img_end = mf.start, span_end(mf)
                ov_start = max(img_start, cs)
                ov_end = min(img_end, ce)
                if ov_start >= ov_end:
                    vis_i += 1
                    continue
                src_full = visuals[vis_i]
                if src_full is None:
                    raise RuntimeError(
                        f"missing visual embeds for uid={mf.uid} in scheduled window"
                    )
                vis_off = ov_start - img_start
                nv = ov_end - ov_start
                local = off + (ov_start - cs)
                src = src_full[vis_off : vis_off + nv]
                if src.device != embeds.device or src.dtype != embeds.dtype:
                    src = src.to(dtype=embeds.dtype, device=device, non_blocking=True)
                embeds[local : local + nv].copy_(src)
                vis_i += 1
            off += seq.num_scheduled_tokens

    def _build_deepstack(
        self,
        seqs: list,
        n: int,
        h: int,
        device: torch.device,
        dtype: torch.dtype,
        deepstack_by_uid: dict[str, list[torch.Tensor] | None],
    ) -> list[torch.Tensor] | None:
        if not self.num_deepstack:
            return None
        if not any(
            deepstack_by_uid.get(mf.uid) for seq in seqs for mf in seq.mm_features
        ):
            return None
        bufs = [torch.zeros(n, h, device=device, dtype=dtype) for _ in range(self.num_deepstack)]
        off = 0
        for seq in seqs:
            cs = seq.num_cached_tokens
            ce = cs + seq.num_scheduled_tokens
            for mf in seq.mm_features:
                cached = deepstack_by_uid.get(mf.uid)
                if cached is None:
                    continue
                img_start, img_end = mf.start, span_end(mf)
                ov_start = max(img_start, cs)
                ov_end = min(img_end, ce)
                if ov_start >= ov_end:
                    continue
                vis_off = ov_start - img_start
                nv = ov_end - ov_start
                local = off + (ov_start - cs)
                for li in range(self.num_deepstack):
                    bufs[li][local : local + nv].copy_(
                        cached[li][vis_off : vis_off + nv].to(
                            device=device, dtype=dtype, non_blocking=True
                        )
                    )
            off += seq.num_scheduled_tokens
        return bufs
