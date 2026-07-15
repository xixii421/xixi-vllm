from __future__ import annotations

import torch

from .rotary import RotaryEmbedding, apply_rotary_emb, _rope_cuda


class MRotaryEmbedding(RotaryEmbedding):
    def __init__(
        self,
        head_size: int,
        rotary_dim: int,
        max_position_embeddings: int,
        base: float,
        mrope_section: tuple[int, ...] = (24, 20, 20),
        mrope_interleaved: bool = False,
    ) -> None:
        self.mrope_section = tuple(int(x) for x in mrope_section)
        self.mrope_interleaved = bool(mrope_interleaved)
        assert sum(self.mrope_section) == rotary_dim // 2, (
            f"mrope_section sum {sum(self.mrope_section)} != rotary_dim//2 {rotary_dim // 2}"
        )
        self._rope_base = float(base)

        init_len = min(max(int(max_position_embeddings), 8192), 16384)
        RotaryEmbedding.__init__(self, head_size, rotary_dim, init_len, base)

    def _rebuild_cache(self, length: int) -> None:
        inv_freq = 1.0 / (
            self._rope_base
            ** (
                torch.arange(
                    0, self.rotary_dim, 2, dtype=torch.float, device=self.cos_sin_cache.device
                )
                / self.rotary_dim
            )
        )
        t = torch.arange(length, dtype=torch.float, device=self.cos_sin_cache.device)
        freqs = torch.einsum("i,j -> ij", t, inv_freq)
        cache = torch.cat((freqs.cos(), freqs.sin()), dim=-1).unsqueeze_(1)
        self.cos_sin_cache = cache

    def _get_cos_sin_3d(self, positions: torch.Tensor):
        if positions.ndim == 1:
            positions = positions.unsqueeze(0).expand(3, -1)
        # Growing the table needs a host sync (``.item()``), which is illegal while
        # a CUDA Graph is capturing. Decode CG positions stay within the initial
        # cache (≤16384); only eager / prefill should ever extend.
        capturing = positions.is_cuda and torch.cuda.is_current_stream_capturing()
        if not capturing:
            need = int(positions.max().item()) + 1
            if need > self.cos_sin_cache.shape[0]:
                self._rebuild_cache(need + 1024)
        cs = self.cos_sin_cache[positions].squeeze(2)  # [3, N, D]
        return cs.chunk(2, dim=-1)

    def _reorder_cos_sin(self, cos_3d: torch.Tensor, sin_3d: torch.Tensor):

        if self.mrope_interleaved:
            cos = cos_3d[0].clone()
            sin = sin_3d[0].clone()
            for dim, offset in ((1, 1), (2, 2)):
                length = self.mrope_section[dim] * 3
                idx = slice(offset, length, 3)
                cos[..., idx] = cos_3d[dim, ..., idx]
                sin[..., idx] = sin_3d[dim, ..., idx]
            return cos.unsqueeze(1), sin.unsqueeze(1)

        t, h, w = self.mrope_section
        cos = torch.cat(
            [cos_3d[0, :, :t], cos_3d[1, :, t : t + h], cos_3d[2, :, t + h :]],
            dim=-1,
        ).unsqueeze(1)
        sin = torch.cat(
            [sin_3d[0, :, :t], sin_3d[1, :, t : t + h], sin_3d[2, :, t + h :]],
            dim=-1,
        ).unsqueeze(1)
        return cos, sin

    def _forward_native(
        self, positions: torch.Tensor, query: torch.Tensor, key: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        cos_3d, sin_3d = self._get_cos_sin_3d(positions)
        cos, sin = self._reorder_cos_sin(cos_3d, sin_3d)
        return apply_rotary_emb(query, cos, sin), apply_rotary_emb(key, cos, sin)

    def _forward_cuda(
        self, positions: torch.Tensor, query: torch.Tensor, key: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        cos_3d, sin_3d = self._get_cos_sin_3d(positions)
        cos, sin = self._reorder_cos_sin(cos_3d, sin_3d)
        return _rope_cuda(
            query,
            key,
            cos.squeeze(1),
            sin.squeeze(1),
            self.head_size,
            self.rotary_dim,
        )


def get_mrope(
    head_size: int,
    rotary_dim: int,
    max_position: int,
    base: float,
    mrope_section: tuple[int, ...] = (24, 20, 20),
    mrope_interleaved: bool = False,
) -> MRotaryEmbedding:
    return MRotaryEmbedding(
        head_size,
        rotary_dim,
        max_position,
        base,
        mrope_section=mrope_section,
        mrope_interleaved=mrope_interleaved,
    )
