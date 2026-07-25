import torch

from .rotary import RotaryEmbedding, _rope_cuda, apply_rotary_emb


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
        assert sum(self.mrope_section) == rotary_dim // 2
        self._rope_base = float(base)
        init_len = min(max(int(max_position_embeddings), 8192), 1 << 20)
        RotaryEmbedding.__init__(self, head_size, rotary_dim, init_len, base)

    def ensure_cache_size(self, need: int) -> None:
        if need <= self.cos_sin_cache.shape[0]:
            return
        length = need + 1024
        dev = self.cos_sin_cache.device
        inv = 1.0 / (
            self._rope_base
            ** (torch.arange(0, self.rotary_dim, 2, dtype=torch.float, device=dev) / self.rotary_dim)
        )
        t = torch.arange(length, dtype=torch.float, device=dev)
        freqs = torch.einsum("i,j -> ij", t, inv)
        self.cos_sin_cache = torch.cat((freqs.cos(), freqs.sin()), dim=-1).unsqueeze_(1)

    def _cos_sin(self, positions: torch.Tensor):
        if positions.ndim == 1:
            positions = positions.unsqueeze(0).expand(3, -1)
        cos3, sin3 = self.cos_sin_cache[positions].squeeze(2).chunk(2, dim=-1)
        if self.mrope_interleaved:
            cos, sin = cos3[0].clone(), sin3[0].clone()
            for dim, off in ((1, 1), (2, 2)):
                idx = slice(off, self.mrope_section[dim] * 3, 3)
                cos[..., idx] = cos3[dim, ..., idx]
                sin[..., idx] = sin3[dim, ..., idx]
            return cos.unsqueeze(1), sin.unsqueeze(1)
        t, h, w = self.mrope_section
        cos = torch.cat([cos3[0, :, :t], cos3[1, :, t : t + h], cos3[2, :, t + h :]], -1).unsqueeze(1)
        sin = torch.cat([sin3[0, :, :t], sin3[1, :, t : t + h], sin3[2, :, t + h :]], -1).unsqueeze(1)
        return cos, sin

    def forward(self, positions, query, key):
        # 1-D → plain RoPE (decode / text); 3-D → MRoPE (MM prefill).
        if positions.ndim == 1 or positions.shape[0] < 3:
            pos = positions if positions.ndim == 1 else positions[0]
            fn = RotaryEmbedding._forward_cuda if query.is_cuda else RotaryEmbedding._forward_native
            return fn(self, pos, query, key)
        cos, sin = self._cos_sin(positions)
        if query.is_cuda:
            return _rope_cuda(query, key, cos.squeeze(1), sin.squeeze(1), self.head_size, self.rotary_dim)
        return apply_rotary_emb(query, cos, sin), apply_rotary_emb(key, cos, sin)


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
