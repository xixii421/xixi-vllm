import torch
from .rotary import RotaryEmbedding


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
        self.mrope_section = tuple(int(size) for size in mrope_section)
        self.mrope_interleaved = mrope_interleaved
        assert len(self.mrope_section) == 3
        assert sum(self.mrope_section) == rotary_dim // 2

        super().__init__(
            head_size,
            rotary_dim,
            max_position_embeddings,
            base,
            build_cache=False,
        )

    def _get_cos_sin_3d(
        self, positions: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if positions.ndim == 1:
            positions = positions.unsqueeze(0).expand(3, -1)
        freqs = positions.to(torch.float32).unsqueeze(-1) * self.inv_freq
        return freqs.cos(), freqs.sin()

    def _merge_modalities(
        self, cos_3d: torch.Tensor, sin_3d: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.mrope_interleaved:
            cos = cos_3d[0].clone()
            sin = sin_3d[0].clone()
            for axis, offset in ((1, 1), (2, 2)):
                indices = slice(offset, self.mrope_section[axis] * 3, 3)
                cos[..., indices] = cos_3d[axis, ..., indices]
                sin[..., indices] = sin_3d[axis, ..., indices]
            return cos.unsqueeze(1), sin.unsqueeze(1)

        time, height, _width = self.mrope_section
        boundaries = (time, time + height)

        def merge(x: torch.Tensor) -> torch.Tensor:
            return torch.cat(
                (
                    x[0, :, : boundaries[0]],
                    x[1, :, boundaries[0] : boundaries[1]],
                    x[2, :, boundaries[1] :],
                ),
                dim=-1,
            ).unsqueeze(1)

        return merge(cos_3d), merge(sin_3d)

    def prepare(
        self, positions: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        cos_3d, sin_3d = self._get_cos_sin_3d(positions)
        cos, sin = self._merge_modalities(cos_3d, sin_3d)
        return cos.contiguous(), sin.contiguous()


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
