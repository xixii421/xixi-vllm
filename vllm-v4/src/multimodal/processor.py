from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Optional

import torch
from transformers import AutoProcessor

from .hasher import hash_image_pixels
from .image import convert_image_mode, normalize_image

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class MMPlaceholder:
    modality: str
    offset: int
    length: int
    mm_hash: str


@dataclass(slots=True)
class MultiModalInputs:

    prompt_token_ids: list[int]
    mm_kwargs: dict[str, Any] = field(default_factory=dict)
    mm_placeholders: list[MMPlaceholder] = field(default_factory=list)
    mm_hashes: list[str] = field(default_factory=list)

    @property
    def pixel_values(self) -> Optional[torch.Tensor]:
        return self.mm_kwargs.get("pixel_values")

    @property
    def image_grid_thw(self) -> Optional[torch.Tensor]:
        return self.mm_kwargs.get("image_grid_thw")


@dataclass(slots=True)
class _CachedImageFeat:
    pixel_values: torch.Tensor  # CPU, contiguous, unpinned
    grid_thw: tuple[int, int, int]
    num_patches: int

    @property
    def num_tokens(self) -> int:
        # Qwen-VL: vision tokens after spatial merge.
        return self.num_patches  # filled with merge-aware count at cache time


def _cpu_unpinned(t: torch.Tensor) -> torch.Tensor:
    t = t.detach().contiguous()
    if t.device.type != "cpu":
        t = t.cpu()
    if t.is_pinned:
        out = torch.empty(t.shape, dtype=t.dtype, device="cpu")
        out.copy_(t)
        return out
    return t


class MultiModalProcessor:
    def __init__(self, model_path: str):
        self.p = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
        tok = self.p.tokenizer
        self.img_tok = getattr(self.p, "image_token_id", getattr(tok, "image_token_id", None))
        self.vs_tok = getattr(
            self.p, "vision_start_token_id", getattr(tok, "vision_start_token_id", None)
        )
        self.ve_tok = getattr(
            self.p, "vision_end_token_id", getattr(tok, "vision_end_token_id", None)
        )
        self._image_pad = getattr(self.p, "image_token", None) or "<|image_pad|>"
        ip = getattr(self.p, "image_processor", None)
        self._merge_size = int(getattr(ip, "merge_size", 2) or 2)

    @staticmethod
    def _prepare_pil(image):
        """Match official vLLM multimodal image prep: EXIF + RGB (RGBA→white bg)."""
        image = normalize_image(image)
        return convert_image_mode(image, "RGB")

    def _encode_image(self, image) -> _CachedImageFeat:
        """Run HF image_processor only (no text)."""
        image = self._prepare_pil(image)
        ip = self.p.image_processor
        out = ip(images=[image], return_tensors="pt")
        pv = _cpu_unpinned(out["pixel_values"])
        grid = out["image_grid_thw"]
        if grid.dim() == 1:
            grid = grid.view(1, -1)
        g = grid[0]
        g_tuple = (int(g[0]), int(g[1]), int(g[2]))
        n_patches = g_tuple[0] * g_tuple[1] * g_tuple[2]
        return _CachedImageFeat(
            pixel_values=pv, grid_thw=g_tuple, num_patches=n_patches
        )

    def _vision_tokens(self, feat: _CachedImageFeat) -> int:
        m = self._merge_size
        return feat.num_patches // (m * m)

    def _expand_image_pads(self, prompt: str, feats: list[_CachedImageFeat]) -> str:
        """Replace each single ``<|image_pad|>`` with grid-sized repeats (Qwen-VL).

        Uses ``split`` so multi-image prompts expand each placeholder in order;
        a naive ``str.replace(..., count=1)`` loop re-matches the pad rows just
        emitted for the previous image (first span over-expands, last stays 1).
        """
        pad = self._image_pad
        parts = prompt.split(pad)
        if len(parts) != len(feats) + 1:
            raise RuntimeError(
                f"expected {len(feats)} pad(s) ({pad!r}) in chat template prompt, "
                f"found {len(parts) - 1}"
            )
        out = [parts[0]]
        for i, feat in enumerate(feats):
            out.append(pad * self._vision_tokens(feat))
            out.append(parts[i + 1])
        return "".join(out)

    def process(
        self, text: str, images: Optional[list] = None
    ) -> MultiModalInputs:
        """Process text (+ optional images) into ``MultiModalInputs``.

        Pixel hash binds the cache key to the *encoded* input (config-aware);
        prompt token ids are always rebuilt (depend on ``text``).
        """
        if not images:
            return MultiModalInputs(prompt_token_ids=self.p.tokenizer.encode(text))

        feats = [self._encode_image(im) for im in images]
        uids = [hash_image_pixels(f.pixel_values) for f in feats]

        msgs = [
            {
                "role": "user",
                "content": [{"type": "image", "image": img} for img in images]
                + [{"type": "text", "text": text}],
            }
        ]
        prompt = self.p.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True
        )
        prompt = self._expand_image_pads(prompt, feats)
        ids = self.p.tokenizer.encode(prompt)

        pv = torch.cat([f.pixel_values for f in feats], dim=0)
        grid = torch.tensor([list(f.grid_thw) for f in feats], dtype=torch.int64)
        placeholders = self._placeholders(ids, uids)
        return MultiModalInputs(
            prompt_token_ids=list(ids),
            mm_kwargs={"pixel_values": pv, "image_grid_thw": grid},
            mm_placeholders=placeholders,
            mm_hashes=list(uids),
        )

    def _placeholders(self, ids: list[int], uids: list[str]) -> list[MMPlaceholder]:
        if self.img_tok is None:
            return []
        out: list[MMPlaceholder] = []
        if self.vs_tok is not None and self.ve_tok is not None:
            vs = [i for i, t in enumerate(ids) if t == self.vs_tok]
            ve = [i for i, t in enumerate(ids) if t == self.ve_tok]
            if len(vs) != len(ve):
                logger.warning(
                    "vision_start/end count mismatch (%s vs %s)", len(vs), len(ve)
                )
            for idx, (s, e) in enumerate(zip(vs, ve)):
                if e <= s:
                    continue
                n = sum(1 for t in ids[s + 1 : e] if t == self.img_tok)
                uid = uids[idx] if idx < len(uids) else f"img-{idx}"
                out.append(
                    MMPlaceholder(
                        modality="image", offset=s + 1, length=n, mm_hash=uid
                    )
                )
            return out

        pos = [i for i, t in enumerate(ids) if t == self.img_tok]
        if not pos:
            return []
        out.append(
            MMPlaceholder(
                modality="image",
                offset=pos[0],
                length=len(pos),
                mm_hash=uids[0] if uids else "img-0",
            )
        )
        return out
