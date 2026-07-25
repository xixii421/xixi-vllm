"""CPU-side multimodal preprocessing (HF AutoProcessor → ids + pixels + spans).

Aligned with official vLLM:
  - hash the *raw image* (not float pixel patches)
  - LRU-cache HF processor outputs by (image hashes, text)
"""

from __future__ import annotations

import hashlib
import logging
from collections import OrderedDict
from typing import Optional

import numpy as np
import torch
from transformers import AutoProcessor

logger = logging.getLogger(__name__)


def hash_pil_image(image) -> str:
    """Cheap stable content hash (official uses blake3; sha1 is fine at this scale)."""
    # Prefer EXIF ImageID when present (matches upstream MultiModalHasher idea).
    try:
        exif = getattr(image, "getexif", lambda: None)()
        if exif:
            # 0xA420 = ImageUniqueID
            uid = exif.get(0xA420)
            if uid:
                return hashlib.sha1(f"exif:{uid}".encode()).hexdigest()
    except Exception:
        pass
    arr = np.asarray(image)
    h = hashlib.sha1()
    h.update(arr.tobytes())
    h.update(f"|{image.size}|{getattr(image, 'mode', '')}".encode())
    return h.hexdigest()


class MultiModalProcessor:
    def __init__(self, model_path: str, cache_size: int = 64):
        self.p = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
        self.img_tok = getattr(
            self.p, "image_token_id", getattr(self.p.tokenizer, "image_token_id", None)
        )
        self.vs_tok = getattr(
            self.p,
            "vision_start_token_id",
            getattr(self.p.tokenizer, "vision_start_token_id", None),
        )
        self.ve_tok = getattr(
            self.p,
            "vision_end_token_id",
            getattr(self.p.tokenizer, "vision_end_token_id", None),
        )
        self._cache: OrderedDict[
            tuple, tuple[list[int], Optional[torch.Tensor], list[dict], list[str]]
        ] = OrderedDict()
        self._cache_size = max(1, cache_size)

    def process(
        self, text: str, images: Optional[list] = None
    ) -> tuple[list[int], Optional[torch.Tensor], list[dict], list[str]]:
        """Returns ``(input_ids, pixel_values, mm_info, image_uids)``."""
        if not images:
            return self.p.tokenizer.encode(text), None, [], []

        img_uids = [hash_pil_image(im) for im in images]
        key = (text, tuple(img_uids))
        hit = self._cache.get(key)
        if hit is not None:
            self._cache.move_to_end(key)
            ids, pv, info, uids = hit
            # Return clones of mutable tensors so callers can pin/move freely.
            pv_out = pv.clone() if pv is not None else None
            return list(ids), pv_out, [dict(x) for x in info], list(uids)

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
        inputs = self.p(text=prompt, images=images, return_tensors="pt")
        ids = inputs["input_ids"][0].tolist()
        pv = inputs.get("pixel_values")
        grid = inputs.get("image_grid_thw")
        if pv is not None and not pv.is_pinned:
            pv = pv.contiguous().pin_memory()
        info = self._info(ids, pv, grid)
        self._cache[key] = (list(ids), pv, [dict(x) for x in info], list(img_uids))
        self._cache.move_to_end(key)
        while len(self._cache) > self._cache_size:
            self._cache.popitem(last=False)
        pv_out = pv.clone() if pv is not None else None
        return list(ids), pv_out, info, list(img_uids)

    def _info(self, ids, pv, grid) -> list[dict]:
        if pv is None or self.img_tok is None:
            return []
        info: list[dict] = []
        if self.vs_tok and self.ve_tok:
            vs = [i for i, t in enumerate(ids) if t == self.vs_tok]
            ve = [i for i, t in enumerate(ids) if t == self.ve_tok]
            for idx, (s, e) in enumerate(zip(vs, ve)):
                n = sum(1 for t in ids[s + 1 : e] if t == self.img_tok)
                g = grid[idx : idx + 1] if grid is not None and grid.dim() > 1 else grid
                # Keep a plain Python tuple to avoid later .cpu().tolist() syncs.
                g_tuple = None
                if g is not None:
                    flat = g.detach().view(-1, 3)[0].tolist()
                    g_tuple = (int(flat[0]), int(flat[1]), int(flat[2]))
                info.append(
                    {
                        "start": s + 1,
                        "end": e,
                        "num_encoder_tokens": n,
                        "grid_thw": g,
                        "grid_thw_tuple": g_tuple,
                    }
                )
        else:
            pos = [i for i, t in enumerate(ids) if t == self.img_tok]
            if pos:
                d = {"start": pos[0], "end": pos[-1] + 1, "num_encoder_tokens": len(pos)}
                if grid is not None:
                    d["grid_thw"] = grid
                    flat = grid.detach().view(-1, 3)[0].tolist()
                    d["grid_thw_tuple"] = (int(flat[0]), int(flat[1]), int(flat[2]))
                info.append(d)
        return info
