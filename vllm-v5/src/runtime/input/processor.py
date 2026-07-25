"""HF AutoProcessor wrapper → token ids + typed image spans."""

from __future__ import annotations

import logging
from typing import Optional

import torch
from PIL import Image
from transformers import AutoProcessor

from .features import ImageSpanInfo

logger = logging.getLogger(__name__)


class MultiModalProcessor:
    def __init__(self, model_path: str):
        self.processor = AutoProcessor.from_pretrained(
            model_path, trust_remote_code=True
        )
        tok = self.processor.tokenizer
        self.image_token_id = getattr(
            self.processor, "image_token_id", getattr(tok, "image_token_id", None)
        )
        self.vision_start_id = getattr(
            self.processor,
            "vision_start_token_id",
            getattr(tok, "vision_start_token_id", None),
        )
        self.vision_end_id = getattr(
            self.processor,
            "vision_end_token_id",
            getattr(tok, "vision_end_token_id", None),
        )

    def process(
        self, text: str, images: Optional[list[Image.Image]] = None
    ) -> tuple[list[int], Optional[torch.Tensor], list[ImageSpanInfo]]:
        if not images:
            return self.processor.tokenizer.encode(text), None, []

        messages = [
            {
                "role": "user",
                "content": [{"type": "image", "image": img} for img in images]
                + [{"type": "text", "text": text}],
            }
        ]
        return self.process_messages(messages, images)

    def process_messages(
        self,
        messages: list[dict],
        images: Optional[list[Image.Image]] = None,
    ) -> tuple[list[int], Optional[torch.Tensor], list[ImageSpanInfo]]:
        """Tokenize HF-style chat messages (OpenAI Serving / multi-turn MM)."""
        prompt = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        if not images:
            return self.processor.tokenizer.encode(prompt), None, []
        inputs = self.processor(text=prompt, images=images, return_tensors="pt")
        ids = inputs["input_ids"][0].tolist()
        pixel_values = inputs.get("pixel_values")
        grid = inputs.get("image_grid_thw")
        return ids, pixel_values, self._spans(ids, pixel_values, grid)

    def apply_chat_template_text(self, messages: list[dict]) -> str:
        """Text-only chat template (no images)."""
        return self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

    def _spans(
        self,
        ids: list[int],
        pixel_values: Optional[torch.Tensor],
        grid: Optional[torch.Tensor],
    ) -> list[ImageSpanInfo]:
        if pixel_values is None or self.image_token_id is None:
            return []

        spans: list[ImageSpanInfo] = []
        if self.vision_start_id is not None and self.vision_end_id is not None:
            starts = [i for i, t in enumerate(ids) if t == self.vision_start_id]
            ends = [i for i, t in enumerate(ids) if t == self.vision_end_id]
            if len(starts) != len(ends):
                logger.warning(
                    "vision_start/end count mismatch (%s vs %s); truncating",
                    len(starts),
                    len(ends),
                )
            for idx, (s, e) in enumerate(zip(starts, ends)):
                if e <= s:
                    continue
                n = sum(1 for t in ids[s + 1 : e] if t == self.image_token_id)
                g = None
                if grid is not None:
                    g = grid[idx : idx + 1] if grid.dim() > 1 else grid
                spans.append(
                    ImageSpanInfo(
                        start=s + 1, end=e, num_encoder_tokens=n, grid_thw=g
                    )
                )
            return spans

        positions = [i for i, t in enumerate(ids) if t == self.image_token_id]
        if not positions:
            return []
        spans.append(
            ImageSpanInfo(
                start=positions[0],
                end=positions[-1] + 1,
                num_encoder_tokens=len(positions),
                grid_thw=grid,
            )
        )
        return spans
