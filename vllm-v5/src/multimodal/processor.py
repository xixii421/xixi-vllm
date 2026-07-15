from typing import Optional
import torch
from PIL import Image
from transformers import AutoProcessor
import logging
logger = logging.getLogger(__name__)


class MultiModalProcessor:
    def __init__(self, model_path: str):
        self.p = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
        self.img_tok = getattr(self.p, "image_token_id",
                               getattr(self.p.tokenizer, "image_token_id", None))
        self.vs_tok = getattr(self.p, "vision_start_token_id",
                              getattr(self.p.tokenizer, "vision_start_token_id", None))
        self.ve_tok = getattr(self.p, "vision_end_token_id",
                              getattr(self.p.tokenizer, "vision_end_token_id", None))

    def process(self, text: str, images: Optional[list[Image.Image]] = None
                ) -> tuple[list[int], Optional[torch.Tensor], list[dict]]:
        if not images:
            return self.p.tokenizer.encode(text), None, []

        msgs = [{"role": "user", "content":
            [{"type": "image", "image": img} for img in images] +
            [{"type": "text", "text": text}]}]
        prompt = self.p.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        inputs = self.p(text=prompt, images=images, return_tensors="pt")
        ids = inputs["input_ids"][0].tolist()
        pv = inputs.get("pixel_values")
        grid = inputs.get("image_grid_thw")
        return ids, pv, self._info(ids, pv, grid)

    def _info(self, ids, pv, grid) -> list[dict]:
        if pv is None or self.img_tok is None:
            return []
        info = []
        if self.vs_tok and self.ve_tok:
            vs = [i for i, t in enumerate(ids) if t == self.vs_tok]
            ve = [i for i, t in enumerate(ids) if t == self.ve_tok]
            for idx, (s, e) in enumerate(zip(vs, ve)):
                n = sum(1 for t in ids[s+1:e] if t == self.img_tok)
                g = grid[idx:idx+1] if grid is not None and grid.dim() > 1 else grid
                info.append({"start": s+1, "end": e, "num_encoder_tokens": n, "grid_thw": g})
        else:
            pos = [i for i, t in enumerate(ids) if t == self.img_tok]
            if pos:
                d = {"start": pos[0], "end": pos[-1]+1, "num_encoder_tokens": len(pos)}
                if grid is not None: d["grid_thw"] = grid
                info.append(d)
        return info
