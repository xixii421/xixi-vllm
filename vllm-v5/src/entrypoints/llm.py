"""Synchronous offline LLM façade (scripts / notebooks)."""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

from ..config import EngineConfig
from ..runtime.engine import Engine
from ..sampling import SamplingParams

if TYPE_CHECKING:
    from PIL import Image


class LLM:
    """Thin sync wrapper around ``Engine`` for offline batch generate."""

    def __init__(self, config: EngineConfig) -> None:
        self.config = config
        self.engine = Engine(config)

    def generate(
        self,
        prompts: str | list[str],
        sampling_params: SamplingParams | None = None,
        *,
        images: Optional[list["Image.Image"] | list[list["Image.Image"]]] = None,
    ) -> list[str]:
        prompt_list = [prompts] if isinstance(prompts, str) else list(prompts)
        image_arg: list | None = None
        if images is not None:
            if isinstance(prompts, str):
                # Single prompt: images is list[Image] or list[list[Image]]
                if images and not isinstance(images[0], list):
                    image_arg = [images]  # type: ignore[list-item]
                else:
                    image_arg = images  # type: ignore[assignment]
            else:
                image_arg = images  # type: ignore[assignment]
        return self.engine.generate_batch(
            prompt_list, sampling_params, images=image_arg
        )

    def shutdown(self) -> None:
        self.engine.destroy()
