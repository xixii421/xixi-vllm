from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from typing import TYPE_CHECKING, Any, Optional

from ..config import EngineConfig
from ..sampling_params import SamplingParams
from .engine import Engine

if TYPE_CHECKING:
    from PIL import Image


class AsyncLLM:
    """Async façade over sync ``Engine`` (ullm-style).

    CPU-side add/step run in a thread pool so FastAPI can await many requests.
    When ``enable_async_scheduling`` is set, the engine may overlap CPU schedule
    with the previous GPU step (placeholder token); PP is rejected in config.
    """

    def __init__(self, config: EngineConfig):
        self.config = config
        self._engine = Engine(config)
        self._lock = asyncio.Lock()

    @property
    def engine(self) -> Engine:
        return self._engine

    async def generate(
        self,
        prompt: str | list[int],
        sampling_params: SamplingParams | None = None,
        *,
        images: Optional[list["Image.Image"]] = None,
        request_id: str | None = None,
    ) -> AsyncIterator[dict[str, Any]]:
        del request_id
        sp = sampling_params or SamplingParams()
        async with self._lock:
            seq = await asyncio.to_thread(
                self._engine.add_request, prompt, sp, images
            )
            emitted = 0
            while not seq.is_finished:
                await asyncio.to_thread(self._engine.step)
                while emitted < seq.num_completion_tokens:
                    tid = seq.token_ids[seq.num_prompt_tokens + emitted]
                    emitted += 1
                    text = self._engine.tokenizer.decode([tid])
                    yield {
                        "token_id": tid,
                        "text": text,
                        "finished": seq.is_finished
                        and emitted >= seq.num_completion_tokens,
                    }

    async def generate_full(
        self,
        prompt: str | list[int],
        sampling_params: SamplingParams | None = None,
        *,
        images: Optional[list["Image.Image"]] = None,
    ) -> str:
        parts: list[str] = []
        async for chunk in self.generate(prompt, sampling_params, images=images):
            parts.append(chunk["text"])
        return "".join(parts)

    async def shutdown(self) -> None:
        await asyncio.to_thread(self._engine.destroy)
