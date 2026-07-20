"""OpenAIServing — programmable OpenAI surface over AsyncLLM (no FastAPI)."""

from __future__ import annotations

import json
import time
import uuid
from collections.abc import AsyncIterator
from typing import Any

from ...runtime.async_llm import AsyncLLM
from ...sampling import SamplingParams
from .chat_parser import parse_chat_messages
from .protocol import (
    ChatCompletionRequest,
    CompletionRequest,
    ModelCard,
    ModelList,
)


def sampling_from_request(req: CompletionRequest | ChatCompletionRequest) -> SamplingParams:
    return SamplingParams(
        temperature=req.temperature,
        top_p=req.top_p if req.top_p < 1.0 else None,
        max_tokens=req.max_tokens,
        ignore_eos=getattr(req, "ignore_eos", False),
    )


class OpenAIServing:
    """Stable API for HTTP servers, bots, and tests.

    Other code should call this class (or ``AsyncLLM`` / ``Engine`` directly),
    not FastAPI route handlers.
    """

    def __init__(self, llm: AsyncLLM, *, model_name: str | None = None) -> None:
        self.llm = llm
        self.model_name = model_name or llm.config.model

    def list_models(self) -> ModelList:
        return ModelList(
            data=[
                ModelCard(
                    id=self.model_name,
                    created=int(time.time()),
                )
            ]
        )

    def _chat_inputs(
        self, request: ChatCompletionRequest
    ) -> tuple[str | None, list[dict] | None, list | None]:
        """Return ``(prompt, chat_messages, images)`` for AsyncLLM."""
        parsed = parse_chat_messages(request.messages)
        images = parsed.images or None
        engine = self.llm.engine
        if engine.mm_processor is not None or hasattr(
            engine.tokenizer, "apply_chat_template"
        ):
            return None, parsed.hf_messages, images
        return parsed.text_fallback, None, images

    async def create_completion(
        self, request: CompletionRequest
    ) -> dict[str, Any] | AsyncIterator[str]:
        prompt = (
            request.prompt if isinstance(request.prompt, str) else request.prompt[0]
        )
        sp = sampling_from_request(request)
        if request.stream:
            return self._stream_completion(request, prompt, sp)
        text = await self.llm.generate_full(prompt, sp)
        return {
            "id": f"cmpl-{uuid.uuid4().hex[:24]}",
            "object": "text_completion",
            "created": int(time.time()),
            "model": request.model,
            "choices": [{"index": 0, "text": text, "finish_reason": "stop"}],
        }

    async def create_chat_completion(
        self, request: ChatCompletionRequest
    ) -> dict[str, Any] | AsyncIterator[str]:
        prompt, chat_messages, images = self._chat_inputs(request)
        sp = sampling_from_request(request)
        if request.stream:
            return self._stream_chat(
                request,
                prompt=prompt,
                chat_messages=chat_messages,
                images=images,
                sp=sp,
            )
        text = await self.llm.generate_full(
            prompt,
            sp,
            images=images,
            chat_messages=chat_messages,
        )
        return {
            "id": f"chatcmpl-{uuid.uuid4().hex[:24]}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": request.model,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": text},
                    "finish_reason": "stop",
                }
            ],
        }

    async def _stream_completion(
        self, request: CompletionRequest, prompt: str, sp: SamplingParams
    ) -> AsyncIterator[str]:
        cid = f"cmpl-{uuid.uuid4().hex[:24]}"
        async for chunk in self.llm.generate(prompt, sp):
            payload = {
                "id": cid,
                "object": "text_completion",
                "created": int(time.time()),
                "model": request.model,
                "choices": [
                    {
                        "index": 0,
                        "text": chunk["text"],
                        "finish_reason": "stop" if chunk["finished"] else None,
                    }
                ],
            }
            yield f"data: {json.dumps(payload)}\n\n"
        yield "data: [DONE]\n\n"

    async def _stream_chat(
        self,
        request: ChatCompletionRequest,
        *,
        prompt: str | None,
        chat_messages: list[dict] | None,
        images: list | None,
        sp: SamplingParams,
    ) -> AsyncIterator[str]:
        cid = f"chatcmpl-{uuid.uuid4().hex[:24]}"
        async for chunk in self.llm.generate(
            prompt,
            sp,
            images=images,
            chat_messages=chat_messages,
        ):
            payload = {
                "id": cid,
                "object": "chat.completion.chunk",
                "created": int(time.time()),
                "model": request.model,
                "choices": [
                    {
                        "index": 0,
                        "delta": {"content": chunk["text"]},
                        "finish_reason": "stop" if chunk["finished"] else None,
                    }
                ],
            }
            yield f"data: {json.dumps(payload)}\n\n"
        yield "data: [DONE]\n\n"
