"""Minimal OpenAI-compatible HTTP API (teaching subset of ullm/vLLM).

Endpoints:
  GET  /v1/models
  POST /v1/completions
  POST /v1/chat/completions
"""

from __future__ import annotations

import argparse
import time
import uuid
from contextlib import asynccontextmanager
from typing import Any, Literal

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel

from ..config import EngineConfig
from ..sampling_params import SamplingParams
from ..runtime.async_llm import AsyncLLM


class CompletionRequest(BaseModel):
    model: str
    prompt: str | list[str]
    max_tokens: int = 64
    temperature: float = 0.0
    top_p: float = 1.0
    stream: bool = False
    ignore_eos: bool = False


class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str


class ChatCompletionRequest(BaseModel):
    model: str
    messages: list[ChatMessage]
    max_tokens: int = 64
    temperature: float = 0.0
    top_p: float = 1.0
    stream: bool = False


def _sp(req: CompletionRequest | ChatCompletionRequest) -> SamplingParams:
    return SamplingParams(
        temperature=req.temperature,
        top_p=req.top_p if req.top_p < 1.0 else None,
        max_tokens=req.max_tokens,
        ignore_eos=getattr(req, "ignore_eos", False),
    )


def _chat_prompt(messages: list[ChatMessage]) -> str:
    parts: list[str] = []
    for m in messages:
        parts.append(f"{m.role}: {m.content}")
    parts.append("assistant:")
    return "\n".join(parts)


def create_app(config: EngineConfig) -> FastAPI:
    state: dict[str, Any] = {}

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        state["llm"] = AsyncLLM(config)
        state["model"] = config.model
        yield
        await state["llm"].shutdown()

    app = FastAPI(title="vllm-v7 OpenAI API", lifespan=lifespan)

    @app.get("/v1/models")
    async def list_models():
        return {
            "object": "list",
            "data": [
                {
                    "id": state["model"],
                    "object": "model",
                    "created": int(time.time()),
                    "owned_by": "vllm-v7",
                }
            ],
        }

    @app.post("/v1/completions")
    async def completions(body: CompletionRequest, request: Request):
        llm: AsyncLLM = state["llm"]
        prompt = body.prompt if isinstance(body.prompt, str) else body.prompt[0]
        sp = _sp(body)
        if body.stream:

            async def gen():
                cid = f"cmpl-{uuid.uuid4().hex[:24]}"
                async for chunk in llm.generate(prompt, sp):
                    payload = {
                        "id": cid,
                        "object": "text_completion",
                        "created": int(time.time()),
                        "model": body.model,
                        "choices": [
                            {
                                "index": 0,
                                "text": chunk["text"],
                                "finish_reason": "stop" if chunk["finished"] else None,
                            }
                        ],
                    }
                    yield f"data: {__import__('json').dumps(payload)}\n\n"
                yield "data: [DONE]\n\n"

            return StreamingResponse(gen(), media_type="text/event-stream")

        text = await llm.generate_full(prompt, sp)
        return {
            "id": f"cmpl-{uuid.uuid4().hex[:24]}",
            "object": "text_completion",
            "created": int(time.time()),
            "model": body.model,
            "choices": [
                {"index": 0, "text": text, "finish_reason": "stop"}
            ],
        }

    @app.post("/v1/chat/completions")
    async def chat_completions(body: ChatCompletionRequest, request: Request):
        llm: AsyncLLM = state["llm"]
        prompt = _chat_prompt(body.messages)
        sp = _sp(body)
        if body.stream:

            async def gen():
                cid = f"chatcmpl-{uuid.uuid4().hex[:24]}"
                async for chunk in llm.generate(prompt, sp):
                    payload = {
                        "id": cid,
                        "object": "chat.completion.chunk",
                        "created": int(time.time()),
                        "model": body.model,
                        "choices": [
                            {
                                "index": 0,
                                "delta": {"content": chunk["text"]},
                                "finish_reason": "stop" if chunk["finished"] else None,
                            }
                        ],
                    }
                    yield f"data: {__import__('json').dumps(payload)}\n\n"
                yield "data: [DONE]\n\n"

            return StreamingResponse(gen(), media_type="text/event-stream")

        text = await llm.generate_full(prompt, sp)
        return {
            "id": f"chatcmpl-{uuid.uuid4().hex[:24]}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": body.model,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": text},
                    "finish_reason": "stop",
                }
            ],
        }

    return app


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="vllm-v7 OpenAI-compatible server")
    p.add_argument("--model", required=True)
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--tp-size", type=int, default=1)
    p.add_argument("--pp-size", type=int, default=1)
    p.add_argument("--device-ids", default=None)
    p.add_argument("--rpc-backend", choices=["zmq", "pipe"], default="zmq")
    p.add_argument("--nccl-host", default="127.0.0.1")
    p.add_argument("--nccl-port", type=int, default=0)
    p.add_argument("--enable-radix", action="store_true", default=True)
    p.add_argument("--no-radix", action="store_true")
    p.add_argument("--async-scheduling", action="store_true")
    p.add_argument("--eager", action="store_true")
    args = p.parse_args(argv)

    device_ids = None
    if args.device_ids:
        device_ids = [int(x) for x in args.device_ids.split(",")]
    cfg = EngineConfig(
        model=args.model,
        tp_size=args.tp_size,
        pp_size=args.pp_size,
        device_ids=device_ids,
        rpc_backend=args.rpc_backend,
        nccl_host=args.nccl_host,
        nccl_port=args.nccl_port,
        enable_radix=not args.no_radix,
        enable_async_scheduling=args.async_scheduling,
        enforce_eager=args.eager,
    )
    import uvicorn

    uvicorn.run(create_app(cfg), host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
