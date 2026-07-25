"""Thin FastAPI OpenAI HTTP server — routes only; business logic in Serving."""

from __future__ import annotations

import argparse
import inspect
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI
from fastapi.responses import StreamingResponse

from ...config import EngineConfig
from ...runtime.async_llm import AsyncLLM
from .protocol import ChatCompletionRequest, CompletionRequest
from .serving import OpenAIServing


def _as_http(result: Any):
    if inspect.isasyncgen(result) or hasattr(result, "__aiter__"):
        return StreamingResponse(result, media_type="text/event-stream")
    return result


def create_app(config: EngineConfig) -> FastAPI:
    state: dict[str, Any] = {}

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        llm = AsyncLLM(config)
        state["serving"] = OpenAIServing(llm, model_name=config.model)
        yield
        await llm.shutdown()

    app = FastAPI(title="vllm-v7 OpenAI API", lifespan=lifespan)

    @app.get("/v1/models")
    async def list_models():
        serving: OpenAIServing = state["serving"]
        return serving.list_models().model_dump()

    @app.post("/v1/completions")
    async def completions(body: CompletionRequest):
        serving: OpenAIServing = state["serving"]
        return _as_http(await serving.create_completion(body))

    @app.post("/v1/chat/completions")
    async def chat_completions(body: ChatCompletionRequest):
        serving: OpenAIServing = state["serving"]
        return _as_http(await serving.create_chat_completion(body))

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
    p.add_argument("--eager", action="store_true")
    p.add_argument(
        "--mm-encoder-tp-mode",
        choices=["data", "weights"],
        default="data",
        help="ViT TP: data=image DP; weights=true Weight-TP on ViT linears",
    )
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
        enforce_eager=args.eager,
        mm_encoder_tp_mode=args.mm_encoder_tp_mode,
    )
    import uvicorn

    uvicorn.run(create_app(cfg), host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
