"""OpenAI-compatible request / response schemas (HTTP-agnostic)."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


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
    # OpenAI multipart: str | list[{type, text|image_url}, ...]
    content: str | list[dict[str, Any]]


class ChatCompletionRequest(BaseModel):
    model: str
    messages: list[ChatMessage]
    max_tokens: int = 64
    temperature: float = 0.0
    top_p: float = 1.0
    stream: bool = False
    ignore_eos: bool = False


class ModelCard(BaseModel):
    id: str
    object: str = "model"
    created: int
    owned_by: str = "vllm-v7"


class ModelList(BaseModel):
    object: str = "list"
    data: list[ModelCard] = Field(default_factory=list)
