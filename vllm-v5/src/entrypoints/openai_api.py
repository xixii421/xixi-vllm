"""Backward-compatible entry: ``python -m src.entrypoints.openai_api``.

Prefer importing from ``src.entrypoints.openai`` for new code.
"""

from __future__ import annotations

from .openai.api_server import create_app, main
from .openai.protocol import ChatCompletionRequest, ChatMessage, CompletionRequest
from .openai.chat_parser import parse_chat_messages

__all__ = [
    "ChatCompletionRequest",
    "ChatMessage",
    "CompletionRequest",
    "create_app",
    "main",
    "parse_chat_messages",
]


if __name__ == "__main__":
    main()
