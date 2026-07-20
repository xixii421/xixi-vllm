"""OpenAI chat messages → HF-style messages + ordered PIL images."""

from __future__ import annotations

import base64
import io
from dataclasses import dataclass
from typing import Any
from urllib.request import urlopen

from .protocol import ChatMessage

try:
    from PIL import Image
except ImportError:  # pragma: no cover
    Image = None  # type: ignore[misc, assignment]


@dataclass(slots=True)
class ParsedChat:
    """Result of parsing an OpenAI chat request.

    ``hf_messages`` uses HuggingFace multimodal content parts
    (``type: text|image``) so callers can ``apply_chat_template``.
    """

    hf_messages: list[dict[str, Any]]
    images: list[Any]  # list[PIL.Image]
    text_fallback: str  # plain role:text join when no tokenizer/template


def load_image_url(url: str) -> "Image.Image":
    if Image is None:
        raise RuntimeError("Pillow is required for multimodal chat (pip install pillow)")
    if url.startswith("data:"):
        _, _, payload = url.partition(",")
        raw = base64.b64decode(payload)
        return Image.open(io.BytesIO(raw)).convert("RGB")
    with urlopen(url, timeout=30) as resp:  # noqa: S310 — user-supplied OpenAI URL
        return Image.open(io.BytesIO(resp.read())).convert("RGB")


def parse_chat_messages(messages: list[ChatMessage]) -> ParsedChat:
    """Convert OpenAI messages into HF chat + image list (order preserved)."""
    hf_messages: list[dict[str, Any]] = []
    images: list[Any] = []
    text_lines: list[str] = []

    for m in messages:
        content = m.content
        if isinstance(content, str):
            hf_messages.append({"role": m.role, "content": content})
            text_lines.append(f"{m.role}: {content}")
            continue

        hf_parts: list[dict[str, Any]] = []
        text_chunks: list[str] = []
        for part in content:
            ptype = part.get("type")
            if ptype == "text":
                text = str(part.get("text", ""))
                hf_parts.append({"type": "text", "text": text})
                text_chunks.append(text)
            elif ptype == "image_url":
                url_obj = part.get("image_url") or {}
                url = url_obj.get("url") if isinstance(url_obj, dict) else url_obj
                if not url:
                    raise ValueError("image_url part missing url")
                img = load_image_url(str(url))
                images.append(img)
                hf_parts.append({"type": "image", "image": img})
                text_chunks.append("<image>")
            # Unknown part types ignored for forward compat.
        if len(hf_parts) == 1 and hf_parts[0].get("type") == "text":
            hf_messages.append({"role": m.role, "content": hf_parts[0]["text"]})
        else:
            hf_messages.append({"role": m.role, "content": hf_parts})
        text_lines.append(f"{m.role}: {''.join(text_chunks)}")

    text_lines.append("assistant:")
    return ParsedChat(
        hf_messages=hf_messages,
        images=images,
        text_fallback="\n".join(text_lines),
    )
