"""Shared streaming token emission helpers (CPU-only)."""

from __future__ import annotations

from collections.abc import Callable, Iterator
from typing import Any

from ..scheduling import Sequence


def iter_new_token_ids(seq: Sequence, emitted: int) -> Iterator[tuple[int, int]]:
    """Yield ``(new_emitted_count, token_id)`` for tokens after ``emitted``."""
    while emitted < seq.num_completion_tokens:
        tid = seq.token_ids[seq.num_prompt_tokens + emitted]
        emitted += 1
        yield emitted, tid


def collect_stream_events(
    seq: Sequence,
    *,
    emitted: int,
    decode_token: Callable[[int], str],
) -> tuple[int, list[dict[str, Any]], bool]:
    """Build AsyncLLM-style chunk events for newly completed tokens.

    Returns ``(new_emitted, events, finished_sentinel)`` where
    ``finished_sentinel`` means the caller should also enqueue a ``None``
    end-of-stream marker.
    """
    events: list[dict[str, Any]] = []
    for emitted, tid in iter_new_token_ids(seq, emitted):
        finished = seq.is_finished and emitted >= seq.num_completion_tokens
        events.append(
            {
                "token_id": tid,
                "text": decode_token(tid),
                "finished": finished,
            }
        )
    finished_sentinel = False
    if seq.is_finished:
        if emitted == 0 and not events:
            events.append(
                {
                    "token_id": None,
                    "text": "",
                    "finished": True,
                }
            )
        finished_sentinel = True
    return emitted, events, finished_sentinel
