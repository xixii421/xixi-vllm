from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Optional

from ..config import EngineConfig
from ..sampling import SamplingParams
from ..scheduling import Sequence
from .engine import Engine

if TYPE_CHECKING:
    from PIL import Image


@dataclass
class _RequestState:
    seq: Sequence
    queue: asyncio.Queue[dict[str, Any] | Exception | None]
    emitted: int = 0
    done: bool = False


class AsyncLLM:
    """Async continuous-batching façade over the sync Engine.

    Concurrent ``generate`` calls only serialize briefly around engine
    mutations. A single background loop drives ``Engine.step`` so multiple
    sequences share batches (ullm-style control plane, in-process).

    Only uses Engine public APIs (no direct scheduler / tokenizer access).
    """

    def __init__(self, config: EngineConfig):
        self.config = config
        self._engine = Engine(config)
        # Short critical section only — never held across a full generate().
        self._lock = asyncio.Lock()
        self._requests: dict[int, _RequestState] = {}
        self._has_work = asyncio.Event()
        self._stopped = False
        self._loop_task: asyncio.Task[None] | None = None
        # Abort must be enqueue-able without await (async-gen aclose / cancel).
        self._abort_ids: set[int] = set()

    @property
    def engine(self) -> Engine:
        return self._engine

    def _ensure_loop(self) -> None:
        if self._loop_task is None or self._loop_task.done():
            self._loop_task = asyncio.create_task(
                self._run_loop(), name="vllm-v7-async-llm-loop"
            )

    def _request_abort(self, seq_id: int) -> None:
        """Non-blocking abort signal safe for GeneratorExit cleanup."""
        self._abort_ids.add(seq_id)
        self._has_work.set()

    def _drain_aborts(self) -> list[tuple[int, asyncio.Queue | None]]:
        """Apply pending aborts. Caller must hold ``_lock`` / engine ownership."""
        notified: list[tuple[int, asyncio.Queue | None]] = []
        for seq_id in list(self._abort_ids):
            self._abort_ids.discard(seq_id)
            state = self._requests.pop(seq_id, None)
            if state is None:
                continue
            self._engine.abort(state.seq)
            notified.append((seq_id, state.queue))
        return notified

    async def _run_loop(self) -> None:
        try:
            while not self._stopped:
                async with self._lock:
                    aborted = self._drain_aborts()
                    idle = (
                        self._engine.is_finished()
                        and not self._requests
                        and not self._abort_ids
                    )
                for _seq_id, queue in aborted:
                    if queue is not None:
                        queue.put_nowait(None)

                if idle:
                    self._has_work.clear()
                    # Recheck after clear to avoid lost wakeups.
                    if (
                        self._stopped
                        or self._requests
                        or self._abort_ids
                        or not self._engine.is_finished()
                    ):
                        continue
                    await self._has_work.wait()
                    continue

                async with self._lock:
                    aborted = self._drain_aborts()
                    pending = (
                        self._engine.begin_step()
                        if not self._engine.is_finished()
                        else None
                    )

                for _seq_id, queue in aborted:
                    if queue is not None:
                        queue.put_nowait(None)

                if pending is None:
                    await asyncio.sleep(0)
                    continue

                # GPU/RPC wait is deliberately outside the engine-state lock.
                # New requests may be admitted while this batch is executing;
                # aborts are committed after completion so KV pages cannot be
                # freed while a worker still references them.
                result = await asyncio.to_thread(self._engine.wait_step, pending)
                async with self._lock:
                    self._engine.finish_step(pending, result)
                    aborted = self._drain_aborts()
                    events = self._collect_events()

                for _seq_id, queue in aborted:
                    if queue is not None:
                        queue.put_nowait(None)

                for seq_id, payload in events:
                    state = self._requests.get(seq_id)
                    if state is None:
                        continue
                    state.queue.put_nowait(payload)
                    if payload is None or isinstance(payload, Exception):
                        self._requests.pop(seq_id, None)
        except Exception as exc:
            for state in list(self._requests.values()):
                state.queue.put_nowait(exc)
            self._requests.clear()
            raise

    def _collect_events(self) -> list[tuple[int, dict[str, Any] | Exception | None]]:
        """Turn committed tokens into per-request output events."""
        from .emit import collect_stream_events

        events: list[tuple[int, dict[str, Any] | Exception | None]] = []
        for seq_id, state in self._requests.items():
            if state.done:
                continue
            state.emitted, chunks, finished_sentinel = collect_stream_events(
                state.seq,
                emitted=state.emitted,
                decode_token=self._engine.decode_token,
            )
            for chunk in chunks:
                events.append((seq_id, chunk))
            if finished_sentinel:
                events.append((seq_id, None))
                state.done = True

        return events

    async def generate(
        self,
        prompt: str | list[int] | None = None,
        sampling_params: SamplingParams | None = None,
        *,
        images: Optional[list["Image.Image"]] = None,
        chat_messages: list[dict] | None = None,
        request_id: str | None = None,
    ) -> AsyncIterator[dict[str, Any]]:
        del request_id
        self._ensure_loop()
        sp = sampling_params or SamplingParams()
        queue: asyncio.Queue[dict[str, Any] | Exception | None] = asyncio.Queue()
        seq_id: int | None = None

        try:
            async with self._lock:
                seq = await asyncio.to_thread(
                    self._engine.add_request,
                    prompt,
                    sp,
                    images,
                    chat_messages=chat_messages,
                )
                seq_id = seq.seq_id
                self._requests[seq_id] = _RequestState(seq=seq, queue=queue)
                self._has_work.set()

            while True:
                item = await queue.get()
                if isinstance(item, Exception):
                    raise item
                if item is None:
                    break
                yield item
        finally:
            # Async-gen aclose / task cancel cannot reliably await here.
            if seq_id is not None and seq_id in self._requests:
                self._request_abort(seq_id)

    async def abort(self, seq_id: int) -> None:
        self._request_abort(seq_id)
        # Wait until the background loop applies it (or request already gone).
        for _ in range(1000):
            if seq_id not in self._requests and seq_id not in self._abort_ids:
                return
            await asyncio.sleep(0)

    async def generate_full(
        self,
        prompt: str | list[int] | None = None,
        sampling_params: SamplingParams | None = None,
        *,
        images: Optional[list["Image.Image"]] = None,
        chat_messages: list[dict] | None = None,
    ) -> str:
        parts: list[str] = []
        async for chunk in self.generate(
            prompt,
            sampling_params,
            images=images,
            chat_messages=chat_messages,
        ):
            parts.append(chunk["text"])
        return "".join(parts)

    async def shutdown(self) -> None:
        self._stopped = True
        self._has_work.set()
        if self._loop_task is not None:
            try:
                await self._loop_task
            except Exception:
                # Active request queues already receive the loop failure.
                pass
            self._loop_task = None
        async with self._lock:
            for seq_id in list(self._requests):
                self._abort_ids.add(seq_id)
            for _seq_id, queue in self._drain_aborts():
                if queue is not None:
                    queue.put_nowait(None)
            self._requests.clear()
            await asyncio.to_thread(self._engine.destroy)
