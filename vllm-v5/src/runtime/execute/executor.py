from __future__ import annotations

import threading
import warnings
from itertools import count
from typing import Any

from ...config import EngineConfig
from ...scheduling import Sequence
from .futures import ModelExecutionFuture, PendingExecute
from .reply import execute_expected_replies
from .request import collect_mm_pixels, compact_execute_sequences
from .transport import Transport, create_transport


class Executor:
    def __init__(self, config: EngineConfig):
        self.config = config
        # Mirrored for tests / stubs that patch sizes without rebuilding config.
        self.tp_size = config.tp_size
        self.pp_size = config.pp_size
        self.world_size = config.world_size
        self.rpc_backend = config.rpc_backend
        self._rpc_ids = count(1)
        self._pending_execute: dict[int, PendingExecute] = {}
        # Exactly one thread may own the response path at a time.
        self._response_lock = threading.Lock()
        # Uids already pushed to workers (avoid re-shipping pixels).
        self._mm_pixels_sent: set[str] = set()
        self._transport: Transport | None = create_transport(
            config, on_fail=self.shutdown
        )
        self._transport.wait_ready()

    @property
    def _tx(self) -> Transport:
        assert self._transport is not None
        return self._transport

    @property
    def workers(self):
        """Process handles (not in-process ``Worker`` modules)."""
        return self._tx.workers

    # --- execute_model (async future) ---------------------------------

    def submit_execute_model(
        self, seqs: list[Sequence], is_prefill: bool
    ) -> ModelExecutionFuture:
        """Enqueue model execution and return without receiving worker replies."""
        new_pixels: dict = {}
        if is_prefill and self.config.is_multimodal:
            for uid, data in collect_mm_pixels(seqs).items():
                if uid not in self._mm_pixels_sent:
                    new_pixels[uid] = data
                    self._mm_pixels_sent.add(uid)
        wire_seqs = compact_execute_sequences(seqs, is_prefill)
        rpc_id = next(self._rpc_ids)
        future = ModelExecutionFuture(self, rpc_id)
        expected = execute_expected_replies(self.pp_size)
        message = (rpc_id, "execute_model", (wire_seqs, is_prefill))
        # Hold the response lock so submit cannot interleave between a control
        # RPC's drain and its send (which would corrupt the reply stream).
        with self._response_lock:
            # Only drain when inserting a control RPC ahead of execute; otherwise
            # concurrent futures must stay pending (out-of-order replies OK).
            if new_pixels:
                self._drain_execute_unlocked()
                self._tx.send_all(("register_mm_pixels", (new_pixels,)))
                for msg in self._tx.recv_control("register_mm_pixels"):
                    status, result, *_ = msg
                    if status == "err":
                        future._set_error(
                            RuntimeError(f"register_mm_pixels: {result}")
                        )
                        return future
            self._pending_execute[rpc_id] = PendingExecute(future, expected)
            try:
                self._tx.send_all(message)
            except BaseException as exc:
                self._pending_execute.pop(rpc_id, None)
                future._set_error(exc)
        return future

    def execute_model(
        self, seqs: list[Sequence], is_prefill: bool
    ) -> tuple[list[int], list | None]:
        """Deprecated sync helper for tests; production uses ``submit_execute_model``."""
        warnings.warn(
            "Executor.execute_model is deprecated; use submit_execute_model().result()",
            DeprecationWarning,
            stacklevel=2,
        )
        from ...utils.profiler import Timer

        with Timer("rpc.execute_model"):
            token_ids, mrope_updates = self.submit_execute_model(
                seqs, is_prefill
            ).result()
        assert token_ids is not None
        return token_ids, mrope_updates

    def _pump_execute(self, timeout: float | None) -> None:
        msg = self._tx.recv_execute_message(timeout)
        if not msg or not isinstance(msg[0], int):
            raise RuntimeError(
                f"unexpected RPC response while execute pending: {msg!r}"
            )
        rpc_id, status, *rest = msg
        pending = self._pending_execute.get(rpc_id)
        if pending is None:
            # Orphan after fail-fast / already completed.
            return
        pending.ingest(status, rest)
        if pending.try_finish():
            self._pending_execute.pop(rpc_id, None)

    def _wait_for_execute(
        self, future: ModelExecutionFuture, timeout: float | None
    ) -> None:
        if future.done():
            return
        with self._response_lock:
            while not future.done():
                if future.rpc_id not in self._pending_execute and not future.done():
                    raise RuntimeError(
                        f"execute RPC {future.rpc_id} is no longer pending"
                    )
                self._pump_execute(timeout)

    def _drain_execute(self) -> None:
        with self._response_lock:
            self._drain_execute_unlocked()

    def _drain_execute_unlocked(self) -> None:
        """Drain pending executes; caller must hold ``_response_lock``."""
        while self._pending_execute:
            self._pump_execute(None)

    # --- control RPCs (all-gather, under response lock) ---------------

    def _control(self, method: str, args: tuple[Any, ...] = ()) -> list[Any]:
        """Broadcast a control method; return raw reply tuples from all ranks."""
        with self._response_lock:
            self._drain_execute_unlocked()
            self._tx.send_all((method, args))
            return self._tx.recv_control(method)

    def get_num_kvcache_blocks(self) -> int:
        replies = self._control("get_num_kvcache_blocks")
        sizes: list[int] = []
        errors: list[str] = []
        for msg in replies:
            status, result, *_ = msg
            if status == "err":
                errors.append(str(result))
            else:
                sizes.append(int(result))
        if errors:
            self.shutdown()
            raise RuntimeError("Worker error(s):\n" + "\n".join(errors))
        if not sizes:
            raise RuntimeError("no workers reported kvcache block count")
        return min(sizes)

    def reset_sys_profile(self) -> None:
        for msg in self._control("reset_sys_profile"):
            status, result, *_ = msg
            if status == "err":
                raise RuntimeError(f"reset_sys_profile: {result}")

    def get_sys_profile(self) -> list[dict]:
        """Collect profiler summaries from all workers."""
        out: list[dict] = []
        errors: list[str] = []
        for msg in self._control("get_sys_profile"):
            status, result, *rest = msg
            if status == "err":
                errors.append(str(result))
                continue
            rank = rest[0] if rest else len(out)
            entry = result if isinstance(result, dict) else {}
            out.append({**entry, "_rank": rank})
        if errors:
            raise RuntimeError("get_sys_profile errors:\n" + "\n".join(errors))
        out.sort(key=lambda d: int(d.get("_rank", 0)))
        return out

    def shutdown(self) -> None:
        error = RuntimeError("executor shut down before execute RPC completed")
        for pending in self._pending_execute.values():
            pending.future._set_error(error)
        self._pending_execute.clear()
        if self._transport is not None:
            try:
                self._transport.shutdown()
            except Exception:
                pass
            self._transport = None

    def __del__(self):
        try:
            self.shutdown()
        except Exception:
            pass
