"""Functional coverage for Executor / futures / reply merge (fake transport)."""

from __future__ import annotations

from itertools import count
from types import SimpleNamespace

import pytest

from src.runtime.execute import Executor
from src.runtime.execute.futures import ModelExecutionFuture, PendingExecute
from src.runtime.execute.zmq_rpc import ZmqCodec
from src.sampling import SamplingParams
from src.scheduling import Sequence


class _FakeTransport:
    def __init__(self) -> None:
        self.sent: list[tuple] = []
        self._execute_q: list[tuple] = []
        self._control_q: list[tuple] = []
        self.workers = []
        self.driver = None
        self.world_size = 1

    def send_all(self, message: tuple) -> None:
        self.sent.append(message)

    def recv_execute_message(self, timeout: float | None) -> tuple:
        del timeout
        if not self._execute_q:
            raise TimeoutError("no execute reply queued")
        return self._execute_q.pop(0)

    def recv_control(self, method: str, timeout: float | None = None) -> list:
        del method, timeout
        n = self.world_size
        if len(self._control_q) < n:
            raise TimeoutError("not enough control replies")
        out = self._control_q[:n]
        del self._control_q[:n]
        return out

    def control_reply_count(self, method: str) -> int:
        del method
        return self.world_size

    def wait_ready(self, timeout: float = 1800.0) -> None:
        del timeout

    def shutdown(self) -> None:
        pass

    def push_execute(self, msg: tuple) -> None:
        self._execute_q.append(msg)

    def push_control(self, *msgs: tuple) -> None:
        self._control_q.extend(msgs)


def _executor(*, pp_size: int = 1, world_size: int = 1) -> Executor:
    import threading

    ex = object.__new__(Executor)
    ex.config = SimpleNamespace(is_multimodal=False)
    ex.tp_size = world_size // pp_size
    ex.pp_size = pp_size
    ex.world_size = world_size
    ex.rpc_backend = "zmq"
    ex._rpc_ids = count(1)
    ex._pending_execute = {}
    ex._mm_pixels_sent = set()
    ex._response_lock = threading.Lock()
    tx = _FakeTransport()
    tx.world_size = world_size
    ex._transport = tx
    return ex


def _seq() -> Sequence:
    seq = Sequence(
        [1, 2, 3],
        SamplingParams(temperature=0.0, max_tokens=2),
        block_size=16,
    )
    seq.block_table = [0]
    seq.num_cached_tokens = 3
    seq.num_scheduled_tokens = 1
    return seq


def test_pending_pp2_merges_driver_and_mrope() -> None:
    ex = _executor(pp_size=2, world_size=2)
    fut = ModelExecutionFuture(ex, 1)
    pending = PendingExecute(fut, expected_replies=2)
    # mRoPE source first (first-PP)
    pending.ingest("ok", [([0], ["mrope"]), False, True])
    assert not pending.try_finish()
    # driver (last-PP) with tokens
    pending.ingest("ok", [([42], None), True, False])
    assert pending.try_finish()
    assert fut.result() == ([42], ["mrope"])


def test_pending_fail_fast_on_error() -> None:
    ex = _executor()
    fut = ModelExecutionFuture(ex, 1)
    pending = PendingExecute(fut, expected_replies=2)
    pending.ingest("err", ["boom"])
    assert pending.try_finish()
    with pytest.raises(RuntimeError, match="boom"):
        fut.result()


def test_submit_wire_format_and_result() -> None:
    ex = _executor()
    tx: _FakeTransport = ex._transport  # type: ignore[assignment]
    fut = ex.submit_execute_model([_seq()], False)
    assert tx.sent[0][0] == 1 and tx.sent[0][1] == "execute_model"
    tx.push_execute((1, "ok", ([3, 4], None), True, True))
    assert fut.result() == ([3, 4], None)


def test_execute_model_sync_wrapper() -> None:
    """``execute_model`` is submit+result; inject reply from send_all hook."""
    ex = _executor()
    tx: _FakeTransport = ex._transport  # type: ignore[assignment]
    orig = tx.send_all

    def _send_and_queue(message: tuple) -> None:
        orig(message)
        rpc_id = message[0]
        tx.push_execute((rpc_id, "ok", ([9], None), True, True))

    tx.send_all = _send_and_queue  # type: ignore[method-assign]
    assert ex.execute_model([_seq()], False) == ([9], None)


def test_pp2_executor_waits_two_replies() -> None:
    ex = _executor(pp_size=2, world_size=2)
    tx: _FakeTransport = ex._transport  # type: ignore[assignment]
    fut = ex.submit_execute_model([_seq()], False)
    tx.push_execute((1, "ok", ([0], ["u"]), False, True))
    tx.push_execute((1, "ok", ([7], None), True, False))
    assert fut.result() == ([7], ["u"])


def test_orphan_reply_ignored() -> None:
    ex = _executor()
    tx: _FakeTransport = ex._transport  # type: ignore[assignment]
    fut = ex.submit_execute_model([_seq()], False)
    tx.push_execute((1, "err", "fail"))
    with pytest.raises(RuntimeError, match="fail"):
        fut.result()
    # Orphan OK for same id must not crash pump
    tx.push_execute((1, "ok", ([1], None), True, True))
    ex._pump_execute(None)


def test_control_get_num_kvcache_blocks() -> None:
    ex = _executor(world_size=2)
    tx: _FakeTransport = ex._transport  # type: ignore[assignment]
    tx.world_size = 2
    tx.push_control(("ok", 100), ("ok", 80))
    assert ex.get_num_kvcache_blocks() == 80
    assert tx.sent[-1] == ("get_num_kvcache_blocks", ())


def test_control_get_sys_profile_sorts_ranks() -> None:
    ex = _executor(world_size=2)
    tx: _FakeTransport = ex._transport  # type: ignore[assignment]
    tx.world_size = 2
    tx.push_control(("ok", {"t": 1}, 1), ("ok", {"t": 0}, 0))
    out = ex.get_sys_profile()
    assert [d["_rank"] for d in out] == [0, 1]


def test_control_drains_pending_execute() -> None:
    ex = _executor()
    tx: _FakeTransport = ex._transport  # type: ignore[assignment]
    fut = ex.submit_execute_model([_seq()], False)
    tx.push_execute((1, "ok", ([5], None), True, True))
    tx.push_control(("ok", None))
    ex.reset_sys_profile()
    assert fut.done()
    assert fut.result() == ([5], None)


def test_shutdown_errors_pending_futures() -> None:
    ex = _executor()
    fut = ex.submit_execute_model([_seq()], False)
    ex.shutdown()
    with pytest.raises(RuntimeError, match="shut down"):
        fut.result()


def test_out_of_order_rpc_ids() -> None:
    ex = _executor()
    tx: _FakeTransport = ex._transport  # type: ignore[assignment]
    a = ex.submit_execute_model([_seq()], False)
    b = ex.submit_execute_model([_seq()], False)
    tx.push_execute((2, "ok", ([22], None), True, True))
    tx.push_execute((1, "ok", ([11], None), True, True))
    assert a.result() == ([11], None)
    assert b.result() == ([22], None)
