from __future__ import annotations

from itertools import count
from types import SimpleNamespace

from src.runtime.execute import Executor
from src.runtime.execute.zmq_rpc import ZmqCodec
from src.sampling import SamplingParams
from src.scheduling import Sequence


class _Push:
    def __init__(self) -> None:
        self.messages: list[bytes] = []

    def send(self, payload: bytes) -> None:
        self.messages.append(payload)


class _Pull:
    def __init__(self) -> None:
        self.messages: list[bytes] = []
        self.recv_calls = 0

    def poll(self, timeout: int) -> bool:
        del timeout
        return bool(self.messages)

    def recv(self) -> bytes:
        self.recv_calls += 1
        return self.messages.pop(0)


class _FakeZmqTransport:
    def __init__(self) -> None:
        self._cmd = _Push()
        self._pull = _Pull()
        self.workers = []
        self.driver = None

    def send_all(self, message: tuple) -> None:
        self._cmd.send(ZmqCodec.dumps(message))

    def recv_execute_message(self, timeout: float | None) -> tuple:
        del timeout
        return ZmqCodec.loads(self._pull.recv())

    def recv_control(self, method: str, timeout: float | None = None) -> list:
        del method, timeout
        raise NotImplementedError

    def control_reply_count(self, method: str) -> int:
        del method
        return 1

    def wait_ready(self, timeout: float = 1800.0) -> None:
        del timeout

    def shutdown(self) -> None:
        pass


def _executor() -> Executor:
    executor = object.__new__(Executor)
    executor.config = SimpleNamespace(is_multimodal=False)
    executor.tp_size = 1
    executor.pp_size = 1
    executor.world_size = 1
    executor.rpc_backend = "zmq"
    executor._rpc_ids = count(1)
    executor._pending_execute = {}
    executor._mm_pixels_sent = set()
    import threading

    executor._response_lock = threading.Lock()
    executor._transport = _FakeZmqTransport()
    return executor


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


def _reply(rpc_id: int, tokens: list[int]) -> bytes:
    result = (tokens, None)
    return ZmqCodec.dumps((rpc_id, "ok", result, True, True))


def test_submit_execute_model_does_not_receive() -> None:
    executor = _executor()
    future = executor.submit_execute_model([_seq()], False)

    assert not future.done()
    assert executor._transport._pull.recv_calls == 0
    rpc_id, method, _args = ZmqCodec.loads(executor._transport._cmd.messages[0])
    assert (rpc_id, method) == (1, "execute_model")

    executor._transport._pull.messages.append(_reply(1, [7]))
    assert future.result() == ([7], None)
    assert executor._transport._pull.recv_calls == 1


def test_futures_route_out_of_order_replies_by_rpc_id() -> None:
    executor = _executor()
    first = executor.submit_execute_model([_seq()], False)
    second = executor.submit_execute_model([_seq()], False)
    executor._transport._pull.messages.extend([_reply(2, [22]), _reply(1, [11])])

    assert first.result() == ([11], None)
    assert second.done()
    assert second.result() == ([22], None)
