"""Async execute_model futures for the multiproc Executor."""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from .merge import merge_execute_model_results

if TYPE_CHECKING:
    from .executor import Executor


class ModelExecutionFuture:
    """FIFO RPC future backed by the executor's single response owner."""

    def __init__(self, executor: Executor, rpc_id: int) -> None:
        self._executor = executor
        self.rpc_id = rpc_id
        self._event = threading.Event()
        self._result: tuple[list[int], list | None] | None = None
        self._error: BaseException | None = None

    def done(self) -> bool:
        return self._event.is_set()

    def result(self, timeout: float | None = None) -> tuple[list[int], list | None]:
        self._executor._wait_for_execute(self, timeout)
        if self._error is not None:
            raise self._error
        assert self._result is not None
        return self._result

    def _set_result(self, result: tuple[list[int], list | None]) -> None:
        self._result = result
        self._event.set()

    def _set_error(self, error: BaseException) -> None:
        self._error = error
        self._event.set()


@dataclass
class PendingExecute:
    future: ModelExecutionFuture
    expected_replies: int
    replies: int = 0
    driver_payload: Any = None
    mrope_payload: Any = None
    errors: list[str] = field(default_factory=list)

    def ingest(self, status: str, rest: list[Any]) -> None:
        self.replies += 1
        if status == "err":
            self.errors.append(str(rest[0] if rest else "unknown worker error"))
            return
        result = rest[0]
        is_driver = bool(rest[1]) if len(rest) > 1 else False
        is_mrope_source = bool(rest[2]) if len(rest) > 2 else False
        if is_driver:
            self.driver_payload = result
        if is_mrope_source:
            self.mrope_payload = result

    def try_finish(self) -> bool:
        """Complete the future on error (fail-fast) or when all OK replies arrive."""
        if self.errors:
            self.future._set_error(
                RuntimeError("Worker error(s):\n" + "\n".join(self.errors))
            )
            return True
        if self.replies < self.expected_replies:
            return False
        try:
            result = merge_execute_model_results(
                driver=self.driver_payload,
                mrope_source=self.mrope_payload,
            )
            self.future._set_result(result)
        except BaseException as exc:
            self.future._set_error(exc)
        return True
