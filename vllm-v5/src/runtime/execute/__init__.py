"""Stage 2 — distributed execution (engine ↔ workers).

Owns process spawn, ZMQ/pipe RPC, and per-GPU ``Worker`` loops.

Layout::

    executor.py       Facade (futures + control RPC)
    transport.py      Pipe / ZMQ spawn + send/recv
    worker_proc.py    Subprocess RPC loop
    futures.py        ModelExecutionFuture / PendingExecute
    reply.py          Who replies to which RPC
    worker.py         Per-GPU Worker (NCCL + ModelRunner)
    request.py        Compact decode RPC payloads
    zmq_rpc.py        ZMQ helpers + pickle codec

Heavy modules (``Worker`` / ``ModelRunner`` / FlashInfer) are imported lazily
so Engine can pin ``FLASHINFER_CUDA_ARCH_LIST`` first.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from .futures import ModelExecutionFuture
from .merge import merge_execute_model_results
from .request import (
    DecodeSequenceMetadata,
    PrefillSequenceWire,
    compact_execute_sequences,
)

if TYPE_CHECKING:
    from .executor import Executor
    from .local import LocalExecutor
    from .worker import Worker

__all__ = [
    "DecodeSequenceMetadata",
    "PrefillSequenceWire",
    "Executor",
    "LocalExecutor",
    "ModelExecutionFuture",
    "Worker",
    "compact_execute_sequences",
    "merge_execute_model_results",
]


def __getattr__(name: str) -> Any:
    if name == "Executor":
        from .executor import Executor

        return Executor
    if name == "LocalExecutor":
        from .local import LocalExecutor

        return LocalExecutor
    if name == "Worker":
        from .worker import Worker

        return Worker
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
