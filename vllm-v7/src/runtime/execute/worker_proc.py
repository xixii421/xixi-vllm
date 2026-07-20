"""Subprocess entry: Worker init + RPC loop (pipe or ZMQ)."""

from __future__ import annotations

import os
import traceback
from multiprocessing.connection import Connection
from typing import Any

from typing import TYPE_CHECKING

from ...config import EngineConfig
from .reply import include_rank_in_reply, should_reply
from .zmq_rpc import ZmqCodec, connect_pull, connect_push

if TYPE_CHECKING:
    from .worker import Worker


def _safe_destroy(worker: Worker | None) -> None:
    if worker is None:
        return
    try:
        worker.destroy_environment()
    except Exception:
        pass


def _send(
    *,
    transport: str,
    pipe_conn: Connection | None,
    push,
    payload: tuple[Any, ...],
) -> None:
    if transport == "pipe":
        assert pipe_conn is not None
        pipe_conn.send(payload)
    else:
        assert push is not None
        push.send(ZmqCodec.dumps(payload))


def _reply_ok(
    *,
    transport: str,
    method: str,
    rpc_id: int | None,
    result: Any,
    is_driver: bool,
    is_mrope_source: bool,
    rank: int,
    pipe_conn: Connection | None,
    push,
) -> None:
    if not should_reply(
        method, is_driver=is_driver, is_mrope_source=is_mrope_source
    ):
        return

    if method == "execute_model":
        if rpc_id is not None:
            payload: tuple[Any, ...] = (
                rpc_id,
                "ok",
                result,
                is_driver,
                is_mrope_source,
            )
        else:
            payload = ("ok", result, is_driver, is_mrope_source)
        _send(transport=transport, pipe_conn=pipe_conn, push=push, payload=payload)
        return

    if include_rank_in_reply(method):
        payload = ("ok", result, rank)
    else:
        payload = ("ok", result)
    _send(transport=transport, pipe_conn=pipe_conn, push=push, payload=payload)


def _reply_err(
    *,
    transport: str,
    rpc_id: int | None,
    err: str,
    pipe_conn: Connection | None,
    push,
) -> None:
    payload: tuple[Any, ...] = (
        (rpc_id, "err", err) if rpc_id is not None else ("err", err)
    )
    # Errors always surface (even from non-driver ranks) so the parent can fail.
    _send(transport=transport, pipe_conn=pipe_conn, push=push, payload=payload)


def worker_main(
    config: EngineConfig,
    rank: int,
    tp_rank: int,
    pp_rank: int,
    local_device: int,
    is_driver: bool,
    is_mrope_source: bool,
    ready_conn: Connection,
    *,
    transport: str,
    cmd_path: str | None = None,
    response_path: str | None = None,
) -> None:
    """Shared worker loop for pipe and ZMQ transports.

    Pipe: ``ready_conn`` is the duplex Pipe used for both ready and RPC.
    ZMQ: ``ready_conn`` is ready-only; commands go over ZMQ sockets.
    """
    os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
    # Pin *this* GPU's arch only. Passing the full hetero device list (e.g.
    # Blackwell SM120 + A100 SM80) makes FlashInfer try to JIT SM120 on the
    # A100 worker → hang / "SM 12.x requires CUDA >= 12.9" with toolkit 12.8.
    from ...utils.cuda_env import apply_flashinfer_arch_list

    apply_flashinfer_arch_list([local_device], overwrite=True)
    from .worker import Worker

    worker: Worker | None = None
    pull = push = None
    pipe_conn = ready_conn if transport == "pipe" else None
    try:
        if transport == "zmq":
            import zmq

            assert cmd_path and response_path
            ctx = zmq.Context.instance()
            pull = connect_pull(ctx, cmd_path)
            push = connect_push(ctx, response_path)

        worker = Worker(config, rank, tp_rank, pp_rank, local_device)
        worker.init_environment()
        ready_conn.send(("ready", None))

        while True:
            if transport == "pipe":
                assert pipe_conn is not None
                msg = pipe_conn.recv()
            else:
                assert pull is not None
                msg = ZmqCodec.loads(pull.recv())

            if msg is None or msg[0] == "shutdown":
                break
            rpc_id = None
            if len(msg) == 3 and isinstance(msg[0], int):
                rpc_id, method, args = msg
            else:
                method, args = msg
            try:
                result = getattr(worker, method)(*args)
                _reply_ok(
                    transport=transport,
                    method=method,
                    rpc_id=rpc_id,
                    result=result,
                    is_driver=is_driver,
                    is_mrope_source=is_mrope_source,
                    rank=rank,
                    pipe_conn=pipe_conn,
                    push=push,
                )
            except Exception as e:
                err = f"{type(e).__name__}: {e}\n{traceback.format_exc()}"
                _reply_err(
                    transport=transport,
                    rpc_id=rpc_id,
                    err=err,
                    pipe_conn=pipe_conn,
                    push=push,
                )
                break
    except Exception as e:
        try:
            ready_conn.send(
                ("err", f"{type(e).__name__}: {e}\n{traceback.format_exc()}")
            )
        except Exception:
            pass
    finally:
        _safe_destroy(worker)
        if pull is not None:
            pull.close(0)
        if push is not None:
            push.close(0)
