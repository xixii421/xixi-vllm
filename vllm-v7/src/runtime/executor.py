from __future__ import annotations

import multiprocessing as mp
import os
import time
import traceback
from dataclasses import dataclass
from multiprocessing.connection import Connection
from typing import Any

from ..config import EngineConfig
from ..scheduling import Sequence
from .worker import Worker
from .zmq_rpc import ZmqCodec, bind_pull, bind_push, connect_pull, connect_push, ipc_path


def _unlink_ipc(path: str) -> None:
    if path.startswith("ipc://"):
        fs = path[len("ipc://") :]
        try:
            os.unlink(fs)
        except FileNotFoundError:
            pass


def _worker_pipe_main(
    config: EngineConfig,
    rank: int,
    tp_rank: int,
    pp_rank: int,
    local_device: int,
    is_driver: bool,
    conn: Connection,
) -> None:
    os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
    worker: Worker | None = None
    try:
        worker = Worker(config, rank, tp_rank, pp_rank, local_device)
        worker.init_environment()
        conn.send(("ready", None))
        while True:
            msg = conn.recv()
            if msg is None or msg[0] == "shutdown":
                break
            method, args = msg
            try:
                result = getattr(worker, method)(*args)
                conn.send(("ok", result))
            except Exception as e:
                conn.send(("err", f"{type(e).__name__}: {e}\n{traceback.format_exc()}"))
                break
    except Exception as e:
        try:
            conn.send(("err", f"{type(e).__name__}: {e}\n{traceback.format_exc()}"))
        except Exception:
            pass
    finally:
        if worker is not None:
            try:
                worker.destroy_environment()
            except Exception:
                pass


def _worker_zmq_main(
    config: EngineConfig,
    rank: int,
    tp_rank: int,
    pp_rank: int,
    local_device: int,
    is_driver: bool,
    ready_conn: Connection,
    cmd_path: str,
    response_path: str,
) -> None:
    """Reliable fan-out: each worker PULL commands; driver PUSH results.

    Avoids ZMQ PUB/SUB slow-joiner drops that hang the first RPC.
    """
    import zmq

    os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
    worker: Worker | None = None
    ctx = zmq.Context.instance()
    pull = connect_pull(ctx, cmd_path)
    push = connect_push(ctx, response_path)
    try:
        worker = Worker(config, rank, tp_rank, pp_rank, local_device)
        worker.init_environment()
        ready_conn.send(("ready", None))
        while True:
            msg = ZmqCodec.loads(pull.recv())
            if msg is None or msg[0] == "shutdown":
                break
            method, args = msg
            try:
                if method == "__report_kv_blocks__":
                    result = worker.get_num_kvcache_blocks()
                    push.send(ZmqCodec.dumps(("ok", result)))
                    continue
                result = getattr(worker, method)(*args)
                if is_driver:
                    push.send(ZmqCodec.dumps(("ok", result)))
            except Exception as e:
                err = f"{type(e).__name__}: {e}\n{traceback.format_exc()}"
                push.send(ZmqCodec.dumps(("err", err)))
                break
    except Exception as e:
        try:
            ready_conn.send(("err", f"{type(e).__name__}: {e}\n{traceback.format_exc()}"))
        except Exception:
            pass
    finally:
        if worker is not None:
            try:
                worker.destroy_environment()
            except Exception:
                pass
        pull.close(0)
        push.close(0)


@dataclass
class _WorkerHandle:
    proc: mp.Process
    rank: int
    is_driver: bool
    conn: Connection | None = None
    cmd_path: str | None = None


class Executor:
    def __init__(self, config: EngineConfig):
        self.config = config
        self.tp_size = config.tp_size
        self.pp_size = config.pp_size
        self.world_size = config.world_size
        self.device_ids = config.resolved_device_ids()
        self.rpc_backend = getattr(config, "rpc_backend", "pipe")
        self._workers: list[_WorkerHandle] = []
        self._driver: _WorkerHandle | None = None
        self._zmq_ctx = None
        self._cmd_socks: list = []
        self._pull = None
        self._ipc_paths: list[str] = []
        if self.rpc_backend == "zmq":
            self._init_zmq()
        else:
            self._init_pipe()
        assert self._driver is not None
        self._wait_ready()

    def _init_pipe(self) -> None:
        ctx = mp.get_context("spawn")
        for pp_rank in range(self.pp_size):
            for tp_rank in range(self.tp_size):
                rank = pp_rank * self.tp_size + tp_rank
                is_driver = tp_rank == 0 and pp_rank == self.pp_size - 1
                parent_conn, child_conn = ctx.Pipe(duplex=True)
                proc = ctx.Process(
                    target=_worker_pipe_main,
                    name=f"vllm-v7-worker-{rank}",
                    args=(
                        self.config,
                        rank,
                        tp_rank,
                        pp_rank,
                        self.device_ids[rank],
                        is_driver,
                        child_conn,
                    ),
                )
                proc.start()
                child_conn.close()
                handle = _WorkerHandle(
                    proc=proc, conn=parent_conn, rank=rank, is_driver=is_driver
                )
                self._workers.append(handle)
                if is_driver:
                    self._driver = handle

    def _init_zmq(self) -> None:
        import zmq

        prefix = f"{self.config.zmq_ipc_prefix}_{os.getpid()}"
        response_path = ipc_path(prefix, "response")
        _unlink_ipc(response_path)
        self._zmq_ctx = zmq.Context.instance()
        self._pull = bind_pull(self._zmq_ctx, response_path)
        self._ipc_paths.append(response_path)

        ctx = mp.get_context("spawn")
        for pp_rank in range(self.pp_size):
            for tp_rank in range(self.tp_size):
                rank = pp_rank * self.tp_size + tp_rank
                is_driver = tp_rank == 0 and pp_rank == self.pp_size - 1
                cmd_path = ipc_path(prefix, f"cmd{rank}")
                _unlink_ipc(cmd_path)
                sock = bind_push(self._zmq_ctx, cmd_path)
                self._cmd_socks.append(sock)
                self._ipc_paths.append(cmd_path)
                parent_conn, child_conn = ctx.Pipe(duplex=True)
                proc = ctx.Process(
                    target=_worker_zmq_main,
                    name=f"vllm-v7-zmq-worker-{rank}",
                    args=(
                        self.config,
                        rank,
                        tp_rank,
                        pp_rank,
                        self.device_ids[rank],
                        is_driver,
                        child_conn,
                        cmd_path,
                        response_path,
                    ),
                )
                proc.start()
                child_conn.close()
                handle = _WorkerHandle(
                    proc=proc,
                    conn=parent_conn,
                    rank=rank,
                    is_driver=is_driver,
                    cmd_path=cmd_path,
                )
                self._workers.append(handle)
                if is_driver:
                    self._driver = handle
        # Allow workers to connect PULL before first send.
        time.sleep(0.5)

    def _wait_ready(self, timeout: float = 300.0) -> None:
        for w in self._workers:
            assert w.conn is not None
            if not w.conn.poll(timeout):
                self.shutdown()
                raise RuntimeError(f"Worker {w.rank} did not become ready in time")
            status, payload = w.conn.recv()
            if status != "ready":
                self.shutdown()
                raise RuntimeError(f"Worker {w.rank} failed to start: {payload}")

    def _broadcast(self, method: str, args: tuple[Any, ...]) -> Any:
        if self.rpc_backend == "zmq":
            return self._broadcast_zmq(method, args)
        return self._broadcast_pipe(method, args)

    def _broadcast_pipe(self, method: str, args: tuple[Any, ...]) -> Any:
        for w in self._workers:
            assert w.conn is not None
            w.conn.send((method, args))
        driver_result = None
        errors = []
        for w in self._workers:
            assert w.conn is not None
            status, payload = w.conn.recv()
            if status == "err":
                errors.append(f"rank {w.rank}: {payload}")
            elif w.is_driver:
                driver_result = payload
        if errors:
            self.shutdown()
            raise RuntimeError("Worker error(s):\n" + "\n".join(errors))
        return driver_result

    def _broadcast_zmq(self, method: str, args: tuple[Any, ...]) -> Any:
        assert self._pull is not None and self._cmd_socks
        payload = ZmqCodec.dumps((method, args))
        for sock in self._cmd_socks:
            sock.send(payload)
        raw = self._pull.recv()
        status, result = ZmqCodec.loads(raw)
        if status == "err":
            self.shutdown()
            raise RuntimeError(f"Driver worker error: {result}")
        return result

    def execute_model(
        self, seqs: list[Sequence], is_prefill: bool
    ) -> tuple[list[int], list | None]:
        result = self._broadcast("execute_model", (seqs, is_prefill))
        assert result is not None
        token_ids, mrope_updates = result
        assert token_ids is not None
        return token_ids, mrope_updates

    def get_num_kvcache_blocks(self) -> int:
        if self.rpc_backend == "zmq":
            return self._get_num_kvcache_blocks_zmq()
        return self._get_num_kvcache_blocks_pipe()

    def _get_num_kvcache_blocks_pipe(self) -> int:
        for w in self._workers:
            assert w.conn is not None
            w.conn.send(("get_num_kvcache_blocks", ()))
        sizes: list[int] = []
        errors = []
        for w in self._workers:
            assert w.conn is not None
            status, payload = w.conn.recv()
            if status == "err":
                errors.append(f"rank {w.rank}: {payload}")
            else:
                sizes.append(int(payload))
        if errors:
            self.shutdown()
            raise RuntimeError("Worker error(s):\n" + "\n".join(errors))
        return min(sizes)

    def _get_num_kvcache_blocks_zmq(self) -> int:
        assert self._pull is not None and self._cmd_socks
        payload = ZmqCodec.dumps(("__report_kv_blocks__", ()))
        for sock in self._cmd_socks:
            sock.send(payload)
        sizes: list[int] = []
        errors: list[str] = []
        for _ in range(self.world_size):
            status, result = ZmqCodec.loads(self._pull.recv())
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

    def shutdown(self) -> None:
        if self.rpc_backend == "zmq" and self._cmd_socks:
            try:
                payload = ZmqCodec.dumps(("shutdown",))
                for sock in self._cmd_socks:
                    sock.send(payload)
            except Exception:
                pass
        for w in self._workers:
            try:
                if self.rpc_backend != "zmq" and w.conn is not None and w.proc.is_alive():
                    w.conn.send(("shutdown",))
            except Exception:
                pass
        for w in self._workers:
            try:
                w.proc.join(timeout=10)
            except Exception:
                pass
            if w.proc.is_alive():
                w.proc.terminate()
                w.proc.join(timeout=5)
            if w.conn is not None:
                try:
                    w.conn.close()
                except Exception:
                    pass
        for sock in self._cmd_socks:
            try:
                sock.close(0)
            except Exception:
                pass
        self._cmd_socks.clear()
        if self._pull is not None:
            try:
                self._pull.close(0)
            except Exception:
                pass
            self._pull = None
        for p in self._ipc_paths:
            _unlink_ipc(p)
        self._ipc_paths.clear()
        self._workers.clear()

    def __del__(self):
        try:
            self.shutdown()
        except Exception:
            pass
