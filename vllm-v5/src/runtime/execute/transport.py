"""Pipe / ZMQ transports for engine ↔ worker control RPC."""

from __future__ import annotations

import multiprocessing as mp
import os
import time
from dataclasses import dataclass, field
from multiprocessing.connection import Connection, wait
from typing import Any, Protocol

from ...config import EngineConfig
from .reply import role_flags
from .worker_proc import worker_main
from .zmq_rpc import ZmqCodec, bind_pull, bind_push, ipc_path


def unlink_ipc(path: str) -> None:
    if path.startswith("ipc://"):
        fs = path[len("ipc://") :]
        try:
            os.unlink(fs)
        except FileNotFoundError:
            pass


@dataclass
class WorkerHandle:
    proc: mp.Process
    rank: int
    is_driver: bool
    is_mrope_source: bool = False
    conn: Connection | None = None
    cmd_path: str | None = None


class Transport(Protocol):
    workers: list[WorkerHandle]
    driver: WorkerHandle

    def wait_ready(self, timeout: float = 1800.0) -> None: ...
    def send_all(self, message: tuple[Any, ...]) -> None: ...
    def control_reply_count(self, method: str) -> int: ...
    def recv_control(self, method: str, timeout: float | None = None) -> list[Any]: ...
    def recv_execute_message(self, timeout: float | None) -> tuple[Any, ...]: ...
    def shutdown(self) -> None: ...


def _spawn_workers(
    config: EngineConfig,
    *,
    transport: str,
    make_child_kwargs,
) -> tuple[list[WorkerHandle], WorkerHandle]:
    ctx = mp.get_context("spawn")
    device_ids = config.resolved_device_ids()
    workers: list[WorkerHandle] = []
    driver: WorkerHandle | None = None
    for pp_rank in range(config.pp_size):
        for tp_rank in range(config.tp_size):
            rank = pp_rank * config.tp_size + tp_rank
            is_driver, is_mrope_source = role_flags(
                tp_rank, pp_rank, config.pp_size
            )
            parent_conn, child_conn = ctx.Pipe(duplex=True)
            kwargs = make_child_kwargs(rank, child_conn)
            proc = ctx.Process(
                target=worker_main,
                name=f"vllm-v7-{transport}-worker-{rank}",
                args=(
                    config,
                    rank,
                    tp_rank,
                    pp_rank,
                    device_ids[rank],
                    is_driver,
                    is_mrope_source,
                    child_conn,
                ),
                kwargs=kwargs,
            )
            proc.start()
            child_conn.close()
            handle = WorkerHandle(
                proc=proc,
                conn=parent_conn,
                rank=rank,
                is_driver=is_driver,
                is_mrope_source=is_mrope_source,
                cmd_path=kwargs.get("cmd_path"),
            )
            workers.append(handle)
            if is_driver:
                driver = handle
    assert driver is not None
    return workers, driver


def _worker_death_hint(w: WorkerHandle) -> str:
    """Extra context when a worker vanishes during RPC / ready handshake."""
    alive = w.proc.is_alive()
    code = w.proc.exitcode
    hint = f"alive={alive} exitcode={code}"
    if not alive and code not in (0, None):
        hint += (
            " — if you launched via `python -` / heredoc / `python -c`, "
            "spawn cannot re-import `<stdin>`; use `python path/to/script.py` "
            "or `python -m ...` instead"
        )
    return hint


def _wait_ready(workers: list[WorkerHandle], on_fail, timeout: float) -> None:
    """Block until every worker sends ``('ready', None)``.

    Polls in short slices so a crashed worker fails fast instead of sitting in
    ``conn.poll(1800)`` with idle CPU/GPU (looks like a hang / deadlock).
    Also ``join(0)`` to reap zombies so ``is_alive()`` cannot stick True.
    """
    slice_s = 1.0
    for w in workers:
        assert w.conn is not None
        deadline = time.monotonic() + timeout
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                on_fail()
                raise RuntimeError(
                    f"Worker {w.rank} did not become ready in time "
                    f"({_worker_death_hint(w)})"
                )
            # Reap exited children so zombies don't look indefinitely "alive".
            if w.proc.exitcode is None:
                w.proc.join(timeout=0)
            if w.conn.poll(min(slice_s, remaining)):
                break
            if not w.proc.is_alive():
                on_fail()
                raise RuntimeError(
                    f"Worker {w.rank} exited before ready ({_worker_death_hint(w)})"
                )
        try:
            status, payload = w.conn.recv()
        except EOFError as e:
            on_fail()
            raise RuntimeError(
                f"Worker {w.rank} closed ready pipe ({_worker_death_hint(w)})"
            ) from e
        if status != "ready":
            on_fail()
            raise RuntimeError(f"Worker {w.rank} failed to start: {payload}")


def _join_workers(workers: list[WorkerHandle]) -> None:
    for w in workers:
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


def _summarize_workers(workers: list[WorkerHandle]) -> str:
    return ", ".join(
        f"rank{w.rank} alive={w.proc.is_alive()} exit={w.proc.exitcode}"
        for w in workers
    )


def _ensure_workers_alive(workers: list[WorkerHandle], where: str) -> None:
    dead = [w for w in workers if not w.proc.is_alive()]
    if dead:
        detail = ", ".join(
            f"rank{w.rank} exit={w.proc.exitcode}" for w in dead
        )
        raise RuntimeError(
            f"worker(s) died during {where}: {detail}. "
            f"{_worker_death_hint(dead[0])}"
        )


class _TransportCommon:
    """Shared ready / liveness helpers for Pipe and ZMQ transports."""

    workers: list[WorkerHandle]
    _on_fail: Any

    def wait_ready(self, timeout: float = 1800.0) -> None:
        _wait_ready(self.workers, self._on_fail, timeout)

    def _ensure_workers_alive(self, where: str) -> None:
        _ensure_workers_alive(self.workers, where)

    def control_reply_count(self, method: str) -> int:
        del method
        return len(self.workers)


@dataclass
class PipeTransport(_TransportCommon):
    workers: list[WorkerHandle]
    driver: WorkerHandle
    world_size: int
    pp_size: int
    _on_fail: Any = field(repr=False, default=None)

    @classmethod
    def spawn(cls, config: EngineConfig, on_fail) -> PipeTransport:
        workers, driver = _spawn_workers(
            config,
            transport="pipe",
            make_child_kwargs=lambda _rank, _conn: {"transport": "pipe"},
        )
        return cls(
            workers=workers,
            driver=driver,
            world_size=config.world_size,
            pp_size=config.pp_size,
            _on_fail=on_fail,
        )

    def send_all(self, message: tuple[Any, ...]) -> None:
        for w in self.workers:
            assert w.conn is not None
            w.conn.send(message)

    def recv_control(self, method: str, timeout: float | None = None) -> list[Any]:
        del method
        self._ensure_workers_alive("control RPC")
        out: list[Any] = []
        for w in self.workers:
            assert w.conn is not None
            if timeout is not None and not w.conn.poll(timeout):
                raise TimeoutError(
                    f"timed out waiting for control RPC from rank {w.rank}"
                )
            try:
                out.append(w.conn.recv())
            except EOFError as e:
                raise RuntimeError(
                    f"pipe closed on control RPC rank={w.rank} "
                    f"({_worker_death_hint(w)})"
                ) from e
        return out

    def recv_execute_message(self, timeout: float | None) -> tuple[Any, ...]:
        """Recv one execute reply from any worker (success or fail-fast error)."""
        self._ensure_workers_alive("execute_model")
        conns = [w.conn for w in self.workers if w.conn is not None]
        # Bounded wait even when timeout is None so dead workers are detected.
        slice_s = 1.0 if timeout is None else timeout
        deadline = None if timeout is None else time.monotonic() + timeout
        while True:
            if deadline is not None and time.monotonic() >= deadline:
                raise TimeoutError("timed out waiting for execute_model RPC")
            ready = wait(conns, timeout=slice_s)
            if ready:
                try:
                    return ready[0].recv()
                except EOFError as e:
                    raise RuntimeError(
                        "pipe closed while waiting for execute_model "
                        f"({_summarize_workers(self.workers)})"
                    ) from e
            self._ensure_workers_alive("execute_model")

    def shutdown(self) -> None:
        for w in self.workers:
            try:
                if w.conn is not None and w.proc.is_alive():
                    w.conn.send(("shutdown",))
            except Exception:
                pass
        _join_workers(self.workers)
        self.workers.clear()


@dataclass
class ZmqTransport(_TransportCommon):
    workers: list[WorkerHandle]
    driver: WorkerHandle
    world_size: int
    pp_size: int
    _cmd_socks: list
    _pull: Any
    _ipc_paths: list[str]
    _on_fail: Any = field(repr=False, default=None)

    @classmethod
    def spawn(cls, config: EngineConfig, on_fail) -> ZmqTransport:
        import zmq

        prefix = f"{config.zmq_ipc_prefix}_{os.getpid()}"
        response_path = ipc_path(prefix, "response")
        unlink_ipc(response_path)
        zmq_ctx = zmq.Context.instance()
        pull = bind_pull(zmq_ctx, response_path)
        ipc_paths = [response_path]
        cmd_socks: list = []

        def make_kwargs(rank: int, _conn: Connection) -> dict:
            cmd_path = ipc_path(prefix, f"cmd{rank}")
            unlink_ipc(cmd_path)
            cmd_socks.append(bind_push(zmq_ctx, cmd_path))
            ipc_paths.append(cmd_path)
            return {
                "transport": "zmq",
                "cmd_path": cmd_path,
                "response_path": response_path,
            }

        workers, driver = _spawn_workers(
            config, transport="zmq", make_child_kwargs=make_kwargs
        )
        return cls(
            workers=workers,
            driver=driver,
            world_size=config.world_size,
            pp_size=config.pp_size,
            _cmd_socks=cmd_socks,
            _pull=pull,
            _ipc_paths=ipc_paths,
            _on_fail=on_fail,
        )

    def send_all(self, message: tuple[Any, ...]) -> None:
        payload = ZmqCodec.dumps(message)
        for sock in self._cmd_socks:
            sock.send(payload)

    def recv_control(self, method: str, timeout: float | None = None) -> list[Any]:
        assert self._pull is not None
        self._ensure_workers_alive(method)
        n = self.control_reply_count(method)
        out: list[Any] = []
        for _ in range(n):
            if timeout is None:
                while not self._pull.poll(1000):
                    self._ensure_workers_alive(method)
            elif not self._pull.poll(max(0, int(timeout * 1000))):
                self._ensure_workers_alive(method)
                raise TimeoutError(f"timed out waiting for {method}")
            out.append(ZmqCodec.loads(self._pull.recv()))
        return out

    def recv_execute_message(self, timeout: float | None) -> tuple[Any, ...]:
        assert self._pull is not None
        self._ensure_workers_alive("execute_model")
        # ZMQ poll is in ms; slice so we can notice dead workers under timeout=None.
        if timeout is None:
            while True:
                if self._pull.poll(1000):
                    return ZmqCodec.loads(self._pull.recv())
                self._ensure_workers_alive("execute_model")
        if not self._pull.poll(max(0, int(timeout * 1000))):
            self._ensure_workers_alive("execute_model")
            raise TimeoutError("timed out waiting for execute_model RPC")
        return ZmqCodec.loads(self._pull.recv())

    def shutdown(self) -> None:
        try:
            payload = ZmqCodec.dumps(("shutdown",))
            for sock in self._cmd_socks:
                sock.send(payload)
        except Exception:
            pass
        _join_workers(self.workers)
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
            unlink_ipc(p)
        self._ipc_paths.clear()
        self.workers.clear()


def create_transport(config: EngineConfig, on_fail) -> PipeTransport | ZmqTransport:
    if config.rpc_backend == "zmq":
        return ZmqTransport.spawn(config, on_fail)
    return PipeTransport.spawn(config, on_fail)
