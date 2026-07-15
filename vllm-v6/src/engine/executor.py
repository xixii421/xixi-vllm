"""Multiprocess executor: spawn TP×PP workers, Pipe RPC broadcast."""

from __future__ import annotations

import multiprocessing as mp
import os
import traceback
from dataclasses import dataclass
from multiprocessing.connection import Connection
from typing import Any

from ..config import EngineConfig
from ..memory import Sequence
from .worker import Worker


def _worker_main(
    config: EngineConfig,
    rank: int,
    tp_rank: int,
    pp_rank: int,
    local_device: int,
    is_driver: bool,
    conn: Connection,
) -> None:
    """Worker process entry (spawn)."""
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
                # Always ack; non-driver execute_model returns None.
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


@dataclass
class _WorkerHandle:
    proc: mp.Process
    conn: Connection
    rank: int
    is_driver: bool


class Executor:
    """Spawn ``tp_size * pp_size`` workers; driver = last PP & tp_rank 0."""

    def __init__(self, config: EngineConfig):
        self.config = config
        self.tp_size = config.tp_size
        self.pp_size = config.pp_size
        self.world_size = config.world_size
        self.device_ids = config.resolved_device_ids()

        ctx = mp.get_context("spawn")
        self._workers: list[_WorkerHandle] = []
        self._driver: _WorkerHandle | None = None

        for pp_rank in range(self.pp_size):
            for tp_rank in range(self.tp_size):
                rank = pp_rank * self.tp_size + tp_rank
                is_driver = tp_rank == 0 and pp_rank == self.pp_size - 1
                parent_conn, child_conn = ctx.Pipe(duplex=True)
                proc = ctx.Process(
                    target=_worker_main,
                    name=f"vllm-v6-worker-{rank}",
                    args=(
                        config,
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

        assert self._driver is not None
        self._wait_ready()

    def _wait_ready(self, timeout: float = 120.0) -> None:
        for w in self._workers:
            if not w.conn.poll(timeout):
                self.shutdown()
                raise RuntimeError(f"Worker {w.rank} did not become ready in time")
            status, payload = w.conn.recv()
            if status != "ready":
                self.shutdown()
                raise RuntimeError(f"Worker {w.rank} failed to start: {payload}")

    def _broadcast(self, method: str, args: tuple[Any, ...]) -> Any:
        for w in self._workers:
            w.conn.send((method, args))
        driver_result = None
        errors = []
        for w in self._workers:
            status, payload = w.conn.recv()
            if status == "err":
                errors.append(f"rank {w.rank}: {payload}")
            elif w.is_driver:
                driver_result = payload
        if errors:
            self.shutdown()
            raise RuntimeError("Worker error(s):\n" + "\n".join(errors))
        return driver_result

    def execute_model(self, seqs: list[Sequence], is_prefill: bool) -> list[int]:
        result = self._broadcast("execute_model", (seqs, is_prefill))
        assert result is not None
        return result

    def get_num_kvcache_blocks(self) -> int:
        for w in self._workers:
            w.conn.send(("get_num_kvcache_blocks", ()))
        sizes: list[int] = []
        errors = []
        for w in self._workers:
            status, payload = w.conn.recv()
            if status == "err":
                errors.append(f"rank {w.rank}: {payload}")
            else:
                sizes.append(int(payload))
        if errors:
            self.shutdown()
            raise RuntimeError("Worker error(s):\n" + "\n".join(errors))
        return min(sizes)

    def shutdown(self) -> None:
        for w in self._workers:
            try:
                if w.proc.is_alive():
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
            try:
                w.conn.close()
            except Exception:
                pass
        self._workers.clear()

    def __del__(self):
        try:
            self.shutdown()
        except Exception:
            pass
