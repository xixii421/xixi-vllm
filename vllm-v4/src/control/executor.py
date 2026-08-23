"""Executor：EngineCore 与 Worker 之间的唯一执行接口。

- ``Executor``：抽象接口，EngineCore 只依赖它——``run``（发射前向 + 异步 D2H）、
  ``read_samples``（回收采样 id）、``collective_rpc``（广播控制命令）、KV 初始化/重建/销毁。
- ``UniprocExecutor``：tp=pp=1 时本进程直接调用 Worker，无跨进程 IPC。
- ``MultiprocExecutor``：每卡一进程，rank0 兼 driver（调度器 + worker）；
  spawn 与 torchrun 共用。控制面（调度广播 + 采样回收）走 Gloo，计算面（TP/PP）走 NCCL。
"""

from __future__ import annotations

import multiprocessing as mp
from abc import ABC, abstractmethod

import msgspec
import torch
import torch.distributed as dist

from ..compute.worker import Worker
from ..config import EngineConfig
from ..data import KVManager, SchedulerOutput, ScheduledBatch
from ..distributed import ControlChannel, get_world_group


def _sampler_rank(tp_size: int, pp_size: int) -> int:
    """采样回报 rank：末 PP stage 的 tp rank 0（全局 rank）。"""
    return (pp_size - 1) * tp_size


class Executor(ABC):
    """EngineCore 与 worker 之间的唯一接口。"""

    def __init__(
        self, config: EngineConfig, device: torch.device, dtype: torch.dtype
    ) -> None:
        self.config = config
        self.device = device
        self.dtype = dtype
        self.sampler_rank = _sampler_rank(config.tp_size, config.pp_size)

    # ------------------------------------------------------------------ #
    # KV 生命周期
    # ------------------------------------------------------------------ #

    @abstractmethod
    def init_kv(self) -> KVManager:
        """初始化并返回 driver 侧逻辑 KV 账本（调度器使用）。"""

    @abstractmethod
    def reset_kv(self, prefix_backend: str | None = None) -> KVManager:
        """重建 GPU KV 池与逻辑账本，可选切换前缀缓存后端。"""

    # ------------------------------------------------------------------ #
    # 一步执行
    # ------------------------------------------------------------------ #

    @abstractmethod
    def run(self, sched: ScheduledBatch) -> tuple[int, int]:
        """发射一步前向（采样 D2H 异步在飞），返回 (prefill 数, decode 数)。"""

    @abstractmethod
    def read_samples(self, n_p: int, n_d: int) -> tuple[list, list]:
        """D2H 完成后读取 (prefill ids, decode ids)。"""

    # ------------------------------------------------------------------ #
    # 控制与销毁
    # ------------------------------------------------------------------ #

    @abstractmethod
    def collective_rpc(self, op: str, **payload) -> object:
        """向 worker 广播一个控制命令（vLLM collective_rpc 的简化形式）。"""

    @abstractmethod
    def shutdown(self) -> None:
        """通知 worker 进程退出。"""

    @abstractmethod
    def destroy(self) -> None:
        """释放执行器占用的显存与子进程。"""


class ControlCommand(msgspec.Struct):
    op: str  # execute | reset_kv | shutdown
    sched: SchedulerOutput | None = None
    prefix_backend: str | None = None


def _spawn_worker(
    config: EngineConfig, rank: int, tp_rank: int, pp_rank: int
) -> None:
    worker = Worker(config, rank, tp_rank, pp_rank)
    try:
        worker.init_environment()
        worker.initialize()
        dist.barrier()
        worker_loop(worker)
    finally:
        worker.destroy_environment()


def worker_loop(worker: Worker) -> None:
    """非 rank0 worker 的忙循环：收 Gloo 命令并执行。

    采样 rank 在 execute 时先回 counts 再回 token ids（异步 D2H 的两段回收）。
    """
    channel = ControlChannel(get_world_group().cpu_group)
    enc = msgspec.msgpack.Encoder()
    dec = msgspec.msgpack.Decoder(ControlCommand)
    sampler_rank = (worker.pp_size - 1) * worker.tp_size
    while True:
        cmd = dec.decode(channel.broadcast(None, src=0))
        if cmd.op == "shutdown":
            break
        if cmd.op == "reset_kv":
            worker.reset_kv(cmd.prefix_backend)
            continue
        if cmd.op != "execute":
            raise ValueError(f"unknown control op: {cmd.op}")
        counts = worker.execute_model(cmd.sched)
        if worker.rank == sampler_rank:
            n_p, n_d = counts
            channel.send(enc.encode([n_p, n_d]), dst=0)
            p_ids, d_ids = worker.read_samples(n_p, n_d)
            channel.send(enc.encode([p_ids, d_ids]), dst=0)


class UniprocExecutor(Executor):
    """单卡短路执行器（对应 vLLM 的 UniProcExecutor）。"""

    def __init__(
        self, config: EngineConfig, device: torch.device, dtype: torch.dtype
    ) -> None:
        super().__init__(config, device, dtype)
        self.worker = Worker(config, rank=0, tp_rank=0, pp_rank=0)
        self.worker.init_environment()
        self.kv: KVManager | None = None
        self._ready = False

    def _logical_kv(self, num_blocks: int) -> KVManager:
        return KVManager(
            num_blocks=num_blocks,
            block_size=self.config.block_size,
            prefix_backend=self.config.prefix_backend,
        )

    def init_kv(self) -> KVManager:
        if not self._ready:
            self.worker.initialize()
            self._ready = True
        self.kv = self._logical_kv(self.worker.get_kv_cache_size())
        return self.kv

    def reset_kv(self, prefix_backend: str | None = None) -> KVManager:
        if prefix_backend is not None:
            self.config.prefix_backend = prefix_backend
        self.worker.reset_kv(prefix_backend)
        self.kv = self._logical_kv(self.worker.get_kv_cache_size())
        return self.kv

    def run(self, sched: ScheduledBatch) -> tuple[int, int]:
        counts = self.worker.execute_model(sched.to_wire())
        assert counts is not None
        return counts

    def read_samples(self, n_p: int, n_d: int) -> tuple[list, list]:
        return self.worker.read_samples(n_p, n_d)

    def collective_rpc(self, op: str, **payload) -> object:
        # 单进程：直接派发到本地 worker。
        if op == "reset_kv":
            self.worker.reset_kv(payload.get("prefix_backend"))
            return None
        raise ValueError(f"unknown control op: {op}")

    def shutdown(self) -> None:
        # 单进程无子进程，无需通知。
        return None

    def destroy(self) -> None:
        self.kv = None
        self.worker.destroy_environment()


class MultiprocExecutor(Executor):
    """多卡统一执行器。

    - spawn：为本机 rank 1..N-1 拉起子进程，rank0 在本进程当 worker。
    - torchrun：传入已 initialize 的 rank0 Worker，其它 rank 跑 worker_loop。
    """

    def __init__(
        self,
        config: EngineConfig,
        device: torch.device,
        dtype: torch.dtype,
        worker: Worker | None = None,
    ) -> None:
        super().__init__(config, device, dtype)
        self._enc = msgspec.msgpack.Encoder()
        self._dec_counts = msgspec.msgpack.Decoder(list[int])
        self._dec_sample = msgspec.msgpack.Decoder(list[list[int]])
        self._procs: list[mp.Process] = []
        self._spawned = worker is None
        self.kv: KVManager | None = None
        self._pending_counts: tuple[int, int] | None = None

        if worker is None:
            ctx = mp.get_context("spawn")
            for pp_rank in range(config.pp_size):
                for tp_rank in range(config.tp_size):
                    rank = pp_rank * config.tp_size + tp_rank
                    if rank == 0:
                        continue
                    p = ctx.Process(
                        target=_spawn_worker,
                        args=(config, rank, tp_rank, pp_rank),
                        daemon=True,
                    )
                    p.start()
                    self._procs.append(p)
            worker = Worker(config, rank=0, tp_rank=0, pp_rank=0)
            worker.init_environment()
            worker.initialize()
            dist.barrier()

        self.worker = worker
        self.num_blocks = worker.get_kv_cache_size()
        self._channel = ControlChannel(get_world_group().cpu_group)

    # ------------------------------------------------------------------ #
    # KV
    # ------------------------------------------------------------------ #

    def _logical_kv(self) -> KVManager:
        return KVManager(
            num_blocks=self.num_blocks,
            block_size=self.config.block_size,
            prefix_backend=self.config.prefix_backend,
        )

    def init_kv(self) -> KVManager:
        self.kv = self._logical_kv()
        return self.kv

    def reset_kv(self, prefix_backend: str | None = None) -> KVManager:
        if prefix_backend is not None:
            self.config.prefix_backend = prefix_backend
        self.collective_rpc("reset_kv", prefix_backend=prefix_backend)
        self.worker.reset_kv(prefix_backend)
        self.num_blocks = self.worker.get_kv_cache_size()
        self.kv = self._logical_kv()
        return self.kv

    # ------------------------------------------------------------------ #
    # 一步执行
    # ------------------------------------------------------------------ #

    def collective_rpc(self, op: str, **payload) -> object:
        """广播控制命令到所有 worker（rank0 自身在调用处本地执行）。"""
        sched = payload.get("sched")
        prefix_backend = payload.get("prefix_backend")
        cmd = ControlCommand(op=op, sched=sched, prefix_backend=prefix_backend)
        self._channel.broadcast(self._enc.encode(cmd), src=0)
        return None

    def run(self, sched: ScheduledBatch) -> tuple[int, int]:
        wire = sched.to_wire()
        self.collective_rpc("execute", sched=wire)
        counts = self.worker.execute_model(wire)
        if self.sampler_rank == 0:
            assert counts is not None
            self._pending_counts = counts
            return counts
        n_p, n_d = self._dec_counts.decode(self._channel.recv(self.sampler_rank))
        self._pending_counts = (n_p, n_d)
        return n_p, n_d

    def read_samples(self, n_p: int, n_d: int) -> tuple[list, list]:
        if self.sampler_rank == 0:
            return self.worker.read_samples(n_p, n_d)
        p_ids, d_ids = self._dec_sample.decode(self._channel.recv(self.sampler_rank))
        return p_ids, d_ids

    # ------------------------------------------------------------------ #
    # 销毁
    # ------------------------------------------------------------------ #

    def shutdown(self) -> None:
        self.collective_rpc("shutdown")

    def destroy(self) -> None:
        if self._spawned:
            try:
                self.shutdown()
            except Exception:
                pass
            for p in self._procs:
                p.join(timeout=10)
                if p.is_alive():
                    p.terminate()
            self._procs.clear()
            self.worker.destroy_environment()
        self.kv = None


def create_executor(
    config: EngineConfig,
    device: torch.device,
    dtype: torch.dtype,
    *,
    rank0_worker: Worker | None = None,
) -> Executor:
    """按并行度创建执行器。

    - tp=pp=1：``UniprocExecutor``（无 IPC）。
    - 否则：``MultiprocExecutor``；torchrun 多机场景传 ``rank0_worker`` 以复用
      已 ``initialize()`` 的 rank0 上下文（其余 rank 由入口跑 ``worker_loop``）。
    """
    if config.tp_size == 1 and config.pp_size == 1:
        return UniprocExecutor(config, device, dtype)
    return MultiprocExecutor(config, device, dtype, worker=rank0_worker)
