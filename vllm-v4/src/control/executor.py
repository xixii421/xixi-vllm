"""Executor：EngineCore 与 Worker 之间的唯一执行桥（TP / PP）。

并行分叉（与 ``client.make_client`` 正交）::

    tp=pp=1  → UniprocExecutor（本进程直接调 Worker）
    tp/pp>1  → MultiprocExecutor（每卡一进程；控制面 Gloo，计算面 NCCL）

DP 不在这里：``dp_size>1`` 由 ``MPClient`` 起多个独立 Engine，每副本内部仍走本模块。

一步执行约定：

1. driver 把同一份 ``SchedulerOutput`` 广播给所有 rank（TP×PP 共享同一 schedule）；
2. 各 rank ``Worker.execute_model``：TP 在层内集合通信，PP 在 stage 边界 P2P；
3. 仅采样 rank（末 PP stage 的 tp rank 0）把 token id 回传给 driver。
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


def _terminate_procs(procs: list[mp.Process]) -> None:
    """终止并回收残留 worker：SIGTERM 超时后升级 SIGKILL，避免卡在 CUDA/NCCL。"""
    for p in procs:
        if p.is_alive():
            p.terminate()
    for p in procs:
        p.join(timeout=3)
    for p in procs:
        if p.is_alive():
            p.kill()
    for p in procs:
        p.join(timeout=5)
    procs.clear()


def _sampler_rank(tp_size: int, pp_size: int) -> int:
    """采样回报的全局 rank：末 PP stage × TP rank 0。"""
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

    @abstractmethod
    def init_kv(self) -> KVManager: ...

    @abstractmethod
    def reset_kv(self, prefix_backend: str | None = None) -> KVManager: ...

    @abstractmethod
    def run(self, sched: ScheduledBatch) -> tuple[int, int]: ...

    @abstractmethod
    def read_samples(self, n_p: int, n_d: int) -> tuple[list, list]: ...

    @abstractmethod
    def collective_rpc(self, op: str, **payload) -> object: ...

    @abstractmethod
    def shutdown(self) -> None: ...

    @abstractmethod
    def destroy(self) -> None: ...


class ControlCommand(msgspec.Struct):
    """Gloo 控制面消息：广播调度或重置 / 关停。"""

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
    """非 driver worker：收 Gloo 命令并执行（torchrun 与 spawn 共用）。"""
    channel = ControlChannel(get_world_group().cpu_group)
    enc = msgspec.msgpack.Encoder()
    dec = msgspec.msgpack.Decoder(ControlCommand)
    sampler_rank = _sampler_rank(worker.tp_size, worker.pp_size)
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
    """单卡短路（tp=pp=1）：无跨进程 IPC。"""

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
        if op == "reset_kv":
            self.worker.reset_kv(payload.get("prefix_backend"))
            return None
        raise ValueError(f"unknown control op: {op}")

    def shutdown(self) -> None:
        return None

    def destroy(self) -> None:
        self.kv = None
        self.worker.destroy_environment()


class MultiprocExecutor(Executor):
    """多卡执行器：WORLD = tp_size × pp_size；rank0 兼 driver + worker。

    单机 ``run.py`` spawn 子进程；多机 ``run_torchrun.py`` 注入已初始化的 rank0 Worker。
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

        if worker is None:
            driver_worker: Worker | None = None
            try:
                ctx = mp.get_context("spawn")
                for pp_rank in range(config.pp_size):
                    for tp_rank in range(config.tp_size):
                        rank = pp_rank * config.tp_size + tp_rank
                        if rank == 0:
                            continue
                        p = ctx.Process(
                            target=_spawn_worker,
                            args=(config, rank, tp_rank, pp_rank),
                            daemon=False,
                        )
                        p.start()
                        self._procs.append(p)
                driver_worker = Worker(config, rank=0, tp_rank=0, pp_rank=0)
                driver_worker.init_environment()
                driver_worker.initialize()
                dist.barrier()
                worker = driver_worker
            except BaseException:
                # 任一 rank 建连失败则清理残留 worker 与本进程 dist 状态，
                # 让上层 retry 从干净状态重来。
                _terminate_procs(self._procs)
                if driver_worker is not None:
                    try:
                        driver_worker.destroy_environment()
                    except Exception:
                        pass
                raise

        self.worker = worker
        self.num_blocks = worker.get_kv_cache_size()
        self._channel = ControlChannel(get_world_group().cpu_group)

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

    def collective_rpc(self, op: str, **payload) -> object:
        cmd = ControlCommand(
            op=op,
            sched=payload.get("sched"),
            prefix_backend=payload.get("prefix_backend"),
        )
        self._channel.broadcast(self._enc.encode(cmd), src=0)
        return None

    def run(self, sched: ScheduledBatch) -> tuple[int, int]:
        wire = sched.to_wire()
        self.collective_rpc("execute", sched=wire)
        counts = self.worker.execute_model(wire)
        if self.sampler_rank == 0:
            assert counts is not None
            return counts
        n_p, n_d = self._dec_counts.decode(self._channel.recv(self.sampler_rank))
        return n_p, n_d

    def read_samples(self, n_p: int, n_d: int) -> tuple[list, list]:
        if self.sampler_rank == 0:
            return self.worker.read_samples(n_p, n_d)
        p_ids, d_ids = self._dec_sample.decode(self._channel.recv(self.sampler_rank))
        return p_ids, d_ids

    def shutdown(self) -> None:
        self.collective_rpc("shutdown")

    def destroy(self) -> None:
        if self._spawned:
            try:
                self.shutdown()
            except Exception:
                pass
            _terminate_procs(self._procs)
            self.worker.destroy_environment()
        self.kv = None


def create_executor(
    config: EngineConfig,
    device: torch.device,
    dtype: torch.dtype,
    *,
    rank0_worker: Worker | None = None,
) -> Executor:
    """按 tp/pp 选择执行器。``dp_size`` 不影响本函数（见 ``make_client``）。"""
    if config.tp_size == 1 and config.pp_size == 1:
        return UniprocExecutor(config, device, dtype)
    return MultiprocExecutor(config, device, dtype, worker=rank0_worker)
