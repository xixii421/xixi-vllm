"""ModelExecutor: 模型 + KV 池 + ModelRunner 的装配与执行入口。"""

from __future__ import annotations

import copy
import gc
import io
import multiprocessing as mp
from contextlib import redirect_stdout

import torch

from ..config import EngineConfig
from ..data import KVManager, ScheduledBatch
from ..distributed import (
    destroy_distributed_environment,
    destroy_model_parallel,
    init_distributed_environment,
    initialize_model_parallel,
    is_initialized,
)
from .model_runner import ModelRunner
from .models.loader import default_dtype_context, load_model, skip_param_init
from .models.qwen3 import Qwen3ForCausalLM
from .sampler import Sampler


def _init_model_parallel(
    config: EngineConfig, *, rank: int = 0, tp_rank: int = 0, pp_rank: int = 0
) -> None:
    """确保进程组就绪（分片层构造前必须调用）。

    单卡（tp=pp=1）也走一次 singleton NCCL init，让 get_tp_group() 等接口
    统一可用。多卡场景由 worker 传入真实 rank（见 stage 4）。
    """
    if is_initialized():
        return
    world_size = config.tp_size * config.pp_size
    init_distributed_environment(
        world_size=world_size,
        rank=rank,
        init_method=f"tcp://{config.master_addr}:{config.nccl_port}",
    )
    initialize_model_parallel(config.tp_size, config.pp_size, tp_rank, pp_rank)


class ModelExecutor:
    """负责模型加载、KV 池分配与 ModelRunner 生命周期。

    Engine（控制面）只依赖它拿到 KV 池引用以构建 Scheduler，并通过
    ``run`` / ``read_samples`` 驱动前向计算，不直接接触模型与 runner 细节。
    """

    def __init__(
        self,
        config: EngineConfig,
        device: torch.device,
        dtype: torch.dtype,
        *,
        rank: int = 0,
        tp_rank: int = 0,
        pp_rank: int = 0,
    ) -> None:
        self.config = config
        self.device = device
        self.dtype = dtype
        _init_model_parallel(config, rank=rank, tp_rank=tp_rank, pp_rank=pp_rank)
        self.model = self._load_model()
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        self.sampler = Sampler()
        self._mem_baseline = torch.cuda.memory_reserved()
        self.kv: KVManager | None = None
        self.runner: ModelRunner | None = None

    def _load_model(self) -> torch.nn.Module:
        with default_dtype_context(self.dtype):
            with skip_param_init():
                model = Qwen3ForCausalLM(self.config.hf_config)
        load_model(model, self.config.model)
        return model.to(dtype=self.dtype, device=self.device).eval()

    def _make_kv(self, num_blocks: int) -> KVManager:
        return KVManager(
            self.config.hf_config,
            num_blocks=num_blocks,
            block_size=self.config.block_size,
            device=self.device,
            dtype=self.dtype,
            prefix_backend=self.config.prefix_backend,
        )

    def _resolve_num_blocks(self, reserve_bytes: int) -> int:
        return KVManager.auto_num_blocks(
            self.config.hf_config,
            block_size=self.config.block_size,
            dtype=self.dtype,
            device=self.device,
            gpu_memory_utilization=self.config.gpu_memory_utilization,
            reserve_bytes=reserve_bytes,
        )

    def _probe_cg_footprint(self) -> int:
        probe_config = copy.copy(self.config)
        probe_config.torch_compile = False
        kv = self._make_kv(num_blocks=8)
        with redirect_stdout(io.StringIO()):
            runner = ModelRunner(
                self.model, kv, self.sampler, config=probe_config, device=self.device
            )
        torch.cuda.synchronize()
        footprint = torch.cuda.memory_reserved() - self._mem_baseline
        runner.destroy()
        del runner, kv
        gc.collect()
        torch.cuda.empty_cache()
        return int(footprint)

    def init_kv(self) -> KVManager:
        """初始化 KV 池与 ModelRunner（含 CG 足迹探测与块数自动计算）。"""
        use_cg = not self.config.enforce_eager
        if self.config.num_kvcache_blocks > 0:
            num_blocks = self.config.num_kvcache_blocks
        else:
            reserve = self._probe_cg_footprint() if use_cg else 0
            num_blocks = self._resolve_num_blocks(reserve)
            self.config.num_kvcache_blocks = num_blocks
            if use_cg:
                print(
                    f"CG footprint measured: {reserve / 2**30:.3f} GiB; "
                    f"KV pool sized to remainder ({num_blocks} blocks)"
                )
        self.kv = self._make_kv(num_blocks)
        self.runner = ModelRunner(
            self.model, self.kv, self.sampler, config=self.config, device=self.device
        )
        return self.kv

    def reset_kv(self, prefix_backend: str | None = None) -> KVManager:
        """重建 KV 池与 ModelRunner（复用已缓存的块数，不重复探针）。"""
        if self.runner is not None:
            self.runner.destroy()
            self.runner = None
        if prefix_backend is not None:
            self.config.prefix_backend = prefix_backend
        return self.init_kv()

    def run(self, sched: ScheduledBatch) -> tuple[int, int]:
        assert self.runner is not None, "init_kv() must be called before run()"
        return self.runner.run(sched)

    def read_samples(self, n_p: int, n_d: int) -> tuple[list, list]:
        assert self.runner is not None, "init_kv() must be called before read_samples()"
        return self.runner.read_samples(n_p, n_d)

    def destroy(self) -> None:
        if self.runner is not None:
            self.runner.destroy()
        self.runner = None
        self.sampler = None
        self.kv = None
        gc.collect()
        torch.cuda.empty_cache()
        destroy_model_parallel()
        destroy_distributed_environment()


def _run_worker(
    config: EngineConfig,
    rank: int,
    tp_rank: int,
    pp_rank: int,
    is_driver: bool,
    input_queue: mp.Queue,
    ready_queue: mp.Queue,
    output_queue: mp.Queue,
) -> None:
    """worker 进程主循环：初始化环境与模型后，按主进程请求执行前向。"""
    from .worker import Worker

    worker = Worker(config, rank, tp_rank, pp_rank)
    try:
        worker.init_environment()
        worker.initialize()
        ready_queue.put(rank)
        while True:
            req = input_queue.get()
            method = req["method"]
            if method == "shutdown":
                break
            if method == "execute_model":
                result = worker.execute_model(req["sched"])
                if is_driver:
                    output_queue.put(result)
            elif method == "get_kv_cache_size":
                if is_driver:
                    output_queue.put(worker.get_kv_cache_size())
            else:
                raise ValueError(f"unknown worker method: {method}")
    finally:
        worker.destroy_environment()


class MultiWorkerClient:
    """多进程执行器：主进程持逻辑 KV（记账/前缀），每卡一个 worker 做前向。

    与 ModelExecutor 暴露相同接口（init_kv / run / read_samples / destroy），
    使 Engine 控制面无需区分单卡与并行。run 直接返回采样 id，read_samples 读取缓存。
    """

    def __init__(self, config: EngineConfig, device: torch.device, dtype: torch.dtype):
        self.config = config
        self.device = device
        self.dtype = dtype
        self.world_size = config.tp_size * config.pp_size
        self._ctx = mp.get_context("spawn")
        self._procs: list[mp.Process] = []
        self._in_queues: list[mp.Queue] = []
        self._ready_queue: mp.Queue = self._ctx.Queue()
        self._out_queue: mp.Queue = self._ctx.Queue()
        self._last_p_ids: list[int] = []
        self._last_d_ids: list[int] = []
        self.kv: KVManager | None = None

        self._start_workers()
        self._wait_ready()
        self.num_blocks = self._query_kv_size()

    # ------------------------------------------------------------------ #
    # 进程管理
    # ------------------------------------------------------------------ #

    def _start_workers(self) -> None:
        for pp_rank in range(self.config.pp_size):
            for tp_rank in range(self.config.tp_size):
                rank = pp_rank * self.config.tp_size + tp_rank
                is_driver = tp_rank == 0 and pp_rank == self.config.pp_size - 1
                q: mp.Queue = self._ctx.Queue()
                p = self._ctx.Process(
                    target=_run_worker,
                    args=(
                        self.config,
                        rank,
                        tp_rank,
                        pp_rank,
                        is_driver,
                        q,
                        self._ready_queue,
                        self._out_queue,
                    ),
                    daemon=True,
                )
                p.start()
                self._procs.append(p)
                self._in_queues.append(q)

    def _wait_ready(self) -> None:
        for _ in range(self.world_size):
            self._ready_queue.get()

    def _broadcast(self, method: str, **kwargs) -> None:
        req = {"method": method, **kwargs}
        for q in self._in_queues:
            q.put(req)

    def _query_kv_size(self) -> int:
        self._broadcast("get_kv_cache_size")
        return int(self._out_queue.get())

    # ------------------------------------------------------------------ #
    # 与 ModelExecutor 对齐的接口
    # ------------------------------------------------------------------ #

    def _make_logical_kv(self) -> KVManager:
        return KVManager(
            self.config.hf_config,
            num_blocks=self.num_blocks,
            block_size=self.config.block_size,
            device=self.device,
            dtype=self.dtype,
            prefix_backend=self.config.prefix_backend,
            allocate_pool=False,
        )

    def init_kv(self) -> KVManager:
        self.kv = self._make_logical_kv()
        return self.kv

    def reset_kv(self, prefix_backend: str | None = None) -> KVManager:
        if prefix_backend is not None:
            self.config.prefix_backend = prefix_backend
        self.kv = self._make_logical_kv()
        return self.kv

    def run(self, sched: ScheduledBatch) -> tuple[int, int]:
        self._broadcast("execute_model", sched=sched)
        self._last_p_ids, self._last_d_ids = self._out_queue.get()
        return len(self._last_p_ids), len(self._last_d_ids)

    def read_samples(self, n_p: int, n_d: int) -> tuple[list, list]:
        return self._last_p_ids, self._last_d_ids

    def destroy(self) -> None:
        try:
            self._broadcast("shutdown")
        except Exception:
            pass
        for p in self._procs:
            p.join(timeout=10)
            if p.is_alive():
                p.terminate()
        for q in self._in_queues:
            q.close()
        self._procs.clear()
        self._in_queues.clear()
        self.kv = None
        gc.collect()
        torch.cuda.empty_cache()


def create_executor(
    config: EngineConfig, device: torch.device, dtype: torch.dtype
) -> ModelExecutor | MultiWorkerClient:
    """按并行度选择单卡 ModelExecutor 或多进程 MultiWorkerClient。"""
    if config.tp_size == 1 and config.pp_size == 1:
        return ModelExecutor(config, device, dtype)
    return MultiWorkerClient(config, device, dtype)


class TorchrunExecutor:
    """torchrun 多机执行器：rank 0 兼调度器，通过 broadcast/gather 与各 worker 同步。

    worker 由 torchrun 在每卡启动（见 run_torchrun.py），本类只在 rank 0 上被
    Engine 使用，负责把调度结果广播出去、并把 last-PP tp0 的采样结果 gather 回来。
    """

    def __init__(
        self,
        config: EngineConfig,
        device: torch.device,
        dtype: torch.dtype,
        worker,
    ) -> None:
        import torch.distributed as dist

        self.config = config
        self.device = device
        self.dtype = dtype
        self.worker = worker
        self.dist = dist
        self.world_size = config.tp_size * config.pp_size
        # 采样由 last PP stage 的 tp0 完成；rank 映射 = pp_rank * tp_size + tp_rank。
        self.sampler_rank = (config.pp_size - 1) * config.tp_size
        self.num_blocks = worker.get_kv_cache_size()
        self.kv: KVManager | None = None
        self._last_p_ids: list[int] = []
        self._last_d_ids: list[int] = []

    def _make_logical_kv(self) -> KVManager:
        return KVManager(
            self.config.hf_config,
            num_blocks=self.num_blocks,
            block_size=self.config.block_size,
            device=self.device,
            dtype=self.dtype,
            prefix_backend=self.config.prefix_backend,
            allocate_pool=False,
        )

    def init_kv(self) -> KVManager:
        self.kv = self._make_logical_kv()
        return self.kv

    def reset_kv(self, prefix_backend: str | None = None) -> KVManager:
        if prefix_backend is not None:
            self.config.prefix_backend = prefix_backend
        self.kv = self._make_logical_kv()
        return self.kv

    def run(self, sched: ScheduledBatch) -> tuple[int, int]:
        self.dist.broadcast_object_list([sched], src=0)
        result = self.worker.execute_model(sched)
        sample = result if result is not None else ([], [])
        gathered = [None] * self.world_size
        self.dist.gather_object(sample, gathered, dst=0)
        self._last_p_ids, self._last_d_ids = gathered[self.sampler_rank]
        return len(self._last_p_ids), len(self._last_d_ids)

    def read_samples(self, n_p: int, n_d: int) -> tuple[list, list]:
        return self._last_p_ids, self._last_d_ids

    def shutdown(self) -> None:
        self.dist.broadcast_object_list([None], src=0)

    def destroy(self) -> None:
        self.kv = None
