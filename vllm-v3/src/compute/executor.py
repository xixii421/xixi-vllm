"""ModelExecutor: 模型 + KV 池 + ModelRunner 的装配与执行入口。"""

from __future__ import annotations

import copy
import gc
import io
from contextlib import redirect_stdout

import torch

from ..config import EngineConfig
from ..data import KVManager, ScheduledBatch
from .layers import RotaryEmbedding
from .model_runner import ModelRunner
from .models.loader import default_dtype_context, load_model, skip_param_init
from .models.qwen3 import Qwen3ForCausalLM
from .sampler import Sampler


class ModelExecutor:
    """负责模型加载、KV 池分配与 ModelRunner 生命周期。

    Engine（控制面）只依赖它拿到 KV 池引用以构建 Scheduler，并通过
    ``run`` / ``read_samples`` 驱动前向计算，不直接接触模型与 runner 细节。
    """

    def __init__(
        self, config: EngineConfig, device: torch.device, dtype: torch.dtype
    ) -> None:
        self.config = config
        self.device = device
        self.dtype = dtype
        self.model = self._load_model()
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        self.sampler = Sampler()
        self._mem_baseline = torch.cuda.memory_reserved()
        # 记录模型权重加载完成之后 CUDA reserved 显存 用于计算CG开销
        self.kv: KVManager | None = None
        self.runner: ModelRunner | None = None

    def _load_model(self) -> torch.nn.Module:
        with default_dtype_context(self.dtype): # 控制全局默认数据类型
            with skip_param_init(): # 跳过 PyTorch 默认参数初始化
                model = Qwen3ForCausalLM(self.config.hf_config)
        load_model(model, self.config.model)
        model = model.to(dtype=self.dtype, device=self.device).eval()
        # .to(dtype) 会把 RoPE cos/sin 缓存一并转成计算 dtype，而 FlashInfer 要求 fp32。
        # 必须在 compile/CUDA-Graph 捕获之前转回：若在 forward 里惰性 .float()，新张量
        # 落在 CUDAGraph 池内存，下次重放会被覆盖 → "overwritten by a subsequent run"。
        for m in model.modules():
            if isinstance(m, RotaryEmbedding):
                m.cos_sin_cache = m.cos_sin_cache.float()
        return model

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
        """ 自动计算 KV Cache 最大可用 block 数量 """
        return KVManager.auto_num_blocks(
            self.config.hf_config,
            block_size=self.config.block_size,
            dtype=self.dtype,
            device=self.device,
            gpu_memory_utilization=self.config.gpu_memory_utilization,
            reserve_bytes=reserve_bytes,
        )

    def _probe_cg_footprint(self) -> int:
        """ 探测 Decode CUDA‑Graph 捕获阶段会额外吃掉多少显存 """
        probe_config = copy.copy(self.config)
        probe_config.torch_compile = False # 关闭torch.compile 只测CUDA-Graph
        kv = self._make_kv(num_blocks=8) # 构造一个8block的KV池，用来实例化ModelRunner,不做真实推理
        with redirect_stdout(io.StringIO()): # 屏蔽 stdout 避免控制台乱输出
            runner = ModelRunner(
                self.model, kv, self.sampler, config=probe_config, device=self.device
            )
        torch.cuda.synchronize()
        footprint = torch.cuda.memory_reserved() - self._mem_baseline
        # 释放临时分配资源
        runner.destroy() # 释放 CG 相关资源
        del runner, kv # 删除变量引用
        gc.collect() # 手动触发 Python 解释器垃圾回收
        torch.cuda.empty_cache() # 清空 CUDA 缓存

        return int(footprint)

    def init_kv(self) -> KVManager:
        """初始化 KV 池与 ModelRunner（含 CG 足迹探测与块数自动计算）。"""
        use_cg = not self.config.enforce_eager
        if self.config.num_kvcache_blocks > 0:
            # 手动规定kv缓存块数，无需自动分配
            num_blocks = self.config.num_kvcache_blocks
        else:
            # 自动计算kvcache最大可用块数
            reserve = self._probe_cg_footprint() if use_cg else 0 #
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
        """ engine 调用，返回已完成 prefill 和 decode 请求的数量 """
        # GPU 调度任务，不需要管 CPU 拷贝是否完成，继续跑下一轮
        assert self.runner is not None, "init_kv() must be called before run()"
        return self.runner.run(sched)

    def read_samples(self, n_p: int, n_d: int) -> tuple[list, list]:
        """ engine 调用，返回已完成 prefill 和 decode 请求的下一个token id列表 """
        # CPU 拷贝任务，跟 GPU 并行运行，需要 token 且 GPU 拷贝完成时调用
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
