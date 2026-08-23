"""EngineCore：与传输/进程模型无关的纯控制循环（对齐 vLLM V1 的 ``EngineCore``）。

职责：持有调度器与执行器，实现「调度 → 前向 → 采样回收 → 更新」这一核心循环。
tokenizer / 输入输出处理不在这一层，而是由 ``Engine`` 门面负责；进程模型
（单进程 / ZMQ 多副本）由 ``EngineCoreClient`` 决定。
"""

from __future__ import annotations

import gc

import torch

from ..config import EngineConfig
from ..data import EngineCoreOutput, KVManager, ScheduledBatch, Sequence
from .executor import create_executor
from .scheduler import Scheduler


class EngineCore:
    """纯控制循环，可运行于本进程，也可被 ``MPClient`` 包装为独立进程。"""

    # 仍有请求但调度器持续返回空批时，超过该步数判定异常（常见为 KV 池耗尽）。
    _MAX_IDLE_STEPS = 1024

    def __init__(
        self,
        config: EngineConfig,
        *,
        eos_token_id: int,
        device: torch.device,
        dtype: torch.dtype,
        executor=None,
    ) -> None:
        self.config = config
        self.eos_token_id = eos_token_id
        # 允许注入自定义 executor（torchrun 多机路径用已初始化的 rank0 Worker）。
        self.executor = executor or create_executor(config, device, dtype)
        self._idle_steps = 0
        self._init_kv()

    # ------------------------------------------------------------------ #
    # 初始化
    # ------------------------------------------------------------------ #

    def _build_scheduler(self) -> Scheduler:
        return Scheduler(
            self.kv,
            max_num_seqs=self.config.max_num_seqs,
            max_num_batched_tokens=self.config.max_num_batched_tokens,
            eos_token_id=self.eos_token_id,
            mix_prefill_decode=self.config.mix_prefill_decode,
        )

    def _init_kv(self) -> None:
        self.kv: KVManager = self.executor.init_kv()
        self.scheduler = self._build_scheduler()
        self.last_num_cached_tokens = 0

    # ------------------------------------------------------------------ #
    # 请求与步进
    # ------------------------------------------------------------------ #

    def add_request(self, seq: Sequence) -> None:
        """把一条已构造好的 Sequence 交给调度器。"""
        self.scheduler.add(seq)

    def has_unfinished(self) -> bool:
        return not self.scheduler.is_finished()

    def step(self) -> tuple[list[EngineCoreOutput], int]:
        """执行一步调度、前向计算与采样后处理。

        Returns:
            (outputs, metric)：本步每条采样序列的输出增量 + 处理的 token 数（吞吐统计）。
        """
        sched: ScheduledBatch = self.scheduler.schedule()
        if sched.is_empty():
            if not self.scheduler.is_finished():
                self._idle_steps += 1
                if self._idle_steps >= self._MAX_IDLE_STEPS:
                    raise RuntimeError(
                        "scheduler returned empty for too long while requests remain "
                        f"(idle_steps={self._idle_steps}). Likely KV pool exhaustion "
                        f"(free_blocks={self.kv.num_free})."
                    )
            return [], 0
        self._idle_steps = 0

        # 先发射前向（采样 D2H 异步在飞），CPU 记账与之重叠，再收回采样。
        n_p, n_d = self.executor.run(sched)
        self.last_num_cached_tokens = self.kv.last_num_cached_tokens

        # 本步 token 指标：prefill token 数 + decode 序列数。
        prefill_toks = sum(s.num_scheduled_tokens for s in sched.prefills)
        metric = prefill_toks + len(sched.decodes)

        p_ids, d_ids = self.executor.read_samples(n_p, n_d)
        outputs = self.scheduler.update(sched, p_ids, d_ids)
        return outputs, metric

    def destroy(self) -> None:
        self.executor.destroy()
        self.kv = None
        self.scheduler = None
        gc.collect()
        torch.cuda.empty_cache()
