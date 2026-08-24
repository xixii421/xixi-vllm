"""EngineCore：与传输无关的纯控制循环（对齐 vLLM V1 ``EngineCore``）。

固定链路（所有并行模式共用；Scheduler 不感知 TP/DP/PP）::

    schedule → executor.run(同一份 schedule) → read_samples → scheduler.update

- TP/PP：``Executor`` 把 schedule 广播到同 Engine 的多 rank，协同算一份模型。
- DP：由 ``MPClient`` 在更外层起多个 EngineCore；本类内部始终是单副本逻辑。
"""

from __future__ import annotations

import gc

import torch

from ..config import EngineConfig
from ..data import EngineCoreOutput, KVManager, ScheduledBatch, Sequence
from .executor import create_executor
from .scheduler import Scheduler


class EngineCore:
    """纯控制循环：可跑在本进程（Inproc），也可被 MPClient 包成独立副本进程。"""

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
        self.executor = executor or create_executor(config, device, dtype)
        self._idle_steps = 0
        self._init_kv()

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

    def add_request(self, seq: Sequence) -> None:
        self.scheduler.add(seq)

    def has_unfinished(self) -> bool:
        return not self.scheduler.is_finished()

    def step(self) -> tuple[list[EngineCoreOutput], int]:
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

        n_p, n_d = self.executor.run(sched)
        self.last_num_cached_tokens = self.kv.last_num_cached_tokens
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
