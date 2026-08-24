"""EngineCoreClient：Engine 门面与 EngineCore 之间的传输抽象。

并行决策树（先 DP，再 TP/PP）::

    dp_size == 1  → InprocClient：EngineCore 在本进程
    dp_size  > 1  → MPClient：spawn dp 个独立 EngineCoreProc（ZMQ）

每个副本内部再用 ``create_executor`` 处理 tp/pp（Gloo + NCCL）。
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import torch

from ..config import EngineConfig
from ..data import EngineCoreOutput, Sequence
from .core import EngineCore


class EngineCoreClient(ABC):
    """客户端抽象：加请求、步进取输出、查询是否空闲、销毁。"""

    @abstractmethod
    def add_request(self, seq: Sequence) -> None: ...

    @abstractmethod
    def step(self) -> tuple[list[EngineCoreOutput], int]: ...

    @abstractmethod
    def has_unfinished(self) -> bool: ...

    @abstractmethod
    def destroy(self) -> None: ...


class InprocClient(EngineCoreClient):
    """单副本（dp=1）：EngineCore 与门面共享调度器与 Sequence 对象。"""

    def __init__(self, engine_core: EngineCore) -> None:
        self.engine_core = engine_core

    def add_request(self, seq: Sequence) -> None:
        self.engine_core.add_request(seq)

    def step(self) -> tuple[list[EngineCoreOutput], int]:
        return self.engine_core.step()

    def has_unfinished(self) -> bool:
        return self.engine_core.has_unfinished()

    def destroy(self) -> None:
        self.engine_core.destroy()


def make_client(
    config: EngineConfig,
    *,
    eos_token_id: int,
    device: torch.device,
    dtype: torch.dtype,
    executor=None,
) -> EngineCoreClient:
    """按 ``dp_size`` 构造客户端；``executor`` 供 torchrun 注入已就绪的 rank0 执行器。"""
    if config.dp_size > 1:
        from .mp_client import MPClient

        if executor is not None:
            raise ValueError(
                "dp_size>1 uses MPClient replicas; cannot inject a shared executor"
            )
        return MPClient(
            config, eos_token_id=eos_token_id, device=device, dtype=dtype
        )
    core = EngineCore(
        config, eos_token_id=eos_token_id, device=device, dtype=dtype, executor=executor
    )
    return InprocClient(core)
