"""EngineCoreClient：Engine 门面与 EngineCore 之间的抽象。

- ``InprocClient``：EngineCore 跑在本进程，直接调用（对应 vLLM 的 InprocClient）。
- ``MPClient``：DP 多副本场景，通过 ZMQ 与各 EngineCore 进程通信（见 mp_client.py）。

对齐 vLLM V1 的 ``v1/engine/core_client.py``：客户端只关心「加请求 / 取输出」，
不关心 EngineCore 的调度与执行细节。
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
    """单副本（dp=1）客户端：EngineCore 与本进程共享调度器与 Sequence 对象。"""

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
    """按并行度构造客户端。

    - dp=1：InprocClient（EngineCore 本进程）。
    - dp>1：MPClient（spawn dp 个 EngineCore 进程，ZMQ 路由）。

    ``executor`` 用于注入已初始化好的执行器（torchrun 多机路径）。
    """
    if config.dp_size > 1:
        from .mp_client import MPClient

        return MPClient(
            config, eos_token_id=eos_token_id, device=device, dtype=dtype
        )
    core = EngineCore(
        config, eos_token_id=eos_token_id, device=device, dtype=dtype, executor=executor
    )
    return InprocClient(core)
