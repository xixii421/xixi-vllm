from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Protocol


class PDRole(Enum):
    COLLOCATED = auto()
    PREFILL = auto()
    DECODE = auto()


@dataclass
class PDConfig:
    role: PDRole = PDRole.COLLOCATED
    kv_transfer_backend: str = "none"
    remote_host: str | None = None
    remote_port: int = 29600
    metadata: dict[str, Any] = field(default_factory=dict)


class KVTransferEngine(Protocol):
    def send_blocks(self, block_ids: list[int], meta: dict[str, Any]) -> None: ...

    def recv_blocks(self, meta: dict[str, Any]) -> list[int]: ...

    def close(self) -> None: ...


class NullKVTransfer:
    def send_blocks(self, block_ids: list[int], meta: dict[str, Any]) -> None:
        raise NotImplementedError(
            "PD KV transfer is a v7 stub. Wire NCCL P2P / Mooncake / shared-memory "
            "backend before enabling pd_mode=prefill|decode."
        )

    def recv_blocks(self, meta: dict[str, Any]) -> list[int]:
        raise NotImplementedError(
            "PD KV transfer is a v7 stub. See src/pd/transfer.py."
        )

    def close(self) -> None:
        return


def make_kv_transfer(cfg: PDConfig) -> KVTransferEngine:
    if cfg.kv_transfer_backend == "none" or cfg.role == PDRole.COLLOCATED:
        return NullKVTransfer()
    raise ValueError(f"unsupported kv_transfer_backend={cfg.kv_transfer_backend!r}")
