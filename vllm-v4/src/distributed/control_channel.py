"""控制面通道：调度广播与采样回收走 Gloo，与 NCCL 计算面分离。"""

from __future__ import annotations

import torch
import torch.distributed as dist


class ControlChannel:
    """WORLD cpu_group 上的字节广播 / 点对点。

    单机 spawn 与 torchrun 共用这一条通道，不再用 ZMQ。
    ``src`` / ``dst`` 是全局 rank（Gloo 默认组语义），不是 TP/PP 组内 rank。
    """

    def __init__(self, group) -> None:
        self.group = group

    def broadcast(self, data: bytes | None, src: int = 0) -> bytes | None:
        rank = dist.get_rank(self.group)
        size = torch.tensor([len(data) if data is not None else 0], dtype=torch.long)
        dist.broadcast(size, src=src, group=self.group)
        n = int(size.item())
        if n == 0:
            return None
        if rank == src:
            buf = torch.frombuffer(bytearray(data), dtype=torch.uint8).clone()
            dist.broadcast(buf, src=src, group=self.group)
            return None
        buf = torch.empty(n, dtype=torch.uint8)
        dist.broadcast(buf, src=src, group=self.group)
        return bytes(buf.tolist())

    def send(self, data: bytes, dst: int) -> None:
        dist.send(
            torch.tensor([len(data)], dtype=torch.long), dst=dst, group=self.group
        )
        if data:
            dist.send(
                torch.frombuffer(bytearray(data), dtype=torch.uint8).clone(),
                dst=dst,
                group=self.group,
            )

    def recv(self, src: int) -> bytes:
        size = torch.tensor([0], dtype=torch.long)
        dist.recv(size, src=src, group=self.group)
        n = int(size.item())
        if n == 0:
            return b""
        buf = torch.empty(n, dtype=torch.uint8)
        dist.recv(buf, src=src, group=self.group)
        return bytes(buf.tolist())
