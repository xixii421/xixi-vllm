"""集合通信原语与 TP/PP 封装。

基础原语都带 world_size == 1 的 fast path：单卡 / 单组直接返回输入，
避免无意义的 NCCL 调用。TP 层用 all_reduce / all_gather / reduce_scatter，
PP 层用 send / recv 传中间张量字典。
"""

from __future__ import annotations

from typing import Any

import torch
import torch.distributed as dist

from .parallel_state import Group, get_tp_group, get_world_group


def all_gather(
    input: torch.Tensor, dim: int = -1, group: Group | None = None
) -> torch.Tensor:
    if dim < 0:
        dim += input.dim()
    group = get_world_group() if group is None else group
    input_size = input.size()
    world_size = group.size
    if world_size == 1:
        return input
    # 用 concat 风格 all_gather（all_gather_single），stack 风格与 torch.compile 有兼容问题。
    output_size = (input_size[0] * world_size, *input_size[1:])
    output_tensor = torch.empty(output_size, dtype=input.dtype, device=input.device)
    dist.all_gather_single(output_tensor, input, group=group.device_group)
    output_tensor = output_tensor.reshape((world_size,) + input_size)
    output_tensor = output_tensor.movedim(0, dim)
    return output_tensor.reshape(
        input_size[:dim] + (world_size * input_size[dim],) + input_size[dim + 1 :]
    )


def all_reduce(
    input: torch.Tensor, op=dist.ReduceOp.SUM, group: Group | None = None
) -> torch.Tensor:
    group = get_world_group() if group is None else group
    if group.size == 1:
        return input
    dist.all_reduce(input, op, group=group.device_group)
    return input


def reduce_scatter(
    input: torch.Tensor,
    dim: int = -1,
    op=dist.ReduceOp.SUM,
    group: Group | None = None,
) -> torch.Tensor:
    group = get_world_group() if group is None else group
    if group.size == 1:
        return input
    assert -input.dim() <= dim < input.dim()
    if dim < 0:
        dim += input.dim()
    input_tensor = input.movedim(0, dim).contiguous()
    assert input_tensor.shape[0] % group.size == 0
    chunk_size = input_tensor.shape[0] // group.size
    output_tensor = torch.empty(
        (chunk_size,) + input_tensor.shape[1:],
        dtype=input_tensor.dtype,
        device=input_tensor.device,
    )
    dist.reduce_scatter_tensor(
        output_tensor, input_tensor, group=group.device_group, op=op
    )
    return output_tensor.movedim(0, dim).contiguous()


def gather(
    input: torch.Tensor, dst: int = 0, dim: int = -1, group: Group | None = None
) -> torch.Tensor | None:
    group = get_world_group() if group is None else group
    if group.size == 1:
        return input
    assert -input.dim() <= dim < input.dim()
    if dim < 0:
        dim += input.dim()
    rank_in_group = group.group_rank
    if rank_in_group == dst:
        gather_list = [torch.empty_like(input) for _ in range(group.size)]
    else:
        gather_list = None
    dist.gather(input, gather_list, group_dst=dst, group=group.device_group)
    if rank_in_group == dst:
        return torch.cat(gather_list, dim=dim)
    return None


def broadcast(input: torch.Tensor, src: int = 0, group: Group | None = None) -> torch.Tensor:
    group = get_world_group() if group is None else group
    assert src < group.size, f"Invalid src rank ({src})"
    if group.size == 1:
        return input
    dist.broadcast(input, group_src=src, group=group.device_group)
    return input


# --------------------------------------------------------------------------- #
# TP 封装
# --------------------------------------------------------------------------- #


def tensor_model_parallel_all_reduce(
    input: torch.Tensor, op=dist.ReduceOp.SUM
) -> torch.Tensor:
    return all_reduce(input, op, group=get_tp_group())


def tensor_model_parallel_all_gather(input: torch.Tensor, dim: int = -1) -> torch.Tensor:
    return all_gather(input, dim=dim, group=get_tp_group())


def tensor_model_parallel_reduce_scatter(
    input: torch.Tensor, dim: int = -1, op=dist.ReduceOp.SUM
) -> torch.Tensor:
    return reduce_scatter(input, dim=dim, op=op, group=get_tp_group())


# --------------------------------------------------------------------------- #
# PP 封装：点对点传中间张量字典
# --------------------------------------------------------------------------- #


def send_tensor(tensor: torch.Tensor, dst: int, group: Group | None = None) -> None:
    """定点发送单个张量（无元数据，形状双方约定）。

    用于 CUDA Graph 捕获路径：decode 中间张量形状恒为 [bs, hidden]，
    双方预先知道形状，直接 dist.send，可被图捕获。
    """
    group = get_world_group() if group is None else group
    if group.size == 1:
        return
    dist.send(tensor, group_dst=dst, group=group.device_group)


def recv_tensor(tensor: torch.Tensor, src: int, group: Group | None = None) -> torch.Tensor:
    """定点接收单个张量到预分配 tensor（无元数据，形状双方约定）。

    接收目标由调用方预先分配（通常为静态 buffer 的切片，保证图重放地址固定）。
    """
    group = get_world_group() if group is None else group
    if group.size == 1:
        return tensor
    dist.recv(tensor, group_src=src, group=group.device_group)
    return tensor


def send_tensor_dict(
    tensor_dict: dict[str, torch.Tensor],
    dst: int | None = None,
    group: Group | None = None,
    all_gather_group: Group | None = None,
) -> None:
    """发送中间张量字典。dst 缺省为组内下一个 rank（环形）。"""
    group = get_world_group() if group is None else group
    if group.size == 1:
        return

    all_gather_size = 1 if all_gather_group is None else all_gather_group.size
    all_gather_rank = 0 if all_gather_group is None else all_gather_group.group_rank

    if dst is None:
        dst = (group.group_rank + 1) % group.size
    assert dst < group.size, f"Invalid dst rank ({dst})"

    metadata_list: list[tuple[str, str, torch.dtype, torch.Size]] = [
        (key, tensor.device.type, tensor.dtype, tensor.shape)
        for key, tensor in tensor_dict.items()
    ]
    dist.send_object_list([metadata_list], group_dst=dst, group=group.cpu_group)

    for _, tensor in tensor_dict.items():
        if tensor.numel() == 0:
            continue
        # send-allgather：只发切片，接收端再 allgather，降低首段带宽压力。
        if all_gather_group is not None and tensor.numel() % all_gather_size == 0:
            tensor = tensor.reshape(all_gather_size, -1)[all_gather_rank]
        if tensor.is_cpu:
            dist.send(tensor, group_dst=dst, group=group.cpu_group)
        else:
            dist.send(tensor, group_dst=dst, group=group.device_group)


def recv_tensor_dict(
    src: int | None = None,
    group: Group | None = None,
    all_gather_group: Group | None = None,
) -> dict[str, torch.Tensor] | None:
    """接收中间张量字典。src 缺省为组内上一个 rank（环形）。"""
    group = get_world_group() if group is None else group
    if group.size == 1:
        return None

    all_gather_size = 1 if all_gather_group is None else all_gather_group.size
    all_gather_rank = 0 if all_gather_group is None else all_gather_group.group_rank

    if src is None:
        src = (group.group_rank - 1) % group.size
    assert src < group.size, f"Invalid src rank ({src})"

    recv_metadata_list: list = [[]]
    dist.recv_object_list(recv_metadata_list, group_src=src, group=group.cpu_group)

    tensor_dict: dict[str, torch.Tensor] = {}
    for key, device, dtype, shape in recv_metadata_list[0]:
        tensor = torch.empty(shape, dtype=dtype, device=device)
        if tensor.numel() == 0:
            tensor_dict[key] = tensor
            continue
        use_all_gather = (
            all_gather_group is not None and tensor.numel() % all_gather_size == 0
        )
        orig_shape = tensor.shape
        if use_all_gather:
            tensor = tensor.reshape(all_gather_size, -1)[all_gather_rank]
        if tensor.is_cpu:
            dist.recv(tensor, group_src=src, group=group.cpu_group)
        else:
            dist.recv(tensor, group_src=src, group=group.device_group)
        if use_all_gather:
            tensor = all_gather(tensor, dim=0, group=all_gather_group)
            tensor = tensor.reshape(orig_shape)
        tensor_dict[key] = tensor
    return tensor_dict


def broadcast_tensor_dict(
    tensor_dict: dict[Any, torch.Tensor] | None = None,
    src: int = 0,
    group: Group | None = None,
) -> dict[Any, torch.Tensor] | None:
    """广播张量字典（CPU 张量走 gloo，GPU 张量走 nccl）。"""
    group = get_world_group() if group is None else group
    if not dist.is_initialized() or group.size == 1:
        return tensor_dict
    assert src < group.size, f"Invalid src rank ({src})"

    metadata_list: list[tuple[Any, str, torch.dtype, torch.Size]] = []
    if group.group_rank == src:
        assert isinstance(tensor_dict, dict)
        for key, tensor in tensor_dict.items():
            metadata_list.append((key, tensor.device.type, tensor.dtype, tensor.shape))
    recv = [metadata_list]
    dist.broadcast_object_list(recv, group_src=src, group=group.cpu_group)
    metadata_list = recv[0]

    result: dict[Any, torch.Tensor] = {}
    handles: list[dist.Work] = []
    for key, device, dtype, shape in metadata_list:
        if group.group_rank == src:
            assert isinstance(tensor_dict, dict)
            tensor = tensor_dict[key]
        else:
            tensor = torch.empty(shape, dtype=dtype, device=device)
            if tensor.numel() == 0:
                result[key] = tensor
                continue
        if tensor.is_cpu:
            handle = dist.broadcast(
                tensor, group_src=src, group=group.cpu_group, async_op=True
            )
        else:
            handle = dist.broadcast(
                tensor, group_src=src, group=group.device_group, async_op=True
            )
        if handle is not None:
            handles.append(handle)
        result[key] = tensor
    for handle in handles:
        handle.wait()
    return result
