from __future__ import annotations

import torch
import torch.distributed as dist

from .parallel_state import Group, get_tp_group, get_world_group


def all_gather(
    input: torch.Tensor, dim: int = -1, group: Group | None = None
) -> torch.Tensor:
    group = get_world_group() if group is None else group
    if group.size == 1:
        return input
    if dim < 0:
        dim += input.dim()
    input_size = input.size()
    output_tensor = torch.empty(
        (input_size[0] * group.size, *input_size[1:]),
        dtype=input.dtype,
        device=input.device,
    )
    dist.all_gather_single(output_tensor, input, group=group.device_group)
    if dim == 0:
        return output_tensor
    output_tensor = output_tensor.reshape((group.size,) + input_size).movedim(0, dim)
    return output_tensor.reshape(
        input_size[:dim] + (group.size * input_size[dim],) + input_size[dim + 1 :]
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
    dist.reduce_scatter_single(
        output_tensor, input_tensor, group=group.device_group, op=op
    )
    if dim == 0:
        return output_tensor
    return output_tensor.movedim(0, dim).contiguous()


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


def send_tensors(
    tensors: tuple[torch.Tensor, ...] | list[torch.Tensor],
    dst: int,
    group: Group | None = None,
) -> None:
    """阻塞点对点发送。``dst`` 为组内 rank（``group_peer``），不是 global rank。

    线性 PP 传 ``group.next_group_rank``。piecewise CG 下在图外调用。
    """
    works = _isend_tensors(tensors, dst, group)
    for work in works:
        work.wait()


def recv_tensors(
    tensors: tuple[torch.Tensor, ...] | list[torch.Tensor],
    src: int,
    group: Group | None = None,
) -> None:
    """阻塞点对点接收。``src`` 为组内 rank（``group_peer``），不是 global rank。"""
    works = _irecv_tensors(tensors, src, group)
    for work in works:
        work.wait()


def _isend_tensors(
    tensors: tuple[torch.Tensor, ...] | list[torch.Tensor],
    dst: int,
    group: Group | None = None,
) -> list:
    group = get_world_group() if group is None else group
    if group.size == 1:
        return []
    if not 0 <= dst < group.size:
        raise ValueError(
            f"send dst={dst} is not a group-local rank in [0, {group.size})"
        )
    ops = [
        dist.P2POp(dist.isend, t, group=group.device_group, group_peer=dst)
        for t in tensors
        if t.numel() > 0
    ]
    if not ops:
        return []
    return list(dist.batch_isend_irecv(ops))


def _irecv_tensors(
    tensors: tuple[torch.Tensor, ...] | list[torch.Tensor],
    src: int,
    group: Group | None = None,
) -> list:
    group = get_world_group() if group is None else group
    if group.size == 1:
        return []
    if not 0 <= src < group.size:
        raise ValueError(
            f"recv src={src} is not a group-local rank in [0, {group.size})"
        )
    ops = [
        dist.P2POp(dist.irecv, t, group=group.device_group, group_peer=src)
        for t in tensors
        if t.numel() > 0
    ]
    if not ops:
        return []
    return list(dist.batch_isend_irecv(ops))
