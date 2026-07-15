from typing import Any, Optional, Union
import torch
import torch.distributed as dist
from .parallel_state import Group, get_tp_group, get_world_group

def all_gather(input: torch.Tensor, dim: int=-1, group: Optional[Group]=None) -> torch.Tensor:
    if dim < 0:
        dim += input.dim()
    group = get_world_group() if group is None else group
    input_size = input.size()
    world_size = group.size
    if world_size == 1:
        return input
    output_size = (input_size[0] * world_size, *input_size[1:])
    output_tensor = torch.empty(output_size, dtype=input.dtype, device=input.device)
    dist.all_gather_into_tensor(output_tensor, input, group=group.device_group)
    output_tensor = output_tensor.reshape((world_size,) + input_size)
    output_tensor = output_tensor.movedim(0, dim)
    output_tensor = output_tensor.reshape(input_size[:dim] + (world_size * input_size[dim],) + input_size[dim + 1:])
    return output_tensor

def all_reduce(input: torch.Tensor, op=dist.ReduceOp.SUM, group: Optional[Group]=None) -> torch.Tensor:
    group = get_world_group() if group is None else group
    world_size = group.size
    if world_size == 1:
        return input
    dist.all_reduce(input, op, group=group.device_group)
    return input

def reduce_scatter(input: torch.Tensor, dim: int=-1, op=dist.ReduceOp.SUM, group: Optional[Group]=None) -> torch.Tensor:
    group = get_world_group() if group is None else group
    world_size = group.size
    if world_size == 1:
        return input
    assert -input.dim() <= dim < input.dim(), f'Invalid dim ({dim}) for input tensor with shape {input.size()}'
    if dim < 0:
        dim += input.dim()
    input_tensor = input.movedim(dim, 0).contiguous()
    assert input_tensor.shape[0] % world_size == 0
    chunk_size = input_tensor.shape[0] // world_size
    output_shape = (chunk_size,) + input_tensor.shape[1:]
    output_tensor = torch.empty(output_shape, dtype=input_tensor.dtype, device=input_tensor.device)
    dist.reduce_scatter_tensor(output_tensor, input_tensor, group=group.device_group, op=op)
    return output_tensor.movedim(0, dim).contiguous()

def gather(input: torch.Tensor, dst: int=0, dim: int=-1, group: Optional[Group]=None):
    group = get_world_group() if group is None else group
    world_size = group.size
    if world_size == 1:
        return input
    assert -input.dim() <= dim < input.dim(), f'Invalid dim ({dim}) for input tensor with shape {input.size()}'
    if dim < 0:
        dim += input.dim()
    rank_in_group = group.group_rank
    if rank_in_group == dst:
        gather_list = [torch.empty_like(input) for _ in range(world_size)]
    else:
        gather_list = None
    dist.gather(input, gather_list, group_dst=dst, group=group.device_group)
    if rank_in_group == dst:
        output_tensor = torch.cat(gather_list, dim=dim)
    else:
        output_tensor = None
    return output_tensor

def broadcast(input: torch.Tensor, src: int=0, group: Optional[Group]=None):
    group = get_world_group() if group is None else group
    world_size = group.size
    assert src < world_size, f'Invalid src rank ({src})'
    if world_size == 1:
        return input
    dist.broadcast(input, group_src=src, group=group.device_group)
    return input

def tensor_model_parallel_all_reduce(input: torch.Tensor, op=dist.ReduceOp.SUM) -> torch.Tensor:
    tp_group = get_tp_group()
    return all_reduce(input, op, group=tp_group)

def tensor_model_parallel_all_gather(input: torch.Tensor, dim: int=-1) -> torch.Tensor:
    tp_group = get_tp_group()
    return all_gather(input, dim=dim, group=tp_group)

def tensor_model_parallel_reduce_scatter(input: torch.Tensor, dim: int=-1, op=dist.ReduceOp.SUM) -> torch.Tensor:
    tp_group = get_tp_group()
    return reduce_scatter(input, dim=dim, op=op, group=tp_group)

def tensor_model_parallel_gather(input: torch.Tensor, dst: int=0, dim: int=-1) -> torch.Tensor | None:
    tp_group = get_tp_group()
    return gather(input, dst=dst, dim=dim, group=tp_group)

def broadcast_tensor_dict(tensor_dict: dict[Any, torch.Tensor] | None=None, src: int=0, group: Optional[Group]=None) -> dict[Any, torch.Tensor] | None:
    group = get_world_group() if group is None else group
    world_size = group.size
    rank_in_group = group.group_rank
    if not dist.is_initialized() or world_size == 1:
        return tensor_dict
    assert src < world_size, f'Invalid src rank ({src})'
    metadata_list: list[tuple[Any, str, torch.dtype, torch.Size]] = []
    if rank_in_group == src:
        assert isinstance(tensor_dict, dict), f'Expecting a dictionary, got {type(tensor_dict)}'
        for key, tensor in tensor_dict.items():
            metadata_list.append((key, tensor.device.type, tensor.dtype, tensor.shape))
    recv = [metadata_list]
    dist.broadcast_object_list(recv, group_src=src, group=group.cpu_group)
    metadata_list = recv[0]
    result: dict[Any, torch.Tensor] = {}
    async_handles: list[dist.Work] = []
    for key, device, dtype, shape in metadata_list:
        if rank_in_group == src:
            assert isinstance(tensor_dict, dict), f'Expecting a dictionary, got {type(tensor_dict)}'
            tensor = tensor_dict[key]
        else:
            tensor = torch.empty(shape, dtype=dtype, device=device)
            if tensor.numel() == 0:
                result[key] = tensor
                continue
        if tensor.is_cpu:
            handle = dist.broadcast(tensor, group_src=src, group=group.cpu_group, async_op=True)
        else:
            handle = dist.broadcast(tensor, group_src=src, group=group.device_group, async_op=True)
        if handle is not None:
            async_handles.append(handle)
        result[key] = tensor
    for async_handle in async_handles:
        async_handle.wait()
    return result

def send_tensor_dict(tensor_dict: dict[str, Union[torch.Tensor, Any]], dst: Optional[int]=None, group: Optional[Group]=None, all_gather_group: Optional[Group]=None):
    group = get_world_group() if group is None else group
    world_size = group.size
    rank_in_group = group.group_rank
    if not dist.is_initialized() or world_size == 1:
        return
    all_gather_size = 1 if all_gather_group is None else all_gather_group.size
    all_gather_rank = 0 if all_gather_group is None else all_gather_group.group_rank
    if dst is None:
        dst = (rank_in_group + 1) % world_size
    assert dst < world_size, f'Invalid dst rank ({dst})'
    metadata_list: list[tuple[Any, str, torch.dtype, torch.Size]] = []
    tensor_list: list[torch.Tensor] = []
    assert isinstance(tensor_dict, dict), f'Expecting a dictionary, got {type(tensor_dict)}'
    for key, tensor in tensor_dict.items():
        metadata_list.append((key, tensor.device.type, tensor.dtype, tensor.shape))
        tensor_list.append(tensor)
    dist.send_object_list([metadata_list], group_dst=dst, group=group.cpu_group)
    for tensor in tensor_list:
        if tensor.numel() == 0:
            continue
        if all_gather_group is not None and tensor.numel() % all_gather_size == 0:
            tensor = tensor.reshape(all_gather_size, -1)[all_gather_rank]
        if tensor.is_cpu:
            dist.send(tensor, group_dst=dst, group=group.cpu_group)
        else:
            dist.send(tensor, group_dst=dst, group=group.device_group)

def recv_tensor_dict(src: Optional[int]=None, group: Optional[Group]=None, all_gather_group: Optional[Group]=None) -> Optional[dict[str, torch.Tensor]]:
    group = get_world_group() if group is None else group
    world_size = group.size
    rank_in_group = group.group_rank
    if not dist.is_initialized() or world_size == 1:
        return None
    all_gather_size = 1 if all_gather_group is None else all_gather_group.size
    all_gather_rank = 0 if all_gather_group is None else all_gather_group.group_rank
    if src is None:
        src = (rank_in_group - 1) % world_size
    assert src < world_size, f'Invalid src rank ({src})'
    recv_metadata_list = [[]]
    dist.recv_object_list(recv_metadata_list, group_src=src, group=group.cpu_group)
    recv_metadata_list = recv_metadata_list[0]
    tensor_dict: dict[str, torch.Tensor] = {}
    for key, device, dtype, shape in recv_metadata_list:
        tensor = torch.empty(shape, dtype=dtype, device=device)
        if tensor.numel() == 0:
            tensor_dict[key] = tensor
            continue
        use_all_gather = all_gather_group is not None and tensor.numel() % all_gather_size == 0
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
