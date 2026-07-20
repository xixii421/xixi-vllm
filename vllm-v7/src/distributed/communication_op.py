from __future__ import annotations

from typing import Optional

import torch
import torch.distributed as dist

from .parallel_state import Communicator, get_tp_group, get_world_group
from .pp import IntermediateTensors

_DTYPE_TO_CODE: dict[torch.dtype, int] = {
    torch.float16: 0,
    torch.bfloat16: 1,
    torch.float32: 2,
    torch.float64: 3,
    torch.int32: 4,
    torch.int64: 5,
    torch.uint8: 6,
    torch.int8: 7,
}
_CODE_TO_DTYPE: dict[int, torch.dtype] = {v: k for k, v in _DTYPE_TO_CODE.items()}

_PP_SHARD_CACHE: dict[tuple[int, torch.dtype, str], torch.Tensor] = {}


def clear_pp_buffer_cache() -> None:
    _PP_SHARD_CACHE.clear()


def all_gather(
    input: torch.Tensor,
    dim: int = -1,
    group: Optional[Communicator] = None,
) -> torch.Tensor:
    """Concatenate shards along ``dim``. Output is contiguous."""
    if dim < 0:
        dim += input.dim()
    if not (0 <= dim < input.dim()):
        raise ValueError(f"Invalid dim ({dim}) for input shape {tuple(input.size())}")

    group = get_world_group() if group is None else group
    world_size = group.size
    if world_size == 1:
        return input

    if group.device_group is None:
        raise RuntimeError("all_gather requires a device process group")

    input_t = input.movedim(dim, 0).contiguous()
    gathered = torch.empty(
        (world_size * input_t.shape[0],) + tuple(input_t.shape[1:]),
        dtype=input.dtype,
        device=input.device,
    )
    dist.all_gather_into_tensor(gathered, input_t, group=group.device_group)
    out = gathered.movedim(0, dim)
    return out if out.is_contiguous() else out.contiguous()


def all_reduce(
    input: torch.Tensor,
    op=dist.ReduceOp.SUM,
    group: Optional[Communicator] = None,
) -> torch.Tensor:
    """In-place all-reduce. Caller must pass a contiguous tensor."""
    group = get_world_group() if group is None else group
    if group.size == 1:
        return input
    if group.device_group is None:
        raise RuntimeError("all_reduce requires a device process group")
    if not input.is_contiguous():
        raise ValueError("all_reduce requires a contiguous input tensor")
    dist.all_reduce(input, op, group=group.device_group)
    return input


def tensor_model_parallel_all_reduce(
    input: torch.Tensor, op=dist.ReduceOp.SUM
) -> torch.Tensor:
    return all_reduce(input, op, group=get_tp_group())


def tensor_model_parallel_all_gather(
    input: torch.Tensor, dim: int = -1
) -> torch.Tensor:
    return all_gather(input, dim=dim, group=get_tp_group())


def _dtype_code(dtype: torch.dtype) -> int:
    code = _DTYPE_TO_CODE.get(dtype)
    if code is None:
        raise TypeError(f"unsupported PP dtype: {dtype}")
    return code


def _shard_payload(
    tensor: torch.Tensor, all_gather_group: Communicator | None
) -> tuple[torch.Tensor, bool]:
    """Return (payload, use_all_gather). Payload may be a shard view."""
    if all_gather_group is None or all_gather_group.size <= 1:
        return tensor, False
    ag_size = all_gather_group.size
    if tensor.numel() % ag_size != 0:
        return tensor, False
    shard = tensor.reshape(ag_size, -1)[all_gather_group.group_rank]
    return shard.contiguous(), True


def _recv_shard_buffer(
    numel: int, dtype: torch.dtype, device: torch.device
) -> torch.Tensor:
    key = (numel, dtype, str(device))
    buf = _PP_SHARD_CACHE.get(key)
    if buf is None or buf.numel() < numel or buf.dtype != dtype or buf.device != device:
        buf = torch.empty(numel, dtype=dtype, device=device)
        _PP_SHARD_CACHE[key] = buf
    return buf[:numel]


def send_intermediate_tensors(
    tensors: IntermediateTensors,
    dst: int,
    *,
    group: Optional[Communicator] = None,
    all_gather_group: Optional[Communicator] = None,
) -> None:
    """Send fixed-schema PP activations: hidden_states then residual."""
    group = get_world_group() if group is None else group
    if not dist.is_initialized() or group.size == 1:
        return
    if not (0 <= dst < group.size):
        raise ValueError(f"Invalid dst rank ({dst})")
    if group.device_group is None:
        raise RuntimeError("PP send requires a device process group")

    hs = tensors.hidden_states
    res = tensors.residual
    if hs.shape != res.shape:
        raise ValueError(
            f"hidden_states shape {tuple(hs.shape)} != residual shape {tuple(res.shape)}"
        )
    if hs.dtype != res.dtype:
        raise ValueError(f"dtype mismatch: {hs.dtype} vs {res.dtype}")
    if hs.dim() != 2:
        raise ValueError(f"expected 2D activations, got shape {tuple(hs.shape)}")

    # Compact header on Gloo: [tokens, hidden, dtype_code, use_tp_shard]
    use_shard = (
        all_gather_group is not None
        and all_gather_group.size > 1
        and hs.numel() % all_gather_group.size == 0
    )
    header = torch.tensor(
        [hs.shape[0], hs.shape[1], _dtype_code(hs.dtype), int(use_shard)],
        dtype=torch.int64,
    )
    if group.cpu_group is None:
        raise RuntimeError("PP send requires a cpu_group for the shape header")
    dist.send(header, group_dst=dst, group=group.cpu_group)

    for tensor in (hs, res):
        if tensor.numel() == 0:
            continue
        payload, _ = _shard_payload(tensor, all_gather_group if use_shard else None)
        dist.send(payload, group_dst=dst, group=group.device_group)


def recv_intermediate_tensors(
    src: int,
    *,
    group: Optional[Communicator] = None,
    all_gather_group: Optional[Communicator] = None,
) -> IntermediateTensors:
    """Receive fixed-schema PP activations."""
    group = get_world_group() if group is None else group
    if not dist.is_initialized() or group.size == 1:
        raise RuntimeError("recv_intermediate_tensors requires an active multi-rank group")
    if not (0 <= src < group.size):
        raise ValueError(f"Invalid src rank ({src})")
    if group.device_group is None or group.cpu_group is None:
        raise RuntimeError("PP recv requires device and cpu process groups")

    header = torch.empty(4, dtype=torch.int64)
    dist.recv(header, group_src=src, group=group.cpu_group)
    tokens = int(header[0].item())
    hidden = int(header[1].item())
    dtype = _CODE_TO_DTYPE[int(header[2].item())]
    use_shard = bool(header[3].item())
    shape = (tokens, hidden)
    device = torch.device("cuda", torch.cuda.current_device())

    outs: list[torch.Tensor] = []
    for _ in range(2):
        full_numel = tokens * hidden
        if full_numel == 0:
            outs.append(torch.empty(shape, dtype=dtype, device=device))
            continue

        if use_shard:
            if all_gather_group is None or all_gather_group.size <= 1:
                raise RuntimeError("sender used TP shard but receiver has no all_gather_group")
            ag_size = all_gather_group.size
            if full_numel % ag_size != 0:
                raise RuntimeError("activation numel not divisible by TP size")
            shard_numel = full_numel // ag_size
            shard = _recv_shard_buffer(shard_numel, dtype, device)
            dist.recv(shard, group_src=src, group=group.device_group)
            gathered = all_gather(shard, dim=0, group=all_gather_group)
            outs.append(gathered.view(shape))
        else:
            buf = torch.empty(shape, dtype=dtype, device=device)
            dist.recv(buf, group_src=src, group=group.device_group)
            outs.append(buf)

    return IntermediateTensors(hidden_states=outs[0], residual=outs[1])
