from __future__ import annotations
import itertools
import math
from typing import Optional
import torch
import torch.distributed as dist
from ..distributed import all_reduce, get_tp_group, tensor_model_parallel_all_gather

def get_load_balance_assignment(sizes: list[int], num_gpus: int) -> tuple[list[int], list[int], list[int]]:
    n = len(sizes)
    if n == 0:
        return ([], [0] * num_gpus, [0] * num_gpus)
    assignments: list[list[int]] = [[] for _ in range(num_gpus)]
    loads = [0] * num_gpus
    for idx in sorted(range(n), key=lambda i: sizes[i], reverse=True):
        g = min(range(num_gpus), key=lambda i: loads[i])
        assignments[g].append(idx)
        loads[g] += sizes[idx]
    shuffle: list[int] = []
    counts: list[int] = []
    for g in range(num_gpus):
        shuffle.extend(assignments[g])
        counts.append(len(assignments[g]))
    return (shuffle, counts, loads)

def run_dp_sharded_vision_encode(vision_tower, pixel_values: torch.Tensor, grid_thw: torch.Tensor) -> tuple[torch.Tensor, Optional[list[torch.Tensor]]]:
    tp = get_tp_group()
    tp_size = tp.size
    tp_rank = tp.group_rank
    merge = int(vision_tower.merge_size)
    merge_area = merge * merge
    out_h = int(vision_tower.out_dim)
    grid_list = grid_thw.tolist() if grid_thw.dim() == 2 else list(grid_thw)
    patches_per_image = [int(math.prod(g)) for g in grid_list]
    tokens_per_image = [p // merge_area for p in patches_per_image]
    cum_patches = [0, *itertools.accumulate(patches_per_image)]
    shuffle, sample_counts, patch_loads = get_load_balance_assignment(patches_per_image, tp_size)
    cum_counts = [0, *itertools.accumulate(sample_counts)]
    local_idxs = shuffle[cum_counts[tp_rank]:cum_counts[tp_rank + 1]]
    if local_idxs:
        local_pv = torch.cat([pixel_values[cum_patches[i]:cum_patches[i + 1]] for i in local_idxs], dim=0)
        local_grid = torch.tensor([grid_list[i] for i in local_idxs], device=pixel_values.device, dtype=grid_thw.dtype if isinstance(grid_thw, torch.Tensor) else torch.long)
        parts = vision_tower(local_pv, grid_thw=local_grid)
        if isinstance(parts, tuple):
            local_flat = torch.cat(list(parts), dim=0)
        else:
            local_flat = parts
        local_dps = vision_tower.deepstack_features
    else:
        param = next(vision_tower.parameters(), None)
        out_dtype = param.dtype if param is not None else pixel_values.dtype
        local_flat = torch.empty((0, out_h), device=pixel_values.device, dtype=out_dtype)
        local_dps = None
    max_tok = max((pl // merge_area for pl in patch_loads)) if patch_loads else 0
    max_tok = max(max_tok, 1)
    cur = local_flat.shape[0]
    if cur < max_tok:
        pad = torch.empty(max_tok - cur, out_h, device=local_flat.device, dtype=local_flat.dtype)
        local_flat = torch.cat([local_flat, pad], dim=0)
    elif cur > max_tok:
        local_flat = local_flat[:max_tok]
    gathered = tensor_model_parallel_all_gather(local_flat.contiguous(), dim=0)
    out_slots: list[torch.Tensor | None] = [None] * len(grid_list)
    cursor_shuf = 0
    for rank in range(tp_size):
        count = sample_counts[rank]
        rank_tok = patch_loads[rank] // merge_area
        block = gathered[rank * max_tok:rank * max_tok + rank_tok]
        if count == 0:
            continue
        rank_imgs = shuffle[cursor_shuf:cursor_shuf + count]
        off = 0
        for img_i in rank_imgs:
            n = tokens_per_image[img_i]
            out_slots[img_i] = block[off:off + n]
            off += n
        cursor_shuf += count
    assert all((t is not None for t in out_slots))
    flat = torch.cat(out_slots, dim=0)
    deepstack: Optional[list[torch.Tensor]] = None
    n_ds = len(local_dps) if local_dps is not None else 0
    depth_t = torch.tensor([n_ds], device=pixel_values.device, dtype=torch.int32)
    all_reduce(depth_t, op=dist.ReduceOp.MAX, group=tp)
    n_ds = int(depth_t.item())
    if n_ds > 0:
        if local_dps is None:
            local_dps = [torch.empty((0, out_h), device=pixel_values.device, dtype=local_flat.dtype) for _ in range(n_ds)]
        gathered_ds: list[torch.Tensor] = []
        for li in range(n_ds):
            chunk = local_dps[li]
            if chunk.shape[0] < max_tok:
                pad = torch.empty(max_tok - chunk.shape[0], out_h, device=chunk.device, dtype=chunk.dtype)
                chunk = torch.cat([chunk, pad], dim=0)
            elif chunk.shape[0] > max_tok:
                chunk = chunk[:max_tok]
            gathered_ds.append(tensor_model_parallel_all_gather(chunk.contiguous(), dim=0))
        deepstack = []
        for li in range(n_ds):
            slots: list[torch.Tensor | None] = [None] * len(grid_list)
            cursor_shuf = 0
            g = gathered_ds[li]
            for rank in range(tp_size):
                count = sample_counts[rank]
                rank_tok = patch_loads[rank] // merge_area
                block = g[rank * max_tok:rank * max_tok + rank_tok]
                if count == 0:
                    continue
                rank_imgs = shuffle[cursor_shuf:cursor_shuf + count]
                off = 0
                for img_i in rank_imgs:
                    n = tokens_per_image[img_i]
                    slots[img_i] = block[off:off + n]
                    off += n
                cursor_shuf += count
            deepstack.append(torch.cat(slots, dim=0))
    return (flat, deepstack)
