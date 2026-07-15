from __future__ import annotations
import torch
from ..scheduling import Sequence

def prepare_prefill(seqs: list[Sequence], block_size: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor, dict]:
    input_ids_list: list[int] = []
    positions_list: list[int] = []
    cu_seqlens_q = [0]
    cu_seqlens_k = [0]
    max_seqlen_q = 0
    max_seqlen_k = 0
    slot_mapping_list: list[int] = []
    block_tables = None
    for seq in seqs:
        start = seq.num_cached_tokens
        seqlen_q = seq.num_scheduled_tokens
        end = start + seqlen_q
        seqlen_k = end
        input_ids_list.extend(seq.token_ids[start:end])
        positions_list.extend(range(start, end))
        cu_seqlens_q.append(cu_seqlens_q[-1] + seqlen_q)
        cu_seqlens_k.append(cu_seqlens_k[-1] + seqlen_k)
        max_seqlen_q = max(seqlen_q, max_seqlen_q)
        max_seqlen_k = max(seqlen_k, max_seqlen_k)
        if not seq.block_table:
            continue
        start_block = start // block_size
        end_block = (end + block_size - 1) // block_size
        for i in range(start_block, end_block):
            slot_start = seq.block_table[i] * block_size
            if i == start_block:
                slot_start += start % block_size
            if i != end_block - 1:
                slot_end = seq.block_table[i] * block_size + block_size
            else:
                slot_end = seq.block_table[i] * block_size + end - i * block_size
            slot_mapping_list.extend(range(slot_start, slot_end))
    if any((seq.block_table for seq in seqs)):
        block_tables = _block_tables(seqs, device)
    input_ids = torch.tensor(input_ids_list, dtype=torch.int64, pin_memory=True).to(device, non_blocking=True)
    positions = torch.tensor(positions_list, dtype=torch.int64, pin_memory=True).to(device, non_blocking=True)
    cu_seqlens_q_t = torch.tensor(cu_seqlens_q, dtype=torch.int32, pin_memory=True).to(device, non_blocking=True)
    cu_seqlens_k_t = torch.tensor(cu_seqlens_k, dtype=torch.int32, pin_memory=True).to(device, non_blocking=True)
    slot_mapping = torch.tensor(slot_mapping_list, dtype=torch.int32, pin_memory=True).to(device, non_blocking=True) if slot_mapping_list else None
    return (input_ids, positions, {'slot_mapping': slot_mapping, 'cu_seqlens_q': cu_seqlens_q_t, 'cu_seqlens_k': cu_seqlens_k_t, 'max_seqlen_q': max_seqlen_q, 'max_seqlen_k': max_seqlen_k, 'block_tables': block_tables})

def prepare_decode(seqs: list[Sequence], block_size: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor, dict]:
    input_ids_list: list[int] = []
    positions_list: list[int] = []
    slot_mapping_list: list[int] = []
    context_lens_list: list[int] = []
    for seq in seqs:
        input_ids_list.append(seq.last_token)
        positions_list.append(len(seq) - 1 + getattr(seq, 'rope_delta', 0))
        context_lens_list.append(len(seq))
        slot_mapping_list.append(seq.block_table[-1] * block_size + seq.last_block_num_tokens - 1)
    input_ids = torch.tensor(input_ids_list, dtype=torch.int64, pin_memory=True).to(device, non_blocking=True)
    positions = torch.tensor(positions_list, dtype=torch.int64, pin_memory=True).to(device, non_blocking=True)
    slot_mapping = torch.tensor(slot_mapping_list, dtype=torch.int32, pin_memory=True).to(device, non_blocking=True)
    context_lens = torch.tensor(context_lens_list, dtype=torch.int32, pin_memory=True).to(device, non_blocking=True)
    block_tables = _block_tables(seqs, device)
    return (input_ids, positions, {'slot_mapping': slot_mapping, 'context_lens': context_lens, 'block_tables': block_tables})

def _block_tables(seqs: list[Sequence], device: torch.device) -> torch.Tensor:
    max_len = max((len(seq.block_table) for seq in seqs))
    tables = [seq.block_table + [-1] * (max_len - len(seq.block_table)) for seq in seqs]
    return torch.tensor(tables, dtype=torch.int32, pin_memory=True).to(device, non_blocking=True)
