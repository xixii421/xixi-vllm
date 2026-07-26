from __future__ import annotations

import torch
from ..memory import Sequence


PREFILL_LONG_SINGLE_MIN_TOKENS = 16_384
BLOCK_TABLE_LARGE_MIN_ELEMS = 100_000


def _list_to_device(values, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    return torch.tensor(values, dtype=dtype, pin_memory=True).to(device,non_blocking=True)

def _tensor_to_device(x: torch.Tensor, device: torch.device) -> torch.Tensor:
    return x.pin_memory().to(device, non_blocking=True)

def _cached_token_tensor(seq: Sequence) -> torch.Tensor:
    end = seq.num_cached_tokens + seq.num_scheduled_tokens

    tokens = getattr(seq, "_tok_tensor", None)
    if tokens is None or len(tokens) < end:
        tokens = seq._tok_tensor = torch.as_tensor(
            seq.token_ids,
            dtype=torch.int64,
        ).pin_memory()

    return tokens


def _cached_block_table_tensor(seq: Sequence) -> torch.Tensor:
    t = getattr(seq, "_bt_tensor", None)
    if t is None or len(t) != len(seq.block_table):
        t = seq._bt_tensor = torch.as_tensor(seq.block_table, dtype=torch.int32)
    return t


def _prepare_prefill_single_long(
    seq: Sequence,
    block_size: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, dict]:
    start = seq.num_cached_tokens
    seqlen_q = seq.num_scheduled_tokens
    end = start + seqlen_q

    token_ids = _cached_token_tensor(seq)[start:end]
    positions = torch.arange(start, end, dtype=torch.int64)

    block_table = torch.as_tensor(seq.block_table, dtype=torch.int64)
    block_index = positions // block_size
    block_offset = positions % block_size

    slot_mapping = (
        block_table[block_index] * block_size + block_offset
    ).to(torch.int32)

    return (
        _tensor_to_device(token_ids, device),
        _tensor_to_device(positions, device),
        {
            "slot_mapping": _tensor_to_device(slot_mapping, device),
            "cu_seqlens_q": _list_to_device([0, seqlen_q], torch.int32, device),
            "cu_seqlens_k": _list_to_device([0, end], torch.int32, device),
            "max_seqlen_q": seqlen_q,
            "max_seqlen_k": end,
            "block_tables": _block_tables([seq], device),
        },
    )


def prepare_prefill(
    seqs: list[Sequence],
    block_size: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, dict]:
    if (
        len(seqs) == 1
        and seqs[0].block_table
        and seqs[0].num_scheduled_tokens >= PREFILL_LONG_SINGLE_MIN_TOKENS
    ):
        return _prepare_prefill_single_long(seqs[0], block_size, device)

    input_ids_list: list[int] = []
    positions_list: list[int] = []

    cu_seqlens_q = [0]
    cu_seqlens_k = [0]

    max_seqlen_q = 0
    max_seqlen_k = 0

    slot_mapping_list: list[int] = []
    has_block_table = False

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

        block_table = seq.block_table
        if not block_table:
            continue

        has_block_table = True

        start_block = start // block_size
        end_block = (end + block_size - 1) // block_size

        for i in range(start_block, end_block):
            block_id = block_table[i]
            slot_start = block_id * block_size

            if i == start_block:
                slot_start += start % block_size

            if i != end_block - 1:
                slot_end = block_id * block_size + block_size
            else:
                slot_end = block_id * block_size + end - i * block_size

            slot_mapping_list.extend(range(slot_start, slot_end))

    block_tables = _block_tables(seqs, device) if has_block_table else None

    input_ids = _list_to_device(input_ids_list, torch.int64, device)
    positions = _list_to_device(positions_list, torch.int64, device)

    cu_seqlens_q_t = _list_to_device(cu_seqlens_q, torch.int32, device)
    cu_seqlens_k_t = _list_to_device(cu_seqlens_k, torch.int32, device)

    slot_mapping = (
        _list_to_device(slot_mapping_list, torch.int32, device)
        if slot_mapping_list
        else None
    )

    return input_ids, positions, {
        "slot_mapping": slot_mapping,
        "cu_seqlens_q": cu_seqlens_q_t,
        "cu_seqlens_k": cu_seqlens_k_t,
        "max_seqlen_q": max_seqlen_q,
        "max_seqlen_k": max_seqlen_k,
        "block_tables": block_tables,
    }


def prepare_decode(
    seqs: list[Sequence],
    block_size: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, dict]:
    input_ids_list: list[int] = []
    positions_list: list[int] = []
    slot_mapping_list: list[int] = []
    context_lens_list: list[int] = []

    for seq in seqs:
        seq_len = len(seq)

        input_ids_list.append(seq.last_token)

        positions_list.append(seq_len - 1 + seq.rope_delta)

        context_lens_list.append(seq_len)

        slot_mapping_list.append(
            seq.block_table[-1] * block_size + seq.last_block_num_tokens - 1
        )

    input_ids = _list_to_device(input_ids_list, torch.int64, device)
    positions = _list_to_device(positions_list, torch.int64, device)
    slot_mapping = _list_to_device(slot_mapping_list, torch.int32, device)
    context_lens = _list_to_device(context_lens_list, torch.int32, device)
    block_tables = _block_tables(seqs, device)

    return input_ids, positions, {
        "slot_mapping": slot_mapping,
        "context_lens": context_lens,
        "block_tables": block_tables,
    }


def _block_tables(seqs: list[Sequence], device: torch.device) -> torch.Tensor:
    max_len = max(len(seq.block_table) for seq in seqs)

    if len(seqs) * max_len < BLOCK_TABLE_LARGE_MIN_ELEMS:
        tables = [
            seq.block_table + [-1] * (max_len - len(seq.block_table))
            for seq in seqs
        ]
        return torch.tensor(tables, dtype=torch.int32, pin_memory=True).to(
            device,
            non_blocking=True,
        )

    out = torch.full((len(seqs), max_len), -1, dtype=torch.int32, pin_memory=True)

    for i, seq in enumerate(seqs):
        t = _cached_block_table_tensor(seq)
        out[i, : len(t)] = t

    return out.to(device, non_blocking=True)