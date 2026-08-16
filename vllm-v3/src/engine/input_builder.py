from __future__ import annotations
from dataclasses import dataclass, field
import torch
from ..memory.sequence import Sequence, slots_for_range
from ..utils import h2d


@dataclass
class ModelInputs:
    """一次模型前向所需的完整输入。

    除 seqs 外，其余字段均为展平或补齐后的 GPU 张量，
    可直接交给 attention backend 和模型前向使用。
    """

    # [total_tokens] 本步参与计算的 token ids，所有序列按顺序展平。
    input_ids: torch.Tensor | None = None

    # [total_tokens] 每个 token 在原始序列中的逻辑位置。
    positions: torch.Tensor | None = None

    # 本步是否为 prefill。该标记会影响采样时如何从 logits 中取行。
    is_prefill: bool = False

    # [n_seq + 1] query token 的累计边界，用于变长 attention kernel。
    cu_seqlens_q: torch.Tensor | None = None

    # [n_seq + 1] key token 的累计边界。
    # 当前实现中本步 key 与 query 使用同一套展平布局，因此与 cu_seqlens_q 相同。
    cu_seqlens_k: torch.Tensor | None = None

    # 本步单个序列中 query token 的最大数量。
    max_seqlen_q: int = 0

    # 本步单个序列中 key token 的最大数量。
    max_seqlen_k: int = 0

    # [total_tokens] 每个 token 对应的物理 KV slot，用于写入 KV cache。
    slot_mapping: torch.Tensor | None = None

    # [n_seq, max_blocks] 补齐后的物理页表，短序列用 -1 填充。
    block_tables: torch.Tensor | None = None

    # 本批序列对象引用，供后续采样、后处理和状态更新使用。
    seqs: list[Sequence] = field(default_factory=list)


def _decode_span(seq: Sequence) -> tuple[int, int]:
    """返回 decode 步骤需要计算的 token 区间。

    decode 阶段每个序列只消费最后一个已生成 token，
    区间为 [len(seq) - 1, len(seq))。
    """
    return len(seq) - 1, len(seq)


def _prefill_span(seq: Sequence) -> tuple[int, int]:
    """返回 prefill 步骤需要计算的 token 区间。

    prefill 阶段只计算未被 prefix cache 命中的 token，
    区间为 [num_cached_tokens, num_cached_tokens + num_scheduled_tokens)。
    """
    start = seq.num_cached_tokens
    return start, start + seq.num_scheduled_tokens


def _pack(
    seqs: list[Sequence],
    spans: list[tuple[int, int]],
    block_size: int,
    device: torch.device,
) -> ModelInputs:
    """将多个序列的待计算 token 区间打包成展平 GPU 输入。

    Args:
        seqs: 参与本步前向的序列列表。
        spans: 与 seqs 一一对应的 (start, end) token 区间。
        block_size: KV cache block 大小。
        device: 目标 GPU 设备。

    Returns:
        打包后的 ModelInputs。
    """
    input_ids: list[int] = []
    positions: list[int] = []
    slots: list[int] = []

    # cumulative query lengths：记录每个序列在展平 query token 中的边界。
    cu_q = [0]
    max_q = 0

    # 展平所有序列的 token、位置和 KV slot，并构建 query 的累计边界。
    for seq, (start, end) in zip(seqs, spans):
        assert seq.block_table, f"seq {seq.seq_id} 缺少 block_table，无法映射 slot"
        qo = end - start
        input_ids.extend(seq.token_ids[start:end])
        positions.extend(range(start, end))
        slots.extend(slots_for_range(seq.block_table, start, end, block_size))
        cu_q.append(cu_q[-1] + qo)
        max_q = max(max_q, qo)

    # 当前实现中 query 与 key 使用同一套展平 token 布局，
    # 因此 cu_seqlens_q 与 cu_seqlens_k 指向同一个张量。
    cu_t = h2d(cu_q, torch.int32, device)

    return ModelInputs(
        input_ids=h2d(input_ids, torch.int64, device),
        positions=h2d(positions, torch.int64, device),
        cu_seqlens_q=cu_t,
        cu_seqlens_k=cu_t,
        max_seqlen_q=max_q,
        max_seqlen_k=max_q,
        slot_mapping=h2d(slots, torch.int32, device),
        block_tables=_block_tables(seqs, device) if seqs else None,
        seqs=seqs,
    )


def prepare_unified(
    decodes: list[Sequence],
    prefills: list[Sequence],
    block_size: int,
    device: torch.device,
) -> tuple[ModelInputs, list[int]]:
    """准备 decode + prefill 混合批次的模型输入。

    序列顺序为 decodes 在前、prefills 在后，便于 attention backend
    按统一批次处理。

    Args:
        decodes: decode 阶段序列列表。
        prefills: prefill 阶段序列列表。
        block_size: KV cache block 大小。
        device: 目标 GPU 设备。

    Returns:
        tuple:
            - ModelInputs: 前向输入。
            - list[int]: 每个序列当前完整 KV 长度。
              decode 使用当前总长度；prefill 使用本次计算结束后的长度。
    """
    decode_spans = [_decode_span(s) for s in decodes]
    prefill_spans = [_prefill_span(s) for s in prefills]

    # 统一批次：先 decode，后 prefill。
    seqs = decodes + prefills

    # decode 的 KV 长度为序列当前总长度；
    # prefill 的 KV 长度为本次计算区间结束位置。
    kv_lens = [len(s) for s in decodes] + [end for _, end in prefill_spans]

    inputs = _pack(seqs, decode_spans + prefill_spans, block_size, device)

    # 混合批次下，key 长度需要反映每个序列当前完整 KV 长度，
    # 而不只是本步参与计算的 query token 长度。
    inputs.max_seqlen_k = max(kv_lens) if kv_lens else 0

    return inputs, kv_lens


def prepare_prefill(
    seqs: list[Sequence],
    block_size: int,
    device: torch.device,
) -> ModelInputs:
    """准备纯 prefill 批次的模型输入。"""
    return _pack(seqs, [_prefill_span(s) for s in seqs], block_size, device)


def prepare_decode(
    seqs: list[Sequence],
    block_size: int,
    device: torch.device,
) -> ModelInputs:
    """准备纯 decode 批次的模型输入。"""
    return _pack(seqs, [_decode_span(s) for s in seqs], block_size, device)


def build_model_inputs(
    seqs: list[Sequence],
    is_prefill: bool,
    block_size: int,
    device: torch.device,
) -> ModelInputs:
    """按阶段构建模型输入的统一入口。

    Args:
        seqs: 本步前向序列列表。
        is_prefill: True 走 prefill 路径，False 走 decode 路径。
        block_size: KV cache block 大小。
        device: 目标 GPU 设备。

    Returns:
        标记了 is_prefill 的 ModelInputs。
    """
    if is_prefill:
        inputs = prepare_prefill(seqs, block_size, device)
    else:
        inputs = prepare_decode(seqs, block_size, device)

    inputs.is_prefill = is_prefill
    return inputs


def _block_tables(seqs: list[Sequence], device: torch.device) -> torch.Tensor:
    """将变长 block_table 补齐为规整的二维张量。

    Args:
        seqs: 序列列表。
        device: 目标 GPU 设备。

    Returns:
        [n_seq, max_blocks] 的物理页表张量，短序列用 -1 填充。
    """
    max_len = max(len(seq.block_table) for seq in seqs)
    tables = [seq.block_table + [-1] * (max_len - len(seq.block_table)) for seq in seqs]
    return h2d(tables, torch.int32, device)