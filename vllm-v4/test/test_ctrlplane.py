"""回归测试 ControlChannel（gloo + msgpack）的 broadcast / send / recv / shutdown。

用法（在 vllm-v4/ 下，需 2 卡）:
    ../.venv2/bin/torchrun --nproc_per_node=2 test/test_ctrlplane.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import msgspec
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.data.batch import SchedulerOutput, SeqSchedule  # noqa: E402
from src.distributed import ControlChannel  # noqa: E402
from src.distributed import (  # noqa: E402
    destroy_distributed_environment,
    get_world_group,
    init_distributed_environment,
)

rank = int(os.environ["RANK"])
world_size = int(os.environ["WORLD_SIZE"])
local_rank = int(os.environ["LOCAL_RANK"])

init_distributed_environment(
    world_size=world_size, rank=rank, local_rank=local_rank, backend="nccl",
    init_method="env://",
)
channel = ControlChannel(get_world_group().cpu_group)
enc = msgspec.msgpack.Encoder()
dec_sched = msgspec.msgpack.Decoder(SchedulerOutput | None)
dec_sample = msgspec.msgpack.Decoder(list[list[int]])

if rank == 0:
    wire = SchedulerOutput(
        decodes=[
            SeqSchedule(seq_id=7, block_table=[0, 1, 2], num_cached_tokens=512, token_ids=[99]),
            SeqSchedule(seq_id=8, block_table=[5, 6], num_cached_tokens=256, token_ids=[123]),
        ]
    )
    for step in range(3):
        channel.broadcast(enc.encode(wire), src=0)
        p_ids, d_ids = dec_sample.decode(channel.recv(1))
        assert p_ids == [1, 2, 3] and d_ids == [4, 5], (p_ids, d_ids)
    channel.broadcast(enc.encode(None), src=0)
    print("rank0 OK", flush=True)
else:
    for step in range(3):
        sched = dec_sched.decode(channel.broadcast(None, src=0))
        assert sched is not None and len(sched.decodes) == 2 and sched.decodes[1].seq_id == 8
        channel.send(enc.encode(([1, 2, 3], [4, 5])), dst=0)
    sched = dec_sched.decode(channel.broadcast(None, src=0))
    assert sched is None
    print("rank1 OK", flush=True)

destroy_distributed_environment()
