"""PP2 fixed-schema IntermediateTensors transport."""

from __future__ import annotations

import os

import pytest
import torch

from tests.conftest import needs_2gpu

pytestmark = [pytest.mark.distributed, needs_2gpu]


@pytest.fixture(scope="module")
def pp_env():
    if "RANK" not in os.environ:
        pytest.skip("requires torchrun")
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    if world_size != 2:
        pytest.skip("pp transport test uses world_size=2")

    from src.distributed import (
        destroy_distributed_environment,
        destroy_model_parallel,
        init_distributed_environment,
        initialize_model_parallel,
    )

    init_distributed_environment(
        world_size=2,
        rank=rank,
        backend="nccl",
        init_method="env://",
        local_device=local_rank,
    )
    initialize_model_parallel(tp_size=1, pp_size=2, tp_rank=0, pp_rank=rank)
    yield rank, local_rank
    destroy_model_parallel()
    destroy_distributed_environment()


def test_pp_is_topology_only(pp_env):
    from src.distributed import get_pp_group, get_tp_group, get_world_group
    from src.distributed.parallel_state import Communicator, RankGroup

    assert isinstance(get_pp_group(), RankGroup)
    assert not isinstance(get_pp_group(), Communicator)
    assert get_tp_group().size == 1
    assert get_tp_group().device_group is None
    assert get_world_group().cpu_group is not None


def test_pp_intermediate_roundtrip(pp_env):
    from src.distributed import (
        get_pp_group,
        recv_intermediate_tensors,
        send_intermediate_tensors,
    )
    from src.distributed.pp import IntermediateTensors

    pp = get_pp_group()
    tokens, hidden = 8, 64
    if pp.is_first_rank:
        h = torch.arange(tokens * hidden, device="cuda", dtype=torch.float16).view(
            tokens, hidden
        )
        r = h + 1
        send_intermediate_tensors(
            IntermediateTensors(hidden_states=h, residual=r), dst=pp.next_rank
        )
    else:
        got = recv_intermediate_tensors(src=pp.prev_rank)
        expect = torch.arange(tokens * hidden, device="cuda", dtype=torch.float16).view(
            tokens, hidden
        )
        assert torch.equal(got.hidden_states, expect)
        assert torch.equal(got.residual, expect + 1)


def test_pp_empty_tensors(pp_env):
    from src.distributed import (
        get_pp_group,
        recv_intermediate_tensors,
        send_intermediate_tensors,
    )
    from src.distributed.pp import IntermediateTensors

    pp = get_pp_group()
    if pp.is_first_rank:
        empty = IntermediateTensors(
            hidden_states=torch.empty(0, 64, device="cuda", dtype=torch.float16),
            residual=torch.empty(0, 64, device="cuda", dtype=torch.float16),
        )
        send_intermediate_tensors(empty, dst=pp.next_rank)
    else:
        got = recv_intermediate_tensors(src=pp.prev_rank)
        assert got.hidden_states.shape == (0, 64)
        assert got.residual.shape == (0, 64)
