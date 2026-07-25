from __future__ import annotations

from src.runtime.execute.reply import (
    execute_expected_replies,
    role_flags,
    should_reply,
)


def test_role_flags_pp1() -> None:
    is_driver, is_mrope = role_flags(0, 0, 1)
    assert is_driver and is_mrope
    assert execute_expected_replies(1) == 1


def test_role_flags_pp2() -> None:
    d0, m0 = role_flags(0, 0, 2)
    d1, m1 = role_flags(0, 1, 2)
    assert (d0, m0) == (False, True)
    assert (d1, m1) == (True, False)
    assert execute_expected_replies(2) == 2


def test_execute_reply_policy() -> None:
    assert should_reply(
        "execute_model", is_driver=True, is_mrope_source=False
    )
    assert should_reply(
        "execute_model", is_driver=False, is_mrope_source=True
    )
    assert not should_reply(
        "execute_model", is_driver=False, is_mrope_source=False
    )


def test_control_all_ranks_reply() -> None:
    assert should_reply(
        "get_num_kvcache_blocks", is_driver=False, is_mrope_source=False
    )
    assert should_reply(
        "get_sys_profile", is_driver=False, is_mrope_source=False
    )
