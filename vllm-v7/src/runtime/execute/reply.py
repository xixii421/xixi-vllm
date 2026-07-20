"""RPC reply policy shared by worker processes and the executor.

``execute_model``: only the driver (last-PP tp0) and mRoPE source (first-PP
tp0) reply.  When ``pp_size == 1`` those roles coincide → one reply.

All control RPCs: every rank replies (barrier / reduce).  Errors always reply
so the parent can fail-fast even from a non-driver rank.
"""

from __future__ import annotations

EXECUTE_METHOD = "execute_model"


def role_flags(tp_rank: int, pp_rank: int, pp_size: int) -> tuple[bool, bool]:
    is_driver = tp_rank == 0 and pp_rank == pp_size - 1
    is_mrope_source = tp_rank == 0 and pp_rank == 0
    return is_driver, is_mrope_source


def execute_expected_replies(pp_size: int) -> int:
    return 1 if pp_size <= 1 else 2


def should_reply(
    method: str, *, is_driver: bool, is_mrope_source: bool
) -> bool:
    if method == EXECUTE_METHOD:
        return is_driver or is_mrope_source
    return True


def include_rank_in_reply(method: str) -> bool:
    """``get_sys_profile`` tags each payload with the worker rank."""
    return method == "get_sys_profile"
