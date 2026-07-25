"""Merge multiproc execute_model replies (driver tokens + mRoPE source)."""

from __future__ import annotations


def merge_execute_model_results(
    *,
    driver: tuple[list[int] | None, list | None] | None,
    mrope_source: tuple[list[int] | None, list | None] | None,
) -> tuple[list[int], list | None]:
    """Tokens from last-PP driver; mRoPE state from first-PP encode rank."""
    token_ids = None if driver is None else driver[0]
    if token_ids is None:
        raise RuntimeError("execute_model returned no token ids from driver worker")
    mrope = None
    if mrope_source is not None and mrope_source[1]:
        mrope = mrope_source[1]
    elif driver is not None and driver[1]:
        mrope = driver[1]
    return token_ids, mrope
