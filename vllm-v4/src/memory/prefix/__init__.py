"""Prefix backends: none | hash | radix."""

from __future__ import annotations

from typing import Literal

from .hash_index import HashPrefixIndex
from .hash_util import compute_block_hash
from .noop import NoopPrefixIndex
from .radix_index import RadixPrefixIndex
from .radix_tree import RadixTree

PrefixBackend = Literal["none", "hash", "radix"]


def make_prefix_index(backend: PrefixBackend):
    if backend == "hash":
        return HashPrefixIndex()
    if backend == "radix":
        return RadixPrefixIndex()
    return NoopPrefixIndex()


__all__ = [
    "HashPrefixIndex",
    "NoopPrefixIndex",
    "PrefixBackend",
    "RadixPrefixIndex",
    "RadixTree",
    "compute_block_hash",
    "make_prefix_index",
]
