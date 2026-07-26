"""Prefix-cache backends for BlockManager.

Default production path is ``RadixPrefixIndex`` (ullm-style).
``HashPrefixIndex`` is the flat ``hash → block_id`` fallback.
"""

from __future__ import annotations

from .hash_index import HashPrefixIndex
from .radix_index import RadixPrefixIndex

__all__ = ["HashPrefixIndex", "RadixPrefixIndex"]
