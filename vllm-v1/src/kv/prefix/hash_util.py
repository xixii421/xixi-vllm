""" 满块链式哈希 """

from __future__ import annotations

import hashlib
import struct


def compute_block_hash(token_ids: list[int], prefix: int = -1) -> int:
    """ 本块 tokens（可选缀上一段块哈希）→ 64bit 无符号哈希。 """
    h = hashlib.blake2b(digest_size=8)
    if prefix != -1:
        h.update(prefix.to_bytes(8, "little", signed=False))
    h.update(struct.pack(f"<{len(token_ids)}i", *token_ids))
    return int.from_bytes(h.digest(), "little", signed=False)
