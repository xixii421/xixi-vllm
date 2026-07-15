from __future__ import annotations
from collections import deque
import numpy as np
from .sequence import Sequence
import xxhash

class Block:

    def __init__(self, block_id: int) -> None:
        self.block_id = block_id
        self.ref_count = 0
        self.hash = -1
        self.token_ids: list[int] = []

    def update(self, h: int, token_ids: list[int]) -> None:
        self.hash = h
        self.token_ids = list(token_ids)

    def reset(self) -> None:
        self.ref_count = 1
        self.hash = -1
        self.token_ids = []

class BlockManager:

    def __init__(self, num_blocks: int, block_size: int) -> None:
        if num_blocks <= 0:
            raise ValueError(f'num_blocks must be positive, got {num_blocks}')
        if block_size <= 0:
            raise ValueError(f'block_size must be positive, got {block_size}')
        self.num_blocks = num_blocks
        self.block_size = block_size
        self.blocks = [Block(i) for i in range(num_blocks)]
        self.hash_to_block_id: dict[int, int] = {}
        self.free_block_ids: deque[int] = deque(range(num_blocks))
        self.used_block_ids: set[int] = set()

    @property
    def num_free(self) -> int:
        return len(self.free_block_ids)

    @property
    def num_used(self) -> int:
        return self.num_blocks - len(self.free_block_ids)

    def blocks_for_tokens(self, num_tokens: int) -> int:
        if num_tokens <= 0:
            return 0
        return (num_tokens + self.block_size - 1) // self.block_size

    @staticmethod
    def compute_hash(token_ids: list[int], prefix: int=-1) -> int:
        if xxhash is not None:
            h = xxhash.xxh64()
            if prefix != -1:
                h.update(prefix.to_bytes(8, 'little'))
            h.update(np.asarray(token_ids, dtype=np.int32).tobytes())
            return h.intdigest()
        seed = prefix if prefix != -1 else 0
        data = np.asarray(token_ids, dtype=np.int32).tobytes()
        return hash((seed, data)) & (1 << 64) - 1

    def _allocate_block(self) -> int:
        if not self.free_block_ids:
            raise RuntimeError(f'KV block pool exhausted: used={self.num_used}/{self.num_blocks}')
        block_id = self.free_block_ids.popleft()
        block = self.blocks[block_id]
        assert block.ref_count == 0
        if block.hash != -1 and self.hash_to_block_id.get(block.hash) == block_id:
            del self.hash_to_block_id[block.hash]
        block.reset()
        self.used_block_ids.add(block_id)
        return block_id

    def _deallocate_block(self, block_id: int) -> None:
        assert self.blocks[block_id].ref_count == 0
        self.used_block_ids.discard(block_id)
        self.free_block_ids.append(block_id)

    def can_allocate(self, seq: Sequence) -> int:
        if getattr(seq, 'has_multimodal', False) and seq.has_multimodal:
            need = seq.num_blocks
            if len(self.free_block_ids) < need:
                return -1
            return 0
        h = -1
        num_cached_blocks = 0
        num_new_blocks = seq.num_blocks
        for i in range(max(seq.num_blocks - 1, 0)):
            token_ids = seq.block_token_ids(i)
            h = self.compute_hash(token_ids, h)
            block_id = self.hash_to_block_id.get(h, -1)
            if block_id == -1 or self.blocks[block_id].token_ids != token_ids:
                break
            num_cached_blocks += 1
            if block_id in self.used_block_ids:
                num_new_blocks -= 1
        if len(self.free_block_ids) < num_new_blocks:
            return -1
        return num_cached_blocks

    def allocate(self, seq: Sequence, num_cached_blocks: int) -> None:
        assert not seq.block_table
        h = -1
        for i in range(num_cached_blocks):
            token_ids = seq.block_token_ids(i)
            h = self.compute_hash(token_ids, h)
            block_id = self.hash_to_block_id[h]
            block = self.blocks[block_id]
            if block_id in self.used_block_ids:
                block.ref_count += 1
            else:
                block.ref_count = 1
                self.free_block_ids.remove(block_id)
                self.used_block_ids.add(block_id)
            seq.block_table.append(block_id)
        for _ in range(num_cached_blocks, seq.num_blocks):
            seq.block_table.append(self._allocate_block())
        seq.num_cached_tokens = num_cached_blocks * self.block_size

    def deallocate(self, seq: Sequence) -> None:
        for block_id in reversed(seq.block_table):
            block = self.blocks[block_id]
            block.ref_count -= 1
            if block.ref_count == 0:
                self._deallocate_block(block_id)
        seq.num_cached_tokens = 0
        seq.block_table.clear()

    def can_append(self, seq: Sequence) -> bool:
        need_new = len(seq) % self.block_size == 1
        return not need_new or len(self.free_block_ids) >= 1

    def may_append(self, seq: Sequence) -> None:
        if len(seq) % self.block_size == 1:
            seq.block_table.append(self._allocate_block())

    def hash_blocks(self, seq: Sequence) -> None:
        if getattr(seq, 'has_multimodal', False) and seq.has_multimodal:
            return
        start = seq.num_cached_tokens // self.block_size
        end = (seq.num_cached_tokens + seq.num_scheduled_tokens) // self.block_size
        if start == end:
            return
        h = self.blocks[seq.block_table[start - 1]].hash if start > 0 else -1
        for i in range(start, end):
            block = self.blocks[seq.block_table[i]]
            token_ids = seq.block_token_ids(i)
            h = self.compute_hash(token_ids, h)
            block.update(h, token_ids)
            old = self.hash_to_block_id.get(h, -1)
            if old == -1 or old == block.block_id or self.blocks[old].token_ids == token_ids:
                self.hash_to_block_id[h] = block.block_id
