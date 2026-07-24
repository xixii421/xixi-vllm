from __future__ import annotations

from collections import OrderedDict, deque
from dataclasses import dataclass, field
from typing import Callable


@dataclass
class Block:
    ref_count: int = 0
    hash: int = -1
    token_ids: list[int] = field(default_factory=list)

    def update(self, h: int, token_ids: list[int]) -> None:
        self.hash = h
        self.token_ids = list(token_ids)

    def clear_meta(self) -> None:
        self.hash = -1
        self.token_ids = []

    def reset(self) -> None:
        self.ref_count = 1
        self.clear_meta()


class BlockManager:
    def __init__(self, num_blocks: int, block_size: int) -> None:
        if num_blocks <= 0:
            raise ValueError(f"num_blocks must be positive, got {num_blocks}")
        if block_size <= 0:
            raise ValueError(f"block_size must be positive, got {block_size}")
        self.num_blocks = num_blocks
        self.block_size = block_size
        self.blocks = [Block() for _ in range(num_blocks)]
        self.free_ids: deque[int] = deque(range(num_blocks))
        self.used_ids: set[int] = set()
        # LRU: oldest at front → evict first under pressure (vLLM free-queue idea).
        self.cached_ids: OrderedDict[int, None] = OrderedDict()
        self.on_evict: Callable[[int], None] | None = None

    @property
    def num_free(self) -> int:
        return len(self.free_ids)

    @property
    def num_cached(self) -> int:
        return len(self.cached_ids)

    @property
    def available_slots(self) -> int:
        return len(self.free_ids) + len(self.cached_ids)

    def allocate_fresh(self) -> int:
        if not self.free_ids:
            if not self.cached_ids:
                raise RuntimeError("KV block pool exhausted")
            bid, _ = self.cached_ids.popitem(last=False)  # oldest
            self._evict_cached(bid)
        bid = self.free_ids.popleft()
        block = self.blocks[bid]
        block.reset()
        self.used_ids.add(bid)
        return bid

    def _evict_cached(self, bid: int) -> None:
        if self.on_evict is not None:
            self.on_evict(bid)
        self.cached_ids.pop(bid, None)
        self.blocks[bid].ref_count = 0
        self.blocks[bid].clear_meta()
        self.free_ids.append(bid)

    def reclaim_cached(self, bid: int) -> None:
        self.cached_ids.pop(bid, None)
        try:
            self.free_ids.remove(bid)
        except ValueError:
            pass
        self.used_ids.add(bid)
        self.blocks[bid].ref_count = 1

    def touch_cached(self, bid: int) -> None:
        if bid in self.cached_ids:
            self.cached_ids.move_to_end(bid)

    def release_to_cached_or_free(self, bid: int, *, keep_cached: bool) -> None:
        block = self.blocks[bid]
        block.ref_count -= 1
        if block.ref_count > 0:
            return
        self.used_ids.discard(bid)
        if keep_cached and block.hash != -1:
            self.cached_ids[bid] = None
            self.cached_ids.move_to_end(bid)
        else:
            block.clear_meta()
            self.cached_ids.pop(bid, None)
            self.free_ids.append(bid)

    def can_append(self, seq) -> bool:
        need_new = len(seq) % self.block_size == 1
        return (not need_new) or self.available_slots >= 1

    def may_append(self, seq) -> None:
        if len(seq) % self.block_size == 1:
            seq.block_table.append(self.allocate_fresh())
