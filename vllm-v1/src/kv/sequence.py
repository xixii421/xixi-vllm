from __future__ import annotations
from dataclasses import dataclass, field


@dataclass
class Sequence:
    token_ids: list[int]
    block_size: int
    seq_id: int = 0
    block_table: list[int] = field(default_factory=list)
    num_cached_tokens: int = 0

    @property
    def num_tokens(self) -> int:
        return len(self.token_ids)

    @property
    def num_blocks(self) -> int:
        n = self.num_tokens
        if n == 0:
            return 0
        return (n + self.block_size - 1) // self.block_size

    def num_full_blocks(self) -> int:
        return self.num_tokens // self.block_size

    def block_token_ids(self, i: int) -> list[int]:
        s = i * self.block_size
        return self.token_ids[s : s + self.block_size]

    def append_token(self, token_id: int) -> None:
        self.token_ids.append(token_id)
