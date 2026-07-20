""" 单请求的逻辑状态：token 序列 + 页表 """

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Sequence:
    """ 一条生成请求的 tokens 与物理页表。 """

    token_ids: list[int]                              # 已有 prompt + 已生成 token
    block_size: int                                   # 每页 token 数
    seq_id: int = 0                                   # 进程内唯一 id（radix 引用用）
    block_table: list[int] = field(default_factory=list)  # 逻辑页 → 物理块号
    num_cached_tokens: int = 0                        # 前缀命中跳过的 token 数

    @property
    def num_tokens(self) -> int:
        """ 当前序列长度。 """
        return len(self.token_ids)

    @property
    def num_blocks(self) -> int:
        """ 容纳当前 token 所需页数（向上取整）。 """
        n = self.num_tokens
        if n == 0:
            return 0
        return (n + self.block_size - 1) // self.block_size

    def num_full_blocks(self) -> int:
        """ 已填满的页数（尾块未满不计入，前缀只共享满块）。 """
        return self.num_tokens // self.block_size

    def block_token_ids(self, i: int) -> list[int]:
        """ 第 i 个逻辑页上的 token（用于算块哈希）。 """
        s = i * self.block_size
        return self.token_ids[s : s + self.block_size]

    def append_token(self, token_id: int) -> None:
        """ decode 追加一个新 token。 """
        self.token_ids.append(token_id)
