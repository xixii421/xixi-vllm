""" 单请求的逻辑状态：token 序列 + 页表 """

from __future__ import annotations

from dataclasses import dataclass,field


@dataclass
class Sequence:
    """ 一条生成请求的 tokens 与物理页表。 """

    token_ids: list[int] # 保存这条请求全部逻辑token，包括输入prompt加decode已经生成出来的token
    block_size: int # 每个物理block可以存放的token数
    seq_id: int = 0 # 每个请求的唯一编号，radix 引用用
    block_table: list[int] = field(default_factory=list) # 逻辑块索引到物理块索引的映射 PagedAttention页表
    # field() 对 dataclass 的字段做精细配置
    # 这里 default_factory=list 每创建一个Sequence实例，就生成全新空列表，防止所有实例共享列表对象
    num_cached_tokens: int = 0 # 前缀命中复用的token数，只需要prefill之后的token

    @property # 把方法伪装成属性，调用时不用写括号，但是内部执行函数逻辑
    def num_tokens(self) -> int:
        """ 当前序列长度 """
        return len(self.token_ids)

    @property
    def num_blocks(self) -> int:
        """ 容纳当前 token 所需页数（向上取整） """
        n = self.num_tokens
        if n == 0:
            return 0
        return (n + self.block_size - 1) // self.block_size

    def num_full_blocks(self) -> int:
        """ 当前请求可以填满的页数（尾块未满不计入，前缀只共享满块） """
        return self.num_tokens // self.block_size

    def block_token_ids(self, i: int) -> list[int]:
        """ 取出第 i 个逻辑页上的 token_id 片段（用于算块哈希） """
        s = i * self.block_size
        return self.token_ids[s : s + self.block_size]

    def append_token(self, token_id: int) -> None:
        """ decode 追加一个新 token """
        self.token_ids.append(token_id)