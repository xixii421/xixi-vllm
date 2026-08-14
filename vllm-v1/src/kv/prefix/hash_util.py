""" 满块链式哈希 """

from __future__ import annotations

import hashlib
import struct


def compute_block_hash(token_ids: list[int], prefix: int = -1) -> int:
    """ 本块 tokens（可选缀上一段块哈希）→ 64bit 无符号哈希。 """
    # token_ids 一个block内部保存的token id列表
    # prefix 前一个块的哈希值，-1 代表没有前驱块
    h = hashlib.blake2b(digest_size=8)
    # blake2 是现代高速密码哈希算法 blake2b 内部规范小端，面向64位系统
    # digest_size=8 代表输出8字节，也就是64bit的哈希摘要
    if prefix != -1: # 不是第一个block，要把上一块的哈希混入
        h.update(prefix.to_bytes(8, "little", signed=False))
        # 把上一节的输入按照小端方式，无符号整数处理成二进制数
    h.update(struct.pack(f"<{len(token_ids)}i", *token_ids))
    # struct.pack python数据转换成二进制数 其中 < 代表小端序 i 代表int 四字节
    return int.from_bytes(h.digest(), "little", signed=False)
    # digest() 填充压缩，输出8字节的小端存储且无符号二进制串
    # int.from_bytes 把二进制字节还原成python整数
