from __future__ import annotations
import torch
import torch.nn.functional as F
# 函数式接口，提供神经网络运算函数，不需要实例化即可调用
from torch import nn

class Attention(nn.Module):
    def __init__(self,num_heads:int,head_dim:int,num_kv_heads:int) -> None:
        """GQA 即分组查询注意力 多个Q公用一组KV"""
        # 1 num_heads Q的总头数
        # 2 head_dim 头维度
        # 3 num_kv_heads KV总头数
        super().__init__()
        if num_heads % num_kv_heads != 0:
            raise ValueError("num_heads mast be divided by num_kv_heads")
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.scaling = head_dim ** -0.5
        # scaling 缩放因子，用来降低标准差，防止注意力过于集中
    def forward(self, q:torch.Tensor,k:torch.Tensor,v:torch.Tensor)->torch.Tensor:
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        # 1 q.shape=[B,S,H,D],transpose用来交换中间两维 [B,H,S,D]
        # 2 SDPA要求H维度在前，内存连续，我们需要对每个头做矩阵乘法
        # 3 B batch_size 批次大小，一般指句子数量
        # 4 S seq_len 序列长度，一般指token数量
        # 5 H num_heads 头数量 D head_dim 单头维度

        out = F.scaled_dot_product_attention(q,k,v,is_causal=True,scale=self.scaling,enable_gqa=self.num_kv_heads!=self.num_heads)
        # 1 is_causal 因果掩码，推理时必须为True
        # 2 scale 缩放因子，不填默认为1/sqrt(head_dim)
        # 3 enable_gqa 分组查询注意力GQA支持，Q头数和KV头数不相等时开启
        # 4 SDPA具体做了什么：QK^T -> scale 缩放 -> 掩码覆盖 -> softmax计算权重 -> 权重乘V，得到每个头输出
        # 5 因为开启了GQA支持 所以内部会自动处理GQA的KV共享
        return out.transpose(1, 2).contiguous()
        # 1 换回维度，以便后续运算
        # 2 contiguous 保证内存连续 经过transpose后，内存存储是乱序的，如果非连续张量调用reshape会报错，因此需要内存连续
        # 3 具体做法是拷贝一份张量，在内存里重新排序，对齐逻辑顺序和内存顺序

if __name__ == "__main__":
    attn = Attention(num_heads=8, head_dim=64, num_kv_heads=2)
    B, S = 1, 8
    q = torch.randn(B, S, 8, 64)
    k = torch.randn(B, S, 2, 64)
    v = torch.randn(B, S, 2, 64)
    out = attn.forward(q, k, v)
    print("out shape", out.shape)