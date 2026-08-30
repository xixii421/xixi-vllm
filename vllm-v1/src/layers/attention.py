from __future__ import annotations
import torch
import torch.nn.functional as F
# 函数式接口，提供神经网络运算函数，不需要实例化即可调用
from torch import nn

from ..kv.context import get_forward_context


class Attention(nn.Module):
    def __init__(self, num_heads: int, head_dim: int, num_kv_heads: int, layer_id: int=0) -> None:
        """GQA 即分组查询注意力 多个Q公用一组KV"""
        # 1 num_heads Q的总头数
        # 2 head_dim 头维度
        # 3 num_kv_heads KV总头数
        # 4 layer_id 本层号，KV Cache 写/读定位物理页用
        super().__init__()
        if num_heads % num_kv_heads != 0:
            raise ValueError("num_heads mast be divided by num_kv_heads")
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.layer_id = layer_id
        self.scaling = head_dim ** -0.5
        # scaling 缩放因子，用来降低标准差，防止注意力过于集中

    def _sdpa(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        *,
        is_causal: bool,
        attn_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # 处理普通 MHA 和 GQA，处理因果掩码、外部传入注意力掩码的分支逻辑
        kwargs: dict = {"scale": self.scaling}
        if attn_mask is not None:
            kwargs["attn_mask"] = attn_mask
        else:
            kwargs["is_causal"] = is_causal
        if k.shape[1] != q.shape[1]:
            kwargs["enable_gqa"] = True
        return F.scaled_dot_product_attention(q, k, v, **kwargs)
        # 1 is_causal 因果掩码，推理时必须为True
        # 2 scale 缩放因子，不填默认为1/sqrt(head_dim)
        # 3 enable_gqa 分组查询注意力GQA支持，Q头数和KV头数不相等时开启
        # 4 SDPA具体做了什么：QK^T -> scale 缩放 -> 掩码覆盖 -> softmax计算权重 -> 权重乘V，得到每个头输出
        # 5 因为开启了GQA支持 所以内部会自动处理GQA的KV共享

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        ctx = get_forward_context()

        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        # 1 q.shape=[B,S,H,D],transpose用来交换中间两维 [B,H,S,D]
        # 2 SDPA要求H维度在前，内存连续，我们需要对每个头做矩阵乘法
        # 3 B batch_size 批次大小，一般指句子数量
        # 4 S seq_len 序列长度，一般指token数量
        # 5 H num_heads 头数量 D head_dim 单头维度


        if ctx is None:
        # 无 ForwardContext（如单独跑模型）：走无 KV Cache SDPA
            out = self._sdpa(q,k,v,is_causal=True)
            return out.transpose(1, 2).contiguous()
            # 1 换回维度，以便后续运算
            # 2 contiguous 保证内存连续 经过transpose后，内存存储是乱序的，如果非连续张量调用reshape会报错，因此需要内存连续
            # 3 具体做法是拷贝一份张量，在内存里重新排序，对齐逻辑顺序和内存顺序

        start = ctx.cache_len   # 本步开始前已有 KV 长度
        n = q.shape[2]          # 本步新增 token 数
        total = ctx.seq_len     # 本步写完后的总 KV 长度
        assert total == start + n

        # 1) write 把本步计算出来的新 K、V 按 slot_mapping 写入 KV 缓存存储池
        ctx.store.write(self.layer_id,ctx.slot_mapping,k.transpose(1,2),v.transpose(1,2))

        # 2) gather 按 block_table 读回逻辑长度 total 的全部历史 K V
        k_c,v_c = ctx.store.gather(self.layer_id,ctx.block_table,total)
        # k_c.shape=[seq_len,num_kv_heads,head_dim]
        k_c = k_c.unsqueeze(0).transpose(1, 2)
        v_c = v_c.unsqueeze(0).transpose(1, 2)
        # k_c.shape=[1,num_kv_heads,seq_len,head_dim]

        # 3) attend：本步 Q 与历史做注意力，掩码随 Prefill / Decode / 部分 Prefill 变化
        if start == 0 and n == total: # prefill 下三角
            out = self._sdpa(q,k_c,v_c,is_causal=True)
        elif n == 1: # Decode 只看历史，无掩码
            out = self._sdpa(q,k_c,v_c,is_causal=False)
        else: # 前缀命中后的部分 Prefill：Q 只覆盖新增区间，需手工下三角掩码
            # 此时本步Q只有后n个token，没有前start个token
            device,dtype = q.device,q.dtype
            q_pos = torch.arange(start,start+n,device=device)[:,None]
            # 本步 Q 逻辑位置 start~start+n-1 [:,None] 变成列向量 [n,1]
            kv_pos = torch.arange(total,device=device)[None,:]
            # 本步 KV 逻辑位置 0~total-1 [None,:] 变成行向量 [1,total]
            attn_mask = torch.zeros(n,total,device=device,dtype=dtype)
            # 初始化掩码矩阵， shape=[n,total]
            attn_mask.masked_fill_(kv_pos > q_pos,torch.finfo(dtype).min)
            # 掩码填充，条件 kv的位置 > q 的全局位置 → 这是未来token，要mask掉
            # 最终实现 每个 query 只能看见全局位置 ≤ 自己位置的 KV，屏蔽未来
            out = self._sdpa(q,k_c,v_c,is_causal=False,attn_mask=attn_mask)

        return out.transpose(1, 2).contiguous()

if __name__ == "__main__":
    attn = Attention(num_heads=8, head_dim=64, num_kv_heads=2)
    B, S = 1, 8
    q = torch.randn(B, S, 8, 64)
    k = torch.randn(B, S, 2, 64)
    v = torch.randn(B, S, 2, 64)
    out = attn.forward(q, k, v)
    print("out shape", out.shape)