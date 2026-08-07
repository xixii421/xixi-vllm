from __future__ import annotations
import torch
from torch import nn

def _rotate(x: torch.Tensor,cos:torch.Tensor,sin:torch.Tensor) -> torch.Tensor:
    # RoPE核心运算函数，用来对Q,K的最后一维执行复数旋转，给token注入位置信息
    x1,x2 = x.chunk(2,dim=-1)
    # chunk 2代表两等分，dim=-1代表最后一维，即最后一维两等分
    return torch.cat((x1 * cos - x2 * sin, x2 * cos + x1 * sin),dim=-1)
    # 1 x1 * cos - x2 * sin为旋转后的实部，x2 * cos + x1 * sin为旋转后的虚部
    # 2 cat,拼接，在最后一维，将旋转后的实部虚部拼接回去，确保输入x与返回的值shape一致

class RotaryEmbedding(nn.Module):
    def __init__(self,head_size:int,rotary_dim:int,max_position_embedding:int,base:float) -> None:
        # 1 head_size 多头注意力机制中每个头的向量长度，QKV操作对象
        # 2 rotary_dim 参与旋转的维度数量，必须为偶数，小于等于head_size，小于的时候只旋转靠前的维度，后面不带位置信息
        # 3 max_position_embedding cos或者sin查找表的大小，模型初始化时需要预计算并存入cache
        # 4 base RoPE的基底基数，一般固定为10000.0 base越大，位置区分变弱，base越小，对短位置更敏感
        super(RotaryEmbedding,self).__init__()
        if rotary_dim % 2 != 0:
            raise ValueError("rotary_dim must be even")
        self.head_size = head_size
        self.rotary_dim = rotary_dim

        inv_freq = 1.0 / (
            base ** (torch.arange(0, rotary_dim, 2,dtype=torch.float) / rotary_dim)
        )
        # 1 inv_freq,就是θ，步长计算公式
        # 2 arange 从0开始到rotary_dim结束，步长为2，数据类型为float RoPE两个维度为一组，只取偶数下标
        t = torch.arange(max_position_embedding,dtype=torch.float)
        # t 从 0 到 max_position_embedding-1 生成token的位置编号
        freqs = torch.outer(t,inv_freq)
        # outer 外积，此时freqs.shape = [max_position_embedding,rotary_dim//2] 存储的就是第pos个token，第i组维度的旋转角度θ
        self.register_buffer(
            'cos_sin_cache',
            torch.cat((freqs.cos(),freqs.sin()),dim=-1),
        persistent=False,
        )
        # 1 对每个元素求余弦和正弦，最后按最后一维拼接，前一半存cos，后一半存sin
        # 2 拼在一起方便一次索引拿到一个token的全部cos和sin
        # 3 register_buffer nn.Module内置方法，把一个张量注册成为模块的buffer，在用GPU运算时可以自动将cache搬到GPU
        # 4 persistent=False 代表cache不会放进state_dict,不会将cache放进权重文件

    def forward(self,positions:torch.Tensor,query:torch.Tensor,key:torch.Tensor) -> tuple[torch.Tensor,torch.Tensor]:
        flat = positions.reshape(-1)
        # 1 positions token的位置索引张量，一开始的position是二维的，positions.shape=[B,N],B代表batch，一次性可以处理的句子数，N代表句长，也就是token数
        # 2 reshape(-1) 将所有token展平成一维，下述index_select的索引参数要求是一维张量
        # 3 此时flat就是所有token的位置编号
        # 4 举个例子 positions = [[0,1,2],[0,1]]代表两个句子，每个句子各3，2个token
        # 5 flat 的结果即是 [0,1,2,0,1]
        # 6 根据索引选取cos sin值时，就是按照这个来，允许不同句子token的值重复，句子内的值不能重复，RoPE只需处理句子内的token位置关系
        cos_sin = self.cos_sin_cache.index_select(0,flat).to(dtype=query.dtype)
        # 1 index_select 从cache里在第0维按照flat做索引挑选，
        # 2 to cache生成时默认float32,但是大模型常用bfloat16，需要类型转换
        # 3 此时cos_sin.shape = [N,rotary_dim]，N代表token数，存储的每一行对应一个token的旋转系数
        cos,sin = cos_sin.chunk(2,dim=-1)
        cos = cos.unsqueeze(-2)
        sin = sin.unsqueeze(-2)
        # 1 unsqueeze,倒数第二的位置增加一个长度为1的新维度，为了广播匹配多头
        # 2 token 多头注意力机制中，每个token会分为多头，会增加一个head_size维度
        # 3 这里cos sin增加维度逻辑上等价把中间的1扩展成head_size，此时，计算时，同一个token下，所有头都用一套cos和sin
        # 4 x.shape=[seq_len,num_heads,rotary_dim] cos.shape=[N,1,rotary_dim//2]
        def _apply(x:torch.Tensor)-> torch.Tensor: # 对 x 应用旋转
            shape = x.shape
            # 1 记录原始shape,后面需要恢复
            # 2 因为QK的输入会把注意力头向量拼接在一起，而不是分开的两个维度，做RoPE需要拆分维度
            x = x.reshape(flat.numel(),-1,self.head_size)
            # 1 numel 计算全部的 token 数，即 seq_len
            # 2 将x.shape=[batch_size,seq_len,num_heads*head_size]变为[seq_len,num_heads,head_size]方便旋转
            # 3 reshape 只是解读方式不同，不改变原有结构
            x_rot = _rotate(x[..., : self.rotary_dim], cos, sin)
            # [...,:self.rotary_dim] 取前 rotary_dim 个向量进行旋转，之后恢复原 shape
            x_pass = x[..., self.rotary_dim:]  # 未旋转部分直接保留
            x_out = torch.cat([x_rot, x_pass], dim=-1)
            return x_out.reshape(shape)
        return _apply(query),_apply(key)

if __name__ == "__main__":
    head_size = 64
    rotary_dim = 32
    max_pos = 128
    base = 10000.0
    rope = RotaryEmbedding(head_size, rotary_dim, max_pos, base)

    # 1 QK shape=[batch_size, seq_len, num_heads, head_size]
    # 2 batch_size 批次大小，一般指句子数量
    # 3 seq_len 序列长度，一般指token数量
    # 4 num_heads 头数量  head_dim 单头维度
    q = torch.randn(1, 8, 4, head_size)
    k = torch.randn_like(q)
    pos = torch.arange(8).unsqueeze(0) # 位置id 加一维符合token位置格式

    q_rot, k_rot = rope.forward(pos,q,k)
    print("q shape", q_rot.shape)
    print("k shape", k_rot.shape)