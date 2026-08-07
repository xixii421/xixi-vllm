from __future__ import annotations
import torch
from torch import nn

class RMSNorm(nn.Module):
    def __init__(self,hidden_size:int,eps:float = 1e-6):
        # 1 hidden_size 隐藏维度，每个token的输出向量长度
        # 2 epx 极小常数，防止分母为0
        super().__init__()  #继承父类nn.Module 也可以用 nn.Module.__init__(self)

        self.weight = nn.Parameter(torch.ones(hidden_size))
        # 1 weight RMS的权重，可训练参数
        # 2 torch.ones() 构建一个一维张量 此时shape为[hidden_size] 意思是每个token对应大小为hidden_size的向量
        # 3 大模型常见的是三维张量，比如 x.shape = [2, 128, 2048]
        # 4 其中，2代表同时处理两条样本，128代表每条样本有128个token，2048代表每个token对应2048维向量
        # 5 nn.Parameter pytorch类，用来开启梯度 requires_grad=True，同时注册到module
        self.eps = eps
    def forward(self, x:torch.Tensor,residual:torch.Tensor|None = None) -> torch.Tensor | tuple[torch.Tensor,torch.Tensor]:
        # 1 tensor 中文名张量 数字类型，就是多维数字数组，专门用来深度学习
        # 2 x 输入特征张量, residual 残差张量
        # 3 residual的输入要么是张量要么是空，如果不输入默认为空
        # 4 函数输出类型为单个张量或者一个张量二元组
        orig_dtype = x.dtype # 张量存放的数据类型,这里保存下来方便后续做更高精度运算，运算完再转回
        x32 = x.float() # 把张量强制转换为float32类型，便于后续计算
        if residual is not None:
            x32 = x32 + residual.float() # 残差存在则转换后运算
            residual =  x32.to(orig_dtype) # 新的原类型残差，用于传给下一层Decoder

        # RMS归一化计算 out = (x/sqrt((mean(x^2)+eps)))*w LayerNorm 计算是先减均值再除方差
        rms = torch.rsqrt(x32.pow(2).mean(-1, keepdim=True) + self.eps)
        # 1 rsqrt 平方根倒数 pow(2) 张量每个元素平方，但是shape不变
        # 2 这里用rsqrt而不用sqrt是因为在GPU上rsqrt算子计算更快
        # 3 mean(-1,keepdim=True) 对元素平方做平均，-1代表倒数第一维 keepdim代表保留最后一维，操作完shape最后一维变为1
        x_normed = x32 * rms # 归一化
        out32 = x_normed * self.weight.float() # 权重缩放
        out = out32.to(orig_dtype) # 数值类型转回
        return (out,residual) if residual is not None else out

if __name__ == "__main__":
    norm = RMSNorm(64)
    x = torch.randn(1, 10, 64)
    # randn 生成服从标准正太分布的随机张量
    # 不带residual
    o1 = norm(x)
    print(o1.shape)

    # 带residual
    res = torch.randn_like(x)
    o2, res_new = norm(x, res)
    print(o2.shape, res_new.shape)