
## 1. RMSNorm
[RMSNorm实现代码](../../src/layers/layernorm.py)

在模型架构中，RMSNorm 的主要作用是对每个 Token 的 Hidden States 沿最后一个维度（通道维度）进行归一化。
其公式如下：

```math
\text{RMSNorm}(x) = \frac{x}{\sqrt{\frac{1}{d}\sum x_i^2 + \epsilon}} \cdot \gamma
```

其中 $\gamma$ 是可学习的 scale 参数（`self.weight`），$\epsilon$ 防止除零（默认 `1e-6`）。

> [!NOTE]
> **RMS和LN的区别？(博客里有写到)**
>
> LLaMA 和 Qwen 系列均采用了 RMSNorm，因为相比于标准的 LayerNorm，它去掉了计算均值和中心化（减去均值）的步骤。研究表明，均值中心化对模型效果影响微乎其微，去掉后可以减少计算量，提升网络的前向推理速度。


在大模型推理中，像加法和归一化这类操作属于内存密集型，真正的瓶颈不在于 GPU 的算力，而在于显存的读写带宽。
为了减少显存的读写次数，vllm-v0 采用了一种极其经典的工程优化：将残差连接与 RMSNorm 融合在一个函数中处理。

vllm-v0 的 RMSNorm 集成了 **residual connection**：

```python
def forward(self, x, residual=None):
    # 如果传入了残差分支
    if residual is not None:
        x = x + residual       # 1. 融合操作：计算 当前分支输出 + 主干残差
        residual = x           # 2. 更新主干：将相加后的结果作为新的残差主干保留
    
    # ... 进行标准的 RMS 归一化 ...
    
    # 同时返回：(归一化后的送入下一层的特征, 更新后的主干残差)
    return (x, residual) if residual is not None else x
```

利用这种 Fused 设计，Transformer 层的代码变得极其简洁。
不过需要注意的是，其中`residual`才是从模型输入以来，所有历史层输出结构的累加总和。
`hidden_states`只代表上一个子层刚计算完的原始输出结果。

```

1. 融合归一化
      norned_h, resiudal = self.rms(hidden_states, residual)
2. Attention计算
      attn_out = self.attn(normed_h)
3. 融合归一化
      normed_h, residual = self.post_rms(attn_out, residual)
      这里的residual是第一步更新后的值，计算完后，再进一步更新residual
4. MLP计算
      mlp_out = self.mlp(normed_h)
5. return mlp_out, residual
```

这种写法并没有显式的加号，而是不断更新residual变量，把累加和传递下去，这样节省了GPU显存中来回搬运中间加法结果的时间，速度更快。



