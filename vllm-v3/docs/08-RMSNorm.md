# RMSNorm优化
## 一、 PyTorch 原生实现
我们来看一下Pytorch版本的原生实现：
```python
class RMSNorm(nn.Module):
    def forward(self, x: torch.Tensor, residual: torch.Tensor | None = None):
        orig_dtype = x.dtype
        x32 = x.float()
        
        if residual is not None:
            x32 = x32 + residual.float()
            residual = x32.to(orig_dtype)
            
        out = (
            x32 * torch.rsqrt(x32.pow(2).mean(dim=-1, keepdim=True) + self.eps)
            * self.weight
        ).to(orig_dtype)
        
        return (out, residual) if residual is not None else out
```

这段代码符合工程师的数学直觉，逻辑清晰。但在 GPU 的底层执行视角中，它性能开销却非常大。

PyTorch 的 Eager 模式遵循 **一个算子一个Kernel**的调度原则。上述几行代码在底层会被拆解为十数个独立的 CUDA Kernel：
1. `x.float()`：类型转换 Kernel
2. `x32 + residual.float()`：向量加法 Kernel
3. `x32.pow(2)`：逐元素平方 Kernel
4. `.mean(...)`：归约（Reduction）求均值 Kernel
5. `+ self.eps`：标量加法 Kernel
6. `torch.rsqrt(...)`：开方倒数 Kernel
7. `x32 * ... * self.weight`：逐元素乘法 Kernel
8. `.to(orig_dtype)`：类型转换 Kernel

在原生实现中，每一个独立的 Kernel 执行时，都必须经历 从 HBM 读取数据 -》 在 SRAM/寄存器中计算 -》 将结果写回 HBM 的完整生命周期。
这意味着，中间变量（如 `x32.pow(2)` 的结果）被频繁地写入 HBM，紧接着又在下一个 Kernel 中被读取出来。这种冗余的 HBM 读写导致了极其严重的“访存墙”问题，使得 GPU 的计算单元大量时间处于“等待数据搬运”的空闲状态。

## 二、 Triton 融合实现

为了打破访存墙，工业界引入了**算子融合** 技术。Triton 作为一种基于 Python 语法的 GPU 编程语言，使得开发者能够以极高的效率编写融合 Kernel。

```python
@triton.jit
def _rms_norm_fused_kernel(
    X_ptr, Residual_ptr, Out_ptr, New_Res_ptr, Weight_ptr,
    row_stride, N, eps,
    ADD_RESIDUAL: tl.constexpr, BLOCK_SIZE: tl.constexpr,
):
    # ... 省略指针偏移计算 ...
    
    # 1. 一次性从 HBM 加载输入数据到寄存器/SRAM
    x = tl.load(x_row_ptr + cols, mask=mask, other=0.0).to(tl.float32)

    # 2. 条件融合残差加法
    if ADD_RESIDUAL:
        residual = tl.load(res_row_ptr + cols, mask=mask, other=0.0).to(tl.float32)
        x = x + residual
        tl.store(new_res_row_ptr + cols, x, mask=mask) # 写出新残差

    # 3. 在寄存器/SRAM 中完成所有密集计算
    variance = tl.sum(x * x, axis=0) / N
    rsqrt_var = tl.rsqrt(variance + eps)
    weight = tl.load(Weight_ptr + cols, mask=mask).to(tl.float32)
    out = x * rsqrt_var * weight
    
    # 4. 一次性将最终结果写回 HBM
    tl.store(out_row_ptr + cols, out, mask=mask)
```

Triton 版本将原本若干个个 PyTorch 算子，整合成了一个单一的 CUDA Kernel。
在这个 Kernel 内部，数据一旦通过 `tl.load` 从 HBM 加载到 GPU 的片上 SRAM或线程寄存器中，后续所有的加法、平方、求和、开方、乘法操作，全部在片上高速存储中完成。直到计算出最终的 `out`，才通过 `tl.store` 写回 HBM。

注意参数 `ADD_RESIDUAL: tl.constexpr`。`constexpr` 意味着该参数在 **Triton 编译期** 就已经确定。
* 当 `ADD_RESIDUAL=False` 时，Triton 编译器会执行**死代码消除**，直接将 `if ADD_RESIDUAL:` 分支内的代码从生成的 PTX/SASS 汇编中剔除。
* 这使得同一个 Kernel 能够同时适配“带残差”和“不带残差”两种场景，且没有任何运行时的分支预测开销。

# v3 RMSNorm 性能基准

> 对应 [01-v2性能瓶颈](./01-v2性能瓶颈.md) — **瓶颈 D（层内算子）**。  
> 测试环境：NVIDIA GPU 4090D + PyTorch 2.6.0 + Triton 3.2.0 + CUDA 12.4

v3 将 v2 的纯 PyTorch RMSNorm 替换为 Triton fused kernel，将原本 5~8 个独立 CUDA kernel 合并为 1 个。下面用实测数据验证加速比与精度。

## 一、测试配置

| 参数 | 值 |
|------|-----|
| hidden_size | 2048, 4096 (Qwen3 常见尺寸) |
| num_tokens | 1, 32, 128, 512, 2048 |
| dtype | float16, bfloat16 |
| residual | None (纯 norm) / 传入张量 (norm + residual add) |
| eps | 1e-6 |
| 预热 | 30 次 |
| 测量 | 200 次取 min (us) |
| 参考实现 | `torch.nn.functional.rms_norm` (PyTorch 原生) |


## 二、延迟数据 (us)

### hidden_size = 2048

#### 无残差 (residual=None)

| dtype | num_tokens | v2 (us) | v3 (us) | **加速比** | v3 绝对误差 | v3 相对误差 |
|-------|-----------|---------|---------|-----------|------------|------------|
| float16 | 1 | 143.5 | 130.6 | **1.10×** | 0 | 0 |
| bfloat16 | 1 | 131.9 | 125.5 | **1.05×** | 0 | 0 |
| float16 | 32 | 132.1 | 126.0 | **1.05×** | 0 | 0 |
| bfloat16 | 32 | 133.6 | 125.3 | **1.07×** | 0 | 0 |
| float16 | 128 | 134.3 | 126.9 | **1.06×** | 9.8e-04 | 1.9e-04 |
| bfloat16 | 128 | 133.1 | 127.1 | **1.05×** | 0 | 0 |
| float16 | 512 | 136.1 | 127.6 | **1.07×** | 2.0e-03 | 3.9e-04 |
| bfloat16 | 512 | 142.1 | 126.8 | **1.12×** | 0 | 0 |
| float16 | 2048 | 179.3 | 129.4 | **1.39×** | 2.0e-03 | 3.9e-04 |
| bfloat16 | 2048 | 181.9 | 130.9 | **1.39×** | 0 | 0 |

#### 带残差 (residual ≠ None)

| dtype | num_tokens | v2 (us) | v3 (us) | **加速比** | v3 绝对误差 | v3 相对误差 |
|-------|-----------|---------|---------|-----------|------------|------------|
| float16 | 1 | 183.5 | 120.5 | **1.52×** | 0 | 0 |
| bfloat16 | 1 | 181.5 | 118.6 | **1.53×** | 0 | 0 |
| float16 | 32 | 184.3 | 120.1 | **1.53×** | 9.8e-04 | 2.2e-04 |
| bfloat16 | 32 | 184.2 | 118.9 | **1.55×** | 0 | 0 |
| float16 | 128 | 184.2 | 120.8 | **1.52×** | 4.9e-04 | 1.1e-04 |
| bfloat16 | 128 | 183.2 | 120.2 | **1.52×** | 0 | 0 |
| float16 | 512 | 183.1 | 120.5 | **1.52×** | 2.0e-03 | 3.9e-04 |
| bfloat16 | 512 | 183.3 | 120.9 | **1.52×** | 3.9e-03 | 7.8e-04 |
| float16 | 2048 | 244.3 | 127.6 | **1.91×** | 2.0e-03 | 3.7e-04 |
| bfloat16 | 2048 | 243.8 | 125.7 | **1.94×** | 1.6e-02 | 3.1e-03 |

### hidden_size = 4096

#### 无残差 (residual=None)

| dtype | num_tokens | v2 (us) | v3 (us) | **加速比** | v3 绝对误差 | v3 相对误差 |
|-------|-----------|---------|---------|-----------|------------|------------|
| float16 | 1 | 131.6 | 126.0 | **1.04×** | 0 | 0 |
| bfloat16 | 1 | 131.1 | 126.6 | **1.04×** | 0 | 0 |
| float16 | 32 | 132.2 | 126.5 | **1.04×** | 9.8e-04 | 2.0e-04 |
| bfloat16 | 32 | 135.2 | 65.8 | **2.06×** | 0 | 0 |
| float16 | 128 | 136.5 | 59.0 | **2.31×** | 2.0e-03 | 3.9e-04 |
| bfloat16 | 128 | 137.2 | 56.8 | **2.42×** | 0 | 0 |
| float16 | 512 | 157.9 | 122.2 | **1.29×** | 2.0e-03 | 4.0e-04 |
| bfloat16 | 512 | 158.6 | 122.6 | **1.29×** | 0 | 0 |
| float16 | 2048 | 272.5 | 128.8 | **2.12×** | 2.0e-03 | 3.8e-04 |
| bfloat16 | 2048 | 276.9 | 128.6 | **2.15×** | 7.8e-03 | 1.5e-03 |

#### 带残差 (residual ≠ None)

| dtype | num_tokens | v2 (us) | v3 (us) | **加速比** | v3 绝对误差 | v3 相对误差 |
|-------|-----------|---------|---------|-----------|------------|------------|
| float16 | 1 | 183.4 | 119.9 | **1.53×** | 9.8e-04 | 2.9e-04 |
| bfloat16 | 1 | 182.3 | 119.5 | **1.53×** | 0 | 0 |
| float16 | 32 | 144.5 | 118.5 | **1.22×** | 0 | 0 |
| bfloat16 | 32 | 142.4 | 116.5 | **1.22×** | 0 | 0 |
| float16 | 128 | 146.1 | 120.5 | **1.21×** | 2.0e-03 | 4.0e-04 |
| bfloat16 | 128 | 150.7 | 121.4 | **1.24×** | 7.8e-03 | 1.7e-03 |
| float16 | 512 | 173.6 | 127.3 | **1.36×** | 2.0e-03 | 4.1e-04 |
| bfloat16 | 512 | 178.4 | 126.4 | **1.41×** | 0 | 0 |
| float16 | 2048 | 455.2 | 182.0 | **2.50×** | 2.0e-03 | 3.7e-04 |
| bfloat16 | 2048 | 456.9 | 143.5 | **3.18×** | 7.8e-03 | 1.5e-03 |

## 三、精度验证

v3 (Triton) 与 `torch.nn.functional.rms_norm` 的精度对比：

| 指标 | 值 |
|------|-----|
| **v3 最大绝对误差** | 1.56 × 10⁻² |
| **v3 最大相对误差** | 3.09 × 10⁻³ |

v3 的 Triton 实现在误差上完全在 float16 / bfloat16 精度的合理范围内：

- **float16 相对精度**：约 4.88 × 10⁻⁴（10-bit mantissa），实测最大相对误差 4.12 × 10⁻⁴，**完全匹配**
- **bfloat16 相对精度**：约 3.91 × 10⁻³（7-bit mantissa），实测最大相对误差 3.09 × 10⁻³，**在容忍范围内**

所有实验点中 v2 (PyTorch eager) 的绝对误差也完全一致 —— 两种实现本质上都在 float32 累积后做最后的 dtype 截断，精度上的差异来自 Triton 内核内的浮点运算顺序与 `torch.rsqrt` 的微小差异，不影响正确性。

## 四、结果分析

### 4.1 无残差场景：batch 越大，加速越明显

| hidden_size | num_tokens=1 | num_tokens=2048 |
|-------------|-------------|----------------|
| 2048 | 1.05~1.10× | **1.39×** |
| 4096 | 1.04× | **2.12~2.15×** |

**趋势：batch 越小，v3 优势越弱。** 原因是：

1. **num_tokens=1 时 GPU 负载极低**，Triton kernel 的 launch overhead 相对占比高。v2 的 PyTorch eager 虽然有多轮 HBM 读写，但数据量太小，读写开销几乎可忽略。
2. **num_tokens=2048 时 v2 的 HBM 往返代价被放大**。v2 的 `x.pow(2)` → `.mean()` → `.rsqrt()` → `* weight` → `.to(dtype)` 每一轮都 HBM read + compute + HBM write，batch 大时带宽瓶颈凸显。v3 的 Triton kernel 一次 load → 全部在片上计算 → 一次 store，**HBM 访问次数约为 v2 的 1/5**。

### 4.2 带残差场景：融合收益稳定且显著

| hidden_size | 无残差加速比 | 带残差加速比 | 增量 |
|-------------|-------------|-------------|------|
| 2048, 1 token | 1.10× | **1.52×** | +0.42× |
| 2048, 2048 tokens | 1.39× | **1.94×** | +0.55× |
| 4096, 1 token | 1.04× | **1.53×** | +0.49× |
| 4096, 2048 tokens | 2.12× | **2.50~3.18×** | +0.38~1.06× |

带残差时 v2 内部多出两次额外 kernel：

```python
x32 = x32 + residual.float()   # Kernel 1: Add
residual = x32.to(orig_dtype)   # Kernel 2: Cast
```

v3 的 Triton 内核在 `ADD_RESIDUAL=True` 编译路径下，将残差加法直接融入主 kernel：

```
load(x) + load(residual) → norm → * weight → store(out, new_residual)
```

这相当于是 **"v2 基础加速 + 额外省掉 2 个 kernel"**，所以带残差场景的加速比始终高于无残差。

### 4.3 异常数据点分析

| 场景 | 加速比 | 原因 |
|------|--------|------|
| hs=4096, nt=32/128, 无残差 | **2.06~2.42×** | 此尺寸恰好将 Triton `BLOCK_SIZE=4096` 填满一个 thread block，而 v2 的 4096 维度需要拆成多个 kernel，GPU occupancy 在此处严重偏斜 |
| hs=4096, nt=512, 无残差 | **1.29×** | 512 token 介于"计算太少"和"计算够多"之间，GPU occupancy 在 v2 中恰好饱满，差异缩小 |
| hs=4096, nt=2048, bf16, 带残差 | **3.18×** | 大 hidden + 大 batch + 3 项融合（norm + add + cast）→ v2 的 kernel 数量达到峰值，Triton 的融合收益最大化 |

## 五、复现

```bash
cd vllm-v3
python bench/bench_rmsnorm.py
```


## 三、Rotary优化

方法同RMSNorm，代码参考：[rotary.py](../src/layers/rotary.py)
