from __future__ import annotations

import glob
# glob 按通配符匹配文件，用来查找权重文件
import os
from collections.abc import Iterable

import torch
from safetensors import safe_open
# 1 safetensors 主流大模型权重格式，替代老旧 .bin
# 2 safe_open 安全读取 safetensors 权重文件
# 3 对比 torch 原生 torch.load 不执行 pickle 反序列化，没有代码注入风险
# 4 pickle 在读文件时，会执行文件里面记录的 Python 代码，bin的恶意代码会执行
from torch import nn
from tqdm import tqdm
# tqdm 进度条工具，可视化进度条

def default_weight_loader(param:torch.Tensor,loaded_weight:torch.Tensor) -> None:
    param.data.copy_(loaded_weight)
    # 1 权重加载函数
    # 2 param 是模型层的nn.Parameter 它包装一层张量，内部有.data属性（张量，存放真实权重）
    # 3 不直接copy param 是因为 首先 copy_也是张量运算，因为param开启了梯度跟踪，这次拷贝操作会被记录进计算图
    # 4 但是加载权重不需要求梯度，不用记录这一步操作
    # 5 loaded_weight 从 safetensors/bin 文件读出来的权重张量

def safetensors_weights_iterator(hf_folder:str) -> Iterable[tuple[str, torch.Tensor]]:
    # 1 hf_folder 模型文件夹路径
    # 2 Iterable 输出，迭代对象，每迭代一项的到一个元组
    files = sorted(glob.glob(os.path.join(hf_folder, "*.safetensors")))
    # glob.glob 路径模式，返回匹配该模式的全部文件绝对路径列表，但是不保证顺序，所以需要sort排序
    if not files:
        raise FileNotFoundError(f"No safetensors found in {hf_folder}")
    for path in tqdm(files,desc="Loading weights"):
        with safe_open(path,framework="pt") as f:
        # 1 safe_open 流式打开上述safetensors文件
        # 2 framework 为 pt 指定框架为pytorch，内部返回张量就是torch.Tensor
        # 3 f 得到一个 SafeFile 对象，不会像load_file()那样一次性读取完整文件并全部放入内存，它是按需读取单张张量
            for name in f.keys():
                yield name,f.get_tensor(name)
def load_model(model:nn.Module,hf_folder:str) -> nn.Module:
    # 模块接口，加载模型
    model.load_weights(safetensors_weights_iterator(hf_folder))
    return model
