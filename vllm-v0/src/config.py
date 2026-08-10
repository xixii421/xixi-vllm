from __future__ import annotations
import os
from dataclasses import dataclass,field
# dataclasses Python数据类，自动生成_init_等样板代码
from transformers import AutoConfig,PretrainedConfig
# AutoConfig HuggingFace工具，自动加载模型config.json
# PretrainedConfig 所有HF模型配置的基类

@dataclass
class EngineConfig:
    model: str # 模型路径，定位加载对应模型
    context_len: int = 2048 # 最大上下文窗口
    enforce_eager: bool = True # 执行eager,不适用CUDA Graph
    dtype: str = "auto"
    hf_config: PretrainedConfig = field(init=False)

    def __post_init__(self) -> None:
        self.model = os.path.expanduser(self.model)
        # expanduser 路径中~替换为真实系统路径
        self.hf_config = AutoConfig.from_pretrained(self.model,trust_remote_code=True)
        # 加载模型 config.json 返回 PretrainedConfig 对象
        # trust_remote_code 允许加载模型自定义配置代码
