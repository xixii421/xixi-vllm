from __future__ import annotations

import torch
from transformers import AutoTokenizer

from config import EngineConfig
from model import Qwen3ForCausalLM
from model_loader import load_model
from sampler import Sampler

class Engine:
    def __init__(self,config:EngineConfig) -> None:
        self.config = config
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.dtype = self._resolve_dtype()

        self.tokenizer = AutoTokenizer.from_pretrained(
            config.model,trust_remote_code=True
        )
        # 1 加载模型分词器 即 tokenizer.json tokenizer_config_json
        # 2 分词器负责将prompt转成token_id给模型，模型输出token_id转回文本
        # 3 trust_remote_code 允许加载模型自定义配置代码
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.eos_token_id = self.tokenizer.eos_token_id
        # 1 pad_token 填充token 做batch批量处理的时候，不同长度输入要补齐到相同长度，就用pad_token补充
        # 2 很多大模型原生没有自定义pad_token 就用eos_token代替

        self.model = Qwen3ForCausalLM(config.hf_config) # 加载超参文件
        load_model(self.model,config.model) # 加载权重文件
        self.model = self.model.to(dtype=self.dtype,device=self.device).eval()
        # 1 把模型所有权重张量转换成对应精度，同时移动到指定设备
        # 2 to()返回新的模型对象，模型所有权重张量复制，不修改
        # 3 eval() 模型设置为评估模式，关闭训练逻辑，推理必须调用eval
        if not config.enforce_eager and self.device.type == "cuda":
            self.model = torch.compile(self.model,mode="reduce-overhead")
        # 1 enforce_eager 强制使用原生eager执行，不编译
        # 2 compile JIT图编译工具，包装原模型
        # 3 reduce-overhead 降低python层算子调度开销，减少python-cuda来回切换的开销

        self.sampler = Sampler() # 采样器无训练参数；放在同一设备上即可

    def _resolve_dtype(self) -> torch.dtype:
        if self.config.dtype != "auto":
            return getattr(torch,self.config.dtype)
            # 1 getattr(obj,name[,default]) 获取对象obj上面名为name的属性
            # 2 如果不指定数据类型为auto，那么就用torch库里的dtype
        cfg = self.config.hf_config # 引用模型配置
        dt = getattr(cfg, "dtype", None) or getattr(cfg, "torch_dtype", None)
        # 读取模型配置里的dtype或者torch_dtype属性，如果不存在返回None
        return dt if isinstance(dt,torch.dtype) else torch.float16
        # isinstance 判断dt是不是合法dtype对象，如果是就用模型配置，如果不是 保底用float16

    @torch.inference_mode()
    # 推理模式，关闭梯度计算，不记录计算图
    def generate(
        self,
        prompt: str | list[int],
        max_new_tokens: int = 128,
        temperature: float = 0.0,
        top_p: float | None = None,
        top_k: int | None = None,
    ):
        ids = (
            list(prompt)
            if isinstance(prompt,list) # 如果输入已经是token_id列表
            else self.tokenizer.encode(prompt) # 否则则调用分词器编码
        ) # token_ids
        ctx = self.config.context_len # 最大上下文长度

        if len(ids) > ctx:
            ids = ids[-ctx:] # 取最后ctx个元素

        def pack(v,dtype): # 普通输入包装为1维张量
            return None if v is None else torch.tensor([v],device=self.device,dtype=dtype)

        temp_t = None if temperature<=0 else pack(temperature,torch.float)
        top_p_t = pack(top_p,torch.float)
        top_k_t = pack(top_k,torch.float)

        for _ in range(max_new_tokens):
            # ids: Python list 长度 N → 张量 [1, N]；positions 为 0..N-1
            input_ids = torch.tensor([ids], device=self.device,dtype=torch.long)
            positions = torch.arange(len(ids),device=self.device).unsqueeze(0)

            h = self.model(input_ids,positions)
            # 向前传播，输出最后一层隐藏状态h,shape=[1,seq_len,hidden_size]
            logits = self.model.compute_logits(h[:,-1:]).squeeze(1)
            # 1 h[:,-1:]，取序列最后一个token ，大模型生成下一个token，只使用最后一个位置的隐状态
            # 2 computer_logits 后 shape=[1,1,vocab_size] 再删掉中间维度 shape=[1,vocab_size]
            next_id = int(self.sampler(logits,temp_t,top_p_t,top_k_t).item())
            # item() 把0维标量张量数字取出，转换成python数字,之后转换成整数得到下一个token_id
            yield self.tokenizer.decode([next_id])
            # 1 解码，首先将int输入包装成列表，之后解码翻译成对应的字符串片段
            # 2 yield 生成的文本片段依次输出
            ids.append(next_id) # 更新token列表
            if next_id==self.eos_token_id or len(ids)>= ctx:
                break