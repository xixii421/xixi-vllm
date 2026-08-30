from __future__ import annotations

from typing import Literal

import torch
from transformers import AutoTokenizer

from .config import EngineConfig
from .kv import ForwardContext, KVManager, Sequence, set_forward_context
from .model import Qwen3ForCausalLM
from .model_loader import load_model
from .pagedattention import slots_tensor
from .sampler import Sampler


class Engine:
    def __init__(self, config:EngineConfig) -> None:
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
        self.context_len = self.config.context_len
        self.model = Qwen3ForCausalLM(config.hf_config) # 加载超参文件
        load_model(self.model,config.model) # 加载权重文件
        self.model = self.model.to(dtype=self.dtype,device=self.device).eval()
        # 1 把模型所有权重张量转换成对应精度，同时移动到指定设备
        # 2 to()返回新的模型对象，模型所有权重张量复制，不修改
        # 3 eval() 模型设置为评估模式，关闭训练逻辑，推理必须调用eval
        self.sampler = Sampler() # 采样器无训练参数；放在同一设备上即可
        self._init_kv()
        self.last_num_cached_tokens=0

    def _init_kv(self) -> None:
        """按当前 config 重建页池与前缀索引（不重载模型权重）"""
        self.kv = KVManager(
            self.model,
            num_blocks=self.config.num_kv_blocks,
            block_size=self.config.block_size,
            prefix_backend=self.config.prefix_backend,
            device=self.device,
            dtype=self.dtype,
        )
    def reset_kv(
        self, prefix_backend: Literal["none","hash","radix"] | None = None,
    ) -> None:
        """清空 KV / 前缀状态；可选切换 prefix_backend。供测试复用同一模型 """
        if prefix_backend is not None:
            if prefix_backend not in ("none","hash","radix"):
                raise ValueError(f"invalid prefix_backend: {prefix_backend!r}")
            self.config.prefix_backend = prefix_backend
        old = getattr(self,"kv",None)
        self._init_kv()
        self.last_num_cached_tokens = 0
        if old is not None:
            del old
            if self.device.type == "cuda":
                torch.cuda.empty_cache()

    def _bind_ctx(
        self,
        seq: Sequence,
        *,
        is_prefill: bool,
        cache_len: int,
        write_end: int,
    ) -> None:
        """安装 ForwardContext：暴露本步要写入的 slot_mapping 与页表给 Attention """
        slots = slots_tensor(
            seq.block_table,
            cache_len,
            write_end,
            self.kv.block_size,
            device=self.device,
        )
        set_forward_context(
            ForwardContext(
                is_prefill=is_prefill,
                cache_len=cache_len,
                seq_len=write_end,
                slot_mapping=slots,
                block_table=seq.block_table,
                store=self.kv.store,
            )
        )

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
        input_ids = (
            list(prompt)
            if isinstance(prompt,list) # 如果输入已经是token_id列表
            else self.tokenizer.encode(prompt) # 否则则调用分词器编码
        ) # token_ids
        if len(input_ids) > self.context_len:
            input_ids = input_ids[-self.context_len:] # 取最后ctx个元素


        def pack(v,dtype): # 普通输入包装为1维张量
            return None if v is None else torch.tensor([v],device=self.device,dtype=dtype)

        temp_t = None if temperature<=0 else pack(temperature,torch.float)
        top_p_t = pack(top_p,torch.float)
        top_k_t = pack(top_k,torch.int)

        seq: Sequence = self.kv.new_sequence(input_ids) # 前缀匹配
        self.kv.allocate(seq) # 分配block资源
        self.last_num_cached_tokens = seq.num_cached_tokens
        cached = seq.num_cached_tokens # 前缀匹配之后已经有kv缓存的token数量
        prompt_len = len(input_ids) # 完整token长度

        try: # prefill逻辑
            if cached < prompt_len: # 部分/全部未命中缓存
                self._bind_ctx(
                    seq,is_prefill=True,cache_len=cached,write_end=prompt_len,
                )
                new_ids = input_ids[cached:] # 取未被缓存的token

                ids = torch.tensor([new_ids], device=self.device,dtype=torch.long)
                positions = torch.arange(cached,prompt_len,device=self.device).unsqueeze(0)
                # token和positions都跳过缓存的token
                h = self.model(ids,positions) #  输出hidden_state
                # 向前传播，输出最后一层隐藏状态h,shape=[1,seq_len,hidden_size]
                logits = self.model.compute_logits(h[:,-1:]).squeeze(1)
                # 1 h[:,-1:]，取序列最后一个token ，大模型生成下一个token，只使用最后一个位置的隐状态
                # 2 computer_logits 后 shape=[1,1,vocab_size] 再删掉中间维度 shape=[1,vocab_size]

            else: # 整个prompt完全命中前缀缓存
                self._bind_ctx(
                    seq, is_prefill=False, cache_len=prompt_len - 1, write_end=prompt_len
                )
                ids = torch.tensor([[input_ids[-1]]], device=self.device, dtype=torch.long)
                positions = torch.tensor(
                    [[prompt_len - 1]], device=self.device, dtype=torch.long
                )
                # 需要最后一个token的输出hidden_state来采样第一个输出token
                h = self.model(ids, positions)
                logits = self.model.compute_logits(h[:, -1:]).squeeze(1)

            # prefill结束
            next_id = int(self.sampler(logits,temp_t,top_p_t,top_k_t).item())
            # item() 把0维标量张量数字取出，转换成python数字,之后转换成整数得到下一个token_id
            yield self.tokenizer.decode([next_id])
            # 1 解码，首先将int输入包装成列表，之后解码翻译成对应的字符串片段
            # 2 yield 生成的文本片段依次输出
            if next_id == self.eos_token_id or prompt_len >= self.context_len:
                return
            seq.append_token(next_id)
            self.kv.ensure_blocks(seq) # 检查物理表，需要新的就分配
            cached_len = prompt_len

            # 开启decode循环
            for _ in range(max_new_tokens - 1): # 已经产出了一个token，需要循环生成最多max_new_token-1个
                if cached_len >= self.context_len:
                    break # 上下文上限保护
                self._bind_ctx(
                    seq,is_prefill=False,cache_len=cached_len,write_end=cached_len+1,
                )
                ids = torch.tensor([[next_id]], device=self.device, dtype=torch.long)
                positions = torch.tensor([[cached_len]],device=self.device, dtype=torch.long)
                h = self.model(ids,positions)
                logits = self.model.compute_logits(h[:,-1:]).squeeze(1)
                next_id = int(self.sampler(logits,temp_t,top_p_t,top_k_t).item())
                cached_len += 1
                seq.append_token(next_id)
                self.kv.ensure_blocks(seq)

                yield self.tokenizer.decode([next_id])
                if next_id == self.eos_token_id:
                    break

        finally:
            set_forward_context(None)
            self.kv.finish(seq)