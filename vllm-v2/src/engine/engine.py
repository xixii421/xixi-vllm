from __future__ import annotations

from collections.abc import Iterator
from typing import Literal

import torch
from transformers import AutoTokenizer

from ..config import EngineConfig
from ..kv import KVManager, Sequence
from ..model import Qwen3ForCausalLM
from ..model_loader import load_model
from ..sampling_params import SamplingParams
from ..sampler import Sampler
from .model_runner import ModelRunner
from .scheduler import Scheduler


class Engine:
    def __init__(self, config: EngineConfig) -> None:
        self.config = config
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.dtype = self._resolve_dtype()

        self.tokenizer = AutoTokenizer.from_pretrained(
            config.model, trust_remote_code=True
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.eos_token_id = int(self.tokenizer.eos_token_id)
        self.context_len = config.context_len
        self.block_size = config.block_size

        self.model = Qwen3ForCausalLM(config.hf_config)
        load_model(self.model, config.model)
        self.model = self.model.to(dtype=self.dtype, device=self.device).eval()

        self.sampler = Sampler()
        self._init_kv()
        self.last_num_cached_tokens = 0

    def _init_kv(self) -> None:
        """ 初始化KV缓存，请求调度以及模型执行 """
        # 管理物理block内存，前缀缓存，对外给Scheduler提供block分配接口，给ModelRunner提供KV缓存张量
        self.kv = KVManager(
            self.model,
            num_blocks=self.config.num_kv_blocks,
            block_size=self.config.block_size,
            prefix_backend=self.config.prefix_backend,
            device=self.device,
            dtype=self.dtype,
        )
        # 请求调度器，请求排队，资源判断，状态更新
        self.scheduler = Scheduler(
            self.kv,
            max_num_seqs=self.config.max_num_seqs, # 最大请求数
            max_num_batched_tokens=self.config.max_num_batched_tokens, # 单轮token上限
            eos_token_id=self.eos_token_id, # 结束符，结合is_finished
            mix_prefill_decode=self.config.mix_prefill_decode, # 调度策略
        )
        # 模型执行器 接收Scheduler输出的序列列表向前传播并返回新的token_id
        self.model_runner = ModelRunner(
            self.model, self.kv, self.sampler, device=self.device
        )

    def reset_kv(
        self, prefix_backend: Literal["none", "hash", "radix"] | None = None
    ) -> None:
        """ 运行时重置整套 KV 缓存子系统 """
        if prefix_backend is not None:
            if prefix_backend not in ("none", "hash", "radix"):
                raise ValueError(f"invalid prefix_backend: {prefix_backend!r}")
            self.config.prefix_backend = prefix_backend # 热切换前缀缓存策略
        self._init_kv() # 初始化 KVManager、Scheduler、ModelRunner
        self.last_num_cached_tokens = 0 # 前缀缓存命中的token数量
        if self.device.type == "cuda":
            torch.cuda.empty_cache() # 释放闲置GPU显存给CUDA

    def _resolve_dtype(self) -> torch.dtype:
        if self.config.dtype != "auto":
            return getattr(torch, self.config.dtype)
        cfg = self.config.hf_config
        dt = getattr(cfg, "dtype", None) or getattr(cfg, "torch_dtype", None)
        return dt if isinstance(dt, torch.dtype) else torch.float16

    def add_request(
        self,
        prompt: str | list[int],
        sampling_params: SamplingParams | None = None,
    ) -> Sequence:
        """ 将 prompt 包装成 Sequence 对象，并进入调度器排队 """
        token_ids = (
            list(prompt) if isinstance(prompt, list) else self.tokenizer.encode(prompt)
        )
        if len(token_ids) > self.context_len:
            token_ids = token_ids[-self.context_len :]
        seq = Sequence.from_prompt(
            token_ids, block_size=self.block_size, sampling=sampling_params
        )
        self.scheduler.add(seq)
        return seq

    def step(self) -> tuple[list[tuple[int, list[int]]], int]:
        """ """
        prefills, decodes = self.scheduler.schedule()
        # prefills 需要执行prefill的序列
        # decodes 正在做迭代生成的序列

        if not prefills and not decodes:
            return [], 0

        # 来自 allocate() 的prefill阶段token命中的数量
        self.last_num_cached_tokens = self.kv.last_num_cached_tokens

        prefill_toks = sum(s.num_scheduled_tokens for s in prefills) # 本轮所有prefill跑的token数量
        metric = prefill_toks + len(decodes) # 本轮所有的token数量

        # 优先执行 decode，保证正在流式输出的请求持续向前推进；之后再执行 prefill 分片
        p_ids, d_ids = self.model_runner.run(prefills, decodes)

        finished: list = []
        # 模型跑完后处理: 追加生成的token_id, 分配新生成token的物理地址
        # 返回真正执行结束的seq列表
        if decodes:
            finished.extend(
                self.scheduler.postprocess(decodes, d_ids, is_prefill=False)
            )
        if prefills:
            finished.extend(
                self.scheduler.postprocess(prefills, p_ids, is_prefill=True)
            )
        # 正常情况下，prefill的seq是不会结束的，因为还要继续decode生成
        # 但是如果prompt过长超过限制token长度或者遇到eos,会提前结束

        outputs = [
            (seq.seq_id, seq.token_ids[seq.num_prompt_tokens :]) for seq in finished
        ] # 返回单请求的请求id，decode输出token
        return outputs, metric


    def generate_batch(
        self,
        prompts: list[str | list[int]],
        sampling_params: SamplingParams | list[SamplingParams] | None = None,
    ) -> list[str]:
        """ 多请求对外API """
        # 处理采样参数，每条prompt对应一套采样配置
        if sampling_params is None:
            # 采用默认参数
            sps = [SamplingParams() for _ in prompts]
        elif isinstance(sampling_params, (list, tuple)):
            # 多prompt，采样配置数量必须和prompt数量对齐
            sps = list(sampling_params)
            if len(sps) != len(prompts):
                raise ValueError("sampling_params length must match prompts")
        else:
            # 单prompt
            sps = [sampling_params for _ in prompts]

        seqs = [self.add_request(p, sp) for p, sp in zip(prompts, sps)]
        # seqs 保存 prompt 输入顺序
        finished_text: dict[int, str] = {}
        while not self.scheduler.is_finished():
            outputs, _ = self.step()
            # outputs的seq顺序是结束顺序
            for seq_id, completion_ids in outputs:
                finished_text[seq_id] = self.tokenizer.decode(completion_ids)
        return [finished_text[seq.seq_id] for seq in seqs]
        # 保证输出顺序和输入一致

    def generate(
        self,
        prompt: str | list[int],
        max_new_tokens: int = 128,
        temperature: float = 0.0,
        top_p: float | None = None,
        top_k: int | None = None,
        min_p: float | None = None,
    ) -> Iterator[str]:
        """ 单请求对外API """
        sp = SamplingParams(
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            min_p=min_p,
            max_tokens=max_new_tokens,
        )
        seq = self.add_request(prompt, sp)
        emitted = 0 # 输出 token 计数器
        while not seq.is_finished:
            self.step() # 模型调度 + GPU 推理 + postprocess
            while emitted < seq.num_completion_tokens:
                # 新token生成，但是还没有输出
                tid = seq.token_ids[seq.num_prompt_tokens + emitted]
                emitted += 1
                yield self.tokenizer.decode([tid]) # 输出
