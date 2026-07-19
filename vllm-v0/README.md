# vllm-v0

最小可运行的 Qwen3 自回归推理：单请求、全序列重算、SDPA Attention、可选采样策略。

## 上手

```bash
cd vllm-v0
# 列出缓存与当前显存能装下的模型
python run.py --list

# 下载注册表中的公开权重（默认缓存见 src/model_hub.py）
# 一定要在src/model_hub.py中修改权重存储路径！！！！！
python run.py --download

# 推理
python run.py --model Qwen3-0.6B --prompt "Hello, my name is"
```

## 目录结构

```
vllm-v0/
├── README.md                 ← 本页
├── run.py                    命令行入口
├── docs/                     讲义（按下方顺序读）
└── src/
    ├── config.py             引擎配置
    ├── engine.py             加载模型 + generate 循环
    ├── model.py              Qwen3 骨架与权重装载
    ├── model_loader.py       safetensors 流式读取
    ├── model_hub.py          下载 / 校验辅助
    ├── sampler.py            logits → token id
    └── layers/
        ├── layernorm.py      RMSNorm（可融合残差）
        ├── rotary.py         RoPE
        ├── attention.py      GQA + SDPA
        └── activation.py     SiluAndMul（SwiGLU）
```

## 推荐阅读顺序

讲义在 `docs/`，源码路径写在各篇开头。建议按这条路径读，和代码依赖方向一致：

| 顺序 | 文档 | 对应源码 | 这一步弄清什么 |
| :---: | :--- | :--- | :--- |
| 1 | [architecture.md](./docs/architecture.md) | 全局 | 为什么需要推理编排；本目录怎么自回归循环 |
| 2 | [layers/rmsnorm.md](./docs/layers/rmsnorm.md) | `src/layers/layernorm.py` | Pre-Norm 与 fused residual |
| 3 | [layers/rotary.md](./docs/layers/rotary.md) | `src/layers/rotary.py` | 位置如何旋进 Q/K |
| 4 | [layers/attention.md](./docs/layers/attention.md) | `src/layers/attention.py` | GQA + 因果 SDPA |
| 5 | [layers/activation.md](./docs/layers/activation.md) | `src/layers/activation.py` | SwiGLU 的 SiLU× 门控 |
| 6 | [layers.md](./docs/layers.md) | `src/layers/*` + 模型侧装配 | Attention / MLP / Decoder 数据流总览 |
| 7 | [model.md](./docs/model.md) | `src/model.py`、`model_loader.py` | 层如何堆成 Qwen3；权重如何灌进骨架 |
| 8 | [sampler.md](./docs/sampler.md) | `src/sampler.py` | temperature / top-k / top-p / min-p |
| 9 | [engine.md](./docs/engine.md) | `src/engine.py`、`run.py` | 编排：encode → forward → sample → yield |



