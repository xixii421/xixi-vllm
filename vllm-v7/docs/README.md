# dzyy-vllm 文档

教学向架构说明（代码目录仍为 `vllm-v7/`，对外名称 **dzyy-vllm**）。  
**先读总览，再分调度流 / 计算流。**

## 架构

| 文档 | 内容 |
|------|------|
| [architecture/00-概述.md](architecture/00-概述.md) | **整体架构总览**：三框架对比、vLLM 本质、两面一桥、模块地图、讲解顺序 |
| [architecture/01-scheduling-flow.md](architecture/01-scheduling-flow.md) | 调度流完整图解（谁、何时、占哪页 KV） |
| [architecture/02-compute-flow.md](architecture/02-compute-flow.md) | 计算流完整图解（GPU 上一步如何出 token） |

## 建议阅读顺序

1. `00-概述` — nano / dzyy / 官方对比 + 控制/计算如何分工 + 包级地图  
2. `01-scheduling-flow` — 只讲控制面决策  
3. `02-compute-flow` — 只讲 GPU 执行  
4. 对照 [`DESIGN.md`](../DESIGN.md) 与 `src/` 目录，再进代码
