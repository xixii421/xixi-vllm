# micro

组件与通信微基准，**不进入** `run_matrix` 主路径。

| 脚本 | 作用 |
| --- | --- |
| `bench_bandwidth.py` | 卡间 P2P + NCCL；支持 `--pairs PIX:0,1;PXB:2,3` 多拓扑 |
| `bench_linear.py` 等 | 算子 / 词表 / 对照实验（归档） |

主吞吐请用上级 `bench_v4.py` / `bench_vllm.py` / `run_matrix.py`。  
服务器通信文档：[docs/02-服务器通信测量.md](../docs/02-服务器通信测量.md)。

```bash
# 单对（工作站 / 指定两卡）
python bench/micro/bench_bandwidth.py --devices 0,1

# 四卡按拓扑分类（本机 A100）
python bench/micro/bench_bandwidth.py --pairs PIX:0,1;PXB:2,3;SYS:0,2 --skip-nccl
```
