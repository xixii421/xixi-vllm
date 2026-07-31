import torch
import torch.nn as nn
from annotate import profile  

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(10, 20)
        self.layer2 = nn.Linear(20, 1)

    # 1. 使用装饰器标记并自定义名称
    @profile("Feature Extractor")
    def extract_features(self, x):
        return torch.relu(self.layer1(x))

    # 2. 默认使用函数名 "classify" 作为 scope 名称
    @profile() 
    def classify(self, x):
        return self.layer2(x)

    def forward(self, x):
        x = self.extract_features(x)
        return self.classify(x)

model = MyModel()
dummy_input = torch.randn(32, 10)

# 启动 PyTorch Profiler
with torch.profiler.profile(
    activities=[torch.profiler.ProfilerActivity.CPU],
    record_shapes=True,
) as prof:
    
    # 执行模型，被装饰的函数会被自动追踪
    output = model(dummy_input)

# 打印 Profiler 结果
print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))

# 可选：导出为 Trace 格式，用于可视化分析
prof.export_chrome_trace("trace.json")