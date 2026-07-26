import torch
import time

def test_serial_execution(size_mb: int, iterations: int = 20):
    device = torch.device("cuda:0")
    
    # 普通内存即可，因为后续是同步阻塞传输
    cpu_tensor = torch.randn(size_mb * 1024 * 1024 // 4, dtype=torch.float32) 
    
    start = time.perf_counter()
    for _ in range(iterations):
        # 1. CPU 计算任务
        _ = torch.sum(cpu_tensor ** 2) 
        # 2. 同步传输
        _ = cpu_tensor.to(device, non_blocking=False)
    torch.cuda.synchronize()
    end = time.perf_counter()
    
    return (end - start) / iterations

def test_overlap_execution(size_mb: int, iterations: int = 20):
    device = torch.device("cuda:0")
    # 必须使用 Pinned Memory，否则 non_blocking 会静默退化为同步
    cpu_tensor = torch.randn(size_mb * 1024 * 1024 // 4, dtype=torch.float32).pin_memory()
    
    start = time.perf_counter()
    for _ in range(iterations):
        # 1. 发起异步传输
        _ = cpu_tensor.to(device, non_blocking=True)
        
        # 2. CPU 计算任务 (此时 PCIe 传输与 CPU 计算在物理上并行)
        _ = torch.sum(cpu_tensor ** 2)
        
        # 3. 同步点：确保当前迭代的传输和计算都完成
        torch.cuda.synchronize() 
    end = time.perf_counter()
    
    return (end - start) / iterations

if __name__ == "__main__":
    if not torch.cuda.is_available():
        print("错误: 未检测到 CUDA 设备，请在 GPU 环境下运行此脚本。")
        exit(1)

    DATA_SIZE_MB = 300
    ITERATIONS = 30

    print("=" * 65)
    print(f" CUDA Graph Non-Blocking 传输重叠 (Overlap) 性能测试")
    print(f" 数据大小: {DATA_SIZE_MB} MB | 迭代次数: {ITERATIONS} 次 (取平均值)")
    print("=" * 65)

    # 预热 GPU 与 CPU 缓存
    _ = torch.randn(1000, device="cuda")
    _ = torch.sum(torch.randn(1000) ** 2)
    torch.cuda.synchronize()

    # 执行测试
    time_serial = test_serial_execution(DATA_SIZE_MB, ITERATIONS)
    time_overlap = test_overlap_execution(DATA_SIZE_MB, ITERATIONS)

    # 规范化输出
    print("\n测试结果对比 (相同工作负载):")
    print("-" * 65)
    print(f" [串行模式 (Blocking)]     : {time_serial * 1000:8.2f} ms / 次  (T_cpu + T_pcie)")
    print(f" [并行模式 (Non-blocking)] : {time_overlap * 1000:8.2f} ms / 次  (max(T_cpu, T_pcie))")
    print("-" * 65)
    