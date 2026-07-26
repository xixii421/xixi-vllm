"""RMSNorm 性能对比测试：v2 (PyTorch eager) vs v3 (Triton fused) vs 官方 torch rms_norm

测试维度:
- hidden_size: Qwen3 常见尺寸 (2048, 4096)
- num_tokens: batch size 变化 (1, 32, 128, 512, 2048, 8192)
- with_residual: 是否带残差加法融合
- dtype: float16 / bfloat16

输出: 延迟 (us) + 加速比 + 相对 torch 官方的最大绝对误差 / 相对误差
"""

from __future__ import annotations

import importlib
import itertools
import sys
import time
from collections import OrderedDict
from pathlib import Path

import torch
import torch.nn.functional as F

# ---- 将 v3 src 加入路径，直接加载 layernorm 模块（绕过 __init__.py 的 Attention 依赖） ----
V3_SRC = Path(__file__).resolve().parents[2] / "src"
sys.path.insert(0, str(V3_SRC))
spec = importlib.util.spec_from_file_location(
    "v3_layernorm", V3_SRC / "layers" / "layernorm.py"
)
v3_layernorm = importlib.util.module_from_spec(spec)
spec.loader.exec_module(v3_layernorm)
RMSNormV3 = v3_layernorm.RMSNorm


# ---- v2 版 RMSNorm (PyTorch eager, 完全复制自 vllm-v2) ----
class RMSNormV2(torch.nn.Module):
    """v2 纯 PyTorch 实现: 多个独立 kernel，逐元素操作。"""

    def __init__(self, hidden_size: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, x: torch.Tensor, residual: torch.Tensor | None = None):
        orig_dtype = x.dtype
        x32 = x.float()
        if residual is not None:
            x32 = x32 + residual.float()
            residual = x32.to(orig_dtype)
        out = (
            x32 * torch.rsqrt(x32.pow(2).mean(dim=-1, keepdim=True) + self.eps) * self.weight
        ).to(orig_dtype)
        return (out, residual) if residual is not None else out


# ---- 官方参考实现 ----
def ref_rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    """使用 PyTorch 内置 rms_norm 作为参考输出 (PyTorch 2.1+)."""
    # torch.nn.functional.rms_norm 需要 x 和 weight 的最后一维匹配
    return F.rms_norm(x.float(), (weight.shape[0],), weight.float(), eps).to(x.dtype)


def ref_rms_norm_with_residual(
    x: torch.Tensor, residual: torch.Tensor, weight: torch.Tensor, eps: float
) -> tuple[torch.Tensor, torch.Tensor]:
    """带残差的参考: x + residual → rms_norm → out."""
    x32 = x.float() + residual.float()
    residual_out = x32.to(x.dtype)
    out = F.rms_norm(x32, (weight.shape[0],), weight.float(), eps).to(x.dtype)
    return out, residual_out


# ---- 精度 / 误差计算 ----
def max_abs_error(a: torch.Tensor, b: torch.Tensor) -> float:
    return (a.float() - b.float()).abs().max().item()


def max_rel_error(a: torch.Tensor, b: torch.Tensor) -> float:
    denom = b.float().abs().max().item()
    if denom == 0:
        return 0.0
    return max_abs_error(a, b) / denom


# ---- 性能测量 ----
def bench(
    fn,
    *args,
    warmup: int = 20,
    repeat: int = 100,
    sync_every: int = 1,
) -> float:
    """微基准: 预热后取 repeat 次的最短耗时 (us)."""
    # warmup
    for _ in range(warmup):
        fn(*args)
    torch.cuda.synchronize()

    times: list[float] = []
    for i in range(repeat):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        fn(*args)
        if (i + 1) % sync_every == 0:
            torch.cuda.synchronize()
        t1 = time.perf_counter()
        times.append(t1 - t0)
    torch.cuda.synchronize()
    return min(times) * 1e6  # us


# ---- 主测试 ----
def run_benchmark():
    device = torch.device("cuda:0")
    HIDDEN_SIZES = [2048, 4096]
    TOKEN_COUNTS = [1, 32, 128, 512, 2048]
    DTYPES = [torch.float16, torch.bfloat16]
    EPS = 1e-6
    WARMUP = 30
    REPEAT = 200

    results: list[dict] = []

    for hs, nt, dt in itertools.product(HIDDEN_SIZES, TOKEN_COUNTS, DTYPES):
        # 构造输入
        x = torch.randn(nt, hs, device=device, dtype=dt)
        residual = torch.randn(nt, hs, device=device, dtype=dt)

        # ---- 无残差 ----
        m_v2 = RMSNormV2(hs, EPS).to(device=device, dtype=dt)
        m_v3 = RMSNormV3(hs, EPS).to(device=device, dtype=dt)

        # 预热 + 对齐权重
        m_v3.weight.data.copy_(m_v2.weight.data)
        w = m_v2.weight.data.clone()

        # 正确性验证 (v2 / v3 分别与官方参考对比)
        with torch.no_grad():
            ref_out = ref_rms_norm(x, w, EPS)
            v2_out = m_v2(x)
            v3_out = m_v3(x)

        abs_err_v2 = max_abs_error(v2_out, ref_out)
        rel_err_v2 = max_rel_error(v2_out, ref_out)
        abs_err_v3 = max_abs_error(v3_out, ref_out)
        rel_err_v3 = max_rel_error(v3_out, ref_out)

        # 性能
        t_v2 = bench(lambda: m_v2(x), warmup=WARMUP, repeat=REPEAT)
        t_v3 = bench(lambda: m_v3(x), warmup=WARMUP, repeat=REPEAT)
        speedup = t_v2 / t_v3 if t_v3 > 0 else float("inf")

        results.append(
            {
                "dtype": str(dt).split(".")[-1],
                "hidden_size": hs,
                "num_tokens": nt,
                "residual": False,
                "v2_us": t_v2,
                "v3_us": t_v3,
                "speedup": speedup,
                "v2_abs_err": abs_err_v2,
                "v2_rel_err": rel_err_v2,
                "v3_abs_err": abs_err_v3,
                "v3_rel_err": rel_err_v3,
            }
        )

        # ---- 带残差 ----
        with torch.no_grad():
            ref_out, ref_res = ref_rms_norm_with_residual(x, residual, w, EPS)
            v2_out, v2_res = m_v2(x, residual.clone())
            v3_out, v3_res = m_v3(x, residual.clone())

        abs_err_v2_out = max_abs_error(v2_out, ref_out)
        abs_err_v2_res = max_abs_error(v2_res, ref_res)
        abs_err_v3_out = max_abs_error(v3_out, ref_out)
        abs_err_v3_res = max_abs_error(v3_res, ref_res)

        rel_err_v2_out = max_rel_error(v2_out, ref_out)
        rel_err_v3_out = max_rel_error(v3_out, ref_out)

        t_v2_res = bench(
            lambda: m_v2(x, residual.clone()), warmup=WARMUP, repeat=REPEAT
        )
        t_v3_res = bench(
            lambda: m_v3(x, residual.clone()), warmup=WARMUP, repeat=REPEAT
        )
        speedup_res = t_v2_res / t_v3_res if t_v3_res > 0 else float("inf")

        results.append(
            {
                "dtype": str(dt).split(".")[-1],
                "hidden_size": hs,
                "num_tokens": nt,
                "residual": True,
                "v2_us": t_v2_res,
                "v3_us": t_v3_res,
                "speedup": speedup_res,
                "v2_abs_err": max(abs_err_v2_out, abs_err_v2_res),
                "v2_rel_err": max(rel_err_v2_out, 0),
                "v3_abs_err": max(abs_err_v3_out, abs_err_v3_res),
                "v3_rel_err": max(rel_err_v3_out, 0),
            }
        )

    del m_v2, m_v3
    torch.cuda.empty_cache()

    return results


def format_markdown(results: list[dict]) -> str:
    lines: list[str] = []

    # 按 hidden_size 分组
    by_hs = OrderedDict()
    for r in results:
        by_hs.setdefault(r["hidden_size"], []).append(r)

    for hs, group in by_hs.items():
        lines.append(f"## hidden_size = {hs}")
        lines.append("")

        # ---- 无残差 ----
        lines.append("### 无残差 (residual=None)")
        lines.append("")
        lines.append(
            "| dtype | num_tokens | v2 (us) | v3 (us) | **加速比** | "
            "v2 绝对误差 | v2 相对误差 | v3 绝对误差 | v3 相对误差 |"
        )
        lines.append(
            "|-------|-----------|---------|---------|-----------|"
            "------------|------------|------------|------------|"
        )

        for r in group:
            if r["residual"]:
                continue
            speedup = f"**{r['speedup']:.2f}×**"
            lines.append(
                f"| {r['dtype']} | {r['num_tokens']:>5d} "
                f"| {r['v2_us']:7.1f} | {r['v3_us']:7.1f} "
                f"| {speedup} "
                f"| {r['v2_abs_err']:.2e} | {r['v2_rel_err']:.2e} "
                f"| {r['v3_abs_err']:.2e} | {r['v3_rel_err']:.2e} |"
            )
        lines.append("")

        # ---- 带残差 ----
        lines.append("### 带残差 (residual ≠ None)")
        lines.append("")
        lines.append(
            "| dtype | num_tokens | v2 (us) | v3 (us) | **加速比** | "
            "v2 绝对误差 | v2 相对误差 | v3 绝对误差 | v3 相对误差 |"
        )
        lines.append(
            "|-------|-----------|---------|---------|-----------|"
            "------------|------------|------------|------------|"
        )

        for r in group:
            if not r["residual"]:
                continue
            speedup = f"**{r['speedup']:.2f}×**"
            lines.append(
                f"| {r['dtype']} | {r['num_tokens']:>5d} "
                f"| {r['v2_us']:7.1f} | {r['v3_us']:7.1f} "
                f"| {speedup} "
                f"| {r['v2_abs_err']:.2e} | {r['v2_rel_err']:.2e} "
                f"| {r['v3_abs_err']:.2e} | {r['v3_rel_err']:.2e} |"
            )
        lines.append("")

    return "\n".join(lines)


def print_summary(results: list[dict]) -> None:
    """控制台汇总: 加速比统计."""
    speeds_no_res = [r["speedup"] for r in results if not r["residual"]]
    speeds_with_res = [r["speedup"] for r in results if r["residual"]]
    max_abs_v3 = max(r["v3_abs_err"] for r in results)
    max_rel_v3 = max(r["v3_rel_err"] for r in results)

    print("=" * 60)
    print("  RMSNorm v2 vs v3 性能汇总")
    print("=" * 60)
    if speeds_no_res:
        print(f"  无残差: 加速比 {min(speeds_no_res):.2f}× ~ {max(speeds_no_res):.2f}× "
              f"(均值 {sum(speeds_no_res)/len(speeds_no_res):.2f}×)")
    if speeds_with_res:
        print(f"  带残差: 加速比 {min(speeds_with_res):.2f}× ~ {max(speeds_with_res):.2f}× "
              f"(均值 {sum(speeds_with_res)/len(speeds_with_res):.2f}×)")
    print(f"  v3 最大绝对误差: {max_abs_v3:.2e}  (相对: {max_rel_v3:.2e})")
    print("=" * 60)


def main() -> None:
    if not torch.cuda.is_available():
        print("CUDA 不可用，请在 GPU 环境下运行。")
        sys.exit(1)

    # 全局预热
    _ = torch.randn(1000, device="cuda:0")
    torch.cuda.synchronize()

    results = run_benchmark()
    print_summary(results)

    md = format_markdown(results)

    out_dir = Path(__file__).resolve().parent
    out_path = out_dir / "rmsnorm_bench_result.md"
    out_path.write_text(md, encoding="utf-8")
    print(f"\nMarkdown 结果已写入: {out_path}")


if __name__ == "__main__":
    main()
