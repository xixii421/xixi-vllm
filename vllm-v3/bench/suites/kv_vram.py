"""多请求 KV 显存对比：真实 Prefill 写入共享分页池。

与旧版 multi_req_vram 的区别
----------------------------
旧版：在 Engine 之外另 ``torch.zeros`` 两块假张量比大小 —— **不是**推理路径上的测试。
本版：

1. 按场景把共享池预留到「本跑所需页数」；
2. 对 R 个请求**依次真实 Prefill**，页表同时占用、KV 写进 ``PagedKVCache``；
3. 再对各请求做 1 步 Decode，确认 gather 路径可读；
4. 「连续对照」= R×M×每token字节（估算标注清楚：若每请求按满上下文预留）。

输出列均为中文。
"""

from __future__ import annotations

from bench.lib.config import BenchConfig
from bench.lib.report import ResultWriter, print_table
from bench.lib.session import BenchSession, collect_meta
from bench.lib.timing import sync


def default_scenarios(context_len: int) -> list[tuple[str, list[int]]]:
    """(中文场景名, 各请求真实长度 T_i)。"""
    raw = [
        ("短请求×8", [65, 130, 250, 500, 120, 99, 200, 390]),
        ("短请求×16", [48, 64, 80, 96, 112, 128, 144, 160,
                       176, 192, 208, 224, 240, 256, 280, 300]),
        ("混合长度×12", [128, 256, 384, 512, 768, 1024, 96, 160, 220, 400, 600, 900]),
        ("极短请求×24", [32 + (i % 7) * 17 for i in range(24)]),
    ]
    out: list[tuple[str, list[int]]] = []
    for name, lens in raw:
        clipped = [min(max(1, t), context_len) for t in lens]
        out.append((name, clipped))
    return out


def pages_for(lengths: list[int], block_size: int) -> int:
    return sum((t + block_size - 1) // block_size for t in lengths)


def needed_gpu_blocks(cfg: BenchConfig) -> int:
    """本套件跑完全部场景所需的最大页数（含 Decode+1 可能扩页，+余量）。"""
    B = cfg.block_size

    def peak_pages(lengths: list[int]) -> int:
        # Prefill 用 T_i；套件默认再 Decode 1 步 → 按 T_i+1 估页。
        return pages_for([t + 1 for t in lengths], B)

    if cfg.paged_lengths:
        return peak_pages(list(cfg.paged_lengths)) + 16
    scenarios = default_scenarios(cfg.context_len)
    need = max(peak_pages(lens) for _, lens in scenarios)
    return need + 16


def run(cfg: BenchConfig, writer: ResultWriter | None = None, session: BenchSession | None = None) -> dict:
    # 池按「场景最大占页」sizing，避免 max_num_seqs×M 虚高，也避免页不够。
    if cfg.num_gpu_blocks is None:
        cfg.num_gpu_blocks = needed_gpu_blocks(cfg)

    session = session or BenchSession(cfg)
    meta = collect_meta(cfg)
    writer = writer or ResultWriter(cfg, meta)

    engine = session.engine
    cache = engine.kv_cache
    B = engine.block_size
    M = cfg.context_len
    bpt = cache.bytes_per_token
    pool_mib = round(cache.pool_nbytes / (1024**2), 2)

    scenarios = default_scenarios(M)
    if cfg.paged_lengths:
        # 用户指定长度时只跑这一组，更快。
        scenarios = [("自定义并发", [min(t, M) for t in cfg.paged_lengths if t > 0])]

    rows: list[dict] = []
    table_rows: list[list] = []

    for name, lengths in scenarios:
        R = len(lengths)
        need_pages = pages_for(lengths, B)
        # Prefill 后再 Decode 1 步，长度变为 T_i+1，可能多借一页。
        need_pages_after = pages_for([t + 1 for t in lengths], B)
        real_tokens = sum(lengths)
        waste = need_pages * B - real_tokens
        paged_logic_mib = round(need_pages * B * bpt / (1024**2), 2)
        cont_mib = round(R * M * bpt / (1024**2), 2)
        ratio = round(cont_mib / paged_logic_mib, 2) if paged_logic_mib else float("inf")

        status = "成功"
        err = ""
        peak_gb = ""
        used_blocks = need_pages
        try:
            stats = session.multi_request_prefill(lengths, decode_steps=1)
            used_blocks = stats["used_blocks"]
            if used_blocks not in (need_pages, need_pages_after):
                status = "页数异常"
                err = f"used={used_blocks} expect={need_pages}或{need_pages_after}"
            peak_gb = stats["peak_gb"]
        except Exception as exc:  # noqa: BLE001 — bench 要记失败原因
            status = "失败"
            err = str(exc)[:120]
        finally:
            session.clear_kv_state()

        row = {
            "场景": name,
            "请求数R": R,
            "真实token合计": real_tokens,
            "占用页数": used_blocks,
            "内部碎片token": waste,
            "分页逻辑MiB": paged_logic_mib,
            "连续对照MiB": cont_mib,
            "连续相对分页倍率": ratio,
            "物理共享池MiB": pool_mib,
            "峰值显存GB": peak_gb,
            "Prefill+Decode": status,
            "备注": err,
            "各请求长度": "|".join(str(x) for x in lengths),
            "context_len_M": M,
            "block_size_B": B,
            "bytes_per_token": bpt,
        }
        rows.append(row)
        table_rows.append([
            name,
            R,
            real_tokens,
            used_blocks,
            paged_logic_mib,
            cont_mib,
            f"{ratio}×",
            status,
        ])

    fields = list(rows[0].keys()) if rows else []
    writer.write_csv("kv_vram", rows, fields)

    footnote = (
        f"模型每token(K+V)={bpt} 字节｜最大上下文 M={M}｜页大小 B={B}｜"
        f"物理共享池={pool_mib} MiB（按本跑最大占页预分配）\n"
        "分页逻辑MiB = 占用页数×B×每token字节（本场景实际占页）。\n"
        "连续对照MiB = 请求数R×M×每token字节（若每请求按满上下文连续预留，估算）。\n"
        "Prefill+Decode=成功：R 路真实写入分页池，且每路完成 1 步 Decode gather。"
    )
    writer.append_report(
        "## 多请求 KV 显存（真实 Prefill）\n\n"
        + "| 场景 | 请求数R | 真实ΣT | 占页 | 分页逻辑MiB | 连续对照MiB | 倍率 | 状态 |\n"
        + "| --- | --- | --- | --- | --- | --- | --- | --- |\n"
        + "\n".join(
            f"| {r['场景']} | {r['请求数R']} | {r['真实token合计']} | {r['占用页数']} | "
            f"{r['分页逻辑MiB']} | {r['连续对照MiB']} | {r['连续相对分页倍率']}× | "
            f"{r['Prefill+Decode']} |"
            for r in rows
        )
        + f"\n\n{footnote}\n"
    )
    print_table(
        "多请求 KV 显存（真实 Prefill + 1 步 Decode）",
        ["场景", "请求数R", "真实ΣT", "占页", "分页逻辑MiB", "连续对照MiB", "倍率", "状态"],
        table_rows,
        footnote=footnote,
    )
    print(f"结果目录: {writer.latest_symlink_hint}")
    sync()
    return {"rows": rows, "pool_mib": pool_mib, "bytes_per_token": bpt}
