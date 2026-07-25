"""Merge v4 + official JSON results into a markdown performance report."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _fmt_ms(x) -> str:
    if x is None:
        return "n/a"
    return f"{float(x) * 1000:.1f}"


def _fmt_f(x, nd=1) -> str:
    if x is None:
        return "n/a"
    return f"{float(x):.{nd}f}"


def _token_agree(a: list[int] | None, b: list[int] | None) -> str:
    if not a or not b:
        return "n/a (missing ids)"
    n = min(len(a), len(b))
    if n == 0:
        return "empty"
    match = sum(1 for i in range(n) if a[i] == b[i])
    return f"{match}/{n} prefix match ({100.0 * match / n:.1f}%), len v4={len(a)} off={len(b)}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--v4", required=True)
    ap.add_argument("--official", required=True)
    ap.add_argument("--out", default="bench/results/mm_compare_report.md")
    args = ap.parse_args()

    v4 = json.loads(Path(args.v4).read_text())
    off = json.loads(Path(args.official).read_text())

    lines: list[str] = []
    lines.append("# vllm-v4 vs Official vLLM — Multimodal Performance")
    lines.append("")
    lines.append(f"- Model: `{v4.get('model')}`")
    lines.append(f"- Image: `{v4.get('image')}`")
    lines.append(f"- max_new: {v4.get('max_new')}")
    lines.append(f"- Device: {v4.get('device')}")
    lines.append(f"- Official vLLM: {off.get('vllm_version')}")
    lines.append("")
    lines.append("## Method notes (no sudo / no nvidia-smi)")
    lines.append("")
    lines.append(
        "NVML driver mismatch only breaks `nvidia-smi`; CUDA compute still works. "
        "All timings use `torch.cuda.synchronize()` + `time.perf_counter()`. "
        "Official API does not expose ViT vs LM-prefill splits, so its `ttft` is "
        "`encode+prefill+1 token`; v4 reports an explicit `vit_encode` / `lm_prefill` split."
    )
    lines.append("")

    for case in ("mm_image", "text_only"):
        a = v4["cases"].get(case, {})
        b = off["cases"].get(case, {})
        if not a or not b:
            continue
        lines.append(f"## Case: `{case}`")
        lines.append("")
        lines.append("| Metric | vllm-v4 | Official vLLM |")
        lines.append("| --- | ---: | ---: |")
        lines.append(f"| TTFT (ms) | {_fmt_ms(a.get('ttft_s'))} | {_fmt_ms(b.get('ttft_s'))} |")
        lines.append(
            f"| ViT encode (ms) | {_fmt_ms(a.get('vit_encode_s'))} | {_fmt_ms(b.get('vit_encode_s'))} |"
        )
        lines.append(
            f"| LM prefill (ms) | {_fmt_ms(a.get('lm_prefill_s'))} | {_fmt_ms(b.get('lm_prefill_s'))} |"
        )
        lines.append(
            f"| Prefill wall (ms) | {_fmt_ms(a.get('prefill_s'))} | {_fmt_ms(b.get('prefill_s'))} |"
        )
        lines.append(
            f"| Decode (tok/s) | {_fmt_f(a.get('decode_tok_s'))} | {_fmt_f(b.get('decode_tok_s'))} |"
        )
        lines.append(
            f"| E2E (tok/s) | {_fmt_f(a.get('e2e_tok_s'))} | {_fmt_f(b.get('e2e_tok_s'))} |"
        )
        lines.append(
            f"| Peak mem (GiB) | {_fmt_f(a.get('peak_mem_gb'), 2)} | {_fmt_f(b.get('peak_mem_gb'), 2)} |"
        )
        lines.append(
            f"| Prompt tokens | {_fmt_f(a.get('prompt_tokens_mean'), 0)} | {_fmt_f(b.get('prompt_tokens_mean'), 0)} |"
        )
        lines.append("")
        lines.append("**Output sample (v4):**")
        lines.append("")
        lines.append(f"> {a.get('text_sample', '').strip()}")
        lines.append("")
        lines.append("**Output sample (official):**")
        lines.append("")
        lines.append(f"> {b.get('text_sample', '').strip()}")
        lines.append("")
        lines.append(
            f"**Greedy token agreement:** {_token_agree(a.get('gen_ids_sample'), b.get('gen_ids_sample'))}"
        )
        lines.append("")

    lines.append("## How to read the numbers")
    lines.append("")
    lines.append(
        "1. **TTFT**: multimodal first-token latency; dominated by ViT + LM prefill. "
        "v4's `vit_encode` tells you how much of TTFT is vision."
    )
    lines.append(
        "2. **Decode tok/s**: steady-state generation; should be close if both use "
        "FlashAttention/FlashInfer + CUDA Graph on the same GPU."
    )
    lines.append(
        "3. **text_only**: isolates LM path; if text decode matches but MM TTFT differs, "
        "the gap is in the vision/encoder path."
    )
    lines.append(
        "4. **Token agreement**: greedy (`temperature=0`). Exact match is ideal; "
        "small prefix divergence can still be OK if chat templates / image resize differ."
    )
    lines.append("")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {out}")
    print("\n".join(lines[:40]))


if __name__ == "__main__":
    main()
