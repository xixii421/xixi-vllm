#!/usr/bin/env bash
# 将 docs 下 Markdown 导出为带图片 / 公式 / 列表的 PDF。
#
# 依赖：pandoc、google-chrome（或 chromium）
# 公式：优先 docs/_vendor/katex 或系统 KaTeX，否则 MathML（离线）
#
# 用法：
#   bash docs/export_pdf.sh
#   bash docs/export_pdf.sh docs/00-MoE概述.md
#   bash docs/export_pdf.sh docs/00-MoE概述.md /tmp/out.pdf

set -euo pipefail

DOCS="$(cd "$(dirname "$0")" && pwd)"

SRC="${1:-$DOCS/00-MoE概述.md}"
if [[ ! -f "$SRC" ]]; then
  echo "missing markdown: $SRC" >&2
  exit 1
fi
SRC="$(cd "$(dirname "$SRC")" && pwd)/$(basename "$SRC")"
BASE="$(basename "$SRC" .md)"
OUT_PDF="${2:-$DOCS/${BASE}.pdf}"

TMPDIR="$(mktemp -d /tmp/md2pdf.XXXXXX)"
trap 'rm -rf "$TMPDIR"' EXIT
MD_PRE="$TMPDIR/pre.md"
HTML="$TMPDIR/out.html"

CHROME=""
for c in google-chrome-stable google-chrome chromium chromium-browser; do
  if command -v "$c" >/dev/null 2>&1; then
    CHROME="$c"
    break
  fi
done
if [[ -z "$CHROME" ]]; then
  echo "need google-chrome or chromium" >&2
  exit 1
fi
if ! command -v pandoc >/dev/null 2>&1; then
  echo "need pandoc" >&2
  exit 1
fi

MATH_ARGS=(--mathml)
VT=5000
KATEX_DIR=""
if [[ -f "$DOCS/_vendor/katex/katex.min.js" ]]; then
  KATEX_DIR="$DOCS/_vendor/katex"
elif [[ -f /usr/share/javascript/katex/katex.min.js ]]; then
  KATEX_DIR="/usr/share/javascript/katex"
else
  # 复用本机 VS Code / Cursor 自带的 KaTeX（无需再下载）
  KATEX_DIR="$(ls -d /home/*/.vscode-server/cli/servers/Stable-*/server/node_modules/katex/dist 2>/dev/null | head -1 || true)"
  if [[ -z "$KATEX_DIR" || ! -f "$KATEX_DIR/katex.min.js" ]]; then
    KATEX_DIR="$(ls -d /home/*/.cursor-server/cli/servers/Stable-*/server/node_modules/katex/dist 2>/dev/null | head -1 || true)"
  fi
fi
if [[ -n "$KATEX_DIR" && -f "$KATEX_DIR/katex.min.js" ]]; then
  MATH_ARGS=(--katex="$KATEX_DIR/")
  VT=20000
  echo "using katex: $KATEX_DIR"
else
  echo "using MathML (optional: put katex under docs/_vendor/katex/ for better glyphs)"
fi

# 预处理（即使 md 已规范，也对旧写法兜底）
python3 - "$SRC" "$MD_PRE" <<'PY'
import re, sys
src, dst = sys.argv[1], sys.argv[2]
text = open(src, encoding="utf-8").read()
text = re.sub(
    r"```math\s*\n(.*?)```",
    lambda m: "\n$$\n" + m.group(1).strip() + "\n$$\n",
    text,
    flags=re.S,
)

def trim_inline(m: re.Match) -> str:
    inner = m.group(1).strip()
    # 含中文的一定是误匹配（例如把「$E_i$ 是第 $i$」中间吞掉）
    if re.search(r"[\u4e00-\u9fff]", inner):
        return m.group(0)
    return f"${inner}$"

# 开口 $ 不能跟在 ASCII 字母/数字/反斜杠后；内容不得跨行（避免吞掉列表）
text = re.sub(
    r"(?<![A-Za-z0-9\\])\$[ \t]+([^$\n]+?)[ \t]+\$(?!\$)",
    trim_inline,
    text,
)
lines = text.splitlines()
out = []
for line in lines:
    if (
        re.match(r"^- ", line)
        and out
        and out[-1].strip() != ""
        and not re.match(r"^- ", out[-1])
        and not re.match(r"^\d+\. ", out[-1])
    ):
        out.append("")
    out.append(line)
text = "\n".join(out) + ("\n" if text.endswith("\n") or not text else "")
open(dst, "w", encoding="utf-8").write(text)
PY

pandoc "$MD_PRE" \
  -f markdown+tex_math_dollars+tex_math_single_backslash \
  -t html5 \
  --standalone \
  "${MATH_ARGS[@]}" \
  --embed-resources \
  --resource-path="$(dirname "$SRC")" \
  --metadata title="$BASE" \
  --css "$DOCS/export_pdf.css" \
  -o "$HTML"

"$CHROME" \
  --headless=new \
  --disable-gpu \
  --no-pdf-header-footer \
  --allow-file-access-from-files \
  --virtual-time-budget="$VT" \
  --run-all-compositor-stages-before-draw \
  --print-to-pdf="$OUT_PDF" \
  "file://$HTML" \
  >/dev/null 2>&1

echo "wrote $OUT_PDF ($(du -h "$OUT_PDF" | cut -f1))"
