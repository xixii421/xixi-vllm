#!/usr/bin/env python3
from __future__ import annotations

import ast
import io
import re
import tokenize
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src"


def strip_file(path: Path) -> None:
    source = path.read_text()
    try:
        tree = ast.parse(source)
    except SyntaxError as e:
        print(f"skip parse {path}: {e}")
        return

    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Module)):
            if (
                node.body
                and isinstance(node.body[0], ast.Expr)
                and isinstance(getattr(node.body[0], "value", None), ast.Constant)
                and isinstance(node.body[0].value.value, str)
            ):
                node.body.pop(0)
                if not node.body:
                    node.body.append(ast.Pass())

    try:
        code = ast.unparse(tree)
    except Exception as e:
        print(f"skip unparse {path}: {e}")
        return

    out: list[str] = []
    prev_end = [1, 0]
    tokens = list(tokenize.tokenize(io.BytesIO(code.encode()).readline))
    for i, tok in enumerate(tokens):
        if tok.type in (tokenize.COMMENT, tokenize.ENCODING, tokenize.NL):
            if tok.type == tokenize.NL:
                out.append("\n")
                prev_end = list(tok.end)
            continue
        if tok.type == tokenize.NEWLINE:
            out.append("\n")
            prev_end = list(tok.end)
            continue
        line = code.splitlines(True)
        if tok.start[0] > prev_end[0]:
            for ln in range(prev_end[0], tok.start[0] - 1):
                out.append("\n")
            if tok.start[0] - 1 < len(line):
                prefix = line[tok.start[0] - 1][: tok.start[1]]
                out.append(prefix)
        elif tok.start[1] > prev_end[1] and tok.start[0] <= len(line):
            out.append(line[tok.start[0] - 1][prev_end[1] : tok.start[1]])
        out.append(tok.string)
        prev_end = list(tok.end)

    result = "".join(out)
    result = re.sub(r"\n{3,}", "\n\n", result).strip() + "\n"
    path.write_text(result)
    print("stripped", path.relative_to(SRC.parent))


def main() -> None:
    for path in sorted(SRC.rglob("*.py")):
        if "__pycache__" in path.parts:
            continue
        strip_file(path)


if __name__ == "__main__":
    main()
