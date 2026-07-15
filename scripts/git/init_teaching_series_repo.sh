#!/usr/bin/env bash
# Initialize a Git repo that tracks ONLY vllm-v0 … vllm-v6 (+ series docs).
#
# Safe w.r.t. nested repos:
#   - Does NOT touch dzyy_vllm/.git, ullm/.git, …
#   - Parent directory may not already be a Git repo
#
# Usage:
#   cd /path/to/dzyy-vllm
#   bash scripts/git/init_teaching_series_repo.sh
#   bash scripts/git/init_teaching_series_repo.sh --commit   # also create first commit

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

DO_COMMIT=0
if [[ "${1:-}" == "--commit" ]]; then
  DO_COMMIT=1
fi

if [[ -d "$ROOT/dzyy_vllm/.git" ]]; then
  echo "[ok] nested repo dzyy_vllm/.git will be ignored (not touched)"
fi

if [[ -d "$ROOT/.git" ]]; then
  echo "[info] $ROOT is already a git repo"
else
  git init -b main
  echo "[ok] git init -b main at $ROOT"
fi

# Ensure ignore file is present
if [[ ! -f "$ROOT/.gitignore" ]]; then
  echo "[error] missing .gitignore at repo root" >&2
  exit 1
fi

# Prefer allow-list adds (never `git add -A` at root — sibling trees exist)
git add .gitignore README.md
git add scripts/git/
git add vllm-v0 vllm-v1 vllm-v2 vllm-v3 vllm-v4 vllm-v5 vllm-v6

# Drop accidental junk / symlink envs / non-git helper scripts
git rm -r --cached --ignore-unmatch \
  '*/.venv' '*/venv' '*/__pycache__' '*/bench/results' 2>/dev/null || true
# If anything under scripts/ except scripts/git leaked into the index, drop it
while IFS= read -r f; do
  case "$f" in
    scripts/git/*) ;;
    *) git rm -f --cached -- "$f" 2>/dev/null || true ;;
  esac
done < <(git ls-files 'scripts/*')

echo
echo "=== git status (should list only series files) ==="
git status -sb | head -80
echo "..."
echo "tracked path count: $(git ls-files | wc -l)"
echo
echo "sample tracked paths:"
git ls-files | head -30

# Sanity: nothing from foreign trees
BAD=$(git ls-files | grep -E '^(dzyy_vllm|ullm|nano-vllm|MinivLLM|Sparse-vLLM|vllm)/' || true)
if [[ -n "$BAD" ]]; then
  echo "[error] foreign paths are staged:" >&2
  echo "$BAD" >&2
  exit 1
fi

VENV=$(git ls-files | grep -E '/\.venv/' || true)
if [[ -n "$VENV" ]]; then
  echo "[error] .venv paths are staged:" >&2
  echo "$VENV" >&2
  exit 1
fi

echo
echo "[ok] staging looks clean (only teaching series)"

if [[ "$DO_COMMIT" -eq 1 ]]; then
  if git rev-parse HEAD >/dev/null 2>&1 && [[ -z "$(git status --porcelain)" ]]; then
    echo "[info] nothing to commit"
  else
    git commit -m "$(cat <<'EOF'
Initial commit: teaching mini-vLLM series (vllm-v0 … vllm-v6).

Track code and docs only; ignore envs, caches, bench results, and sibling projects.
EOF
)"
    echo "[ok] created initial commit: $(git rev-parse --short HEAD)"
  fi
else
  echo
  echo "Next:"
  echo "  git commit   # or re-run with --commit"
  echo "  git remote add origin <NEW-empty-repo-url>"
  echo "  git push -u origin main"
  echo
  echo "Do NOT force-push onto the existing dzyy_vllm remote history unless you intend to."
fi
