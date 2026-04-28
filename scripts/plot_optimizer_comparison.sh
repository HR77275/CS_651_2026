#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

PYTHON_BIN="${PYTHON_BIN:-/mnt/c/Users/HIMANSHU/Desktop/langchain/practice/myenv/Scripts/python.exe}"

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "Python interpreter not found or not executable: $PYTHON_BIN"
  exit 1
fi

"$PYTHON_BIN" scripts/plot_optimizer_comparison.py "$@"
