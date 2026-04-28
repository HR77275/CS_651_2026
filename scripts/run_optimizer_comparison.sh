#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

PYTHON_BIN="${PYTHON_BIN:-/mnt/c/Users/HIMANSHU/Desktop/langchain/practice/myenv/Scripts/python.exe}"

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "Python interpreter not found or not executable: $PYTHON_BIN"
  exit 1
fi

if ! "$PYTHON_BIN" -c "import yaml, tqdm" >/dev/null 2>&1; then
  echo "The configured interpreter is missing required packages (pyyaml, tqdm)."
  exit 1
fi

if ! "$PYTHON_BIN" -c "import lion_pytorch" >/dev/null 2>&1; then
  echo "The configured interpreter is missing lion-pytorch, which is required for the Lion run."
  echo "Install it with: $PYTHON_BIN -m pip install lion-pytorch"
  exit 1
fi

configs=(
  "configs/deeplabv3_resnet50_voc_adam_50ep.yaml"
  "configs/deeplabv3_resnet50_voc_adamw_50ep.yaml"
  "configs/deeplabv3_resnet50_voc_lion_50ep.yaml"
)

for cfg in "${configs[@]}"; do
  echo "============================================================"
  echo "Starting run for: $cfg"
  echo "Using interpreter: $PYTHON_BIN"
  echo "============================================================"
  "$PYTHON_BIN" scripts/train.py --config "$cfg"
  echo
  echo "Completed run for: $cfg"
  echo
  sleep 1
done
